#!/usr/bin/env python3
"""Cross-family AsymSpec — heterogeneous drafter + verifier with different tokenizers.

Answers Reviewers qJdq / RXVL (#1): a small open drafter paired with a large
verifier from a *different* model family and vocabulary.

WHAT THIS IS, AND WHY IT LOOKS LIKE THIS
----------------------------------------
Our production method (vllm_specsteer/v0.10.mm) is *speculative*: the drafter
proposes K tokens and the verifier scores them in one parallel forward. That
requires the drafter's token ids to be valid verifier inputs and the two models
to tokenize text identically — both false across families. So the speculative
*speedup* cannot survive a vocabulary mismatch without UAG-style
detokenize/retokenize at every boundary (a systems change, orthogonal to the
method).

The *method* — the accuracy-recovery mechanism reviewers asked about — does
survive. We run it at K=1 (per-token δ-fusion, no speculation), which removes the
draft-token-into-verifier step entirely, and keep the v0.10.mm decision logic
byte-for-byte:

    delta      = a - b                       # context gain, in the drafter vocab
    D          = JSD(a, b)  or  KL/log|V|     # CDA divergence  (== specsteer_sampler)
    gamma_eff  = gamma * exp(-lambda * D)     # CDA gate        (== specsteer_sampler)
    d          = argmax(a)                    # drafter's full-context pick
    accept d  iff  p_llm[d] > gamma_eff * p_base[d]     # == specsteer_sampler
    else emit  argmax(t + beta * delta)                 # == specsteer_sampler

The ONLY cross-family addition is the vocabulary-intersection alignment (adapted
from the user's HF CoSteer script — the single reusable piece): a and b live in
the drafter's vocab, t in the verifier's; we project all three onto the shared
token set V_S ∩ V_L before the arithmetic above. Tokens outside the intersection
get no delta signal (delta=0 there), which we report as coverage.

Backend is HF (per-step full-vocab logits + KV cache) — not vLLM — because
per-token cross-vocab fusion needs arbitrary-token logits that vLLM's API does not
expose, and the accuracy numbers are engine-invariant. Output is a
`*_responses.jsonl` scored by the SAME `scripts/eval_lb.py` as every other
LongBench cell, so the cross-family accuracy is directly comparable.

Usage:
  python asym_crossfamily.py \
      --verifier zai-org/GLM-... --drafter Qwen/Qwen3-4B \
      --gamma 0.5 --beta 1.0 --method cma_vnorm \
      --dataset longbench --n 100 \
      --out cross_glm_qwen --responses cross_glm_qwen_responses.jsonl
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- alignment
def build_intersection(tok_L, tok_S, device_L, device_S):
    """Aligned (L_ids, S_ids): position i is the same token string in both vocabs.

    Adapted verbatim in spirit from the user's `_create_vocab_intersection_map`;
    this is the one reusable piece of the HF cross-family script.
    """
    vL, vS = tok_L.get_vocab(), tok_S.get_vocab()
    inter = sorted(set(vL) & set(vS))            # sorted -> deterministic
    L_ids = torch.tensor([vL[t] for t in inter], dtype=torch.long, device=device_L)
    S_ids = torch.tensor([vS[t] for t in inter], dtype=torch.long, device=device_S)
    cov_S = len(inter) / len(vS)
    cov_L = len(inter) / len(vL)
    print(f"[align] |V_S|={len(vS)} |V_L|={len(vL)} |inter|={len(inter)} "
          f"(S {100*cov_S:.0f}% / L {100*cov_L:.0f}%)", flush=True)
    return L_ids, S_ids, inter


# ----------------------------------------------------------------- CDA gate (== v0.10.mm)
def jsd(a_log, b_log):
    """Verbatim port of specsteer_sampler._jsd (bounded by ln2)."""
    p_a, p_b = a_log.exp(), b_log.exp()
    m = 0.5 * (p_a + p_b)
    log_m = m.clamp_min(1e-30).log()
    return (0.5 * ((p_a * (a_log - log_m)).sum(-1) + (p_b * (b_log - log_m)).sum(-1))).clamp_min(0.0)


def gamma_eff(a_log, b_log, gamma, method, Vsize):
    """γ_eff exactly as specsteer_greedy_sample computes it, per method."""
    if method == "gamma_rule":
        return torch.tensor(float(gamma), device=a_log.device)
    if method == "jsd":
        lam = float(os.environ.get("JSD_LAMBDA", "1.0"))
        return gamma * torch.exp(-lam * jsd(a_log[None], b_log[None])[0])
    if method == "cma":
        lam = float(os.environ.get("CMA_LAMBDA", "1.0"))
        kl = (a_log.exp() * (a_log - b_log)).sum(-1).clamp_min(0.0)
        return gamma * torch.exp(-lam * kl)
    if method == "cma_vnorm":                    # the headline CDA (λ = 1/log|V|)
        lam = 1.0 / math.log(Vsize)
        kl = (a_log.exp() * (a_log - b_log)).sum(-1).clamp_min(0.0)
        return gamma * torch.exp(-lam * kl)
    raise ValueError(method)


# ----------------------------------------------------------------- HF stepper
@torch.inference_mode()
def prefill(model, ids):
    out = model(ids, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values


@torch.inference_mode()
def step(model, tok_id, past):
    out = model(tok_id.view(1, 1), past_key_values=past, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values


@torch.inference_mode()
def generate_one(rec, mV, mS, tokV, tokS, L_ids, S_ids, Vsize,
                 gamma, beta, method, max_new, dev_V, dev_S):
    """Per-token cross-family AsymSpec. Returns the verifier-decoded answer."""
    # each model tokenizes its OWN view with its OWN tokenizer
    ids_V = tokV(rec["comp"], return_tensors="pt").input_ids.to(dev_V)   # verifier: x_comp
    ids_Sa = tokS(rec["full"], return_tensors="pt").input_ids.to(dev_S)  # drafter aug: x_full
    ids_Sb = tokS(rec["comp"], return_tensors="pt").input_ids.to(dev_S)  # drafter base: x_comp

    tV, pV = prefill(mV, ids_V)
    aS, pSa = prefill(mS, ids_Sa)
    bS, pSb = prefill(mS, ids_Sb)

    eos_V = tokV.eos_token_id
    out_L = []
    for _ in range(max_new):
        # project onto the shared token set (this is the whole cross-family delta).
        # verifier logits live on dev_V, drafter on dev_S; do all fusion arithmetic
        # on dev_S (move the projected verifier row over, like the HF CoSteer script
        # moved everything to one device).
        t = torch.log_softmax(tV.index_select(-1, L_ids), -1).to(dev_S)  # verifier, V∩
        a = torch.log_softmax(aS.index_select(-1, S_ids), -1)            # drafter full, V∩
        b = torch.log_softmax(bS.index_select(-1, S_ids), -1)            # drafter comp, V∩

        delta = a - b                                                 # == v0.10.mm ours
        g_eff = gamma_eff(a, b, gamma, method, Vsize)
        d = int(a.argmax())                                           # drafter's context pick
        p_llm, p_base = t[d].exp(), b[d].exp()
        if p_llm > g_eff * p_base:
            emit = d                                                 # accept
        else:
            emit = int((t + beta * delta).argmax())                  # δ-fuse

        eL = int(L_ids[emit]); eS = int(S_ids[emit])                 # back to real ids
        if eL == eos_V:
            break
        out_L.append(eL)
        # advance all three sequences by the emitted token (in each own vocab)
        tV, pV = step(mV, torch.tensor(eL, device=dev_V), pV)
        aS, pSa = step(mS, torch.tensor(eS, device=dev_S), pSa)
        bS, pSb = step(mS, torch.tensor(eS, device=dev_S), pSb)

    return tokV.decode(out_L, skip_special_tokens=True)


# ----------------------------------------------------------------- data (LongBench)
def load_longbench(n):
    raw = {}
    for ds in ["hotpotqa", "2wikimqa", "musique"]:
        for l in open(REPO / "data" / "longbench" / "raw" / f"{ds}.jsonl"):
            r = json.loads(l); raw[r["_id"]] = r
    summ = {json.loads(l)["_id"]: json.loads(l)["summary"]
            for l in open(REPO / "data" / "longbench" / "summaries.jsonl")}
    out = []
    for _id, s in summ.items():
        if _id not in raw:
            continue
        q = raw[_id].get("input", "")
        out.append({
            "_id": _id, "dataset": raw[_id].get("dataset", "hotpotqa"),
            "gold_answers": raw[_id].get("answers", raw[_id].get("gold_answers", [])),
            "full": raw[_id].get("context", "")[:48000] + f"\n\nQuestion: {q}\nAnswer:",
            "comp": s + f"\n\nQuestion: {q}\nAnswer:",
        })
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--method", default="jsd",
                    choices=["gamma_rule", "cma", "jsd", "cma_vnorm"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--verifier_gpu", type=int, default=0)
    ap.add_argument("--drafter_gpu", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--responses", required=True)
    a = ap.parse_args()
    dev_V, dev_S = f"cuda:{a.verifier_gpu}", f"cuda:{a.drafter_gpu}"

    print(f"[load] verifier {a.verifier}", flush=True)
    tokV = AutoTokenizer.from_pretrained(a.verifier, trust_remote_code=True)
    mV = AutoModelForCausalLM.from_pretrained(a.verifier, dtype=torch.bfloat16,
                                              trust_remote_code=True,
                                              attn_implementation="sdpa").to(dev_V).eval()
    print(f"[load] drafter {a.drafter}", flush=True)
    tokS = AutoTokenizer.from_pretrained(a.drafter, trust_remote_code=True)
    mS = AutoModelForCausalLM.from_pretrained(a.drafter, dtype=torch.bfloat16,
                                              trust_remote_code=True,
                                              attn_implementation="sdpa").to(dev_S).eval()

    L_ids, S_ids, inter = build_intersection(tokV, tokS, dev_V, dev_S)
    Vsize = len(inter)

    recs = load_longbench(a.n)
    # eval_lb.py --exp_dir scans k_indep/k2/k4/k6 for *_responses.jsonl
    outdir = REPO / "experiments" / a.out
    (outdir / "k2").mkdir(parents=True, exist_ok=True)
    rp = outdir / "k2" / a.responses
    n_done = 0
    with open(rp, "w") as f:
        for i, rec in enumerate(recs):
            ans = generate_one(rec, mV, mS, tokV, tokS, L_ids, S_ids, Vsize,
                               a.gamma, a.beta, a.method, a.max_new, dev_V, dev_S)
            f.write(json.dumps({"idx": i, "_id": rec["_id"], "dataset": rec["dataset"],
                                "gold_answers": rec["gold_answers"], "response": ans}) + "\n")
            f.flush()
            n_done += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(recs)} done", flush=True)

    meta = {"verifier": a.verifier, "drafter": a.drafter, "method": a.method,
            "gamma": a.gamma, "beta": a.beta, "K": 1, "per_token": True,
            "intersection_size": Vsize, "n": n_done, "responses": str(rp)}
    (outdir / f"{a.out}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[written] {rp}  ({n_done} responses)")
    print(f"[next] score with: python scripts/eval_lb.py --exp_dir experiments/{a.out}")


if __name__ == "__main__":
    main()
