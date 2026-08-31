"""MathVista multimodal benchmark for AsymSpec and its baselines.

Subset: GPS + VQA + FQA from testmini, restricted to PIDs that have official
Bard captions (587 samples after filter; 69 VQA dropped).

Cells:
  --cfg b   B1_aug_vl    : Qwen3-VL-2B alone, sees image + official query
  --cfg c   B1_main_cap  : Qwen3-32B alone, sees Bard caption + EasyOCR + query
  --cfg ss  SpecSteer    : verifier=Qwen3-32B (cap+ocr+q), drafter=Qwen3-VL (img+q)

Run order: prepare data → b/c (independent) → ss with K=2 then K=4.
Eval: regex extract per official MathVista guidance — letter for multi_choice,
last number for free_form. Compare to gold with normalize.
"""
from __future__ import annotations
import argparse, ast, io, json, os, re, sys, time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LLM_PATH as TEXT_PATH, VL_PATH, slm_model, MV_CAPTIONS, MV_OCRS

DRAFTER_TEXT_PATH = slm_model("1.7B")
CAPS = str(MV_CAPTIONS)
OCRS = str(MV_OCRS)
KEEP_TASKS = {"geometry problem solving", "visual question answering",
              "figure question answering"}
OUT_DIR = "outputs/mathvista"
CAP_TRUNC_CHARS = 1600   # ~400 tok p90
OCR_TRUNC_CHARS = 800
MAX_OUT_TOKENS = 1024
COT_SUFFIX = "\nLet's think step by step, then give the final answer."


def _ocr_text(s: str) -> str:
    if not s: return ""
    try:
        items = ast.literal_eval(s)
    except Exception:
        return ""
    return " ".join(t[1] for t in items
                    if isinstance(t, (list, tuple)) and len(t) >= 2 and isinstance(t[1], str))


def load_samples(n: int | None = None, seed: int = 0):
    """Load filtered MathVista testmini samples with caption + ocr texts."""
    from datasets import load_dataset
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    caps = json.load(open(CAPS))["texts"]
    ocrs = json.load(open(OCRS))["texts"]
    out = []
    for ex in ds:
        if ex["metadata"]["task"] not in KEEP_TASKS:
            continue
        pid = str(ex["pid"])
        if pid not in caps:
            continue
        out.append({
            "pid": pid,
            "task": ex["metadata"]["task"],
            "question_type": ex["question_type"],
            "answer_type": ex["answer_type"],
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": ex["answer"],
            "query": ex["query"],
            "image": ex["decoded_image"],
            "caption": caps[pid][:CAP_TRUNC_CHARS],
            "ocr": _ocr_text(ocrs.get(pid, ""))[:OCR_TRUNC_CHARS],
        })
    print(f"[load] filtered={len(out)} (GPS+VQA+FQA with caption)", flush=True)
    if n is not None and n < len(out):
        import random
        random.seed(seed)
        # stratified by task
        by = {}
        for ex in out: by.setdefault(ex["task"], []).append(ex)
        per = {t: max(1, round(n * len(lst) / len(out))) for t, lst in by.items()}
        # adjust to exactly n
        total = sum(per.values())
        diff = n - total
        keys = list(per.keys())
        i = 0
        while diff != 0:
            per[keys[i % len(keys)]] += (1 if diff > 0 else -1)
            diff += (-1 if diff > 0 else 1)
            i += 1
        sub = []
        for t, k in per.items():
            random.shuffle(by[t])
            sub.extend(by[t][:k])
        random.shuffle(sub)
        out = sub
        print(f"[load] subsampled n={len(out)} stratified by task: "
              f"{ {t: sum(1 for e in out if e['task']==t) for t in KEEP_TASKS} }",
              flush=True)
    return out


def build_main_user_text(s: dict) -> str:
    parts = [f"Image caption: {s['caption']}"]
    if s["ocr"].strip():
        parts.append(f"OCR text from image: {s['ocr']}")
    parts.append(s["query"] + COT_SUFFIX)
    return "\n\n".join(parts)


_VERIFIER_TOK = None
def _verifier_tok():
    global _VERIFIER_TOK
    if _VERIFIER_TOK is None:
        from transformers import AutoTokenizer
        _VERIFIER_TOK = AutoTokenizer.from_pretrained(TEXT_PATH,
                                                       trust_remote_code=True)
    return _VERIFIER_TOK


def build_main_prompt(s: dict) -> str:
    """Apply the Qwen3-32B chat template used by the text benchmarks."""
    msgs = [{"role": "user", "content": build_main_user_text(s)}]
    return _verifier_tok().apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)


# ---------- answer extraction (official-aligned) ----------
_LETTER_RE = re.compile(r"\b([A-F])\b")
_NUM_RE = re.compile(r"-?\d+\.?\d*")

def _extract_anchor(text: str) -> str:
    """Find the anchor segment that contains the actual answer.
    Priority: \\boxed{...} → Final answer: ... → answer is X / Answer: X →
              last non-empty line."""
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1]
    m = re.search(r"(?:final\s*answer\s*[:\-]?\s*)\*{0,2}([^\n*]+)\*{0,2}",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    m = re.search(r"(?:answer\s*(?:is|:)\s*)\*{0,2}([^\n*]+?)\*{0,2}\s*$",
                  text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip().rstrip(".")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def extract_pred(text: str, sample: dict) -> str:
    """Extract predicted answer per MathVista convention.
    Multi-choice → letter A-F. Free-form → number."""
    t = text.strip()
    anchor = _extract_anchor(t)
    if sample["question_type"] == "multi_choice":
        # 1. letter in anchor
        mm = _LETTER_RE.search(anchor)
        if mm:
            return mm.group(1).upper()
        # 2. anchor matches a choice text
        if sample["choices"]:
            for i, c in enumerate(sample["choices"]):
                if normalize(anchor).startswith(normalize(c)) or \
                   normalize(c) == normalize(anchor):
                    return chr(ord("A") + i)
        # 3. fallback: last letter in entire text
        all_letters = _LETTER_RE.findall(t)
        if all_letters:
            return all_letters[-1].upper()
        return anchor[:20]
    # free_form: number in anchor
    nums = _NUM_RE.findall(anchor) or _NUM_RE.findall(t.split("\n")[-1]) \
           or _NUM_RE.findall(t)
    if nums:
        v = nums[-1]
        try:
            f = float(v); return str(int(f)) if f.is_integer() else f"{f:g}"
        except Exception:
            return v
    return anchor[:30]


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", str(s).lower().strip().rstrip(".°"))


def is_correct(pred: str, sample: dict) -> bool:
    gold = sample["answer"]
    if sample["question_type"] == "multi_choice":
        # gold is the choice TEXT; map to letter via choices index
        if sample["choices"] and gold in sample["choices"]:
            letter = chr(ord("A") + sample["choices"].index(gold))
            return pred.upper() == letter
        return normalize(pred) == normalize(gold)
    return normalize(pred) == normalize(gold)


# ---------- cells ----------
def cfg_b_aug_vl(samples, out_path):
    print("\n=== CFG B: B1_aug_vl (Qwen3-VL-2B + image) ===")
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    llm = LLM(model=VL_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=16384, gpu_memory_utilization=0.55,
              enforce_eager=True, limit_mm_per_prompt={"image": 1})
    proc = AutoProcessor.from_pretrained(VL_PATH, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUT_TOKENS)
    outs = []
    n_out_total = 0
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": s["query"] + COT_SUFFIX}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
        try:
            r = llm.generate([{"prompt": prompt,
                               "multi_modal_data": {"image": s["image"]}}],
                             sp, use_tqdm=False)
            ans = r[0].outputs[0].text.strip()
            n_out_total += len(r[0].outputs[0].token_ids)
            pred = extract_pred(ans, s)
            ok = is_correct(pred, s)
            outs.append({"ans": ans, "pred": pred, "ok": ok})
        except Exception as e:
            print(f"  [{i}] FAIL: {type(e).__name__}: {str(e)[:200]}")
            outs.append({"ans": f"FAIL:{type(e).__name__}", "pred": "",
                         "ok": False})
        if (i + 1) % 25 == 0:
            n_ok = sum(o["ok"] for o in outs)
            el = time.perf_counter() - t0
            print(f"  [{i+1}/{len(samples)}] acc={100*n_ok/(i+1):.1f}% "
                  f"tps={n_out_total/el:.1f} el={el:.0f}s", flush=True)
    el = time.perf_counter() - t0
    metrics = {"tps": n_out_total / el if el else 0, "elapsed": el,
               "n_out_tokens_total": n_out_total}
    print(f"\n[B metrics] {metrics}")
    save_results("b1_aug_vl", samples, outs, out_path, spec_metrics=metrics)
    return outs


def cfg_c_main_cap(samples, out_path):
    print("\n=== CFG C: B1_main_cap (Qwen3-32B + caption + ocr) ===")
    from vllm import LLM, SamplingParams
    llm = LLM(model=TEXT_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=8192, gpu_memory_utilization=0.85,
              enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUT_TOKENS)
    outs = []
    n_out_total = 0
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        prompt = build_main_prompt(s)
        try:
            r = llm.generate([prompt], sp, use_tqdm=False)
            ans = r[0].outputs[0].text.strip()
            n_out_total += len(r[0].outputs[0].token_ids)
            pred = extract_pred(ans, s)
            ok = is_correct(pred, s)
            outs.append({"ans": ans, "pred": pred, "ok": ok})
        except Exception as e:
            print(f"  [{i}] FAIL: {type(e).__name__}: {str(e)[:200]}")
            outs.append({"ans": f"FAIL:{type(e).__name__}", "pred": "",
                         "ok": False})
        if (i + 1) % 25 == 0:
            n_ok = sum(o["ok"] for o in outs)
            el = time.perf_counter() - t0
            print(f"  [{i+1}/{len(samples)}] acc={100*n_ok/(i+1):.1f}% "
                  f"tps={n_out_total/el:.1f} el={el:.0f}s", flush=True)
    el = time.perf_counter() - t0
    metrics = {"tps": n_out_total / el if el else 0, "elapsed": el,
               "n_out_tokens_total": n_out_total}
    print(f"\n[C metrics] {metrics}")
    save_results("b1_main_cap", samples, outs, out_path, spec_metrics=metrics)
    return outs


def cfg_d_classical(samples, out_path, K=2):
    """Classical SpS: Qwen3-32B verifier + Qwen3-1.7B text drafter, both see
    main_cap (no image). Apples-to-apples speed comparison vs SS."""
    print(f"\n=== CFG D: classical SpS (Qwen3-32B + Qwen3-1.7B drafter) K={K} ===")
    from vllm import LLM, SamplingParams
    llm = LLM(model=TEXT_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=8192, gpu_memory_utilization=0.85,
              enforce_eager=True, disable_log_stats=False,
              speculative_config={
                  "method": "draft_model",
                  "model": DRAFTER_TEXT_PATH,
                  "num_speculative_tokens": K,
              })
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUT_TOKENS)
    outs = []
    n_out_total = 0
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        prompt = build_main_prompt(s)
        try:
            r = llm.generate([prompt], sp, use_tqdm=False)
            ans = r[0].outputs[0].text.strip()
            n_out_total += len(r[0].outputs[0].token_ids)
            pred = extract_pred(ans, s)
            ok = is_correct(pred, s)
            outs.append({"ans": ans, "pred": pred, "ok": ok})
        except Exception as e:
            print(f"  [{i}] FAIL: {type(e).__name__}: {str(e)[:200]}")
            outs.append({"ans": f"FAIL:{type(e).__name__}", "pred": "",
                         "ok": False})
        if (i + 1) % 25 == 0:
            n_ok = sum(o["ok"] for o in outs)
            el = time.perf_counter() - t0
            print(f"  [{i+1}/{len(samples)}] acc={100*n_ok/(i+1):.1f}% "
                  f"tps={n_out_total/el:.1f} el={el:.0f}s", flush=True)
    el = time.perf_counter() - t0
    metrics = {"K": K, "tps": n_out_total / el if el else 0, "elapsed": el,
               "n_out_tokens_total": n_out_total}
    print(f"\n[D metrics] {metrics}")
    save_results(f"classical_K{K}", samples, outs, out_path,
                 spec_metrics=metrics)
    return outs


def cfg_ss(samples, out_path, K=4, beta=1.0, gamma=0.5,
           asym_method="jsd"):
    print(f"\n=== CFG SS: AsymSpec multimodal K={K} β={beta} γ={gamma} method={asym_method} ===")
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    import vllm.config.speculative as _sc
    _sc.SpeculativeConfig.verify_equal_vocab_size_if_draft_model = lambda self: None

    # Asym method dispatch via env var (sampler reads ASYMSPEC_METHOD).
    os.environ["ASYMSPEC_METHOD"] = asym_method
    os.environ.pop("ASYMSPEC_BETA_OVERRIDE", None)

    # Speculative-decoding metrics capture.
    import vllm.v1.spec_decode.metrics as _sdm
    SPEC = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0,
            "per_pos": None}
    _orig = _sdm.SpecDecodingStats.observe_draft

    def _capture(self, *args, **kwargs):
        ndt = kwargs.get("num_draft_tokens", args[0] if args else 0)
        nat = kwargs.get("num_accepted_tokens",
                         args[1] if len(args) > 1 else 0)
        SPEC["drafts"] += 1
        SPEC["draft_tokens"] += ndt
        SPEC["accepted_tokens"] += nat
        if SPEC["per_pos"] is None:
            SPEC["per_pos"] = [0] * self.num_spec_tokens
        for j in range(nat):
            if j < len(SPEC["per_pos"]):
                SPEC["per_pos"][j] += 1
        return _orig(self, *args, **kwargs)
    _sdm.SpecDecodingStats.observe_draft = _capture

    proc = AutoProcessor.from_pretrained(VL_PATH, trust_remote_code=True)

    aug_prepped = []
    for s in samples:
        # Setup A: aug includes the SAME text as main + image (information
        # superset over main). Drafter sees image PLUS what verifier sees.
        # Without this, drafter & verifier outputs live on different
        # distributions → δ noise → SS underperforms.
        aug_user_text = build_main_user_text(s)
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": aug_user_text}]}]
        aug_text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
        po = proc(text=[aug_text], images=[s["image"]], return_tensors="pt",
                  padding=False)
        aug_prepped.append({
            "aug_ids": po["input_ids"][0].tolist(),
            "pixel_values": po["pixel_values"],
            "image_grid_thw": po["image_grid_thw"],
        })

    llm = LLM(model=TEXT_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=32768, gpu_memory_utilization=0.85,
              enforce_eager=True, disable_log_stats=False,
              speculative_config={
                  "method": "specsteer", "model": VL_PATH,
                  "num_speculative_tokens": K,
                  "specsteer_beta": beta, "specsteer_gamma": gamma,
              })
    print("[CFG SS] LLM init OK")

    sp_base = dict(temperature=0, max_tokens=MAX_OUT_TOKENS)
    AUG_MAX = 32768 - MAX_OUT_TOKENS - 64
    outs = []
    n_skip = 0; n_out_total = 0
    t0 = time.perf_counter()
    for i, (s, prep) in enumerate(zip(samples, aug_prepped)):
        if len(prep["aug_ids"]) > AUG_MAX:
            print(f"  [{i}] SKIP aug_len={len(prep['aug_ids'])} > {AUG_MAX}")
            outs.append({"ans": "SKIP", "pred": "", "ok": False})
            n_skip += 1
            continue
        main = build_main_prompt(s)
        sp_kw = dict(sp_base, extra_args={
            "specsteer_aug_prompt_ids": prep["aug_ids"],
            "specsteer_aug_pixel_values": prep["pixel_values"],
            "specsteer_aug_image_grid_thw": prep["image_grid_thw"],
        })
        try:
            r = llm.generate([{"prompt": main}], SamplingParams(**sp_kw),
                             use_tqdm=False)
            ans = r[0].outputs[0].text.strip()
            n_out_total += len(r[0].outputs[0].token_ids)
            pred = extract_pred(ans, s)
            ok = is_correct(pred, s)
            outs.append({"ans": ans, "pred": pred, "ok": ok})
        except Exception as e:
            print(f"  [{i}] FAIL: {type(e).__name__}: {str(e)[:200]}")
            outs.append({"ans": f"FAIL:{type(e).__name__}", "pred": "",
                         "ok": False})
            print("  [STOP] engine state corrupt — saving partial")
            break
        if (i + 1) % 25 == 0:
            n_ok = sum(o["ok"] for o in outs)
            el = time.perf_counter() - t0
            ar = SPEC["accepted_tokens"] / SPEC["draft_tokens"] if SPEC["draft_tokens"] else 0
            mal = 1 + SPEC["accepted_tokens"] / SPEC["drafts"] if SPEC["drafts"] else 1
            print(f"  [{i+1}/{len(samples)}] acc={100*n_ok/(i+1):.1f}% "
                  f"tps={n_out_total/el:.1f} AR={ar:.3f} MAL={mal:.2f} "
                  f"skip={n_skip}", flush=True)
    el = time.perf_counter() - t0
    metrics = {
        "K": K,
        "tps": n_out_total / el if el else 0,
        "AR": SPEC["accepted_tokens"] / SPEC["draft_tokens"] if SPEC["draft_tokens"] else 0,
        "MAL": 1 + SPEC["accepted_tokens"] / SPEC["drafts"] if SPEC["drafts"] else 1,
        "elapsed": el, "n_out_tokens_total": n_out_total,
        "drafts": SPEC["drafts"],
        "draft_tokens": SPEC["draft_tokens"],
        "accepted_tokens": SPEC["accepted_tokens"],
        "per_position_acceptance_rate":
            [c / SPEC["drafts"] for c in (SPEC["per_pos"] or [])],
        "n_skipped": n_skip,
    }
    print(f"\n[SS metrics] {json.dumps(metrics, indent=2)}")
    save_results(f"ss_K{K}", samples, outs, out_path, spec_metrics=metrics)
    return outs


def save_results(cfg, samples, outs, out_path, spec_metrics=None):
    n_ok = sum(o["ok"] for o in outs)
    rec = {
        "cfg": cfg,
        "n_total": len(outs),
        "n_correct": n_ok,
        "accuracy_pct": 100.0 * n_ok / len(outs) if outs else 0.0,
        "per_task": {},
        "spec_metrics": spec_metrics,
        "results": [
            {"idx": i, "pid": s["pid"], "task": s["task"],
             "question_type": s["question_type"],
             "question": s["question"], "gold": s["answer"],
             "pred": o["pred"], "ans": o["ans"], "ok": o["ok"]}
            for i, (s, o) in enumerate(zip(samples[:len(outs)], outs))
        ],
    }
    # per-task accuracy
    for t in KEEP_TASKS:
        tsamps = [(s, o) for s, o in zip(samples[:len(outs)], outs)
                  if s["task"] == t]
        if tsamps:
            rec["per_task"][t] = {
                "n": len(tsamps),
                "n_correct": sum(o["ok"] for _, o in tsamps),
                "accuracy_pct": 100.0 * sum(o["ok"] for _, o in tsamps) / len(tsamps),
            }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"\n[saved] {cfg} → {out_path}  acc={rec['accuracy_pct']:.1f}%")
    for t, d in rec["per_task"].items():
        print(f"  {t}: {d['n_correct']}/{d['n']} = {d['accuracy_pct']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", choices=["b", "c", "d", "ss"], required=True)
    ap.add_argument("--n", type=int, default=0,
                    help="CANONICAL paper config: 0 = all 587 (smoke=20 only for testing).")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="CANONICAL paper config: β=1.0 (best for MathVista per paper).")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--asym_method", default="jsd",
                    choices=["gamma_rule", "cma", "jsd", "jsd_pos",
                             "cma_vnorm", "cma_hbase"])
    ap.add_argument("--tag", default="paper")
    args = ap.parse_args()

    samples = load_samples(args.n if args.n > 0 else None)
    out_path = os.path.join(
        OUT_DIR,
        f"{args.tag}_n{len(samples)}_{args.cfg}"
        + (f"_K{args.K}" if args.cfg in ("ss", "d") else "")
        + ".json")

    if args.cfg == "b":
        cfg_b_aug_vl(samples, out_path)
    elif args.cfg == "c":
        cfg_c_main_cap(samples, out_path)
    elif args.cfg == "d":
        cfg_d_classical(samples, out_path, K=args.K)
    else:
        cfg_ss(samples, out_path, K=args.K, beta=args.beta, gamma=args.gamma,
               asym_method=args.asym_method)


if __name__ == "__main__":
    main()
