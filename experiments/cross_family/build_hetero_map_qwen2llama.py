#!/usr/bin/env python3
"""Build the Llama-3.2 (drafter, A) <-> Qwen3 (verifier, B) vocab translator.

String-equal token matching over tokenizer vocabs (both byte-level BPE with 'Ġ'
space sign — verified identical convention, no normalization needed), plus
explicit special-token pairs so EOS semantics cross the boundary.

Artifact (torch.save dict):
  a2b      int64 [V_A_model]  Llama id -> Qwen id, -1 unmapped
  b2a      int64 [V_B_model]  Qwen id -> Llama id, -1 unmapped
  allow_b  bool  [V_B_model]  emission mask: image(a2b) ∪ Qwen eos specials
  src_a / dst_b int64 [n_pairs]  parallel index tensors for δ scatter
  meta     dict
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--drafter", default="Qwen/Qwen3-4B")
ap.add_argument("--verifier", default="meta-llama/Llama-3.3-70B-Instruct")
ap.add_argument("--output", required=True)
args = ap.parse_args()

A_PATH = args.drafter
B_NAME = args.verifier
OUT = args.output

V_A = AutoConfig.from_pretrained(A_PATH).vocab_size
V_B = AutoConfig.from_pretrained(B_NAME).vocab_size

ta = AutoTokenizer.from_pretrained(A_PATH)
tb = AutoTokenizer.from_pretrained(B_NAME)
va, vb = ta.get_vocab(), tb.get_vocab()

# space-sign sanity: both must tokenize " x" with the same marker
sa = ta.convert_ids_to_tokens(ta(" x", add_special_tokens=False)["input_ids"])[0][0]
sb = tb.convert_ids_to_tokens(tb(" x", add_special_tokens=False)["input_ids"])[0][0]
assert sa == sb == "Ġ", f"space-sign mismatch: {sa!r} vs {sb!r}"

a2b = torch.full((V_A,), -1, dtype=torch.int64)
b2a = torch.full((V_B,), -1, dtype=torch.int64)
n = 0
for tok, aid in va.items():
    bid = vb.get(tok)
    if bid is not None and aid < V_A and bid < V_B:
        a2b[aid] = bid
        b2a[bid] = aid
        n += 1

# explicit special pairs (turn-end and text-end semantics)
SPECIALS = [
    (151645, 128009),   # qwen <|im_end|>    -> llama <|eot_id|>   (A=Qwen -> B=Llama)
    (151643, 128001),   # qwen <|endoftext|> -> llama <|end_of_text|>
]
for aid, bid in SPECIALS:
    a2b[aid] = bid
    b2a[bid] = aid

allow_b = torch.zeros(V_B, dtype=torch.bool)
valid = a2b >= 0
allow_b[a2b[valid]] = True                       # includes specials via pairs

src_a = valid.nonzero(as_tuple=True)[0]          # [n_pairs] Llama ids
dst_b = a2b[src_a]                               # [n_pairs] Qwen ids

# roundtrip spot checks
for probe in ["The", "Ġthe", "Ġanswer", "1", "Ġ1974", ",", "ĠParis"]:
    aid, bid = va.get(probe), vb.get(probe)
    if aid is not None and bid is not None:
        assert a2b[aid] == bid and b2a[bid] == aid, probe
assert int(a2b[151645]) == 128009 and int(b2a[128009]) == 151645

# total surrogate map: EVERY Qwen id -> some Llama id (exact where possible,
# else first token of re-encoding its decoded text; last resort: llama eot).
# Makes drafter-side translation a total function (no -1 anywhere).
b2a_sur = b2a.clone()
miss = (b2a_sur < 0).nonzero(as_tuple=True)[0]
fallback_eot = 151645  # qwen <|im_end|>
n_text, n_eot = 0, 0
for bid in miss.tolist():
    txt = tb.decode([bid], skip_special_tokens=False)
    ids = ta(txt, add_special_tokens=False)["input_ids"] if txt else []
    if ids:
        b2a_sur[bid] = ids[0]; n_text += 1
    else:
        b2a_sur[bid] = fallback_eot; n_eot += 1
assert int((b2a_sur < 0).sum()) == 0
print(f"surrogate: {n_text} via re-encode first-token, {n_eot} eot-fallback")

Path(OUT).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
meta = dict(drafter=A_PATH, verifier=B_NAME, V_A=V_A, V_B=V_B,
            n_string_pairs=n, n_pairs=int(src_a.numel()),
            cov_A=n / len(va), cov_B=n / len(vb))
torch.save(dict(a2b=a2b, b2a=b2a, b2a_sur=b2a_sur, allow_b=allow_b,
                src_a=src_a, dst_b=dst_b, meta=meta), OUT)
print(f"saved {OUT}")
print(f"string pairs={n}  total pairs={src_a.numel()}  "
      f"cov_A={meta['cov_A']:.1%}  cov_B={meta['cov_B']:.1%}  "
      f"allow_b={int(allow_b.sum())}")
