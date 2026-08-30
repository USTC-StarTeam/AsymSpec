"""LongBench v1: LLMLingua-2 extractive compression of x_main.

Mirrors gen_lbv2_llmlingua.py / gen_lme_llmlingua.py but for LB v1's three
multi-hop tasks (hotpotqa, 2wikimqa, musique, n=200 each → 600 total).
Compresses each row's `context` to `--target_tokens` (default 1500 to
match the summary cap), enabling apples-to-apples compression-method
ablation against `--main_context summary` and `--main_context truncate`.

Output: experiments/cache/lb_llmlingua.json
  { "config": {...},
    "compressions": { "<_id>": "<compressed context>" } }

Usage:  python experiments/gen_lb_llmlingua.py     # CPU, ~15-30 min
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

print = functools.partial(print, flush=True)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from paths import LB_RAW  # noqa: E402

LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
TASKS = ("hotpotqa", "2wikimqa", "musique")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target_tokens", type=int, default=1500,
                   help="LLMLingua-2 target for x_main (matches summary cap)")
    p.add_argument("--model_name", default=LLMLINGUA2_MODEL)
    p.add_argument("--output", default=os.path.join(
        _REPO_ROOT, "experiments", "cache", "lb_llmlingua.json"))
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 70)
    print("LongBench v1: LLMLingua-2 extractive compression (CPU, no GPU)")
    print("=" * 70)
    print(f"  tasks         : {TASKS}")
    print(f"  target_tokens : {args.target_tokens}")
    print(f"  output        : {args.output}")

    rows = []
    for t in TASKS:
        fp = os.path.join(LB_RAW, f"{t}.jsonl")
        for line in open(fp):
            d = json.loads(line)
            rows.append({"_id": d["_id"], "task": t, "context": d["context"]})
    print(f"\n[loaded] {len(rows)} rows")

    compressions: dict[str, str] = {}
    if os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            compressions = prev.get("compressions", {})
            print(f"[resume] {len(compressions)} compressions already cached")
        except Exception:
            compressions = {}

    todo = [r for r in rows if r["_id"] not in compressions]
    print(f"[todo] {len(todo)} rows to compress")
    if todo:
        print("[load] LLMLingua-2 (CPU) …")
        t0 = time.perf_counter()
        from llmlingua import PromptCompressor
        comp = PromptCompressor(model_name=args.model_name,
                                use_llmlingua2=True, device_map="cpu")
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")

        def flush():
            json.dump({"config": {"tasks": list(TASKS),
                                  "model_name": args.model_name,
                                  "target_tokens": args.target_tokens,
                                  "n_total": len(rows)},
                       "compressions": compressions},
                      open(args.output, "w"), ensure_ascii=False, indent=1)

        t_start = time.perf_counter()
        for i, r in enumerate(todo):
            try:
                res = comp.compress_prompt(r["context"],
                                           target_token=args.target_tokens,
                                           force_tokens=["\n", ".", "==="])
                compressions[r["_id"]] = res["compressed_prompt"]
            except Exception as e:
                print(f"  [warn] {r['_id']}: {e!r} — skipped")
            if (i + 1) % 10 == 0 or i == len(todo) - 1:
                flush()
                el = time.perf_counter() - t_start
                print(f"  [{i+1:4d}/{len(todo)}] cached={len(compressions)} "
                      f"rate={(i+1)/el:.2f}/s elapsed={el:.0f}s (flushed)")
        flush()

    print(f"\n[done] n_total={len(rows)} compressions={len(compressions)} "
          f"→ {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
