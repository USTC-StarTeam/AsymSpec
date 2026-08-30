"""Generate LLMLingua-2 compressed earlier turns for MultiChallenge.

Mirrors gen_mc_summaries.py but uses extractive token-level compression
(LLMLingua-2) instead of an LLM-generated abstractive summary.

The compressed text replaces the *earlier* portion of the conversation
(all turns except the last K), which then gets prepended to the last K
turns as msg_main — exactly mirroring the summary_last_k pattern.
The resulting loader variant is `main_context="llmlingua_last_k"`.

Compression model: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
  — ~180 MB, runs on CPU, no GPU required.

Output: experiments/cache/mc_llmlingua.json
  {
    "config": {...},
    "compressions": {
      "<qid>": {
        "<k>": "compressed earlier turns text"   # k in args.ks
      }
    }
  }

Usage:
  python experiments/gen_mc_llmlingua.py          # default: K in {4} to match summary_last_k4
  python experiments/gen_mc_llmlingua.py --ks 4   # same explicit

~3-8 min for 273 MC examples on CPU (BERT model is fast).
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

# Compression budget (in tokens) for the "earlier turns" text.
# We target ~200 tokens — close to the summary's ≤250-word cap — so the
# two compression methods are comparable in output length.
TARGET_TOKENS = 200

# Model: multilingual LLMLingua-2 fine-tuned on MeetingBank transcripts.
# Falls back gracefully to English monolingual if download fails.
LLMLINGUA2_MODEL = (
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)


def _conv_to_text(turns: list[dict]) -> str:
    """Render conversation turns as plain transcript text."""
    lines: list[str] = []
    for t in turns:
        role = t["role"].upper()
        content = t["content"].strip()
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default=os.path.join(
            _REPO_ROOT, "data", "multi-challenge", "data",
            "benchmark_questions.jsonl",
        ),
    )
    p.add_argument("--max_n", type=int, default=273)
    p.add_argument(
        "--ks", type=str, default="4",
        help="Comma-separated list of last-K values to generate (default: 4)",
    )
    p.add_argument(
        "--target_tokens", type=int, default=TARGET_TOKENS,
        help="Target token count for compressed 'earlier turns' text",
    )
    p.add_argument(
        "--model_name", default=LLMLINGUA2_MODEL,
    )
    p.add_argument(
        "--output",
        default=os.path.join(
            _REPO_ROOT, "experiments", "cache", "mc_llmlingua.json"
        ),
    )
    args = p.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 70)
    print("MultiChallenge LLMLingua-2 compression (for matrix llmlingua_last_k)")
    print("=" * 70)
    print(f"  model           : {args.model_name}")
    print(f"  data            : {args.data}")
    print(f"  max_n           : {args.max_n}")
    print(f"  ks              : {ks}")
    print(f"  target_tokens   : {args.target_tokens}")
    print(f"  output          : {args.output}")
    print()

    print("[load] LLMLingua-2 compressor …")
    t0 = time.perf_counter()
    from llmlingua import PromptCompressor
    compressor = PromptCompressor(
        model_name=args.model_name,
        use_llmlingua2=True,
        device_map="cpu",
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    examples = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.max_n > 0:
        examples = examples[: args.max_n]
    print(f"\n[loaded] {len(examples)} MC examples")

    compressions: dict[str, dict[str, str]] = {}
    t_start = time.perf_counter()
    n_passthrough = 0  # examples too short to need compression

    for i, d in enumerate(examples):
        qid = str(d["QUESTION_ID"])
        conv = d["CONVERSATION"]
        compressions[qid] = {}

        for k in ks:
            # "earlier turns" = all turns except the last k
            earlier = conv[:-k] if len(conv) > k else []
            if not earlier:
                # Nothing to compress (conv is short) — store empty string.
                compressions[qid][str(k)] = ""
                n_passthrough += 1
                continue

            text = _conv_to_text(earlier)
            n_words = len(text.split())

            if n_words <= args.target_tokens:
                # Already short enough — no compression needed.
                compressions[qid][str(k)] = text
                n_passthrough += 1
                continue

            result = compressor.compress_prompt(
                text,
                target_token=args.target_tokens,
                force_tokens=["\n", "."],
                # context_budget="+100%" avoids the ±10% constraint from default
            )
            compressions[qid][str(k)] = result["compressed_prompt"]

        if (i + 1) % 25 == 0 or i == len(examples) - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i+1:4d}/{len(examples)}]  "
                f"passthrough={n_passthrough}  "
                f"rate={rate:.2f}/s  elapsed={elapsed:.0f}s"
            )

    payload = {
        "config": {
            "model_name": args.model_name,
            "target_tokens": args.target_tokens,
            "ks": ks,
            "n_examples": len(examples),
        },
        "compressions": compressions,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[done] {len(compressions)} examples written → {args.output}")
    print(f"  passthrough (too short): {n_passthrough}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
