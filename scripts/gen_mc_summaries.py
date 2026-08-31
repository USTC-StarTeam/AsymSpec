"""Generate the MultiChallenge compressed-history summaries used in the paper.

For each conversation and each requested ``k``, all but the final ``k`` turns
are summarized offline. The benchmark then prepends that summary to the final
turns. Output matches ``scripts/bench_multichallenge.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import MC_QUESTIONS, MC_SUMMARIES


SUMMARY_INSTRUCTION = (
    "You are a faithful conversation summarizer. The user will provide a "
    "multi-turn dialogue. Produce a SHORT, DENSE summary (≤150 words) of the "
    "dialogue that captures:\n"
    "  1. Any explicit user instructions or constraints (e.g. 'do not use bold')\n"
    "  2. User preferences and facts about themselves\n"
    "  3. Key entities, names, and prior decisions made in the dialogue\n"
    "  4. Any prior assistant commitments\n"
    "Use plain text. No headers. No bullet points. Write as one paragraph. "
    "Do not invent information not in the dialogue."
)


def render_dialogue(turns: list[dict]) -> str:
    return "\n\n".join(
        f"[{turn['role'].upper()}] {turn['content'].strip()}" for turn in turns
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=MC_QUESTIONS)
    parser.add_argument("--output", type=Path, default=MC_SUMMARIES)
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-n", type=int, default=271)
    parser.add_argument("--max-summary-tokens", type=int, default=200)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = []
    with args.data.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if args.max_n > 0:
        rows = rows[: args.max_n]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts: list[str] = []
    keys: list[tuple[str, int]] = []
    summaries: dict[str, dict[str, str]] = {
        str(row["QUESTION_ID"]): {} for row in rows
    }

    for row in rows:
        qid = str(row["QUESTION_ID"])
        conversation = row["CONVERSATION"]
        for k in args.k_values:
            earlier = conversation[:-k] if len(conversation) > k else []
            if not earlier:
                summaries[qid][str(k)] = ""
                continue
            messages = [
                {"role": "system", "content": SUMMARY_INSTRUCTION},
                {"role": "user", "content": (
                    "Dialogue to summarize:\n\n" + render_dialogue(earlier)
                )},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            keys.append((qid, k))

    model = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    outputs = model.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=args.max_summary_tokens),
        use_tqdm=True,
    )
    for (qid, k), output in zip(keys, outputs):
        summaries[qid][str(k)] = output.outputs[0].text.strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {
            "model": args.model,
            "k_values": args.k_values,
            "max_n": args.max_n,
            "max_summary_tokens": args.max_summary_tokens,
        },
        "summary_instruction": SUMMARY_INSTRUCTION,
        "summaries": summaries,
    }, ensure_ascii=False, indent=2))
    print(f"Saved {len(summaries)} examples to {args.output}")


if __name__ == "__main__":
    main()
