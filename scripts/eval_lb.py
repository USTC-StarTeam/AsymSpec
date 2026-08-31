"""LongBench eval — F1/EM + (optional) LLM-judge.

Reads response JSONL files below a user-supplied experiment directory and
writes per-cell evaluation JSON plus an aggregate ``eval_summary.jsonl``.

For each (cell, response) pair:
  - Extract final answer via regex (last "the answer is X" with period-tolerant match)
  - LongBench official F1 (token-overlap, normalized)
  - EM (exact normalized match)
  - Optional LLM-judge (gpt-4o-mini, binary CORRECT/INCORRECT, reasoning-then-verdict)

Per-dataset breakdown (HotpotQA / 2WikiMQA / MuSiQue) for each cell.

Usage:
  python eval_lb.py --exp_dir <dir>            # F1/EM only
  python eval_lb.py --exp_dir <dir> --judge    # also LLM-judge
"""
from __future__ import annotations
import argparse, json, os, re, string, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p_toks)
    rec = n / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


# Priority-ordered patterns. The PRIMARY one (kept for fmt_compliance back-compat).
# Captures "the/my/our [final] answer is X" — handles "the final answer is X" too.
_ANS_RE = re.compile(
    r"(?i)(?:the|my|our)\s+(?:final\s+|correct\s+)?answer\s+is\s*:?\s*(.+?)(?:\n|$)",
    re.MULTILINE,
)

# Full fallback chain. extract_answer tries them in order; fmt_ok = True iff
# any of these matched (NOT the bare last-sentence fallback).
_ANS_PATTERNS = [
    _ANS_RE,
    # "Final answer: X" / "**Final answer:** X" — REQUIRE ":" to avoid matching
    # "the final answer is" (handled by pattern 1 already).
    re.compile(r"(?i)\**\s*final\s+answer\s*:\s*\**\s*(.+?)(?:\n|$)", re.MULTILINE),
    # "Answer: X" / "**Answer:** X" — must START a line (avoid matching
    # mid-sentence "answer:") via ^
    re.compile(r"(?im)^\s*\**\s*answer\s*:\s*\**\s*(.+?)(?:\n|$)"),
    # Markdown header "### Answer\n<line>" — capture the line below the header
    re.compile(r"(?im)^#+\s*answer\s*\n+\s*(.+?)(?:\n|$)"),
    # Chinese: "答案是 X" / "答案：X" / "答案为 X"
    re.compile(r"答案[是为：:]\s*(.+?)(?:\n|$)"),
    # Chinese: "最终答案：X"
    re.compile(r"最终答案[：:]\s*(.+?)(?:\n|$)"),
]

# Leading-noise prefixes commonly added before the actual answer; strip them.
_LEADING_NOISE_RE = re.compile(
    r"^(?:therefore|so|thus|hence|then|finally|in\s+conclusion|"
    r"to\s+summarize|to\s+conclude|all\s+in\s+all|in\s+summary|"
    r"所以|因此|于是|综上所述|综上|总结|结论|最终|最后)"
    r"[,，:：\s]+",
    re.IGNORECASE,
)

# Markdown wrappers we strip from extracted answers.
_TRAILING_STRIP_CHARS = ".,!?;:。，！？；：*_\"'`)】]》"
_LEADING_STRIP_CHARS = "*_\"'`(【[《"


def _clean(s: str) -> str:
    s = s.strip()
    s = _LEADING_NOISE_RE.sub("", s).strip()
    s = s.strip(_LEADING_STRIP_CHARS)
    s = s.rstrip(_TRAILING_STRIP_CHARS).strip()
    return s


def _last_sentence(text: str) -> str:
    """Take the last sentence (NOT whole last paragraph) as a final fallback."""
    text = text.strip()
    if not text:
        return ""
    # Take last non-empty line first, then split into sentences within it.
    last_line = next((l.strip() for l in reversed(text.split("\n")) if l.strip()), "")
    # Split on . ! ? 。！？ but keep multi-char like "Mr." protected by trailing space
    sents = re.split(r"(?<=[.!?。！？])\s+", last_line)
    sents = [s.strip() for s in sents if s.strip()]
    return sents[-1] if sents else last_line


def extract_answer(text: str) -> tuple[str, bool]:
    """Return (answer, fmt_matched_explicit_pattern)."""
    for pat in _ANS_PATTERNS:
        matches = pat.findall(text)
        if matches:
            cand = _clean(matches[-1])
            if cand:
                return cand, True
    # Fallback: last sentence (NOT whole paragraph) cleaned of noise prefix
    cand = _clean(_last_sentence(text))
    return cand, False


def best_score(pred: str, golds: list[str], scorer):
    return max((scorer(pred, g) for g in golds), default=0.0)


def eval_responses(jsonl_path: str) -> dict:
    """Return per-dataset + aggregate F1/EM stats, plus extracted answers."""
    per_ds = defaultdict(lambda: {"n": 0, "f1": 0.0, "em": 0.0, "fmt_ok": 0,
                                    "out_lens": []})
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            ds = r["dataset"]
            golds = r["gold_answers"]
            text = r["response"]
            pred, fmt_ok = extract_answer(text)
            f1 = best_score(pred, golds, f1_score)
            em = best_score(pred, golds, em_score)
            per_ds[ds]["n"] += 1
            per_ds[ds]["f1"] += f1
            per_ds[ds]["em"] += em
            per_ds[ds]["fmt_ok"] += int(fmt_ok)
            per_ds[ds]["out_lens"].append(len(text))
            rows.append({
                "_id": r["_id"], "dataset": ds, "golds": golds,
                "pred": pred, "f1": f1, "em": em, "fmt_ok": fmt_ok,
            })
    # Reduce
    summary = {}
    total_n, total_f1, total_em, total_fmt = 0, 0.0, 0.0, 0
    for ds, s in per_ds.items():
        n = s["n"]
        if n == 0:
            continue
        summary[ds] = {
            "n": n,
            "f1": s["f1"] / n,
            "em": s["em"] / n,
            "fmt_compliance": s["fmt_ok"] / n,
            "mean_resp_chars": sum(s["out_lens"]) / n,
        }
        total_n += n
        total_f1 += s["f1"]
        total_em += s["em"]
        total_fmt += s["fmt_ok"]
    summary["overall"] = {
        "n": total_n,
        "f1": total_f1 / max(total_n, 1),
        "em": total_em / max(total_n, 1),
        "fmt_compliance": total_fmt / max(total_n, 1),
    }
    return {"per_dataset": summary, "rows": rows}


# === LLM-judge (optional) ===

JUDGE_PROMPT = """You are evaluating a question-answering system's response.

Question: {question}
Reference answer: {gold}
Model response: {response}

The reference answer is the ground truth. Determine whether the model's response correctly answers the question. Consider semantic equivalence (e.g., "JFK" matches "John F. Kennedy", "Paris" matches "Paris, France"). The model response may include reasoning; focus on the final answer.

First explain your reasoning in one sentence. Then output your verdict on a new line as exactly one of: CORRECT or INCORRECT.

Format:
Reasoning: <one sentence>
Verdict: <CORRECT|INCORRECT>"""


def parse_verdict(text: str) -> str | None:
    m = re.search(r"verdict\s*:\s*(correct|incorrect)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def llm_judge_batch(rows: list[dict], questions: dict[str, str],
                    model: str = "gpt-4o-mini",
                    concurrency: int = 16) -> list[str | None]:
    """rows have _id; questions maps _id → question text. Returns verdict list aligned with rows."""
    try:
        from api_key_util import get_logged_openai_client
    except ImportError:
        print("ERROR: openai/api_key_util not available.", file=sys.stderr)
        sys.exit(1)
    import concurrent.futures

    client = get_logged_openai_client(user="eval_lb_judge")  # logs to api_logs/
    verdicts: list[str | None] = [None] * len(rows)

    def judge_one(idx: int) -> tuple[int, str | None]:
        r = rows[idx]
        q = questions.get(r["_id"], "")
        gold = " | ".join(r["golds"])
        prompt = JUDGE_PROMPT.format(question=q, gold=gold, response=r.get("response_full", r["pred"]))
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0,
                )
                return idx, parse_verdict(resp.choices[0].message.content)
            except Exception as e:
                if attempt == 2:
                    print(f"[judge fail idx={idx}]: {e}", file=sys.stderr)
                    return idx, None
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(judge_one, i) for i in range(len(rows))]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            idx, v = fut.result()
            verdicts[idx] = v
            done += 1
            if done % 100 == 0:
                print(f"  judged {done}/{len(rows)}", flush=True)
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--judge", action="store_true",
                    help="Also run LLM-judge (requires OPENAI_API_KEY env)")
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    args = ap.parse_args()

    # Load all questions per _id (used by judge for context).
    questions: dict[str, str] = {}
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/longbench/raw")
    for task in ["hotpotqa", "2wikimqa", "musique"]:
        with open(os.path.join(raw_dir, f"{task}.jsonl")) as f:
            for line in f:
                ex = json.loads(line)
                questions[ex["_id"]] = ex["input"]

    # Find all *_responses.jsonl
    resp_files = []
    for sub in ["k_indep", "k2", "k4", "k6"]:
        d = os.path.join(args.exp_dir, sub)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith("_responses.jsonl"):
                resp_files.append(os.path.join(d, fname))
    print(f"[setup] {len(resp_files)} response files found", flush=True)

    summary_rows = []
    for rp in resp_files:
        cell_K = os.path.basename(rp).replace("_responses.jsonl", "")
        print(f"\n=== eval {cell_K} ===")
        result = eval_responses(rp)
        summary = result["per_dataset"]
        for ds in ("hotpotqa", "2wikimqa", "musique", "overall"):
            if ds in summary:
                s = summary[ds]
                print(f"  [{ds}] n={s['n']}  F1={s['f1']:.3f}  EM={s['em']:.3f}  "
                      f"fmt={s['fmt_compliance']:.3f}")
        # Save per-cell eval
        eval_path = rp.replace("_responses.jsonl", "_eval.json")
        eval_obj = {"cell_K": cell_K, "per_dataset": summary}

        if args.judge:
            # Inject full response text for judge.
            full_resps = {}
            with open(rp) as f:
                for line in f:
                    r = json.loads(line)
                    full_resps[r["_id"]] = r["response"]
            for row in result["rows"]:
                row["response_full"] = full_resps.get(row["_id"], "")
            print(f"  [judge] running on {len(result['rows'])} samples...")
            t0 = time.perf_counter()
            verdicts = llm_judge_batch(result["rows"], questions,
                                        model=args.judge_model)
            elapsed = time.perf_counter() - t0
            # Compute judge accuracy per dataset
            per_ds_judge = defaultdict(lambda: {"n": 0, "correct": 0, "n_judged": 0})
            for r, v in zip(result["rows"], verdicts):
                per_ds_judge[r["dataset"]]["n"] += 1
                if v is not None:
                    per_ds_judge[r["dataset"]]["n_judged"] += 1
                    if v == "CORRECT":
                        per_ds_judge[r["dataset"]]["correct"] += 1
            tot_n, tot_c, tot_j = 0, 0, 0
            judge_summary = {}
            for ds, s in per_ds_judge.items():
                judge_summary[ds] = {
                    "n": s["n"], "n_judged": s["n_judged"],
                    "correct": s["correct"],
                    "judge_acc": s["correct"] / max(s["n_judged"], 1),
                }
                tot_n += s["n"]; tot_c += s["correct"]; tot_j += s["n_judged"]
            judge_summary["overall"] = {
                "n": tot_n, "n_judged": tot_j, "correct": tot_c,
                "judge_acc": tot_c / max(tot_j, 1),
            }
            eval_obj["judge"] = judge_summary
            print(f"  [judge] {elapsed:.1f}s; ", end="")
            for ds in ("hotpotqa", "2wikimqa", "musique", "overall"):
                if ds in judge_summary:
                    print(f"{ds}={judge_summary[ds]['judge_acc']:.3f}  ", end="")
            print()

        with open(eval_path, "w") as f:
            json.dump(eval_obj, f)
        summary_rows.append(eval_obj)

    # Aggregate summary
    out_summary = os.path.join(args.exp_dir, "eval_summary.jsonl")
    with open(out_summary, "w") as f:
        for r in summary_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {out_summary}")


if __name__ == "__main__":
    main()
