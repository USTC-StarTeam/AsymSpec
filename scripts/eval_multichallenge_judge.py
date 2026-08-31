#!/usr/bin/env python3
"""Evaluate MultiChallenge response files with the official judge protocol.

Benchmark response rows contain ``idx``, ``qid``, and ``response``. This
wrapper joins the target question, pass criterion, and axis from the official
question file before judging each response.

Per the official MultiChallenge evaluator (ekwinox117/multi-challenge
src/evaluator.py): YES/NO judgment with the verbatim prompt template.

Usage:
  python scripts/eval_multichallenge_judge.py <responses1.jsonl> [<responses2.jsonl> ...]
  → writes <name>_judge_report.json next to each input.

Env:
  N_WORKERS         (default 4)   parallel API calls per file
  MAX_RETRIES       (default 5)
  JUDGE_MODEL       (default gpt-4o-2024-08-06; matches official MC repo
                    ekwinox117/multi-challenge src/evaluator.py)
  SWE_DRY           (set to 1 → skip API calls, just verify JOIN works)
"""
import json
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from api_key_util import get_logged_openai_client
from paths import REPO_ROOT, MC_QUESTIONS

REPO = str(REPO_ROOT)
MC_PATH = str(MC_QUESTIONS)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-2024-08-06")
N_WORKERS = int(os.environ.get("N_WORKERS", "4"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
DRY = os.environ.get("SWE_DRY") == "1"

client = get_logged_openai_client(user="multichallenge_judge")


def load_mc_index() -> dict:
    """Returns {QUESTION_ID: {axis, target_question, pass_criteria}}."""
    idx = {}
    for line in open(MC_PATH):
        d = json.loads(line)
        idx[d["QUESTION_ID"]] = {
            "axis": d.get("AXIS", "UNK"),
            "target_question": d.get("TARGET_QUESTION", ""),
            "pass_criteria": d.get("PASS_CRITERIA", ""),
        }
    return idx


def judge_one(response_text: str, target_question: str) -> bool:
    """Official MultiChallenge judge prompt (verbatim from
    ekwinox117/multi-challenge src/evaluator.py).
    Returns True if YES, False if NO or error.
    """
    prompt = (
        "You are tasked with evaluating a model response to see if it meets "
        "a specific criteria.\nThe criteria will always be YES/NO evaluation."
        "\n\nThe model response is as follows:\n"
        f"<MODEL_RESPONSE>\n{response_text}\n</MODEL_RESPONSE>\n\n"
        "The criteria that the model response must meet is as follows. "
        "Be VERY STRICT!:\n"
        f"<CRITERIA>\n{target_question}\n</CRITERIA>\n\n"
        'Print your reasoning followed by your verdict, either "YES" or "NO".'
    )
    if DRY:
        return True  # skip API call in dry mode
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
                timeout=120,
            )
            txt = (res.choices[0].message.content or "").strip()
            if not txt:
                return False
            # Parse verdict — last line / "YES"/"NO" anywhere near end
            tl = txt.lower()
            # Find the verdict (last YES/NO in the text)
            yes_idx = tl.rfind("yes")
            no_idx = tl.rfind("no")
            if yes_idx > no_idx:
                return True
            return False
        except Exception as e:
            last_err = str(e)
            if "rate" in last_err.lower() or "timeout" in last_err.lower():
                time.sleep(2 + attempt * 2)
                continue
            else:
                if attempt == MAX_RETRIES - 1:
                    print(f"  [judge err {attempt+1}] {last_err[:200]}", file=sys.stderr)
                    return False
                time.sleep(2)
    print(f"  [judge final fail] {last_err[:200]}", file=sys.stderr)
    return False


def eval_file(resp_path: str, mc_idx: dict):
    import re as _re
    _slug = _re.sub(r"[^A-Za-z0-9.]+", "-", JUDGE_MODEL)
    for _suf in ("_responses.jsonl", "_resp.jsonl", ".jsonl"):
        if resp_path.endswith(_suf):
            _base = resp_path[: -len(_suf)]
            break
    else:
        _base = os.path.splitext(resp_path)[0]
    # Per-judge-model report name: never collides with the source file and
    # never overwrites the legacy gpt-4o reports (provenance: keep both).
    out_path = f"{_base}_judge_report__{_slug}.json"
    if os.path.exists(out_path):
        print(f"[SKIP] {os.path.basename(resp_path)} — report exists at {out_path}")
        return None
    if not os.path.exists(resp_path):
        print(f"[SKIP] {resp_path} not found")
        return None

    rows = [json.loads(l) for l in open(resp_path)]
    print(f"[{os.path.basename(resp_path)}] {len(rows)} responses to judge", flush=True)

    # Two response formats are supported: compact benchmark rows joined by
    # qid, and self-contained rows with embedded evaluation metadata.
    n_no_match = 0
    join_rows = []
    for r in rows:
        # Prefer embedded criteria when present.
        if "meta" in r and isinstance(r.get("meta"), dict) and r["meta"].get("target_question"):
            meta = {
                "axis": r["meta"].get("axis", "UNK"),
                "target_question": r["meta"].get("target_question", ""),
                "pass_criteria": r["meta"].get("pass_criteria", ""),
            }
            # Use embedded meta directly; r needs qid alias
            r_aug = dict(r)
            r_aug["qid"] = r.get("question_id", r.get("qid", "?"))
            join_rows.append((r_aug, meta))
            continue
        # Otherwise join the compact row by qid.
        qid = r.get("qid", r.get("question_id"))
        meta = mc_idx.get(qid)
        if meta is None:
            n_no_match += 1
            continue
        r_aug = dict(r)
        r_aug["qid"] = qid
        join_rows.append((r_aug, meta))
    if n_no_match:
        print(f"  [warn] {n_no_match} responses had qid not in MC index", flush=True)

    passed = 0
    total = 0
    by_axis = {}
    by_qid = {}
    t0 = time.time()

    def _work(item):
        r, meta = item
        text = r.get("response", "")
        ok = judge_one(text, meta["target_question"])
        return (r["qid"], meta["axis"], ok)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(_work, item): item for item in join_rows}
        for fut in as_completed(futs):
            try:
                qid, ax, ok = fut.result(timeout=300)
            except Exception as e:
                print(f"  [fut err] {e}", file=sys.stderr)
                qid, ax, ok = "?", "UNK", False
            total += 1
            by_qid[qid] = bool(ok)
            d = by_axis.setdefault(ax, {"p": 0, "t": 0})
            d["t"] += 1
            if ok:
                passed += 1
                d["p"] += 1
            if total % 25 == 0:
                el = time.time() - t0
                print(f"  [{os.path.basename(resp_path)}] {total}/{len(join_rows)} "
                      f"({passed} passed, {el:.0f}s)", flush=True)

    acc = 100.0 * passed / total if total else 0
    print(f"\n[{os.path.basename(resp_path)}] {passed}/{total} = {acc:.1f}%")
    for k, v in sorted(by_axis.items()):
        print(f"  {k}: {v['p']}/{v['t']} = {100*v['p']/v['t']:.1f}%")

    report = {
        "input": resp_path,
        "judge_model": JUDGE_MODEL,
        "n_responses": len(rows),
        "n_judged": total,
        "n_skipped_no_qid_match": n_no_match,
        "passed": passed,
        "total": total,
        "accuracy_pct": acc,
        "by_axis": {k: {"passed": v["p"], "total": v["t"],
                         "accuracy_pct": 100.0*v["p"]/v["t"]}
                    for k, v in sorted(by_axis.items())},
        "by_qid": by_qid,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {out_path}\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="One or more *_responses.jsonl paths")
    args = ap.parse_args()

    print(f"[setup] judge model: {JUDGE_MODEL}, N_WORKERS={N_WORKERS}, "
          f"DRY={DRY}", flush=True)
    mc_idx = load_mc_index()
    print(f"[setup] loaded MC index n={len(mc_idx)}", flush=True)

    for p in args.paths:
        try:
            eval_file(p, mc_idx)
        except Exception as e:
            import traceback
            print(f"[FAIL] {p}: {e}", file=sys.stderr)
            traceback.print_exc()


if __name__ == "__main__":
    main()
