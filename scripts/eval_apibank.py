#!/usr/bin/env python3
"""APIB (API-Bank) evaluator — official metrics, self-contained.

Reads our bench_apibank_v2.py response files (each row = {id, file, level,
ground_truth, prediction, ...}) and computes:

  1. **API-call accuracy** (rows where ground_truth['role'] == 'API'):
     parse prediction → extract (api_name, param_dict);
     compare with ground_truth['api_name'] + ground_truth['param_dict'].
     Strict: api_name match AND param_dict subset match.
     This avoids the official check_api_call_correctness path which requires
     instantiating live tools (some need network/API keys we don't have).

  2. **ROUGE-L** (rows where ground_truth['role'] == 'AI'):
     compute rouge-l F1 between ground_truth['text'] and prediction.

Per-cell output: {api_acc, rouge_l, by_level, n_api_rows, n_ai_rows}.

Usage:
  python scripts/eval_apibank.py <responses1.jsonl> [<responses2.jsonl> ...]
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import defaultdict


def parse_api_call(text: str) -> tuple:
    """From DAMO-ConvAI/api-bank/api_call_extraction.py — verbatim regex."""
    pattern = r"\[(\w+)\((.*)\)\]"
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None, {}
    api_name = match.group(1)
    params = match.group(2)
    param_pattern = r"(\w+)\s*=\s*['\"](.+?)['\"]|(\w+)\s*=\s*(\[.*\])|(\w+)\s*=\s*(\w+)"
    param_dict = {}
    for m in re.finditer(param_pattern, params):
        if m.group(1):
            param_dict[m.group(1)] = m.group(2)
        elif m.group(3):
            param_dict[m.group(3)] = m.group(4)
        elif m.group(5):
            param_dict[m.group(5)] = m.group(6)
    return api_name, param_dict


def api_match(pred_text: str, gt: dict) -> tuple[bool, str]:
    """Returns (passed, reason).

    Pass criteria:
      - api_name extracted matches gt['api_name'] (case-insensitive)
      - All keys in gt['param_dict'] appear in pred's param_dict
      - For each shared key: value match (string equality after strip)
    """
    api_name, param_dict = parse_api_call(pred_text)
    if api_name is None:
        return False, "no_api_call_in_prediction"
    gt_name = gt.get("api_name", "")
    if api_name.lower() != gt_name.lower():
        return False, f"name_mismatch: pred={api_name} gt={gt_name}"
    gt_params = gt.get("param_dict", {})
    if not isinstance(gt_params, dict):
        return False, "gt_params_not_dict"
    missing = [k for k in gt_params if k not in param_dict]
    if missing:
        return False, f"missing_params: {missing}"
    # value comparison: strict string equality after str() + strip()
    mismatches = []
    for k, v_gt in gt_params.items():
        v_pred = param_dict.get(k)
        if str(v_pred).strip() != str(v_gt).strip():
            mismatches.append(f"{k}: pred={v_pred!r} gt={v_gt!r}")
    if mismatches:
        return False, f"value_mismatch: {mismatches[:3]}"
    return True, "ok"


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 — uses rouge package (same as official evaluator)."""
    try:
        from rouge import Rouge
        rouge = Rouge()
        # rouge package crashes on empty strings
        if not reference.strip() or not hypothesis.strip():
            return 0.0
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]["rouge-l"]["f"]
    except Exception:
        return 0.0


def eval_file(path: str):
    out_path = path.replace("_responses.jsonl", "_eval_report.json")
    if not os.path.exists(path):
        print(f"[SKIP] {path}")
        return None

    rows = [json.loads(l) for l in open(path)]
    print(f"\n[{os.path.basename(path)}] {len(rows)} rows", flush=True)

    api_correct = 0; api_total = 0
    api_fail_reasons: dict[str, int] = defaultdict(int)
    rouge_scores: list[float] = []
    by_level: dict[str, dict] = defaultdict(lambda: {"api_pass": 0, "api_total": 0,
                                                       "rouge_sum": 0.0, "ai_total": 0})

    for r in rows:
        gt = r.get("ground_truth", {})
        pred = r.get("prediction", "") or ""
        level = r.get("level", "unknown")
        if gt.get("role") == "API":
            ok, why = api_match(pred, gt)
            api_total += 1
            by_level[level]["api_total"] += 1
            if ok:
                api_correct += 1
                by_level[level]["api_pass"] += 1
            else:
                api_fail_reasons[why.split(":")[0]] += 1
        elif gt.get("role") == "AI":
            score = rouge_l_f1(gt.get("text", ""), pred)
            rouge_scores.append(score)
            by_level[level]["rouge_sum"] += score
            by_level[level]["ai_total"] += 1

    api_acc = api_correct / api_total if api_total else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

    report = {
        "input": path,
        "n_rows": len(rows),
        "api_accuracy": api_acc * 100,
        "api_pass": api_correct,
        "api_total": api_total,
        "rouge_l_avg": avg_rouge,
        "rouge_l_count": len(rouge_scores),
        "api_fail_reasons": dict(api_fail_reasons),
        "by_level": {
            lvl: {
                "api_pass": d["api_pass"],
                "api_total": d["api_total"],
                "api_acc_pct": (100*d["api_pass"]/d["api_total"]) if d["api_total"] else None,
                "ai_total": d["ai_total"],
                "rouge_l_avg": (d["rouge_sum"]/d["ai_total"]) if d["ai_total"] else None,
            } for lvl, d in by_level.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  api: {api_correct}/{api_total} = {100*api_acc:.1f}%")
    print(f"  rouge-L (n={len(rouge_scores)}): {avg_rouge:.3f}")
    for lvl, d in sorted(by_level.items()):
        api_s = f"{d['api_pass']}/{d['api_total']}={100*d['api_pass']/d['api_total']:.1f}%" if d['api_total'] else "-"
        rouge_s = f"{d['rouge_sum']/d['ai_total']:.3f}" if d['ai_total'] else "-"
        print(f"  [{lvl}] api={api_s}  rouge={rouge_s}  ai_n={d['ai_total']}")
    if api_fail_reasons:
        print(f"  api fail buckets: {dict(api_fail_reasons)}")
    print(f"[saved] {out_path}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        try:
            eval_file(p)
        except Exception as e:
            import traceback
            print(f"[FAIL] {p}: {e}", file=sys.stderr)
            traceback.print_exc()


if __name__ == "__main__":
    main()
