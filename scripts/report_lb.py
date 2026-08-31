"""LongBench results consolidation — merge bench JSON + eval JSON, produce
comparison table per dataset and overall. Print Markdown table + save CSV.

Usage:
  python report_lb.py --exp_dir <dir>
"""
from __future__ import annotations
import argparse, json, os, csv
from collections import defaultdict


def load_cell(out_json: str, eval_json: str | None) -> dict:
    """Load bench output + (optional) eval result for one cell."""
    d = json.load(open(out_json))
    cell = d["cell"]
    K = d["K"]
    tps = d.get("tps", 0.0)
    n = d.get("n_kept", 0)
    sm = d.get("spec_metrics") or {}
    ar = sm.get("draft_acceptance_rate")
    mal = sm.get("mean_acceptance_length")
    per_pos = sm.get("per_position_acceptance_rate") or []
    per_ds = d.get("per_dataset") or {}

    f1_per_ds, em_per_ds, judge_per_ds = {}, {}, {}
    if eval_json and os.path.exists(eval_json):
        ev = json.load(open(eval_json))
        for ds, s in ev.get("per_dataset", {}).items():
            f1_per_ds[ds] = s["f1"]
            em_per_ds[ds] = s["em"]
        if "judge" in ev:
            for ds, s in ev["judge"].items():
                judge_per_ds[ds] = s["judge_acc"]

    return {
        "cell": cell, "K": K, "tps": tps, "n": n,
        "AR": ar, "MAL": mal, "per_pos_AR": per_pos,
        "per_ds_n": {ds: per_ds.get(ds, {}).get("n", 0)
                     for ds in ("hotpotqa", "2wikimqa", "musique")},
        "f1": f1_per_ds, "em": em_per_ds, "judge": judge_per_ds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    args = ap.parse_args()

    cells = []
    for sub in ("k_indep", "k2", "k4", "k6"):
        d = os.path.join(args.exp_dir, sub)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".json") and not fname.endswith("_eval.json") \
                    and not fname.endswith(".partial"):
                out_path = os.path.join(d, fname)
                eval_path = out_path.replace(".json", "_eval.json")
                # Check if this is a bench json (not eval/responses)
                try:
                    test = json.load(open(out_path))
                    if "cell" not in test or "tps" not in test:
                        continue
                except Exception:
                    continue
                cells.append(load_cell(out_path, eval_path))

    if not cells:
        print("No cells found.")
        return

    # Sort: K-independent first, then by (K, mode, slm)
    def sort_key(c):
        cell, K = c["cell"], c["K"]
        if cell.startswith("B1"):
            return (0, K, cell)
        is_ss = cell.startswith("ss_")
        return (1, K, 0 if is_ss else 1, cell)
    cells.sort(key=sort_key)

    # Markdown table
    print("\n# LongBench Results\n")
    print(f"Experiment dir: `{args.exp_dir}`\n")
    cols = ["cell", "K", "n", "tps", "AR", "MAL",
            "F1_hotpot", "F1_2wiki", "F1_musique", "F1_mean",
            "EM_mean", "judge_mean"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    rows_csv = [cols]
    for c in cells:
        f1_h = c["f1"].get("hotpotqa", float("nan"))
        f1_w = c["f1"].get("2wikimqa", float("nan"))
        f1_m = c["f1"].get("musique", float("nan"))
        f1_mean = c["f1"].get("overall", float("nan"))
        em_mean = c["em"].get("overall", float("nan"))
        judge = c["judge"].get("overall", float("nan"))
        ar_s = f"{c['AR']:.3f}" if c["AR"] is not None else "—"
        mal_s = f"{c['MAL']:.2f}" if c["MAL"] is not None else "—"
        row = [
            c["cell"], str(c["K"]), str(c["n"]), f"{c['tps']:.1f}", ar_s, mal_s,
            f"{f1_h:.3f}" if f1_h == f1_h else "—",
            f"{f1_w:.3f}" if f1_w == f1_w else "—",
            f"{f1_m:.3f}" if f1_m == f1_m else "—",
            f"{f1_mean:.3f}" if f1_mean == f1_mean else "—",
            f"{em_mean:.3f}" if em_mean == em_mean else "—",
            f"{judge:.3f}" if judge == judge else "—",
        ]
        print("| " + " | ".join(row) + " |")
        rows_csv.append(row)

    # CSV
    csv_path = os.path.join(args.exp_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows_csv:
            w.writerow(r)
    print(f"\n[saved] CSV → {csv_path}")


if __name__ == "__main__":
    main()
