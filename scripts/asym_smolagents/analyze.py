"""Aggregate all smolagents AsymSpec demo results into a summary."""
import json
import sys
from pathlib import Path


DEMO_DIRS = {
    "demo2_benchmark100": "Demo 2: SimpleQA n=100 (main_compression=truncate)",
    "demo2_benchmark100_llmlingua": "Demo 2 ablation: SimpleQA n=100 (main_compression=llmlingua)",
    "demo3_gaia_l1_web": "Demo 3: GAIA L1 web-only n=42 (main_compression=truncate)",
}
MODES = ["asym_cda", "b1_main", "b1_aug"]
ROOT = Path("experiments/v010_smolagents_agent")


def load_mode(demo_dir: str, mode: str) -> dict:
    p = ROOT / demo_dir / f"{mode}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def summarize_mode(d: dict) -> dict:
    r = d["results"]
    n_total = d.get("n_total", len(r))
    n_done = len(r)
    n_correct = sum(x["correct"] for x in r)
    n_err = sum(1 for x in r if x.get("err"))
    elapsed = sum(x["elapsed_s"] for x in r)
    return {
        "n_total": n_total, "n_done": n_done,
        "n_correct": n_correct, "n_err": n_err,
        "acc_pct": round(n_correct / max(n_done, 1) * 100, 1),
        "err_rate_pct": round(n_err / max(n_done, 1) * 100, 1),
        "total_elapsed_s": round(elapsed),
        "mean_per_q_s": round(elapsed / max(n_done, 1), 1),
    }


def main():
    summary = {"demos": {}}
    print("=" * 78)
    print("AsymSpec smolagents — 6-cell results summary")
    print("=" * 78)
    for demo_dir, title in DEMO_DIRS.items():
        print(f"\n### {title}")
        demo_summary = {}
        for mode in MODES:
            d = load_mode(demo_dir, mode)
            if d is None:
                print(f"  [{mode}] MISSING")
                demo_summary[mode] = None
                continue
            s = summarize_mode(d)
            diag = d.get("asym_diag", {})
            print(f"  [{mode:9}] acc={s['n_correct']:>3}/{s['n_done']:<3} "
                  f"= {s['acc_pct']:>5.1f}%  err={s['n_err']} "
                  f"mean_s={s['mean_per_q_s']:.1f}", end="")
            if diag:
                print(f"  | asym_ratio={diag.get('ratio_mean', 0):.2f}× "
                      f"(aug={diag.get('aug_tokens_mean', 0):.0f} tok, "
                      f"main={diag.get('main_tokens_mean', 0):.0f} tok)")
            else:
                print()
            demo_summary[mode] = {"summary": s, "diag": diag}
        # Recovery rate
        if all(demo_summary.get(m) for m in MODES):
            a = demo_summary["asym_cda"]["summary"]["acc_pct"]
            f_ = demo_summary["b1_main"]["summary"]["acc_pct"]
            c = demo_summary["b1_aug"]["summary"]["acc_pct"]
            gap = c - f_
            asym_delta_floor = a - f_
            recovery = (a - f_) / max(gap, 0.01) * 100 if gap > 0 else 0
            print(f"\n  Floor→Ceiling gap: {gap:+.1f} pp")
            print(f"  Asym vs Floor:     {asym_delta_floor:+.1f} pp "
                  f"({'WIN' if asym_delta_floor > 0 else 'LOSS' if asym_delta_floor < 0 else 'TIE'})")
            print(f"  Gap recovery:      {recovery:.0f}%")
            demo_summary["_delta"] = {
                "floor_ceiling_gap_pp": round(gap, 1),
                "asym_vs_floor_pp": round(asym_delta_floor, 1),
                "gap_recovery_pct": round(recovery),
                "verdict": ("WIN" if asym_delta_floor > 1 else
                            "TIE" if abs(asym_delta_floor) <= 1 else "LOSS"),
            }
        summary["demos"][demo_dir] = demo_summary

    print("\n" + "=" * 78)
    summary_path = ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
