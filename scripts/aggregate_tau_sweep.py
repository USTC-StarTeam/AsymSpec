#!/usr/bin/env python3
"""Aggregate τ-sweep cells into a SUMMARY.md.

Reads experiments/tau_sweep_2026-05-24/k2/{cell}_eval.json (per-subset F1
from eval_lb.py) + {cell}.json (spec_metrics, tps) → writes SUMMARY.md.

Usage:
  # First run F1 eval on all cells:
  python scripts/eval_lb.py --exp_dir experiments/tau_sweep_2026-05-24
  # Then aggregate:
  python scripts/aggregate_tau_sweep.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "tau_sweep_2026-05-24" / "k2"

MODES = ["floor", "asym_cda"]  # cell-name short forms in launch_tau_sweep.sh
TAUS = [0.0, 0.7, 1.0]


def load_cell(short: str, tau: float) -> dict | None:
    cell = f"{short}_t{tau}"
    metric_path = EXP / f"{cell}.json"
    eval_path = EXP / f"{cell}_eval.json"
    if not metric_path.exists():
        return None
    d = json.load(open(metric_path))
    eval_d = json.load(open(eval_path)) if eval_path.exists() else None
    return {"metric": d, "eval": eval_d, "cell": cell}


def main():
    rows = {}
    for short in MODES:
        for tau in TAUS:
            r = load_cell(short, tau)
            rows[(short, tau)] = r

    out = []
    out.append("# τ-sweep on LongBench (n=600, K=2, 4B drafter)\n\n")
    out.append("Config: identical to headline (`lb_g05_vnorm_K2.json`) except for ")
    out.append("`--temperature {0.0, 0.7, 1.0}` + `--top_p {1.0, 0.9, 0.9}` + `--seed 42`.\n\n")

    # Overall F1 grid
    out.append("## Overall F1 (mean of 3 multi-hop subsets)\n\n")
    out.append("| Mode | τ=0.0 | τ=0.7 | τ=1.0 |\n")
    out.append("|---|---:|---:|---:|\n")
    for short in MODES:
        cells = [(rows.get((short, t)) or {}) for t in TAUS]
        row_label = {"floor": "Floor (b1_main, summary)",
                     "asym_cda": "AsymSpec (specsteer+CDA)"}[short]
        vals = []
        for r in cells:
            if not r or not r.get("eval"):
                vals.append("—")
                continue
            ev = r["eval"]["per_dataset"].get("overall", {})
            f1 = ev.get("f1")
            vals.append(f"{f1*100:.1f}" if f1 is not None else "—")
        out.append(f"| {row_label} | {vals[0]} | {vals[1]} | {vals[2]} |\n")

    # Gain row
    floor_vals, asym_vals = [], []
    for tau in TAUS:
        for short, target in [("floor", floor_vals), ("asym_cda", asym_vals)]:
            r = rows.get((short, tau))
            if r and r.get("eval"):
                ev = r["eval"]["per_dataset"].get("overall", {})
                f1 = ev.get("f1")
                target.append(f1 * 100 if f1 is not None else None)
            else:
                target.append(None)
    out.append("| **Gain (Δ)** | ")
    for f, a in zip(floor_vals, asym_vals):
        if f is None or a is None:
            out.append("— | ")
        else:
            out.append(f"**{a - f:+.1f}** | ")
    out.append("\n\n")

    # Per-subset breakdown
    for subset in ("hotpotqa", "2wikimqa", "musique"):
        out.append(f"## Per-subset F1: {subset}\n\n")
        out.append("| Mode | τ=0.0 | τ=0.7 | τ=1.0 |\n")
        out.append("|---|---:|---:|---:|\n")
        for short in MODES:
            row_label = {"floor": "Floor", "asym_cda": "AsymSpec"}[short]
            vals = []
            for tau in TAUS:
                r = rows.get((short, tau))
                if not r or not r.get("eval"):
                    vals.append("—")
                    continue
                ev = r["eval"]["per_dataset"].get(subset, {})
                f1 = ev.get("f1")
                vals.append(f"{f1*100:.1f}" if f1 is not None else "—")
            out.append(f"| {row_label} | {vals[0]} | {vals[1]} | {vals[2]} |\n")
        out.append("\n")

    # Spec metrics (acceptance, MAL) — only for specsteer cells
    out.append("## AsymSpec acceptance metrics\n\n")
    out.append("| τ | AR | MAL | per-pos AR | tps |\n")
    out.append("|---|---:|---:|---|---:|\n")
    for tau in TAUS:
        r = rows.get(("asym_cda", tau))
        if not r:
            out.append(f"| {tau} | — | — | — | — |\n")
            continue
        m = r["metric"]
        sm = m.get("spec_metrics") or {}
        ar = sm.get("draft_acceptance_rate")
        mal = sm.get("mean_acceptance_length")
        pp = sm.get("per_position_acceptance_rate") or []
        tps = m.get("tps")
        out.append(
            f"| {tau} | "
            f"{ar:.3f} | " if ar is not None else f"| {tau} | — | "
        )
        out.append(
            f"{mal:.2f} | " if mal is not None else "— | "
        )
        out.append(
            ("[" + ", ".join(f"{p:.3f}" for p in pp) + "]") if pp else "—"
        )
        out.append(" | ")
        out.append(f"{tps:.1f} |\n" if tps is not None else "— |\n")

    summary_path = EXP.parent / "SUMMARY.md"
    summary_path.write_text("".join(out))
    print(f"✓ wrote {summary_path}")
    print()
    print("".join(out))


if __name__ == "__main__":
    main()
