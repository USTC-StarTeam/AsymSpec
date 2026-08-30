#!/usr/bin/env python3
"""Phase 1.3 fidelity analysis on τ-sweep outputs.

Two signals for "did AsymSpec drift catastrophically from Floor?":
  1. Acceptance metrics (AR / MAL / per-pos AR) from spec_metrics —
     paper claims ~0.85 AR on LongBench K=2 4B drafter.
  2. Token-level response overlap (F1) between Floor and AsymSpec at each
     τ — both should produce semantically similar outputs (or AsymSpec
     should systematically improve on Floor, not orthogonally drift).

Writes: experiments/fidelity_analysis_2026-05-24/results.json
        experiments/fidelity_analysis_2026-05-24/SUMMARY.md
"""
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAU_DIR = REPO / "experiments" / "tau_sweep_2026-05-24" / "k2"
OUT_DIR = REPO / "experiments" / "fidelity_analysis_2026-05-24"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAUS = [0.0, 0.7, 1.0]


def normalize(s: str) -> str:
    """LongBench-style: lowercase, strip articles, strip punctuation."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def token_f1(a: str, b: str) -> float:
    """Token-level F1 between two response strings."""
    a_toks = normalize(a).split()
    b_toks = normalize(b).split()
    if not a_toks or not b_toks:
        return 0.0
    common = Counter(a_toks) & Counter(b_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(a_toks)
    rec = n / len(b_toks)
    return 2 * prec * rec / (prec + rec)


def load_responses(cell: str) -> dict[str, str]:
    """Returns {_id: response_text} for a given cell."""
    path = TAU_DIR / f"{cell}_responses.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d["response"]
    return out


def load_spec_metrics(cell: str) -> dict | None:
    """Returns spec_metrics dict (or None if Floor cell / missing)."""
    path = TAU_DIR / f"{cell}.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    return d.get("spec_metrics")


def main():
    results = {"per_tau": {}, "acceptance": {}}

    for tau in TAUS:
        floor = load_responses(f"floor_t{tau}")
        asym = load_responses(f"asym_cda_t{tau}")
        if not floor or not asym:
            print(f"[skip] τ={tau} missing cells (floor={len(floor)} asym={len(asym)})")
            continue
        shared_ids = set(floor) & set(asym)
        f1s = [token_f1(floor[i], asym[i]) for i in shared_ids]
        # Exact-after-normalize match
        ems = [int(normalize(floor[i]) == normalize(asym[i])) for i in shared_ids]
        results["per_tau"][tau] = {
            "n_shared": len(shared_ids),
            "token_f1_mean": sum(f1s) / len(f1s) if f1s else 0.0,
            "token_f1_min": min(f1s) if f1s else 0.0,
            "token_f1_max": max(f1s) if f1s else 0.0,
            "exact_match_rate": sum(ems) / len(ems) if ems else 0.0,
        }
        # Acceptance from AsymSpec cell only
        sm = load_spec_metrics(f"asym_cda_t{tau}")
        if sm:
            results["acceptance"][tau] = {
                "draft_acceptance_rate": sm.get("draft_acceptance_rate"),
                "mean_acceptance_length": sm.get("mean_acceptance_length"),
                "per_position_acceptance_rate": sm.get("per_position_acceptance_rate"),
                "num_drafts": sm.get("num_drafts"),
                "num_draft_tokens": sm.get("num_draft_tokens"),
                "num_accepted_tokens": sm.get("num_accepted_tokens"),
            }

    # Write JSON
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    # Write SUMMARY.md
    md = []
    md.append("# Phase 1.3 fidelity analysis (τ-sweep on LongBench)\n\n")
    md.append("Two questions, two signals:\n\n")
    md.append("1. **AR stays high under sampling?** → AsymSpec acceptance "
              "doesn't collapse as τ grows.\n")
    md.append("2. **Outputs stay similar between Floor and AsymSpec?** → token-F1 ")
    md.append("of paired responses stays high (semantic drift bounded).\n\n")

    md.append("## Acceptance under sampling\n\n")
    md.append("| τ | AR | MAL | per-pos AR | n_drafts | accepted_tokens |\n")
    md.append("|---:|---:|---:|---|---:|---:|\n")
    for tau in TAUS:
        a = results["acceptance"].get(tau)
        if not a:
            md.append(f"| {tau} | — | — | — | — | — |\n")
            continue
        pp = "[" + ", ".join(f"{p:.3f}" for p in (a["per_position_acceptance_rate"] or [])) + "]"
        md.append(
            f"| {tau} | {a['draft_acceptance_rate']:.3f} | "
            f"{a['mean_acceptance_length']:.2f} | {pp} | "
            f"{a['num_drafts']} | {a['num_accepted_tokens']} |\n"
        )

    md.append("\n## Floor ↔ AsymSpec response overlap\n\n")
    md.append("Token-F1 (LongBench-style normalization) between paired (Floor, AsymSpec) ")
    md.append("responses, per `_id`. High F1 = same answer; low = orthogonal drift.\n\n")
    md.append("| τ | n_shared | token-F1 mean | min | max | exact-match rate |\n")
    md.append("|---:|---:|---:|---:|---:|---:|\n")
    for tau in TAUS:
        p = results["per_tau"].get(tau)
        if not p:
            md.append(f"| {tau} | — | — | — | — | — |\n")
            continue
        md.append(
            f"| {tau} | {p['n_shared']} | "
            f"{p['token_f1_mean']:.3f} | {p['token_f1_min']:.3f} | "
            f"{p['token_f1_max']:.3f} | {p['exact_match_rate']:.3f} |\n"
        )

    md.append("\n## Reading\n\n")
    md.append("- AR > 0.80 at τ=1.0 → method is robust to sampling.\n")
    md.append("- Floor↔AsymSpec F1 > 0.60 → outputs stay semantically aligned.\n")
    md.append("- If F1 collapses but accuracy stays similar, AsymSpec found a ")
    md.append("different valid answer (still good).\n")

    (OUT_DIR / "SUMMARY.md").write_text("".join(md))
    print(f"✓ wrote {OUT_DIR / 'results.json'}")
    print(f"✓ wrote {OUT_DIR / 'SUMMARY.md'}")
    print()
    print("".join(md))


if __name__ == "__main__":
    main()
