#!/usr/bin/env python3
"""
Phase 1.1: GAIA Ablation - Ceiling vs AsymSpec

Runs spike_gaia_rag.py in both Ceiling (b1_aug) and AsymSpec (asym_cda) modes,
then parses results to determine if the 22% vs 18.9% gain comes from δ-fusion
or context management.

Usage:
  python scripts/run_gaia_ablation.py --run-experiments
  python scripts/run_gaia_ablation.py --parse-only
"""
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, Optional
import sys
import argparse

REPO_ROOT = Path(__file__).parent.parent
SPIKE_SCRIPT = REPO_ROOT / "scripts" / "spike_gaia_rag.py"
GAIA_SPIKE_DIR = REPO_ROOT / "experiments" / "v010_gaia_spike"
OUTPUT_DIR = REPO_ROOT / "experiments" / "gaia_ablation_2026-05-23"


def run_gaia_mode(mode: str, verbose: bool = True) -> bool:
    """
    Run spike_gaia_rag.py in a specific mode.

    Args:
        mode: 'b1_aug' (Ceiling, full context) or 'asym_cda' (AsymSpec)
        verbose: Print output in real-time

    Returns:
        True if successful, False otherwise
    """
    cmd = ["python", str(SPIKE_SCRIPT), "--mode", mode]

    print(f"\n{'='*70}")
    print(f"Running GAIA {mode}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=not verbose, text=True)

    if result.returncode != 0:
        print(f"✗ GAIA {mode} failed (exit code {result.returncode})")
        if result.stderr:
            print(f"Error: {result.stderr[:500]}")
        return False

    print(f"✓ GAIA {mode} completed successfully")
    return True


def parse_gaia_results() -> Dict[str, Dict]:
    """
    Parse GAIA spike results from summary.log.

    Returns:
        {
            "b1_aug": {"accuracy": 0.18, "numerator": 1, "denominator": 5},
            "asym_cda": {"accuracy": 0.22, "numerator": 1, "denominator": 5},
            "raw_summary": "...",
            "timestamp": "2026-05-23T..."
        }
    """
    summary_log = GAIA_SPIKE_DIR / "summary.log"

    if not summary_log.exists():
        print(f"✗ Summary log not found: {summary_log}")
        return {}

    results = {}
    summary_text = summary_log.read_text()
    results["raw_summary"] = summary_text

    # Parse lines like:
    #   b1_aug  acc=1/5
    #   asym_cda  acc=2/5

    for line in summary_text.splitlines():
        line = line.strip()

        # Match "method_name  acc=X/Y"
        match = re.search(r'(b1_aug|asym_cda|asym_cma|b1_main)\s+acc=(\d+)/(\d+)', line)
        if match:
            method, numerator, denominator = match.groups()
            accuracy = int(numerator) / int(denominator)

            results[method] = {
                "numerator": int(numerator),
                "denominator": int(denominator),
                "accuracy": accuracy,
                "accuracy_pct": f"{100*accuracy:.1f}%"
            }
            print(f"✓ Parsed {method}: {numerator}/{denominator} = {100*accuracy:.1f}%")

    if not results or len(results) == 1:  # Only raw_summary found
        print(f"⚠ No accuracy lines found in summary.log")
        print(f"Trying alternative parsing...")

        # Alternative: Look for standalone accuracy mentions
        for line in summary_text.splitlines():
            for method in ["b1_aug", "asym_cda", "asym_cma", "b1_main"]:
                if method in line.lower():
                    print(f"  Found '{method}' mention: {line.strip()}")

    return results


def analyze_results(results: Dict) -> str:
    """
    Analyze parsed results to determine gain source.

    Returns:
        Analysis text for paper revision
    """
    if "b1_aug" not in results or "asym_cda" not in results:
        return "Incomplete data: missing b1_aug or asym_cda results"

    ceiling_acc = results["b1_aug"]["accuracy"]
    asym_acc = results["asym_cda"]["accuracy"]
    gain = asym_acc - ceiling_acc

    analysis = f"""
GAIA Ablation Analysis
{'='*50}

Ceiling (b1_aug, full context):     {results['b1_aug']['numerator']}/{results['b1_aug']['denominator']} = {results['b1_aug']['accuracy_pct']}
AsymSpec (asym_cda, compressed):    {results['asym_cda']['numerator']}/{results['asym_cda']['denominator']} = {results['asym_cda']['accuracy_pct']}

Gain: {gain:+.3f} ({100*gain:+.1f}pp)

INTERPRETATION:
"""

    if abs(gain) < 0.02:  # < 2pp difference
        analysis += """
→ Negligible difference (< 2pp)
  The 22% vs 18.9% gap in the paper may arise from:
    • Different question sets or difficulty subsets
    • Variation in RAG/search quality
    • Or the above experiment used different context management

→ Paper Revision: Reframe as "consistent performance under compression"
"""
    elif gain > 0.05:  # > 5pp gain for AsymSpec
        analysis += f"""
→ Significant gain ({100*gain:.1f}pp) for AsymSpec
  The δ-fusion mechanism provides measurable benefit even when
  Ceiling gets full context.

→ Paper Revision: "AsymSpec achieves {results['asym_cda']['accuracy_pct']}
  vs Ceiling's {results['b1_aug']['accuracy_pct']}, demonstrating that
  δ-fusion recovery sustains reasoning performance under compression."
"""
    else:
        analysis += f"""
→ Modest gain ({100*gain:.1f}pp) for AsymSpec
  Suggests partial contribution from both context management
  and δ-fusion mechanism.

→ Paper Revision: "AsymSpec achieves {results['asym_cda']['accuracy_pct']}
  on GAIA, with {100*gain:.1f}pp over Ceiling, attributed to a combination
  of compression-aware context management and δ-fusion recovery."
"""

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.1: GAIA Ablation experiment"
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run spike_gaia_rag.py for b1_aug and asym_cda (requires GPU)"
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse existing results from summary.log"
    )
    parser.add_argument(
        "--mode",
        choices=["b1_aug", "asym_cda", "both"],
        default="both",
        help="Which mode(s) to run (if --run-experiments)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print subprocess output"
    )

    args = parser.parse_args()

    # Default: run experiments if no args given
    if not args.run_experiments and not args.parse_only:
        args.run_experiments = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments if requested
    if args.run_experiments:
        if args.mode in ["b1_aug", "both"]:
            if not run_gaia_mode("b1_aug", verbose=args.verbose):
                print("Failed to run b1_aug; aborting")
                return 1

        if args.mode in ["asym_cda", "both"]:
            if not run_gaia_mode("asym_cda", verbose=args.verbose):
                print("Failed to run asym_cda; continuing to parse existing results")

    # Parse results
    print(f"\n{'='*70}")
    print("Parsing GAIA Results")
    print(f"{'='*70}\n")

    results = parse_gaia_results()

    if not results or len(results) == 1:
        print("⚠ No complete results to analyze")
        print(f"  Check {GAIA_SPIKE_DIR}/summary.log")
        return 1

    # Analyze
    analysis = analyze_results(results)
    print(analysis)

    # Save results
    results["analysis"] = analysis
    results["phase"] = "1.1"
    results["timestamp"] = __import__("datetime").datetime.now().isoformat()

    output_file = OUTPUT_DIR / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    analysis_file = OUTPUT_DIR / "analysis.txt"
    with open(analysis_file, "w") as f:
        f.write(analysis)

    print(f"\n{'='*70}")
    print(f"✓ Results saved to:")
    print(f"  - {output_file}")
    print(f"  - {analysis_file}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
