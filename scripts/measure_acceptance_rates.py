#!/usr/bin/env python3
"""
Phase 1.3: Distributional Fidelity via Acceptance Rates

Measures token-level acceptance rates as a lightweight proxy for
distributional drift. High acceptance → stable distribution.

Usage:
  python scripts/measure_acceptance_rates.py --source experiments/campaign_wave1_2026-05-19
  python scripts/measure_acceptance_rates.py --analyze-only
"""
import json
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "experiments" / "fidelity_analysis_2026-05-23"


def parse_logs_for_acceptance(log_dir: Path) -> Dict[str, Dict]:
    """
    Parse benchmark logs for acceptance/rejection statistics.

    Logs may contain lines like:
      [accept] step=123 accepted=1 rejected=0
      [decode] tokens_accepted=45 tokens_rejected=5

    Returns:
        {
            "total_steps": 1000,
            "total_accepted": 850,
            "total_rejected": 150,
            "acceptance_rate": 0.85,
            "by_log": {...}
        }
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"⚠ Log directory not found: {log_dir}")
        return {}

    stats = {
        "total_steps": 0,
        "total_accepted": 0,
        "total_rejected": 0,
        "acceptance_rate": 0.0,
        "by_log": {},
        "search_patterns": [
            "\\[accept\\].*?accepted=(\\d+).*?rejected=(\\d+)",
            "tokens_accepted=(\\d+).*?tokens_rejected=(\\d+)",
            "accept.*?(\\d+).*?reject.*(\\d+)",
        ]
    }

    log_files = list(log_dir.glob("*.log"))
    print(f"Found {len(log_files)} log files")

    for log_file in log_files:
        file_stats = {
            "accepted": 0,
            "rejected": 0,
            "rate": 0.0,
            "matches": 0
        }

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

                # Try multiple patterns
                for pattern in stats["search_patterns"]:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match) == 2:
                            accepted = int(match[0])
                            rejected = int(match[1])
                            file_stats["accepted"] += accepted
                            file_stats["rejected"] += rejected
                            file_stats["matches"] += 1

        except Exception as e:
            print(f"  ⚠ Error reading {log_file.name}: {e}")

        if file_stats["matches"] > 0:
            total = file_stats["accepted"] + file_stats["rejected"]
            if total > 0:
                file_stats["rate"] = file_stats["accepted"] / total
                stats["by_log"][log_file.name] = file_stats

                stats["total_accepted"] += file_stats["accepted"]
                stats["total_rejected"] += file_stats["rejected"]

    # Compute overall acceptance rate
    total = stats["total_accepted"] + stats["total_rejected"]
    if total > 0:
        stats["acceptance_rate"] = stats["total_accepted"] / total
        stats["total_steps"] = total

    return stats


def analyze_acceptance(stats: Dict) -> str:
    """
    Analyze acceptance rates to infer distributional fidelity.
    """
    analysis = []
    analysis.append("\nAcceptance Rate Analysis")
    analysis.append("=" * 60)

    if not stats or stats["total_steps"] == 0:
        analysis.append("\n⚠ No acceptance data found in logs")
        analysis.append("  Possible reasons:")
        analysis.append("    1. Logs don't contain acceptance metrics")
        analysis.append("    2. Search patterns don't match log format")
        analysis.append("    3. Logs are from different benchmark variant")
        return "\n".join(analysis)

    analysis.append(f"\nTotal steps measured: {stats['total_steps']}")
    analysis.append(f"Total accepted: {stats['total_accepted']}")
    analysis.append(f"Total rejected: {stats['total_rejected']}")
    analysis.append(f"Acceptance rate: {stats['acceptance_rate']:.1%}")

    # Interpretation
    analysis.append(f"\nINTERPRETATION:")
    if stats["acceptance_rate"] >= 0.85:
        analysis.append(f"""
✓ High acceptance rate (>85%)
  → Token-level agreement is very high
  → Indicates stable distributional recovery
  → AsymSpec's δ-fusion does not cause major divergence

  Paper text: "Token-level acceptance rate of {stats['acceptance_rate']:.1%}
  demonstrates that δ-fusion maintains close alignment with the
  verifier's distribution, validating the distributional fidelity claim."
""")
    elif stats["acceptance_rate"] >= 0.70:
        analysis.append(f"""
≈ Moderate acceptance rate (70-85%)
  → Some distributional divergence present
  → But not catastrophic
  → May warrant additional validation

  Paper text: "Token-level acceptance rate of {stats['acceptance_rate']:.1%}
  indicates moderate distributional alignment. While δ-fusion introduces
  controlled divergence, acceptance rates remain high enough for stable
  agentic loop performance."
""")
    else:
        analysis.append(f"""
✗ Low acceptance rate (<70%)
  → Significant distributional shift
  → May indicate method instability
  → Needs investigation

  Action: Consider tweaking β or JSD threshold
""")

    if stats["by_log"]:
        analysis.append(f"\n\nPer-log breakdown ({len(stats['by_log'])} files):")
        for name, log_stats in sorted(stats["by_log"].items())[:5]:  # Top 5
            analysis.append(
                f"  {name[:40]:<40} rate={log_stats['rate']:.1%} "
                f"({log_stats['accepted']}/{log_stats['accepted']+log_stats['rejected']})"
            )

    return "\n".join(analysis)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.3: Acceptance Rate Analysis"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Log directory to analyze"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip log parsing, analyze existing results.json"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {}

    if args.analyze_only:
        results_file = OUTPUT_DIR / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                stats = json.load(f)
            print(f"Loaded existing results from {results_file}")
        else:
            print(f"⚠ No existing results found at {results_file}")
            print(f"  Run with --source first to generate results")
            return 1

    else:
        # Parse logs
        if args.source is None:
            # Default: try campaign_wave1_2026-05-19
            args.source = REPO_ROOT / "experiments" / "campaign_wave1_2026-05-19"

        print(f"\n{'='*70}")
        print(f"Parsing logs from: {args.source}")
        print(f"{'='*70}\n")

        stats = parse_logs_for_acceptance(args.source)

        # Save raw stats
        stats["source"] = str(args.source)
        stats["timestamp"] = datetime.now().isoformat()

        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Results saved to {OUTPUT_DIR}/results.json")

    # Analyze
    if stats:
        print(f"\n{'='*70}")
        print("Analysis")
        print(f"{'='*70}")

        analysis = analyze_acceptance(stats)
        print(analysis)

        # Save analysis
        with open(OUTPUT_DIR / "analysis.txt", "w") as f:
            f.write(analysis)

        print(f"\n✓ Analysis saved to {OUTPUT_DIR}/analysis.txt")

    else:
        print("\n⚠ No stats collected. Check log format.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
