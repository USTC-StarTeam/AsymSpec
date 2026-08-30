#!/usr/bin/env python3
"""
Phase 1.2: Tau Sampling Stability

Note: Current bench_lb.py/bench_mc_v07.py do not natively support temperature parameter.
This script documents the limitation and provides a workaround plan.

For now, we use greedy (τ=0) baseline and note that τ>0 testing requires:
  1. Custom vLLM configuration, or
  2. Fork of bench_lb.py with temperature support, or
  3. Manual integration in Phase 2

Usage:
  python scripts/evaluate_tau_stability.py --status      # Check framework support
  python scripts/evaluate_tau_stability.py --plan         # Generate workaround plan
"""
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "experiments" / "tau_stability_2026-05-23"


def check_framework_support() -> dict:
    """
    Check if existing bench scripts support temperature parameter.
    """
    bench_lb = REPO_ROOT / "scripts" / "bench_lb.py"

    support = {
        "temperature_cli": False,
        "temperature_env": False,
        "recommendation": "See plan.json"
    }

    # Check for --temperature in bench_lb.py
    if bench_lb.exists():
        content = bench_lb.read_text()
        if "--temperature" in content or "temperature" in content:
            support["temperature_cli"] = True
        if "TEMPERATURE" in content or "TEMP" in content:
            support["temperature_env"] = True

    return support


def generate_workaround_plan() -> dict:
    """
    Generate a concrete plan for τ>0 testing.
    """
    plan = {
        "status": "Tau parameter not natively supported in CLI",
        "blocking": True,
        "workarounds": [
            {
                "option": "A: Minimal vLLM config modification",
                "effort": "1-2 hours",
                "approach": """
                    Modify bench_lb.py to pass temperature to vLLM:

                    # In bench_lb.py, around generation loop:
                    generation_config = GenerationConfig(
                        temperature=args.temperature,  # Add this
                        top_p=0.9,
                        max_new_tokens=128,
                    )

                    # Add CLI argument:
                    ap.add_argument("--temperature", type=float, default=0.0,
                                    help="Sampling temperature (0.0=greedy)")
                """,
                "phase": "Can be done in Phase 1.2 or Phase 2"
            },
            {
                "option": "B: Environment variable override",
                "effort": "30 minutes",
                "approach": """
                    Control vLLM sampling via VLLM_* environment variables.
                    May require forking vLLM or custom sampling logic.
                    Lower reliability than Option A.
                """,
                "phase": "Phase 2 - lower priority"
            },
            {
                "option": "C: Defer to Phase 2",
                "effort": "0 (defer)",
                "approach": """
                    Run greedy (τ=0) now for Phase 1 completeness.
                    Plan τ>0 testing as Phase 2 task.
                    Document as "limitation" in paper revision.
                """,
                "phase": "Recommended for Phase 1"
            }
        ],
        "data": {
            "greedy_baseline": {
                "available": True,
                "source": "experiments/campaign_wave1_2026-05-19/",
                "note": "Existing data sufficient for Phase 1"
            },
            "sampling_experiments": {
                "available": False,
                "blocker": "No τ>0 framework support",
                "phase": "Phase 2"
            }
        }
    }

    return plan


def generate_phase_1_status() -> dict:
    """
    Current status: what can be done in Phase 1 with existing framework.
    """
    return {
        "phase": "1.2",
        "task": "Tau Sampling Stability",
        "current_status": "Framework limitation detected",
        "available_data": [
            {
                "dataset": "LongBench",
                "task": "2wikimqa, hotpotqa",
                "temperature": 0.0,
                "source": "experiments/campaign_wave1_2026-05-19/",
                "completeness": "Available"
            },
            {
                "dataset": "MultiChallenge",
                "task": "conversation",
                "temperature": 0.0,
                "source": "experiments/campaign_*",
                "completeness": "Available"
            }
        ],
        "missing_data": [
            {
                "dataset": "Any",
                "temperature": [0.7, 1.0],
                "blocking": True,
                "solution": "Implement Option A or C from workaround_plan"
            }
        ],
        "action_items": [
            "1. Choose workaround (recommend: Option C - defer τ>0 to Phase 2)",
            "2. If Option A: modify bench_lb.py, rerun with --temperature 0.7 1.0",
            "3. If Option C: note limitation in paper, schedule for Phase 2",
            "4. Use greedy baseline for Phase 1 evaluation"
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.2: Tau Stability Assessment"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check framework support for temperature parameter"
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Generate workaround plan"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        print("\n" + "="*70)
        print("Tau Framework Support Check")
        print("="*70)

        support = check_framework_support()
        print(f"Temperature CLI support: {support['temperature_cli']}")
        print(f"Temperature env support: {support['temperature_env']}")

        with open(OUTPUT_DIR / "support_check.json", "w") as f:
            json.dump(support, f, indent=2)

        if not support["temperature_cli"] and not support["temperature_env"]:
            print("\n⚠ Current framework does NOT support temperature parameter")
            print("  → See 'plan.json' for workaround options")

    if args.plan:
        print("\n" + "="*70)
        print("Tau Stability Workaround Plan")
        print("="*70)

        plan = generate_workaround_plan()
        status = generate_phase_1_status()

        # Print summary
        print(f"\nStatus: {plan['status']}")
        print(f"Blocking: {plan['blocking']}")
        print(f"\nWorkaround Options:")
        for i, wk in enumerate(plan["workarounds"], 1):
            print(f"\n{i}. {wk['option']}")
            print(f"   Effort: {wk['effort']}")
            print(f"   Phase: {wk['phase']}")

        print(f"\n\nRecommendation:")
        print(f"→ Option C: Defer τ>0 to Phase 2")
        print(f"  Use greedy (τ=0) data for Phase 1 completeness")
        print(f"  This allows Phase 1 to finish on schedule")

        # Save
        with open(OUTPUT_DIR / "plan.json", "w") as f:
            json.dump(plan, f, indent=2)

        with open(OUTPUT_DIR / "phase1_status.json", "w") as f:
            json.dump(status, f, indent=2)

        print(f"\n\nDetails saved to:")
        print(f"  - {OUTPUT_DIR}/plan.json")
        print(f"  - {OUTPUT_DIR}/phase1_status.json")

    if not args.status and not args.plan:
        # Default: show summary
        print("\nPhase 1.2: Tau Sampling Stability")
        print("="*70)
        print("\nCurrent Status: Framework limitation detected")
        print("  - bench_lb.py does not natively support --temperature")
        print("  - Greedy (τ=0) data is available from existing campaigns")
        print("\nNext Steps:")
        print("  1. Run: python scripts/evaluate_tau_stability.py --plan")
        print("  2. Choose workaround (recommend: defer τ>0 to Phase 2)")
        print("  3. Continue Phase 1 with greedy baseline")

        # Minimal output
        status = generate_phase_1_status()
        with open(OUTPUT_DIR / "readme.txt", "w") as f:
            f.write("Phase 1.2: Tau Stability\n")
            f.write("========================\n\n")
            f.write("Status: Temperature parameter not supported in CLI\n")
            f.write("Recommendation: Defer τ>0 testing to Phase 2\n")
            f.write("Use greedy (τ=0) baseline for Phase 1\n")

        print(f"\nStatus saved to {OUTPUT_DIR}/readme.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
