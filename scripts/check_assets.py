#!/usr/bin/env python3
"""Check whether local benchmark assets are ready before a GPU run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (  # noqa: E402
    APIBANK_DIR,
    LB_RAW,
    LB_SUMMARIES,
    MC_QUESTIONS,
    MC_SUMMARIES,
    MV_CAPTIONS,
    MV_OCRS,
)


ASSETS: dict[str, list[tuple[Path, str]]] = {
    "longbench": [
        *((LB_RAW / f"{task}.jsonl", "bash scripts/download_datasets.sh")
          for task in ("hotpotqa", "2wikimqa", "musique")),
        (LB_SUMMARIES, "python scripts/gen_lb_summaries.py"),
    ],
    "multichallenge": [
        (MC_QUESTIONS, "bash scripts/download_datasets.sh"),
        (MC_SUMMARIES, "python scripts/gen_mc_summaries.py"),
    ],
    "api-bank": [
        (APIBANK_DIR / "apis", "bash scripts/download_datasets.sh"),
        (APIBANK_DIR / "lv1-lv2-samples", "bash scripts/download_datasets.sh"),
    ],
    "mathvista": [
        (MV_CAPTIONS, "prepare the official Bard-caption artifact"),
        (MV_OCRS, "prepare the EasyOCR artifact"),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["all", *ASSETS],
        default="all",
        help="asset group to inspect (default: all)",
    )
    args = parser.parse_args()

    selected = ASSETS if args.benchmark == "all" else {
        args.benchmark: ASSETS[args.benchmark]
    }
    missing: list[tuple[str, Path, str]] = []
    for benchmark, assets in selected.items():
        print(f"[{benchmark}]")
        for path, remedy in assets:
            exists = path.exists()
            print(f"  {'ok' if exists else 'MISSING'}  {path}")
            if not exists:
                missing.append((benchmark, path, remedy))

    if not missing:
        print("All selected local assets are ready.")
        return 0

    print("\nPreparation needed:")
    seen: set[str] = set()
    for _, _, remedy in missing:
        if remedy not in seen:
            print(f"  - {remedy}")
            seen.add(remedy)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
