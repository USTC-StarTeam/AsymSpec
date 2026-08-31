#!/usr/bin/env python3
"""Deploy the AsymSpec patches into the installed vLLM package.

Hot-patches vLLM's pip-installed location. No vLLM fork is required.
``vllm_specsteer/vllm_0_19/`` is the single release patch set and supports
both text and multimodal drafters.

Files deployed:
  specsteer_model.py  → vllm/v1/spec_decode/specsteer_model.py
  specsteer_sampler.py → vllm/v1/sample/specsteer_sampler.py
  eagle.py → vllm/v1/spec_decode/eagle.py
  speculative.py → vllm/config/speculative.py
  gpu_model_runner.py → vllm/v1/worker/gpu_model_runner.py

Originals are backed up before being overwritten. ``--revert`` restores them.

Usage:
  python scripts/deploy_specsteer.py --check
  python scripts/deploy_specsteer.py --apply
  python scripts/deploy_specsteer.py --revert
"""
import argparse
import hashlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import REPO_ROOT, vllm_specsteer_targets

VLLM_SPECSTEER = REPO_ROOT / "vllm_specsteer"
PATCH_DIR = VLLM_SPECSTEER / "vllm_0_19"
SUPPORTED_VLLM_VERSION = "0.19.0"


def validate_vllm_version() -> str:
    """Reject full-file patching against an incompatible vLLM layout."""
    try:
        installed = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "vLLM is not installed. Install the pinned requirements first."
        ) from exc
    if installed != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"Installed vLLM is {installed}, but these full-file patches target "
            f"vLLM {SUPPORTED_VLLM_VERSION}. Refusing to overwrite an "
            "incompatible installation; see VLLM_COMPATIBILITY.md."
        )
    return installed


def md5_short(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:10]


def deployment_map() -> list[tuple[Path, Path]]:
    """Return the complete release patch mapping for vLLM 0.19.0."""
    if not PATCH_DIR.exists():
        raise FileNotFoundError(f"patch directory not found: {PATCH_DIR}")
    spec_model_tgt, sampler_tgt = vllm_specsteer_targets()
    vllm_pkg = spec_model_tgt.parents[2]
    return [
        (PATCH_DIR / "specsteer_model.py", spec_model_tgt),
        (PATCH_DIR / "specsteer_sampler.py", sampler_tgt),
        (PATCH_DIR / "eagle.py", spec_model_tgt.parent / "eagle.py"),
        (PATCH_DIR / "speculative.py", vllm_pkg / "config" / "speculative.py"),
        (PATCH_DIR / "gpu_model_runner.py",
         vllm_pkg / "v1" / "worker" / "gpu_model_runner.py"),
    ]


def cmd_check() -> None:
    installed = validate_vllm_version()
    print(f"vLLM:     {installed} (supported)")
    print(f"Source:   {PATCH_DIR.relative_to(REPO_ROOT)}/")
    pairs = deployment_map()
    print(f"Package:  {pairs[0][1].parents[2]}/\n")
    for src, tgt in pairs:
        s_md5 = md5_short(src) if src.exists() else "MISSING   "
        t_md5 = md5_short(tgt) if tgt.exists() else "MISSING   "
        same = src.exists() and tgt.exists() and md5_short(src) == md5_short(tgt)
        mark = "≡" if same else "≠"
        print(f"  {mark}  src {s_md5}  {src.relative_to(REPO_ROOT)}")
        print(f"     tgt {t_md5}  {tgt}")


def cmd_apply(backup_dir: Path) -> None:
    validate_vllm_version()
    pairs = deployment_map()
    backup_dir.mkdir(parents=True, exist_ok=True)
    deployed = 0
    skipped = 0
    for src, tgt in pairs:
        if not src.exists():
            print(f"  ⚠  source missing: {src}")
            continue
        if not tgt.parent.exists():
            print(f"  ✗  target dir missing: {tgt.parent}  (vllm not installed?)")
            continue
        if tgt.exists() and md5_short(src) == md5_short(tgt):
            print(f"  ≡  unchanged: {tgt.name}")
            skipped += 1
            continue
        if tgt.exists():
            bak = backup_dir / tgt.name
            if not bak.exists():
                shutil.copy2(tgt, bak)
                print(f"  💾 backup: {tgt.name} → {bak.relative_to(REPO_ROOT)}")
        shutil.copy2(src, tgt)
        print(f"  ✓  {src.relative_to(REPO_ROOT)} → {tgt}")
        deployed += 1

    print(f"\nDeployed {deployed}/{len(pairs)} ({skipped} unchanged).")
    if deployed > 0:
        print(f"Backup:  {backup_dir.relative_to(REPO_ROOT)}/")
        print("Revert:  python scripts/deploy_specsteer.py --revert")


def cmd_revert(backup_dir: Path) -> None:
    validate_vllm_version()
    if not backup_dir.exists():
        print(f"✗  No backup at {backup_dir}")
        return
    name_to_tgt = {tgt.name: tgt for _, tgt in deployment_map()}
    restored = 0
    for bak in backup_dir.iterdir():
        if bak.name in name_to_tgt:
            tgt = name_to_tgt[bak.name]
            shutil.copy2(bak, tgt)
            print(f"  ✓  restore: {bak.name} → {tgt}")
            restored += 1
    print(f"\nRestored {restored} files from {backup_dir.relative_to(REPO_ROOT)}/")


def main():
    ap = argparse.ArgumentParser(
        description="Deploy AsymSpec patches into the pinned vLLM installation")
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Status only, no changes")
    action.add_argument("--apply", action="store_true", help="Deploy patches (backs up originals)")
    action.add_argument("--revert", action="store_true", help="Restore originals from backup")
    args = ap.parse_args()

    backup_dir = REPO_ROOT / ".backups" / "vllm_0_19"

    try:
        if args.check:
            cmd_check()
        elif args.apply:
            cmd_apply(backup_dir)
        elif args.revert:
            cmd_revert(backup_dir)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main() or 0)
