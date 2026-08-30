#!/usr/bin/env python3
"""Deploy SpecSteer patches into the installed vLLM package.

Hot-patches vLLM's pip-installed location. No vLLM fork needed —
``vllm_specsteer/<ver>/`` contains the .py files to overwrite.

Default version: ``v0.10.mm`` (multimodal — superset of text-only v0.10).
Override with ``--ver v0.10`` for text-only.

Files deployed:
  vllm_specsteer/<ver>/specsteer_model.py   → vllm/v1/spec_decode/specsteer_model.py
  vllm_specsteer/<ver>/specsteer_sampler.py → vllm/v1/sample/specsteer_sampler.py
  vllm_specsteer/v0.10.mm/eagle_patch.py    → vllm/v1/spec_decode/eagle.py  (mm only)

Originals are backed up to ``.backups/vllm_specsteer_<ver>/`` before being
overwritten. ``--revert`` restores them.

Usage:
  python scripts/deploy_specsteer.py --check
  python scripts/deploy_specsteer.py --apply             # default v0.10.mm
  python scripts/deploy_specsteer.py --apply --ver v0.10 # text-only
  python scripts/deploy_specsteer.py --revert
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import REPO_ROOT, vllm_specsteer_targets

VLLM_SPECSTEER = REPO_ROOT / "vllm_specsteer"


def md5_short(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:10]


def deployment_map(version: str) -> list[tuple[Path, Path]]:
    """Returns [(source, target), ...] for the given version.

    v0.10:    [specsteer_model, specsteer_sampler]
    v0.10.mm: [specsteer_model, specsteer_sampler, eagle_patch→eagle,
               speculative_config_patch→config/speculative.py]
    """
    src_dir = VLLM_SPECSTEER / version
    if not src_dir.exists():
        raise FileNotFoundError(f"version dir not found: {src_dir}")

    spec_model_tgt, sampler_tgt = vllm_specsteer_targets()
    pairs = [
        (src_dir / "specsteer_model.py", spec_model_tgt),
        (src_dir / "specsteer_sampler.py", sampler_tgt),
    ]
    # v0.10.mm only: eagle_patch.py replaces eagle.py (same dir as spec_decode)
    eagle_src = src_dir / "eagle_patch.py"
    if eagle_src.exists():
        eagle_tgt = spec_model_tgt.parent / "eagle.py"
        pairs.append((eagle_src, eagle_tgt))
    # v0.10.mm only: speculative.py adds "specsteer" to SpeculativeMethod
    # Literal + specsteer_beta/gamma fields. vllm_pkg/config/speculative.py.
    spec_cfg_src = src_dir / "speculative_config_patch.py"
    if spec_cfg_src.exists():
        vllm_pkg = spec_model_tgt.parents[2]  # .../vllm/v1/spec_decode → .../vllm
        spec_cfg_tgt = vllm_pkg / "config" / "speculative.py"
        pairs.append((spec_cfg_src, spec_cfg_tgt))
    # v0.10.mm only: gpu_model_runner.py routes method=="specsteer" → SpecSteerProposer
    # and adds _specsteer_sample method for greedy δ-fusion sampler dispatch.
    gpu_runner_src = src_dir / "gpu_model_runner_patch.py"
    if gpu_runner_src.exists():
        vllm_pkg = spec_model_tgt.parents[2]
        gpu_runner_tgt = vllm_pkg / "v1" / "worker" / "gpu_model_runner.py"
        pairs.append((gpu_runner_src, gpu_runner_tgt))

    return pairs


def cmd_check(version: str) -> None:
    print(f"Version:  {version}")
    print(f"Source:   {(VLLM_SPECSTEER / version).relative_to(REPO_ROOT)}/")
    pairs = deployment_map(version)
    print(f"vLLM:     {pairs[0][1].parents[2]}/\n")
    for src, tgt in pairs:
        s_md5 = md5_short(src) if src.exists() else "MISSING   "
        t_md5 = md5_short(tgt) if tgt.exists() else "MISSING   "
        same = src.exists() and tgt.exists() and md5_short(src) == md5_short(tgt)
        mark = "≡" if same else "≠"
        print(f"  {mark}  src {s_md5}  {src.relative_to(REPO_ROOT)}")
        print(f"     tgt {t_md5}  {tgt}")


def cmd_apply(version: str, backup_dir: Path) -> None:
    pairs = deployment_map(version)
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
        print(f"Revert:  python scripts/deploy_specsteer.py --revert --ver {version}")


def cmd_revert(version: str, backup_dir: Path) -> None:
    if not backup_dir.exists():
        print(f"✗  No backup at {backup_dir}")
        return
    name_to_tgt = {tgt.name: tgt for _, tgt in deployment_map(version)}
    restored = 0
    for bak in backup_dir.iterdir():
        if bak.name in name_to_tgt:
            tgt = name_to_tgt[bak.name]
            shutil.copy2(bak, tgt)
            print(f"  ✓  restore: {bak.name} → {tgt}")
            restored += 1
    print(f"\nRestored {restored} files from {backup_dir.relative_to(REPO_ROOT)}/")


def main():
    ap = argparse.ArgumentParser(description="Deploy SpecSteer patches into installed vLLM")
    ap.add_argument("--ver", default="v0.10.mm", choices=["v0.10", "v0.10.mm"],
                    help="SpecSteer version (default: v0.10.mm)")
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Status only, no changes")
    action.add_argument("--apply", action="store_true", help="Deploy patches (backs up originals)")
    action.add_argument("--revert", action="store_true", help="Restore originals from backup")
    args = ap.parse_args()

    backup_dir = REPO_ROOT / ".backups" / f"vllm_specsteer_{args.ver}"

    if args.check:
        cmd_check(args.ver)
    elif args.apply:
        cmd_apply(args.ver, backup_dir)
    elif args.revert:
        cmd_revert(args.ver, backup_dir)


if __name__ == "__main__":
    sys.exit(main() or 0)
