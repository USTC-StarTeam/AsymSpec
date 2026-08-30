"""Cached GAIA web-only across Levels 1/2/3, via smolagents CodeAgent + AsymSpec.

Filters GAIA validation metadata to questions WITHOUT file attachments at the
selected levels. Uses CachedDuckDuckGoSearchTool + CachedVisitWebpageTool for
reproducibility.

Counts (validation set, file_name empty):
  L1: 42 / 53 web-only
  L2: 66 / 86 web-only
  L3: 19 / 26 web-only
  Total L1+L2+L3 web-only: 127
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for paths.py
from asym_vllm_model import AsymSpecVLLMModel  # noqa: F401  (re-export for build_model_v2)
from smolagents import CodeAgent, VLLMModel  # noqa: F401
from cached_tools import (CachedDuckDuckGoSearchTool as DuckDuckGoSearchTool,
                          CachedVisitWebpageTool as VisitWebpageTool)
from run_benchmark100 import build_model_v2, run_one_question
from paths import LLM_PATH, slm_model, gaia_validation_dir


GAIA_BASE = str(gaia_validation_dir())


def gaia_score(answer: str, gold: str) -> bool:
    """GAIA scoring: case-insensitive exact match after normalization."""
    def norm(s):
        s = str(s).strip().lower()
        s = re.sub(r"^(the |a |an )", "", s)
        s = s.replace(",", "")
        s = s.rstrip(".!?,;:")
        return s
    a, g = norm(answer), norm(gold)
    if a == g:
        return True
    try:
        return abs(float(a) - float(g)) < 0.01
    except ValueError:
        pass
    return g in a or a in g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["asym_cda", "b1_main", "b1_aug"])
    ap.add_argument("--llm", default=LLM_PATH)
    ap.add_argument("--drafter", default=slm_model("4B"))
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--asym_method", default="jsd",
                    choices=["jsd", "jsd_pos", "gamma_rule", "cma",
                             "cma_vnorm", "cma_hbase"])
    ap.add_argument("--main_compression", default="llmlingua",
                    choices=["truncate", "llmlingua", "none"])
    ap.add_argument("--skip_system_compress", type=lambda s: s.lower() == "true",
                    default=True)
    ap.add_argument("--llmlingua_rate", type=float, default=0.3,
                    help="LLMLingua-2 target retention ratio (lower = more aggressive)")
    ap.add_argument("--keep_last_k", type=int, default=2,
                    help="preserve last k non-system messages verbatim (MC summary_last_k style)")
    ap.add_argument("--code_zone_suppress", type=lambda s: s.lower() == "true", default=False,
                    help="Enable zone-aware δ-fusion gating (see --cz_variant)")
    ap.add_argument("--cz_variant", default="in_code",
                    choices=["in_code", "outside_code", "in_final_arg"],
                    help="Where to suppress δ-fusion: code blocks / outside code / final_answer arg")
    ap.add_argument("--preserve_fn_sigs", type=lambda s: s.lower() == "true", default=False,
                    help="APIB Method A-style: preserve function signature lines, compress prose")
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--levels", default="1,2,3",
                    help="comma-separated GAIA levels to include")
    ap.add_argument("--n_limit", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0,
                    help="0-indexed shard for parallel split")
    ap.add_argument("--shard_total", type=int, default=1,
                    help="total number of shards (round-robin by row)")
    ap.add_argument("--start_idx", type=int, default=0,
                    help="skip first N samples (within shard) — for resuming")
    ap.add_argument("--cell_tag", default="",
                    help="suffix for output filename to distinguish runs")
    ap.add_argument("--out_dir",
                    default="experiments/v010_smolagents_agent/cache_gaia_web")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    print(f"=== Cached GAIA web-only L{','.join(map(str,levels))} mode={args.mode} ===")
    print(f"  main_compression: {args.main_compression}")
    print(f"  skip_system_compress: {args.skip_system_compress}")

    # Load + merge requested levels
    dfs = []
    for L in levels:
        df_L = pd.read_parquet(f"{GAIA_BASE}/metadata.level{L}.parquet")
        df_L["_level"] = L
        dfs.append(df_L)
    df_all = pd.concat(dfs, ignore_index=True)
    df_web = df_all[df_all["file_name"].astype(str).str.len() == 0].reset_index(drop=True)
    print(f"[setup] levels={levels} total={len(df_all)} web-only={len(df_web)}")
    if args.n_limit > 0:
        df_web = df_web.head(args.n_limit)
        print(f"[setup] limited to first {len(df_web)}")
    if args.shard_total > 1:
        df_web = df_web.iloc[args.shard_idx::args.shard_total].reset_index(drop=True)
        print(f"[setup] shard {args.shard_idx}/{args.shard_total}: n={len(df_web)} samples")
    if args.start_idx > 0:
        df_web = df_web.iloc[args.start_idx:].reset_index(drop=True)
        print(f"[setup] resume from start_idx={args.start_idx}: n={len(df_web)} samples")

    # Build model + agent
    t0 = time.perf_counter()
    model = build_model_v2(args.mode, args)
    print(f"[setup] model init: {time.perf_counter()-t0:.1f}s")
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model, max_steps=args.max_steps,
        additional_authorized_imports=["numpy"],
    )

    results = []
    fname = f"{args.mode}{('_' + args.cell_tag) if args.cell_tag else ''}.json"
    out_path = Path(args.out_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_run0 = time.perf_counter()
    for i, (_, row) in enumerate(df_web.iterrows()):
        q, gold = row["Question"], row["Final answer"]
        task_id = row["task_id"]
        level = int(row.get("_level", 0))
        r = run_one_question(agent, q, gold, max_run_s=300)
        r["correct"] = gaia_score(r["answer"], gold)
        r["question"] = q[:300]
        r["gold"] = gold
        r["task_id"] = task_id
        r["level"] = level
        results.append(r)
        n_correct = sum(1 for x in results if x["correct"])
        elapsed_total = time.perf_counter() - t_run0
        print(f"  [{i+1}/{len(df_web)}] L{level} {task_id[:8]} "
              f"{'✓' if r['correct'] else '✗'} "
              f"gold={gold[:30]!r} ans={r['answer'][:40]!r} | "
              f"acc={n_correct}/{i+1} | tot={elapsed_total:.0f}s")
        if (i + 1) % 5 == 0 or i == len(df_web) - 1:
            by_level = {}
            for x in results:
                lv = x["level"]
                by_level.setdefault(lv, {"n": 0, "c": 0})
                by_level[lv]["n"] += 1
                by_level[lv]["c"] += int(x["correct"])
            out_path.write_text(json.dumps({
                "mode": args.mode, "config": vars(args),
                "n_total": len(df_web), "n_done": len(results),
                "by_level": by_level,
                "asym_diag": (model.get_diag_summary()
                              if hasattr(model, "get_diag_summary") else {}),
                "results": results,
            }, indent=2, default=str))

    n_correct = sum(1 for r in results if r["correct"])
    print(f"\n=== Done mode={args.mode} ===")
    print(f"Overall acc: {n_correct}/{len(results)} = {n_correct/len(results)*100:.1f}%")
    by_level = {}
    for x in results:
        lv = x["level"]
        by_level.setdefault(lv, {"n": 0, "c": 0})
        by_level[lv]["n"] += 1
        by_level[lv]["c"] += int(x["correct"])
    for lv in sorted(by_level):
        s = by_level[lv]
        print(f"  L{lv}: {s['c']}/{s['n']} = {s['c']/s['n']*100:.1f}%")
    if hasattr(model, "get_diag_summary"):
        print(f"asym diag: {model.get_diag_summary()}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
