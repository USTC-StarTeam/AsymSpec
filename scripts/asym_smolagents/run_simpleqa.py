"""SimpleQA evaluation with smolagents CodeAgent and AsymSpec.

Replaces gated smolagents/benchmark-v1 with public SimpleQA — same purpose
(short-form factual web-search QA, designed to be hard for frontier models).

Run 3 modes:
  - asym_cda: full AsymSpec (CDA, paper-headline)
  - b1_main : verifier-only baseline on truncated context (Floor analog)
  - b1_aug  : verifier-only on full context (Ceiling analog)

For each mode, agent has DuckDuckGoSearch + VisitWebpage; CodeAgent ReAct loop.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import datasets
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for paths.py
from asym_vllm_model import AsymSpecVLLMModel

from smolagents import CodeAgent, VLLMModel
from cached_tools import CachedDuckDuckGoSearchTool as DuckDuckGoSearchTool, \
                         CachedVisitWebpageTool as VisitWebpageTool
from paths import LLM_PATH, slm_model


def normalize(s: str) -> str:
    return str(s).strip().lower().rstrip(".").rstrip(",")


def is_correct(ans: str, gold: str) -> bool:
    a, g = normalize(ans), normalize(gold)
    if a == g:
        return True
    # numeric tolerance
    try:
        return abs(float(a.replace(",", "")) - float(g.replace(",", ""))) < 0.01
    except ValueError:
        pass
    # substring containment (gold short factoid in answer is acceptable)
    return g in a or a in g


class CompressingVLLMModel(VLLMModel):
    """Plain VLLMModel that compresses messages before generation (Floor baseline)."""
    def __init__(self, *args, compression="truncate", skip_system_compress=True,
                 llmlingua_rate=0.3, keep_last_k=0, preserve_fn_sigs=False, **kwargs):
        # Match AsymSpec model: disable Qwen3 thinking-mode by default.
        actk = dict(kwargs.pop("apply_chat_template_kwargs", None) or {})
        actk.setdefault("enable_thinking", False)
        kwargs["apply_chat_template_kwargs"] = actk
        super().__init__(*args, **kwargs)
        self._compression = compression
        self._skip_system_compress = skip_system_compress
        self._llmlingua_rate = llmlingua_rate
        self._keep_last_k = keep_last_k
        self._preserve_fn_sigs = preserve_fn_sigs

    def generate(self, messages, **kw):
        from asym_vllm_model import _compress_messages
        compressed = _compress_messages(
            [m.dict() if hasattr(m, "dict") else dict(m) for m in messages],
            self._compression, skip_system=self._skip_system_compress,
            llmlingua_rate=self._llmlingua_rate,
            keep_last_k=self._keep_last_k,
            preserve_fn_sigs=self._preserve_fn_sigs,
        )
        return super().generate(compressed, **kw)


def build_model(mode: str, args):
    """Build AsymSpec, compressed-context, or full-context evaluation model."""
    common_kwargs = dict(
        dtype="bfloat16", trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    llmlingua_rate = getattr(args, "llmlingua_rate", 0.3)
    keep_last_k = getattr(args, "keep_last_k", 0)
    code_zone_suppress = getattr(args, "code_zone_suppress", False)
    cz_variant = getattr(args, "cz_variant", "in_code")
    preserve_fn_sigs = getattr(args, "preserve_fn_sigs", False)
    if mode == "asym_cda":
        return AsymSpecVLLMModel(
            model_id=args.llm, drafter_id=args.drafter,
            K=args.K, gamma=args.gamma, beta=args.beta,
            asym_method=args.asym_method,
            main_compression=args.main_compression,
            skip_system_compress=args.skip_system_compress,
            llmlingua_rate=llmlingua_rate,
            keep_last_k=keep_last_k,
            code_zone_suppress=code_zone_suppress,
            cz_variant=cz_variant,
            preserve_fn_sigs=preserve_fn_sigs,
            model_kwargs=common_kwargs,
        )
    elif mode == "b1_main":
        return CompressingVLLMModel(
            model_id=args.llm, compression=args.main_compression,
            skip_system_compress=args.skip_system_compress,
            llmlingua_rate=llmlingua_rate,
            keep_last_k=keep_last_k,
            preserve_fn_sigs=preserve_fn_sigs,
            model_kwargs=common_kwargs,
        )
    elif mode == "b1_aug":
        return VLLMModel(
            model_id=args.llm, model_kwargs=common_kwargs,
            apply_chat_template_kwargs={"enable_thinking": False},
        )
    raise ValueError(mode)


def run_one_question(agent, question: str, gold: str, max_run_s: int = 180):
    t0 = time.perf_counter()
    try:
        answer = str(agent.run(question))
        err = None
    except Exception as e:
        answer = f"<ERROR: {type(e).__name__}: {str(e)[:200]}>"
        err = type(e).__name__
    elapsed = time.perf_counter() - t0
    correct = is_correct(answer, gold)
    return {"answer": answer, "correct": correct, "elapsed_s": round(elapsed, 1), "err": err}


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
                    default=True, help="true|false: keep system prompt intact")
    ap.add_argument("--llmlingua_rate", type=float, default=0.3)
    ap.add_argument("--keep_last_k", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--n", type=int, default=500, help="paper subset size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="outputs/simpleqa")
    args = ap.parse_args()

    print(f"=== SimpleQA n={args.n} mode={args.mode} ===")
    print(f"  main_compression: {args.main_compression}")

    # Load SimpleQA + sample
    ds = datasets.load_dataset("basicv8vc/SimpleQA", split="test")
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), args.n)
    samples = [ds[i] for i in indices]
    print(f"[setup] loaded {len(samples)}/{len(ds)} questions (seed={args.seed})")

    # Build model + agent
    t0 = time.perf_counter()
    model = build_model(args.mode, args)
    init_elapsed = time.perf_counter() - t0
    print(f"[setup] model init: {init_elapsed:.1f}s")

    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model, max_steps=args.max_steps,
        additional_authorized_imports=["numpy"],
    )

    results = []
    out_path = Path(args.out_dir) / f"{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_run0 = time.perf_counter()
    for i, ex in enumerate(samples):
        q, gold = ex["problem"], ex["answer"]
        r = run_one_question(agent, q, gold)
        r["question"] = q[:300]
        r["gold"] = gold
        r["index"] = indices[i]
        results.append(r)
        n_correct = sum(x["correct"] for x in results)
        elapsed_total = time.perf_counter() - t_run0
        print(f"  [{i+1}/{args.n}] {'✓' if r['correct'] else '✗'} "
              f"gold={gold[:30]!r} ans={r['answer'][:40]!r} | "
              f"acc={n_correct}/{i+1} | tot={elapsed_total:.0f}s")
        # Incremental save
        if (i + 1) % 5 == 0 or i == len(samples) - 1:
            out_path.write_text(json.dumps({
                "mode": args.mode, "config": vars(args),
                "n_total": args.n, "n_done": len(results),
                "asym_diag": (model.get_diag_summary()
                              if hasattr(model, "get_diag_summary") else {}),
                "results": results,
            }, indent=2, default=str))

    n_correct = sum(r["correct"] for r in results)
    print(f"\n=== SimpleQA done: mode={args.mode} acc={n_correct}/{len(results)} ===")
    if hasattr(model, "get_diag_summary"):
        print(f"asym diag: {model.get_diag_summary()}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
