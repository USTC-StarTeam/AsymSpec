"""Shared helper to set ASYMSPEC_METHOD env var for bench scripts using setup_asym_method().

Usage (bench_mc_v07.py and similar):
    from _asym_method_util import setup_asym_method
    setup_asym_method(args.asym_method, beta=args.beta)
"""
import os


def setup_asym_method(method: str, beta: float = 1.0) -> None:
    assert method in {"gamma_rule", "cma", "jsd", "jsd_pos", "cma_vnorm", "cma_hbase"}, method
    os.environ["ASYMSPEC_METHOD"] = method
    os.environ.pop("ASYMSPEC_BETA_OVERRIDE", None)
    print(f"[asym_method] method={method} beta={beta}", flush=True)
