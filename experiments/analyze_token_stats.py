"""Analyze per-token statistics for adaptive β offline analysis.

Reads the output of dump_token_stats.py and computes:
1. Correlation between acceptance and each signal (δ_mag, KL, H_llm, sparsity)
2. Distribution of signals for accepted vs rejected tokens
3. Per-example acceptance rate vs mean signal values

Usage:
  python experiments/analyze_token_stats.py experiments/cache/token_stats_mc.jsonl
"""
import json, sys
import numpy as np
from collections import defaultdict


def load_stats(path):
    """Load per-token records from jsonl."""
    tokens = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for t in rec["tokens"]:
                t["qid"] = rec["qid"]
                t["step"] = rec["step"]
                t["n_match"] = rec["n_match"]
                t["draft_len"] = rec["draft_len"]
                tokens.append(t)
    return tokens


def analyze(tokens):
    signals = ["delta_mag", "kl_aug_base", "h_llm_norm", "sparsity", "delta_l1"]
    accepted = np.array([t["accepted"] for t in tokens], dtype=float)
    n = len(tokens)
    n_acc = int(accepted.sum())
    n_rej = n - n_acc
    ar = n_acc / n if n > 0 else 0

    print(f"Total tokens: {n}  accepted: {n_acc} ({ar:.1%})  rejected: {n_rej}")
    print()

    # 1. Point-biserial correlation (accepted vs signal)
    print("=== Correlation: acceptance × signal ===")
    for sig in signals:
        vals = np.array([t[sig] for t in tokens])
        if vals.std() == 0:
            print(f"  {sig:15s}: constant (no variance)")
            continue
        corr = np.corrcoef(accepted, vals)[0, 1]
        print(f"  {sig:15s}: r = {corr:+.4f}")
    print()

    # 2. Mean signal for accepted vs rejected
    print("=== Mean signal | accepted vs rejected ===")
    print(f"  {'signal':15s} {'accepted':>10s} {'rejected':>10s} {'diff':>10s}")
    for sig in signals:
        acc_vals = [t[sig] for t in tokens if t["accepted"]]
        rej_vals = [t[sig] for t in tokens if not t["accepted"]]
        m_acc = np.mean(acc_vals) if acc_vals else 0
        m_rej = np.mean(rej_vals) if rej_vals else 0
        print(f"  {sig:15s} {m_acc:10.4f} {m_rej:10.4f} {m_acc - m_rej:+10.4f}")
    print()

    # 3. Per-example: acceptance rate vs mean signals
    print("=== Per-example: acceptance rate vs mean KL ===")
    by_qid = defaultdict(lambda: {"acc": 0, "total": 0, "kl_sum": 0, "dmag_sum": 0})
    for t in tokens:
        q = by_qid[t["qid"]]
        q["total"] += 1
        q["acc"] += int(t["accepted"])
        q["kl_sum"] += t["kl_aug_base"]
        q["dmag_sum"] += t["delta_mag"]

    qids = sorted(by_qid.keys(), key=lambda q: by_qid[q]["acc"] / max(by_qid[q]["total"], 1))
    ars = [by_qid[q]["acc"] / by_qid[q]["total"] for q in qids]
    kls = [by_qid[q]["kl_sum"] / by_qid[q]["total"] for q in qids]
    dmags = [by_qid[q]["dmag_sum"] / by_qid[q]["total"] for q in qids]

    # Binned summary
    bins = [(0, 0.3, "low AR"), (0.3, 0.6, "mid AR"), (0.6, 1.01, "high AR")]
    print(f"  {'bin':10s} {'n_examples':>10s} {'mean_AR':>8s} {'mean_KL':>8s} {'mean_δ':>8s}")
    for lo, hi, label in bins:
        idxs = [i for i, a in enumerate(ars) if lo <= a < hi]
        if not idxs:
            continue
        n_ex = len(idxs)
        m_ar = np.mean([ars[i] for i in idxs])
        m_kl = np.mean([kls[i] for i in idxs])
        m_dm = np.mean([dmags[i] for i in idxs])
        print(f"  {label:10s} {n_ex:10d} {m_ar:8.3f} {m_kl:8.4f} {m_dm:8.4f}")
    print()

    # 4. Quantiles
    print("=== Signal quantiles ===")
    print(f"  {'signal':15s} {'p10':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p90':>8s}")
    for sig in signals:
        vals = np.array([t[sig] for t in tokens])
        ps = np.percentile(vals, [10, 25, 50, 75, 90])
        print(f"  {sig:15s} {ps[0]:8.4f} {ps[1]:8.4f} {ps[2]:8.4f} {ps[3]:8.4f} {ps[4]:8.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <token_stats.jsonl>")
        sys.exit(1)
    tokens = load_stats(sys.argv[1])
    analyze(tokens)
