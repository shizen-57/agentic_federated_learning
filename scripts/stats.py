#!/usr/bin/env python3
"""Paired significance tests between conditions in one results directory.

Runs with the same seed share the data split, device profiles and per-round
random draws, so conditions are compared seed-by-seed (paired), not as two
independent samples. For each metric and each condition vs. the reference:

- mean paired difference (reference - other) and a 95% bootstrap CI;
- an exact sign-flip permutation p-value (all 2^n sign patterns when n <= 22,
  so no distributional assumption and no scipy needed);
- Holm-corrected p-values across the comparisons made for that metric.

    python scripts/stats.py --dir results/adv/weak_mnist --ref c2_rl_fedprox
    python scripts/stats.py --dir results/adv/weak_mnist --ref c2_rl_fedprox --vs ablation_rawfail

With 5 seeds the smallest possible p-value is 2/32 = 0.0625, so nothing can be
significant at 0.05 -- one reason the advanced sweep uses 10 seeds (min 0.002).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import accuracy_curve, load, load_meta  # noqa: E402

# (column, how to reduce a run to one number, higher is better)
METRICS = [
    ("accuracy", "final", True),
    ("acc_worst10", "final", True),
    ("accuracy", "mean", True),          # area under the accuracy curve: speed
    ("cv", "final", False),
    ("fail_ratio", "mean", False),
    ("arl_oracle", "mean", False),
    ("cost", "mean", False),
    ("latency", "mean", False),
]


def reduce(rows: list[dict], col: str, how: str) -> float:
    if how == "final":
        return rows[-1][col]
    return float(np.mean([r[col] for r in rows]))


def sign_flip_p(d: np.ndarray, n_random: int = 1_000_000, seed: int = 0,
                alternative: str = "two-sided", exact_max: int = 22) -> float:
    """p-value for H0: the paired differences are symmetric about 0.

    alternative: "two-sided", "greater" (mean difference > 0) or "less".
    Exact (all 2^n sign patterns, enumerated in chunks) up to n = exact_max;
    beyond that, n_random random patterns. Exact matters: with 20 seeds a
    Monte Carlo estimate of 0.0249 was in fact 0.0252, on the other side of a
    pre-specified 0.025 threshold.
    """
    n = len(d)
    obs = d.mean()
    if n == 0:
        return float("nan")

    def count(null):
        if alternative == "greater":
            return int((null >= obs - 1e-12).sum())
        if alternative == "less":
            return int((null <= obs + 1e-12).sum())
        return int((np.abs(null) >= abs(obs) - 1e-12).sum())

    if n <= exact_max:
        bits = np.arange(n, dtype=np.int64)
        hits, total, chunk = 0, 2 ** n, 1 << 18
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
            signs = ((idx[:, None] >> bits) & 1) * 2.0 - 1.0
            hits += count((signs * d).mean(axis=1))
        return hits / total
    signs = np.random.default_rng(seed).choice([1.0, -1.0], size=(n_random, n))
    return count((signs * d).mean(axis=1)) / n_random


def bootstrap_ci(d: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    if len(d) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[k]))
        adj[k] = running
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default="c2_rl_fedprox")
    ap.add_argument("--vs", nargs="*", help="conditions to compare against (default: all)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--one-sided", action="store_true",
                    help="test only 'reference is better' (for a pre-specified directional test)")
    args = ap.parse_args()

    runs: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    metas: dict[str, dict[int, dict | None]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(path)[:-4])
        if m:
            runs[m.group(1)][int(m.group(2))] = load(path)
            metas[m.group(1)][int(m.group(2))] = load_meta(path)
    if args.ref not in runs:
        print(f"reference {args.ref} not found in {args.dir}/ (have {sorted(runs)})")
        return 1
    others = args.vs or [c for c in sorted(runs) if c != args.ref]

    out_rows = []
    first = next(iter(runs[args.ref].values()))[0]
    curve_mean: dict[tuple[str, int], float] = {}
    for col, how, higher in METRICS:
        if col == "accuracy" and how == "mean":
            # A curve average must use one test population in every round
            # (report.accuracy_curve); refuse it for runs that cannot show one.
            ok = True
            for c, by_seed in runs.items():
                for sd, rows in by_seed.items():
                    curve = accuracy_curve(rows, metas[c].get(sd))
                    if curve is None:
                        ok = False
                    else:
                        curve_mean[(c, sd)] = float(np.mean([a for _, a in curve]))
            if not ok:
                print("note: curve average skipped -- some runs have no fixed-test-set series\n")
                continue
            col = "accuracy_fixed" if "accuracy_fixed" in first else "accuracy"
        if col not in first:
            continue  # older CSVs predate acc_worst10
        label = f"{how}_{col}" if how == "mean" else f"final_{col}"
        recs = []
        for other in others:
            seeds = sorted(set(runs[args.ref]) & set(runs.get(other, {})))
            if not seeds:
                continue
            if how == "mean" and col in ("accuracy", "accuracy_fixed"):
                a = np.array([curve_mean[(args.ref, s)] for s in seeds])
                b = np.array([curve_mean[(other, s)] for s in seeds])
            else:
                a = np.array([reduce(runs[args.ref][s], col, how) for s in seeds])
                b = np.array([reduce(runs[other][s], col, how) for s in seeds])
            d = a - b
            lo, hi = bootstrap_ci(d)
            recs.append({"metric": label, "other": other, "n": len(seeds),
                         "ref_mean": a.mean(), "other_mean": b.mean(), "diff": d.mean(),
                         "ci_lo": lo, "ci_hi": hi, "higher": higher,
                         "p": sign_flip_p(d, alternative=("greater" if higher else "less")
                                          if args.one_sided else "two-sided")})
        for r, p_adj in zip(recs, holm([r["p"] for r in recs])):
            r["p_holm"] = p_adj
            better = (r["diff"] > 0) == r["higher"]
            r["verdict"] = ("ref better" if better else "ref worse") \
                if p_adj < args.alpha else "no significant difference"
        out_rows.extend(recs)

    print(f"# {args.dir}: {args.ref} vs others (paired by seed; diff = ref - other; "
          f"Holm-corrected within each metric; "
          f"{'one-sided: ref better' if args.one_sided else 'two-sided'})\n")
    cur = None
    for r in out_rows:
        if r["metric"] != cur:
            cur = r["metric"]
            arrow = "higher is better" if r["higher"] else "lower is better"
            print(f"\n## {cur} ({arrow})\n")
            print("| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |")
            print("|---|---|---|---|---|---|---|---|---|")
        print(f"| {r['other']} | {r['n']} | {r['ref_mean']:.4f} | {r['other_mean']:.4f} | "
              f"{r['diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
              f"{r['p']:.4f} | {r['p_holm']:.4f} | {r['verdict']} |")

    suffix = ("_onesided" if args.one_sided else "") + ("_vs_" + "_".join(args.vs) if args.vs else "")
    path = os.path.join(args.dir, f"stats_{args.ref}{suffix}.csv")
    with open(path, "w", newline="") as fh:
        cols = ["metric", "other", "n", "ref_mean", "other_mean", "diff", "ci_lo", "ci_hi",
                "p", "p_holm", "verdict"]
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
