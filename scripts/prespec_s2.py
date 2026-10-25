#!/usr/bin/env python3
"""The pre-specified tests of the s2_* groups, exactly as fixed in
docs/ADVANCED_RESULTS.md ("Simulator fix, ARL-v3 and stronger baselines")
before any of those runs.

    python scripts/prespec_s2.py                 # reads results/adv/s2_*
    python scripts/prespec_s2.py --root some/dir

Families (exact paired sign-flip on seeds 100-119, Holm within each family):
  A   arl3_full vs v2_rawfail, final acc_worst10, one-sided, 4 settings
  B   arl2_full vs rl_light, mean fail_ratio, one-sided (fewer), 2 settings
  C   c2_rl_fedprox vs 6 baselines, final accuracy, two-sided, 2 settings
  C2  the same 12 pairs, mean fail_ratio, two-sided
A family member passes when its Holm-adjusted p is below 0.05.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import load  # noqa: E402
from stats import holm, reduce, sign_flip_p  # noqa: E402

SEEDS = set(range(100, 120))
BASELINES = ["b_favor", "b_refl", "b_delta", "b_random_ew", "b_oort_ew", "b_poc_ew"]

FAMILIES = {
    "A": [(g, "arl3_full", "v2_rawfail", "acc_worst10", "final", "greater")
          for g in ("s2_cifar", "s2_femnist", "s2_churn_cifar", "s2_churn_femnist")],
    "B": [(g, "arl2_full", "rl_light", "fail_ratio", "mean", "less")
          for g in ("s2_cifar", "s2_femnist")],
    "C": [(g, "c2_rl_fedprox", b, "accuracy", "final", "two-sided")
          for g in ("s2_cifar", "s2_femnist") for b in BASELINES],
    "C2": [(g, "c2_rl_fedprox", b, "fail_ratio", "mean", "two-sided")
           for g in ("s2_cifar", "s2_femnist") for b in BASELINES],
}


def runs(d: str, cond: str) -> dict[int, list[dict]]:
    out = {}
    for path in glob.glob(os.path.join(d, f"{cond}_s*.csv")):
        m = re.match(rf"{re.escape(cond)}_s(\d+)$", os.path.basename(path)[:-4])
        if m:
            out[int(m.group(1))] = load(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "adv"))
    args = ap.parse_args()
    ok = True
    for fam, tests in FAMILIES.items():
        rows = []
        for g, ref, other, col, how, alt in tests:
            a_runs, b_runs = runs(os.path.join(args.root, g), ref), runs(os.path.join(args.root, g), other)
            seeds = sorted(SEEDS & set(a_runs) & set(b_runs))
            if len(seeds) < len(SEEDS):
                print(f"{fam}: {g} {ref} vs {other}: only {len(seeds)}/20 paired seeds")
                ok = False
            if not seeds:
                rows.append((g, ref, other, col, float("nan"), float("nan"), float("nan"), 0, float("nan")))
                continue
            a = np.array([reduce(a_runs[s], col, how) for s in seeds])
            b = np.array([reduce(b_runs[s], col, how) for s in seeds])
            rows.append((g, ref, other, col, a.mean(), b.mean(), (a - b).mean(), len(seeds),
                         sign_flip_p(a - b, alternative=alt)))
        adj = holm([r[-1] if np.isfinite(r[-1]) else 1.0 for r in rows])
        print(f"\n## Family {fam} ({tests[0][5]}, {tests[0][4]} {tests[0][3]})\n")
        print("| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | passes |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r, p_adj in zip(rows, adj):
            g, ref, other, col, am, bm, d, n, p = r
            print(f"| {g} | {ref} | {other} | {am:.4f} | {bm:.4f} | {d:+.4f} | {n} | "
                  f"{p:.4f} | {p_adj:.4f} | {'yes' if p_adj < 0.05 else 'no'} |")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
