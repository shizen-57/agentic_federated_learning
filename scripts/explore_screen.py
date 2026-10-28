#!/usr/bin/env python3
"""Exploratory look at a finished screen, AFTER its pre-registered rule was
applied (never a decision): paired per-seed differences with exact sign-flip
p-values, and steadier endpoint versions -- the mean of the last 10 rounds
instead of the last round alone (screen B: one seed's final accuracy was
decided by a single dip at round 200).

    python scripts/explore_screen.py results/rl2_screen_b/full/s2_femnist \
        --a ppo_full_mimic --b rl_light ppo_full arl3_full_pg
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_dev_screen as rep  # noqa: E402

KEYS = (("eq", "equal-spend acc"), ("acc", "final acc"), ("worst10", "final worst-10%"),
        ("acc10", "last-10 mean acc"), ("w10", "last-10 mean worst-10%"))


def load(d: str) -> dict:
    runs: dict[str, dict[int, dict]] = {}
    for p in sorted(glob.glob(os.path.join(d, "*_s*.csv"))):
        cond, seed = os.path.basename(p)[:-4].rsplit("_s", 1)
        m = rep.measures(p)
        rows = list(csv.DictReader(open(p)))
        acc = [float(r["accuracy"]) for r in rows]
        w = [float(r["acc_client_worst10"]) for r in rows]
        m.update(acc10=st.mean(acc[-10:]), w10=st.mean(w[-10:]), wobble=st.pstdev(acc[-20:]))
        runs.setdefault(cond, {})[int(seed)] = m
    return runs


def signflip(d: list[float]) -> float:
    """Exact two-sided paired sign-flip p-value (all 2^n sign patterns)."""
    obs = abs(sum(d))
    hits = sum(abs(sum(s * x for s, x in zip(sg, d))) >= obs - 1e-12
               for sg in itertools.product((1, -1), repeat=len(d)))
    return hits / 2 ** len(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    args = ap.parse_args()
    runs = load(args.dir)
    print("# Exploratory (after the rule; not a decision)\n")
    print("| condition | median last-10 acc | median last-10 worst-10% | median wobble (sd, last 20) |")
    print("|---|---|---|---|")
    for c in sorted(runs):
        v = runs[c].values()
        print(f"| {c} | {st.median(m['acc10'] for m in v):.3f} | {st.median(m['w10'] for m in v):.3f} | "
              f"{st.median(m['wobble'] for m in v):.3f} |")
    for b in args.b:
        seeds = sorted(set(runs[args.a]) & set(runs[b]))
        print(f"\n## {args.a} minus {b} (seeds {seeds[0]}-{seeds[-1]})\n")
        print("| measure | per seed | median | wins | sign-flip p |")
        print("|---|---|---|---|---|")
        for k, name in KEYS:
            d = [runs[args.a][s][k] - runs[b][s][k] for s in seeds]
            print(f"| {name} | {' '.join(f'{x:+.3f}' for x in d)} | {st.median(d):+.3f} | "
                  f"{sum(x > 0 for x in d)}/{len(d)} | {signflip(d):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
