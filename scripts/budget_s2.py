#!/usr/bin/env python3
"""Is a policy's efficiency better decisions, or just less training?

Descriptive (not pre-specified), fixed before the s2_* runs. For one group:

- accuracy at equal spend: for each seed, B = the smallest total cost any
  condition spent over the run; each condition's accuracy at the first round
  its cumulative cost reaches B. A policy that is cheap only because it
  trains less gains nothing here.
- cost to a target: cumulative cost until accuracy first reaches
  T = 0.9 x the mean final accuracy of b_random in the group (share of seeds
  that reach it, and the median cost among those that do).

Paired sign-flip p-values against --ref are shown for the equal-spend
accuracy only, and are exploratory.

    python scripts/budget_s2.py --dir results/adv/s2_cifar
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import load  # noqa: E402
from stats import sign_flip_p  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default="c2_rl_fedprox")
    args = ap.parse_args()
    runs: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(path)[:-4])
        if m:
            runs[m.group(1)][int(m.group(2))] = load(path)
    seeds = sorted(set.intersection(*[set(v) for v in runs.values()]))
    if not seeds or "b_random" not in runs:
        print("need complete seeds and b_random")
        return 1
    target = 0.9 * np.mean([runs["b_random"][s][-1]["accuracy"] for s in seeds])
    at_budget: dict[str, dict[int, float]] = defaultdict(dict)
    to_target: dict[str, list[float]] = defaultdict(list)
    for s in seeds:
        B = min(sum(r["cost"] for r in runs[c][s]) for c in runs)
        for c in runs:
            cum, hit = 0.0, None
            for r in runs[c][s]:
                cum += r["cost"]
                if hit is None and cum >= B - 1e-9:
                    at_budget[c][s] = r["accuracy"]
                    hit = True
            cum = 0.0
            for r in runs[c][s]:
                cum += r["cost"]
                if r["accuracy"] >= target:
                    to_target[c].append(cum)
                    break
    print(f"# {args.dir}: {len(seeds)} seeds; target accuracy {target:.3f}\n")
    print(f"| condition | acc at equal spend | diff vs {args.ref} | p | reach target | median cost to target |")
    print("|---|---|---|---|---|---|")
    order = sorted(runs, key=lambda c: -np.mean(list(at_budget[c].values())))
    for c in order:
        a = np.array([at_budget[c][s] for s in seeds])
        if c != args.ref and args.ref in at_budget:
            d = np.array([at_budget[args.ref][s] for s in seeds]) - a
            cmp = f"{d.mean():+.4f} | {sign_flip_p(d):.4f}"
        else:
            cmp = "-- | --"
        reach = len(to_target[c]) / len(seeds)
        med = f"{np.median(to_target[c]):.0f}" if to_target[c] else "--"
        print(f"| {c} | {a.mean():.4f} | {cmp} | {reach:.0%} | {med} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
