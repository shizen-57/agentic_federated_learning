#!/usr/bin/env python3
"""Is a policy's efficiency better decisions, or just less training?

Descriptive (not pre-specified), fixed before the s2_* runs. For one group:

- accuracy at equal spend: for each seed, B = the smallest total cost any
  condition spent over the run; each condition's accuracy at the *last*
  round whose cumulative cost is still at or below B, so no method is given
  more than B (the share of B actually used is shown; a run with no round
  within B counts as missing). A policy that is cheap only because it trains
  less gains nothing here.
- cost to a target: cumulative cost until accuracy first reaches
  T = 0.9 x the mean final fixed-set accuracy of b_random in the group (share of seeds
  that reach it, and the median cost among those that do).

Paired sign-flip p-values against --ref are shown for the equal-spend
accuracy only, and are exploratory.
Both analyses require accuracy_fixed, evaluated on the same examples every
round including the last. Legacy CSVs without this series must be re-run;
the mixed subset/full-set accuracy column is deliberately not substituted.

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
            rows = load(path)
            if not rows or any("accuracy_fixed" not in r or
                               not np.isfinite(r["accuracy_fixed"]) or
                               not np.isfinite(r.get("cost", float("nan"))) or r["cost"] < 0
                               for r in rows):
                print(f"cannot compare budgets: {path} needs finite accuracy_fixed and nonnegative cost; re-run legacy results")
                return 1
            runs[m.group(1)][int(m.group(2))] = rows
    if not runs:
        print("need complete seeds and b_random")
        return 1
    seeds = sorted(set.intersection(*[set(v) for v in runs.values()]))
    if not seeds or "b_random" not in runs:
        print("need complete seeds and b_random")
        return 1
    target = 0.9 * np.mean([runs["b_random"][s][-1]["accuracy_fixed"] for s in seeds])
    at_budget: dict[str, dict[int, float]] = defaultdict(dict)
    used: dict[str, list[float]] = defaultdict(list)
    to_target: dict[str, list[float]] = defaultdict(list)
    for s in seeds:
        B = min(sum(r["cost"] for r in runs[c][s]) for c in runs)
        for c in runs:
            cum, last, spent = 0.0, None, 0.0
            for r in runs[c][s]:
                cum += r["cost"]
                if cum > B + 1e-9:
                    break
                last, spent = r["accuracy_fixed"], cum
            if last is not None:
                at_budget[c][s] = last
                used[c].append(spent / B if B > 0 else 1.0)
            cum = 0.0
            for r in runs[c][s]:
                cum += r["cost"]
                if r["accuracy_fixed"] >= target:
                    to_target[c].append(cum)
                    break
    print(f"# {args.dir}: {len(seeds)} seeds; target accuracy {target:.3f} (accuracy_fixed)\n")
    print(f"| condition | acc at equal spend | seeds | share of budget used | diff vs {args.ref} "
          "| p | reach target | median cost to target |")
    print("|---|---|---|---|---|---|---|---|")
    order = sorted(runs, key=lambda c: -np.mean(list(at_budget[c].values()) or [0.0]))
    for c in order:
        have = sorted(at_budget[c])
        a = np.array([at_budget[c][s] for s in have])
        both = [s for s in have if s in at_budget.get(args.ref, {})]
        if c != args.ref and both:
            d = np.array([at_budget[args.ref][s] - at_budget[c][s] for s in both])
            cmp = f"{d.mean():+.4f} | {sign_flip_p(d):.4f}"
        else:
            cmp = "-- | --"
        reach = len(to_target[c]) / len(seeds)
        med = f"{np.median(to_target[c]):.0f}" if to_target[c] else "--"
        share = f"{np.mean(used[c]):.0%}" if used[c] else "--"
        mean = f"{a.mean():.4f}" if len(a) else "--"
        print(f"| {c} | {mean} | {len(have)}/{len(seeds)} | {share} | {cmp} | {reach:.0%} | {med} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
