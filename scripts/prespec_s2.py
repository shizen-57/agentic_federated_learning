#!/usr/bin/env python3
"""The pre-specified tests of the s2_* groups, exactly as fixed in
docs/ADVANCED_RESULTS.md ("Simulator fix, ARL-v3 and stronger baselines")
before any of those runs.

    python scripts/prespec_s2.py                 # reads results/adv/s2_*
    python scripts/prespec_s2.py --root some/dir

Families (exact paired sign-flip on seeds 100-119, Holm within each family):
  A   arl3_full vs v2_rawfail, final acc_worst10, one-sided, 4 settings
  A2  the same 4 pairs, final acc_client_worst10 (worst 10% of clients)
  B   arl2_full vs rl_light, mean fail_ratio, one-sided (fewer), 2 settings
  C   c2_rl_fedprox vs 9 baselines (FAVOR, REFL, DELTA as published and with
      the shared mode rule; Random, Oort, Power-of-Choice with it), final
      accuracy, two-sided, 2 settings
  C2  the same 18 pairs, mean fail_ratio, two-sided
  D   learning vs a rule, same reward: v2_rawfail vs rl_ew, rl_light,
      b_cov_ew, final accuracy, two-sided, 2 settings
  D2  the same 6 pairs, mean fail_ratio, two-sided
A family member passes when its Holm-adjusted p is below 0.05. No verdict is
printed for a family unless all its runs are present, finished, made with
the protocol fix and by one code version.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import load  # noqa: E402
from stats import holm, reduce, sign_flip_p  # noqa: E402

SEEDS = set(range(100, 120))
BASELINES = ["b_favor", "b_favor_ew", "b_refl", "b_refl_ew", "b_delta", "b_delta_ew",
             "b_random_ew", "b_oort_ew", "b_poc_ew"]
RULES = ["rl_ew", "rl_light", "b_cov_ew"]
ARL_GROUPS = ("s2_cifar", "s2_femnist", "s2_churn_cifar", "s2_churn_femnist")
MAIN = ("s2_cifar", "s2_femnist")

FAMILIES = {
    "A": [(g, "arl3_full", "v2_rawfail", "acc_worst10", "final", "greater") for g in ARL_GROUPS],
    "A2": [(g, "arl3_full", "v2_rawfail", "acc_client_worst10", "final", "greater")
           for g in ARL_GROUPS],
    "B": [(g, "arl2_full", "rl_light", "fail_ratio", "mean", "less") for g in MAIN],
    "C": [(g, "c2_rl_fedprox", b, "accuracy", "final", "two-sided") for g in MAIN for b in BASELINES],
    "C2": [(g, "c2_rl_fedprox", b, "fail_ratio", "mean", "two-sided") for g in MAIN for b in BASELINES],
    "D": [(g, "v2_rawfail", b, "accuracy", "final", "two-sided") for g in MAIN for b in RULES],
    "D2": [(g, "v2_rawfail", b, "fail_ratio", "mean", "two-sided") for g in MAIN for b in RULES],
}


def runs(d: str, cond: str) -> tuple[dict[int, list[dict]], list[str]]:
    """Seed -> rows for one condition, and the problems that make a run unusable
    for a confirmatory test (missing config, unfinished, protocol fix off)."""
    out, problems = {}, []
    for path in glob.glob(os.path.join(d, f"{cond}_s*.csv")):
        m = re.match(rf"{re.escape(cond)}_s(\d+)$", os.path.basename(path)[:-4])
        if not m:
            continue
        seed, rows = int(m.group(1)), load(path)
        meta_path = path[:-4] + ".config.json"
        if not os.path.exists(meta_path):
            problems.append(f"{cond}_s{seed}: no saved config")
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        if len(rows) != meta["fl"]["rounds"]:
            problems.append(f"{cond}_s{seed}: {len(rows)}/{meta['fl']['rounds']} rounds")
            continue
        if not (meta["fl"].get("val_frac", 0) > 0 and meta["fl"].get("clean_rng")):
            problems.append(f"{cond}_s{seed}: made without the protocol fix")
            continue
        rows[0]["_code"] = meta.get("code_hash")
        out[seed] = rows
    return out, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "adv"))
    args = ap.parse_args()
    ok = True
    for fam, tests in FAMILIES.items():
        rows, problems, codes = [], [], set()
        for g, ref, other, col, how, alt in tests:
            d = os.path.join(args.root, g)
            (a_runs, pa), (b_runs, pb) = runs(d, ref), runs(d, other)
            problems += [f"{g}/{x}" for x in pa + pb]
            seeds = sorted(SEEDS & set(a_runs) & set(b_runs))
            codes |= {r[0].get("_code") for r in list(a_runs.values()) + list(b_runs.values())}
            if len(seeds) < len(SEEDS):
                problems.append(f"{g}: {ref} vs {other}: only {len(seeds)}/20 paired seeds")
            if not seeds:
                rows.append((g, ref, other, col, float("nan"), float("nan"), float("nan"), 0, float("nan")))
                continue
            a = np.array([reduce(a_runs[s], col, how) for s in seeds])
            b = np.array([reduce(b_runs[s], col, how) for s in seeds])
            rows.append((g, ref, other, col, a.mean(), b.mean(), (a - b).mean(), len(seeds),
                         sign_flip_p(a - b, alternative=alt)))
        if len(codes) > 1:
            problems.append(f"runs come from {len(codes)} different code versions")
        # A confirmatory verdict needs every run of the family, complete and
        # made the same way; otherwise numbers are shown but no verdict is.
        complete = not problems
        ok &= complete
        adj = holm([r[-1] if np.isfinite(r[-1]) else 1.0 for r in rows])
        print(f"\n## Family {fam} ({tests[0][5]}, {tests[0][4]} {tests[0][3]})\n")
        if not complete:
            print(f"**INCOMPLETE -- no verdict.** {len(problems)} problem(s), first ones:")
            for x in problems[:8]:
                print(f"- {x}")
            print()
        print("| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | passes |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r, p_adj in zip(rows, adj):
            g, ref, other, col, am, bm, d, n, p = r
            verdict = ("yes" if p_adj < 0.05 else "no") if complete else "--"
            print(f"| {g} | {ref} | {other} | {am:.4f} | {bm:.4f} | {d:+.4f} | {n} | "
                  f"{p:.4f} | {p_adj:.4f} | {verdict} |")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
