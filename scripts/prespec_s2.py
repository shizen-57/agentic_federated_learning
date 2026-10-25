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
printed for a family unless every run is in the frozen plan, matches its
config exactly, has all its rounds, was made by the frozen code version, and
has finite values.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import load  # noqa: E402
from stats import holm, reduce, sign_flip_p  # noqa: E402
from sweep import ROOT, build_jobs, run_status  # noqa: E402

# Fingerprint (afl.provenance.code_hash) of the simulator code the s2 runs
# must have been made with, fixed when the design was frozen.
FROZEN_CODE_HASH = "cf531573f0858241"   # commit c8a91ef

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


def load_plan(root: str) -> dict:
    """(group, condition, seed) -> the job's raw config, from the frozen plan."""
    plan = yaml.safe_load(open(os.path.join(ROOT, "configs", "sweep_advanced.yaml")))
    groups = sorted({t[0] for tests in FAMILIES.values() for t in tests})
    jobs = build_jobs(plan, groups, None, root, [])
    return {(j["group"], j["cond"], j["seed"]): j["raw"] for j in jobs}


def runs(plan: dict, g: str, cond: str, col: str) -> tuple[dict[int, list[dict]], list[str], set]:
    """Seed -> rows for one condition, the problems that make runs unusable for
    a confirmatory test, and the code versions seen. A run is usable only if
    it is in the frozen plan, matches its config exactly, has every round in
    order, was made by one known code version (FROZEN_CODE_HASH when set), and
    its measurement is a finite number in every round."""
    out, problems, codes = {}, [], set()
    for seed in sorted(SEEDS):
        raw = plan.get((g, cond, seed))
        if raw is None:
            problems.append(f"{cond}_s{seed}: not in the frozen plan")
            continue
        st = run_status(raw)
        if st != "ok":
            problems.append(f"{cond}_s{seed}: {st}")
            continue
        base = os.path.join(raw["out"], raw["tag"])
        with open(base + ".config.json") as fh:
            code = json.load(fh).get("code_hash")
        if not code:
            problems.append(f"{cond}_s{seed}: no code fingerprint")
            continue
        if FROZEN_CODE_HASH and code != FROZEN_CODE_HASH:
            problems.append(f"{cond}_s{seed}: code {code}, frozen {FROZEN_CODE_HASH}")
            continue
        rows = load(base + ".csv")
        if col not in rows[0]:
            problems.append(f"{cond}_s{seed}: no column {col}")
            continue
        vals = np.array([r[col] for r in rows], dtype=float)
        if not np.isfinite(vals).all():
            problems.append(f"{cond}_s{seed}: non-finite {col}")
            continue
        codes.add(code)
        out[seed] = rows
    return out, problems, codes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "adv"))
    args = ap.parse_args()
    plan = load_plan(args.root)
    ok = True
    for fam, tests in FAMILIES.items():
        rows, problems, codes = [], [], set()
        for g, ref, other, col, how, alt in tests:
            a_runs, pa, ca = runs(plan, g, ref, col)
            b_runs, pb, cb = runs(plan, g, other, col)
            problems += [f"{g}/{x}" for x in pa + pb]
            codes |= ca | cb
            seeds = sorted(SEEDS & set(a_runs) & set(b_runs))
            if len(seeds) < len(SEEDS):
                problems.append(f"{g}: {ref} vs {other}: only {len(seeds)}/20 usable paired seeds")
            if not seeds:
                rows.append((g, ref, other, col, float("nan"), float("nan"), float("nan"), 0, float("nan")))
                continue
            a = np.array([reduce(a_runs[s], col, how) for s in seeds])
            b = np.array([reduce(b_runs[s], col, how) for s in seeds])
            rows.append((g, ref, other, col, a.mean(), b.mean(), (a - b).mean(), len(seeds),
                         sign_flip_p(a - b, alternative=alt)))
        if len(codes) > 1:
            problems.append(f"runs come from {len(codes)} different code versions")
        # A confirmatory verdict needs every run of the family, valid and made
        # the same way; otherwise numbers are shown but no verdict is.
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
