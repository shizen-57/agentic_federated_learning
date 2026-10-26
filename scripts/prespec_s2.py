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
A family member supports superiority only when its Holm-adjusted p is below
0.05 and its difference has the favourable direction. Two-sided tests can
also establish inferiority; their pre-specified alternatives are unchanged. No verdict is
printed for a family unless every run is in the frozen plan, matches its
config exactly, has all its rounds, was made by the frozen code version, and
has finite values.
"""
from __future__ import annotations

import argparse
import hashlib
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
FROZEN_CODE_HASH = "055c03e6c17728ed"
FROZEN_MANIFEST_SHA256 = "96fdf82f6baab893c84ad394a46fe7a428c3bd470fcb6de51d400b9eebfbd5c2"
MANIFEST_PATH = os.path.join(ROOT, "configs", "s2_frozen.json")
MANIFEST_INPUTS = (
    "configs/sweep_advanced.yaml", "configs/adv/cifar10.yaml", "configs/adv/femnist.yaml",
    "scripts/sweep.py", "scripts/stats.py", "scripts/report.py", "scripts/budget_s2.py",
)

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
HIGHER_IS_BETTER = {"accuracy": True, "acc_worst10": True,
                    "acc_client_worst10": True, "fail_ratio": False}


def validate_manifest() -> None:
    """Pin both simulator and experiment inputs; never silently follow live YAML."""
    if not FROZEN_CODE_HASH or not re.fullmatch(r"[0-9a-f]{16}", FROZEN_CODE_HASH):
        raise ValueError("frozen code fingerprint is unset or invalid")
    if not FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen manifest fingerprint is unset")
    with open(MANIFEST_PATH, "rb") as fh:
        data = fh.read()
    if hashlib.sha256(data).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen manifest fingerprint mismatch")
    manifest = json.loads(data)
    if manifest.get("schema_version") != 1 or manifest.get("code_hash") != FROZEN_CODE_HASH:
        raise ValueError("frozen manifest has a different simulator version")
    protocol = {"seeds": sorted(SEEDS), "families": FAMILIES,
                "higher_is_better": HIGHER_IS_BETTER}
    if manifest.get("protocol") != json.loads(json.dumps(protocol)):
        raise ValueError("test definitions differ from the frozen manifest")
    inputs = manifest.get("input_sha256", {})
    if set(inputs) != set(MANIFEST_INPUTS):
        raise ValueError("frozen manifest inputs are missing or unexpected")
    for path, expected in inputs.items():
        with open(os.path.join(ROOT, path), "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        if actual != expected:
            raise ValueError(f"{path} differs from the frozen manifest")
    from afl.provenance import code_hash
    if code_hash() != FROZEN_CODE_HASH:
        raise ValueError("simulator checkout differs from the frozen code; use its frozen snapshot")


def conclusion(col: str, diff: float, p_adj: float, alternative: str,
               complete: bool = True) -> str:
    """Interpret the original test without turning a significant loss into a win."""
    if not complete or not np.isfinite([diff, p_adj]).all():
        return "no verdict"
    if p_adj >= 0.05 or diff == 0:
        return "inconclusive"
    favourable = diff > 0 if HIGHER_IS_BETTER[col] else diff < 0
    if favourable:
        return "significantly better"
    # A one-sided superiority test provides no inferiority verdict.
    return "significantly worse" if alternative == "two-sided" else "inconclusive"


def load_plan(root: str) -> dict:
    """(group, condition, seed) -> the job's raw config, from the frozen plan."""
    validate_manifest()
    with open(os.path.join(ROOT, "configs", "sweep_advanced.yaml")) as fh:
        plan = yaml.safe_load(fh)
    groups = sorted({t[0] for tests in FAMILIES.values() for t in tests})
    jobs = build_jobs(plan, groups, None, root, [])
    return {(j["group"], j["cond"], j["seed"]): j["raw"] for j in jobs}


def runs(plan: dict, g: str, cond: str, col: str) -> tuple[dict[int, list[dict]], list[str], set]:
    """Seed -> rows for one condition, the problems that make runs unusable for
    a confirmatory test, and the code versions seen. A run is usable only if
    it is in the frozen plan, matches its config exactly, has every round in
    order, was made by the mandatory FROZEN_CODE_HASH version, and
    its measurement is a finite number in every round."""
    out, problems, codes = {}, [], set()
    if not FROZEN_CODE_HASH or not re.fullmatch(r"[0-9a-f]{16}", FROZEN_CODE_HASH):
        return {}, ["frozen code fingerprint is unset or invalid"], set()
    for seed in sorted(SEEDS):
        raw = plan.get((g, cond, seed))
        if raw is None:
            problems.append(f"{cond}_s{seed}: not in the frozen plan")
            continue
        st = run_status(raw, current_code=FROZEN_CODE_HASH)
        if st != "ok":
            problems.append(f"{cond}_s{seed}: {st}")
            continue
        base = os.path.join(raw["out"], raw["tag"])
        with open(base + ".config.json") as fh:
            code = json.load(fh).get("code_hash")
        if not code:
            problems.append(f"{cond}_s{seed}: no code fingerprint")
            continue
        if code != FROZEN_CODE_HASH:
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
    try:
        plan = load_plan(args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"**INVALID PROTOCOL -- no verdict.** {exc}")
        return 1
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
        print("| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | conclusion |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r, p_adj, test in zip(rows, adj, tests):
            g, ref, other, col, am, bm, d, n, p = r
            verdict = conclusion(col, d, p_adj, test[5], complete)
            print(f"| {g} | {ref} | {other} | {am:.4f} | {bm:.4f} | {d:+.4f} | {n} | "
                  f"{p:.4f} | {p_adj:.4f} | {verdict} |")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
