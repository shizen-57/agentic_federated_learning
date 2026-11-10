#!/usr/bin/env python3
"""RL2 M9: freeze the confirmation (docs/RL2_CONFIRMATION.md).

    python scripts/freeze_rl2.py --decision results/rl2_screen_c/full/DECISION.json

From screen C's decision: the final method (tuned settings, pre-trained or
not, and the pre-training recipe), the comparators (the tuned always-Light
with MimiC, and screen C's best tuned comparator on each endpoint), the 8
primary tests, the secondary comparisons, the secondary worlds, and the
number of seeds from a power analysis on screen C's paired spread (taken
cautiously: an 80% upper confidence limit of the spread). Writes configs/rl2_frozen.json and pins its SHA-256
and the simulator code hash into scripts/prespec_rl2.py, both at once.
The confirmation runs on the reserved confirmation pool (data.eval_pool
confirm; configs/rl2_data_roles.json, pinned) and pre-trains its methods
afresh under the frozen code. Refuses: a checkout with uncommitted changes,
data roles other than the recorded ones, a screen C decision not made on
the development pool of those roles, or an earlier freeze (pins set).
Seeds: confirmation seeds 300 upward; never run before (configs/seed_registry.yaml).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import yaml  # noqa: E402

import rl2lib  # noqa: E402

DELTA = {"P1": 0.01, "P2": 0.02}   # smallest differences worth detecting (1 and 2 points)
POWER, ALPHA, TESTS = 0.8, 0.05, 8
MIN_SEEDS, MAX_SEEDS, FIRST_SEED = 20, 100, 300
SD_UCL = 0.8                       # the spread used for power: its 80% upper confidence limit (gap review B8)
ENDPOINTS = {
    "eq5": "P1, accuracy at equal spend: mean fixed-set accuracy over the last 5 rounds whose "
           "cumulative cost is <= 1,400",
    "w10b": "P2, weakest clients at equal spend: mean worst-10% client accuracy over the last 10 "
            "rounds whose cumulative cost is <= 1,400",
}
# Secondary worlds (gap review B3-B5), descriptive: the final method --
# pre-trained in the usual world, if it is pre-trained -- and the control.
WORLDS = {
    "fedscale_devices": {"system.device_trace": "fedscale"},   # measured phone speeds and bandwidths
    "weak_random": {"system.weak_select": "random"},          # weak devices hold random data, not the rarest
    "no_weak": {"system.weak_frac": 0.0},                     # no weak devices at all
    "churn": {"system.churn_after": 3, "system.churn_prob": 0.5},   # clients may leave for good
}
WORLD_SEEDS = 20


def sd_upper(sd: float, n: int, level: float = SD_UCL) -> float:
    """One-sided upper confidence limit of a standard deviation from n paired
    values: sd * sqrt((n-1) / chi2_{1-level}(n-1)), the chi-square quantile by
    the Wilson-Hilferty approximation (no scipy needed)."""
    from statistics import NormalDist
    k = max(n - 1, 1)
    z = NormalDist().inv_cdf(1 - level)
    q = k * (1 - 2 / (9 * k) + z * (2 / (9 * k)) ** 0.5) ** 3
    return sd * (k / q) ** 0.5


def seed_count(tests: list[dict]) -> tuple[int, list[dict]]:
    rows = []
    for t in tests:
        sd = sd_upper(max(t["sd"], 1e-9), t.get("n", 10))
        need = rl2lib.seeds_needed(sd, DELTA[t["family"]], TESTS, POWER, ALPHA)
        rows.append({**{k: t[k] for k in ("family", "dataset", "vs", "endpoint", "sd")},
                     "sd_used": sd, "delta": DELTA[t["family"]], "n_needed": need})
    return min(MAX_SEEDS, max(MIN_SEEDS, max(r["n_needed"] for r in rows))), rows


def main() -> int:
    import prespec_rl2 as pre
    import screen_c_rl2 as sc
    from afl.provenance import code_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", required=True, help="screen C's DECISION.json (committed)")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "rl2_frozen.json"))
    ap.add_argument("--prespec", default=os.path.join(ROOT, "scripts", "prespec_rl2.py"),
                    help="the file whose pins are written (tests use a copy)")
    args = ap.parse_args()
    if pre.FROZEN_SHA256 is not None:
        raise SystemExit("an RL2 confirmation is already frozen; a freeze is never redone to admit new code")
    if rl2lib.git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit everything first: the freeze records the commit it was made from")
    with open(args.decision) as fh:
        dec = json.load(fh)
    import data_roles
    with open(data_roles.PATH) as fh:
        if json.load(fh) != data_roles.roles():
            raise SystemExit(f"{data_roles.PATH} does not match the data roles this code makes")
    # The method must be chosen on development data only. Screen C runs made
    # before the reserved pools existed read the whole evaluation pool (now
    # largely confirmation data): they stay exploratory and cannot be frozen.
    if dec.get("eval_pool") != "dev" or dec.get("data_roles_sha256") != rl2lib._sha(data_roles.PATH):
        raise SystemExit("screen C's decision was not made on the development pool of the recorded data "
                         "roles (a screen C from before the reserved pools?); re-run screen C on this code")
    spec_path = os.path.join(ROOT, dec["spec"])
    with open(spec_path) as fh:
        spec = yaml.safe_load(fh)
    if rl2lib._sha(spec_path) != dec["spec_sha256"]:
        raise SystemExit(f"{dec['spec']} changed since screen C")
    if rl2lib._sha(os.path.join(ROOT, dec["tuned"])) != dec["tuned_sha256"]:
        raise SystemExit(f"{dec['tuned']} changed since screen C")
    with open(os.path.join(ROOT, dec["tuned"])) as fh:
        tuned = json.load(fh)
    # each tuned comparator's settings: as screen C used them, and unchanged since
    for name, tr in dec["tuned_references"].items():
        with open(os.path.join(ROOT, tr["file"])) as fh:
            t = json.load(fh)
        if (t["base"], t["overrides"]) != (tr["base"], tr["overrides"]):
            raise SystemExit(f"{tr['file']} changed since screen C")

    n, power_rows = seed_count(dec["tests"])
    seeds = list(range(FIRST_SEED, FIRST_SEED + n))
    control, best, best_p2 = dec["control"], dec["best_baseline"], dec["best_baseline_p2"]
    final = dec["final"]["name"]
    trefs = {k: dec["tuned_references"][k] for k in dict.fromkeys([control, best, best_p2])}
    # The confirmation is measured on the reserved confirmation pool, and its
    # pre-trained methods are pre-trained afresh in M10 under the frozen code
    # (practice data only), into their own folder.
    cspec = {**spec, "seeds": seeds, "references": ["rl_light"], "tuned_references": trefs,
             "control": control, "purpose": "confirmation",
             "set": {**(spec.get("set") or {}), "data.eval_pool": "confirm"},
             "worlds": {"settings": WORLDS, "seeds": seeds[:WORLD_SEEDS], "final": final, "control": control},
             "pretrain": {**spec["pretrain"], "checkpoints": os.path.join(spec["pretrain"]["checkpoints"], "confirm")}}
    plan = sc.build_plan(cspec, tuned)
    rl2lib.check_pool(plan, "confirmation")
    primary = [{"family": fam, "dataset": d, "vs": vs, "endpoint": k, "alternative": "greater",
                "delta": DELTA[fam]}
               for d in spec["datasets"] for fam, k in sc.PRIMARY
               for vs in (control, best if fam == "P1" else best_p2)]
    others = [c for c in [*(f"{m}_pt" for m in spec["pretrained"]), *(f"{m}_scratch" for m in spec["scratch"]),
                          "rl_light"] if c != final]
    secondary = [{"dataset": d, "a": final, "b": c, "endpoint": k}
                 for d in spec["datasets"] for c in others for k in ("eq5", "w10b", "w10", "acc10")]
    secondary += [{"dataset": d, "a": final, "b": c, "endpoint": "w10"}       # the whole-run fairness too
                  for d in spec["datasets"] for c in trefs]
    secondary += [{"dataset": d, "a": "tuned_arl4_pt", "b": "tuned_noarl_pt", "endpoint": k}
                  for d in spec["datasets"] for k in ("eq5", "w10b")]
    inputs = {p: rl2lib._sha(os.path.join(ROOT, p)) for p in
              [spec["source_plan"], dec["spec"], dec["tuned"], os.path.relpath(data_roles.PATH, ROOT),
               *sorted(tr["file"] for tr in trefs.values()),
               *sorted({e["config"] for e in plan["groups"].values()})]}
    frozen = {
        "schema_version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "code_hash": code_hash(), "git_commit": rl2lib.git("rev-parse", "HEAD"),
        "final": dec["final"], "control": control, "best_baseline": best, "best_baseline_p2": best_p2,
        "worlds": {"settings": WORLDS, "n_seeds": WORLD_SEEDS,
                   "note": "secondary, descriptive: final method vs the control in each world"},
        "spec": cspec, "tuned_overrides": tuned["overrides"], "seeds": seeds, "plan": plan,
        "endpoints": ENDPOINTS, "primary_tests": primary, "secondary_tests": secondary,
        "alpha": ALPHA, "multiplicity": f"Holm over the {TESTS} primary tests",
        "test": "paired sign-flip on the per-seed difference, one-sided (final better); exact for "
                "n <= 20, else Monte Carlo with 200,000 draws, seed 20261028",
        "power": {"target": POWER, "rows": power_rows, "n": n, "bounds": [MIN_SEEDS, MAX_SEEDS],
                  "note": "Bonferroni-safe for 8 tests (Holm is never less powerful); the spread used is "
                          f"the {SD_UCL:.0%} upper confidence limit of screen C's"},
        "pretraining": "in M10, under the frozen code, by spec['pretrain'] (practice seeds and "
                       "practice data only); the checkpoints' SHA-256 go into the confirmation's MANIFEST",
        "inputs": inputs, "screen_c_code_hash": dec["code_hash"],
        "screen_c_decision_sha256": rl2lib._sha(args.decision),
    }
    data = (json.dumps(frozen, indent=1, sort_keys=True) + "\n").encode()
    with open(args.out, "wb") as fh:
        fh.write(data)
    sha = rl2lib._sha(args.out)
    path = args.prespec
    with open(path) as fh:
        src = fh.read()
    src, n1 = re.subn(r'^FROZEN_SHA256 = None$', f'FROZEN_SHA256 = "{sha}"', src, flags=re.M)
    src, n2 = re.subn(r'^FROZEN_CODE = None$', f'FROZEN_CODE = "{code_hash()}"', src, flags=re.M)
    if n1 != 1 or n2 != 1:
        raise SystemExit("could not find the pins in prespec_rl2.py")
    with open(path, "w") as fh:
        fh.write(src)
    runs = sum(len(g["conditions"]) * len(g["seeds"]) for g in plan["groups"].values())
    print(f"frozen: {args.out} (sha256 {sha})\nfinal method {final}; control {control}; "
          f"P1 comparator {best}; P2 comparator {best_p2}; "
          f"seeds {seeds[0]}-{seeds[-1]} (n={n}); {runs} runs\npins written into scripts/prespec_rl2.py "
          "-- commit both files now, then the confirmation can run")
    if any(r["n_needed"] > MAX_SEEDS for r in power_rows):
        print(f"note: some tests would need more than {MAX_SEEDS} seeds for {POWER:.0%} power; "
              "their power is lower, and the report must say so")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
