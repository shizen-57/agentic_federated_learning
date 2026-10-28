#!/usr/bin/env python3
"""RL2 M9: freeze the confirmation (docs/RL2_CONFIRMATION.md).

    python scripts/freeze_rl2.py --decision results/rl2_screen_c/full/DECISION.json

From screen C's decision: the final method (tuned settings, pre-trained or
not, and the pre-training recipe), the comparators (always-Light, screen C's
best baseline, the repaired original RL), the 8 primary tests, the
secondary comparisons, and the number of seeds from a power analysis on
screen C's paired spread. Writes configs/rl2_frozen.json and pins its SHA-256
and the simulator code hash into scripts/prespec_rl2.py, both at once.
The confirmation runs on the reserved confirmation pool (data.eval_pool
confirm; configs/rl2_data_roles.json, pinned) and pre-trains its methods
afresh under the frozen code. Refuses: a checkout with uncommitted changes,
data roles other than the recorded ones, or an earlier freeze (pins set).
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
ENDPOINTS = {
    "eq5": "P1, accuracy at equal spend: mean fixed-set accuracy over the last 5 rounds whose "
           "cumulative cost is <= 1,400",
    "w10": "P2, weakest clients: mean worst-10% client accuracy over the last 10 rounds",
}


def seed_count(tests: list[dict]) -> tuple[int, list[dict]]:
    rows = []
    for t in tests:
        need = rl2lib.seeds_needed(max(t["sd"], 1e-9), DELTA[t["family"]], TESTS, POWER, ALPHA)
        rows.append({**{k: t[k] for k in ("family", "dataset", "vs", "endpoint", "sd")},
                     "delta": DELTA[t["family"]], "n_needed": need})
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
    spec_path = os.path.join(ROOT, dec["spec"])
    with open(spec_path) as fh:
        spec = yaml.safe_load(fh)
    if rl2lib._sha(spec_path) != dec["spec_sha256"]:
        raise SystemExit(f"{dec['spec']} changed since screen C")
    if rl2lib._sha(os.path.join(ROOT, dec["tuned"])) != dec["tuned_sha256"]:
        raise SystemExit(f"{dec['tuned']} changed since screen C")
    with open(os.path.join(ROOT, dec["tuned"])) as fh:
        tuned = json.load(fh)

    n, power_rows = seed_count(dec["tests"])
    seeds = list(range(FIRST_SEED, FIRST_SEED + n))
    best, p2 = dec["best_baseline"], dec["p2_reference"]
    refs = list(dict.fromkeys(["rl_light", best, p2, "rl_light_mimic"]))
    # The confirmation is measured on the reserved confirmation pool, and its
    # pre-trained methods are pre-trained afresh in M10 under the frozen code
    # (practice data only), into their own folder.
    cspec = {**spec, "seeds": seeds, "references": refs, "purpose": "confirmation",
             "set": {**(spec.get("set") or {}), "data.eval_pool": "confirm"},
             "pretrain": {**spec["pretrain"], "checkpoints": os.path.join(spec["pretrain"]["checkpoints"], "confirm")}}
    plan = sc.build_plan(cspec, tuned)
    rl2lib.check_pool(plan, "confirmation")
    final = dec["final"]["name"]
    primary = [{"family": fam, "dataset": d, "vs": vs, "endpoint": k, "alternative": "greater",
                "delta": DELTA[fam]}
               for d in spec["datasets"] for fam, k in sc.PRIMARY
               for vs in ("rl_light", best if fam == "P1" else p2)]
    others = [c for c in [*(f"{m}_pt" for m in spec["pretrained"]), *(f"{m}_scratch" for m in spec["scratch"]),
                          "rl_light_mimic"] if c != final]
    secondary = [{"dataset": d, "a": final, "b": c, "endpoint": k}
                 for d in spec["datasets"] for c in others for k in ("eq5", "w10", "acc10")]
    secondary += [{"dataset": d, "a": "tuned_arl4_pt", "b": "tuned_noarl_pt", "endpoint": k}
                  for d in spec["datasets"] for k in ("eq5", "w10")]
    inputs = {p: rl2lib._sha(os.path.join(ROOT, p)) for p in
              [spec["source_plan"], dec["spec"], dec["tuned"], os.path.relpath(data_roles.PATH, ROOT),
               *sorted({e["config"] for e in plan["groups"].values()})]}
    frozen = {
        "schema_version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "code_hash": code_hash(), "git_commit": rl2lib.git("rev-parse", "HEAD"),
        "final": dec["final"], "best_baseline": best, "p2_reference": p2,
        "spec": cspec, "tuned_overrides": tuned["overrides"], "seeds": seeds, "plan": plan,
        "endpoints": ENDPOINTS, "primary_tests": primary, "secondary_tests": secondary,
        "alpha": ALPHA, "multiplicity": f"Holm over the {TESTS} primary tests",
        "test": "paired sign-flip on the per-seed difference, one-sided (final better); exact for "
                "n <= 20, else Monte Carlo with 200,000 draws, seed 20261028",
        "power": {"target": POWER, "rows": power_rows, "n": n, "bounds": [MIN_SEEDS, MAX_SEEDS],
                  "note": "Bonferroni-safe for 8 tests (Holm is never less powerful)"},
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
    print(f"frozen: {args.out} (sha256 {sha})\nfinal method {final}; best baseline {best}; "
          f"seeds {seeds[0]}-{seeds[-1]} (n={n}); {runs} runs\npins written into scripts/prespec_rl2.py "
          "-- commit both files now, then the confirmation can run")
    if any(r["n_needed"] > MAX_SEEDS for r in power_rows):
        print(f"note: some tests would need more than {MAX_SEEDS} seeds for {POWER:.0%} power; "
              "their power is lower, and the report must say so")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
