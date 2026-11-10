#!/usr/bin/env python3
"""RL2 M10: run the frozen confirmation (docs/RL2_CONFIRMATION.md).

    python scripts/confirm_rl2.py --out runs/rl2_confirm                      # checks + what would run
    python scripts/confirm_rl2.py --out runs/rl2_confirm --jobs 6 --set fl.device=auto --go

Runs only with --go (the user's explicit go-ahead), and only if: the frozen
manifest is the one pinned in scripts/prespec_rl2.py; this checkout's
simulator code is the frozen code and has no uncommitted changes; every
pinned input is byte-identical; the seeds are confirmation seeds run from
this frozen manifest. Then: pre-training of the
methods under the frozen code (practice data only), a 2-round pre-flight of
every condition (seed 900), the runs on the reserved confirmation pool
(resumable), and scripts/prespec_rl2.py's pre-specified tests on the
complete evidence.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import rl2lib  # noqa: E402


def preconditions(frozen: dict, frozen_path: str) -> list[str]:
    from afl.provenance import code_hash
    problems = []
    if code_hash() != frozen["code_hash"]:
        problems.append(f"simulator code {code_hash()} is not the frozen {frozen['code_hash']}")
    if rl2lib.git("status", "--porcelain", "--untracked-files=no"):
        problems.append("the checkout has uncommitted changes")
    for p, sha in frozen["inputs"].items():
        if rl2lib._sha(os.path.join(ROOT, p)) != sha:
            problems.append(f"{p} is missing or not the frozen file")
    import seeds as seedreg
    try:
        seedreg.check(frozen["seeds"], "confirmation", frozen_path)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def main() -> int:
    import prespec_rl2 as pre
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=os.path.join(ROOT, "configs", "rl2_frozen.json"))
    ap.add_argument("--out", default=os.path.join("runs", "rl2_confirm"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="operational override, e.g. fl.device=auto")
    ap.add_argument("--go", action="store_true", help="the user's go-ahead: actually run")
    ap.add_argument("--max-procs", type=int, default=0, help="pre-training: episodes at once (GPU memory)")
    args = ap.parse_args()
    if rl2lib.non_operational(args.set):
        raise SystemExit(f"--set may only be operational, not {rl2lib.non_operational(args.set)}")
    frozen = pre.verify_frozen(args.frozen)
    problems = preconditions(frozen, args.frozen)
    plan = frozen["plan"]
    rl2lib.check_pool(plan, "confirmation")
    n = sum(len(g["conditions"]) * len(g["seeds"]) for g in plan["groups"].values())
    print(f"frozen confirmation: final method {frozen['final']['name']}, seeds {frozen['seeds'][0]}-"
          f"{frozen['seeds'][-1]}, {len(plan['groups'])} groups, {n} runs")
    if problems:
        print("NOT READY:\n" + "\n".join(f"- {p}" for p in problems))
        return 1
    if not args.go:
        print("all checks pass; nothing run (add --go to start, with the user's go-ahead)")
        return 0
    import screen_c_rl2 as sc
    spec, tuned = frozen["spec"], pre.tuned_of(frozen)
    # 1. pre-train the methods under the frozen code (practice data only;
    #    resumable; a valid checkpoint is kept)
    if sc.pretrain_methods(spec, tuned, args.max_procs, args.set):
        return 1
    cwd = os.getcwd()
    os.chdir(ROOT)                          # checkpoints are found relative to the repository
    try:
        if rl2lib.smoke(plan, args.set):    # 2. every condition, 2 rounds, seed 900
            return 1
    finally:
        os.chdir(cwd)
    rl2lib.write_plan(plan, args.out)       # 3. the runs, on the confirmation pool
    rl2lib.write_manifest(args.out, {**pre.inputs_of(frozen, args.frozen), **sc.checkpoint_inputs(spec, tuned)},
                          ["fl.rounds", "diag.enabled", "out", *args.set], stage="confirmation")
    rl2lib.run_sweep(os.path.join(args.out, "plan.yaml"), args.out, args.jobs, args.set)
    code, report, results = pre.evaluate(frozen, args.frozen, args.out)
    print(report)
    if not code:
        import json
        with open(os.path.join(args.out, "report.md"), "w") as fh:
            fh.write(report)
        with open(os.path.join(args.out, "RESULTS.json"), "w") as fh:
            json.dump(results, fh, indent=1, sort_keys=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
