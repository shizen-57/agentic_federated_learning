#!/usr/bin/env python3
"""Pre-flight check for a screen: every condition of the spec runs for a few
rounds on THIS machine's libraries and device, before the real runs start.

    python scripts/smoke_screen.py --spec configs/rl2_screen_b.yaml --set fl.device=auto

Screen B's first attempt lost 65 of 100 runs to a crash that only the server's
torch version showed (a tensor built from numpy bools); this finds such a
crash in about a minute. It uses the GPU-smoke seed 900 (configs/seed_registry.yaml),
never a screen seed, writes into a temporary folder that is deleted, and
prints only ok / FAILED -- no result is looked at. Exit 1 if anything fails.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

SMOKE_SEED = 900


def main() -> int:
    import yaml
    from diagnose_collapse import OPERATIONAL_OVERRIDES, non_operational, plan_for
    from sweep import build_jobs
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--set", action="append", default=[], help="operational override, e.g. fl.device=auto")
    args = ap.parse_args()
    if non_operational(args.set):
        raise SystemExit(f"--set may only change {OPERATIONAL_OVERRIDES}")
    with open(args.spec) as fh:
        spec = yaml.safe_load(fh)
    variant = next(iter(spec["variants"]))
    plan, _ = plan_for(spec, variant)
    out = tempfile.mkdtemp(prefix="afl_smoke_")
    failed = []
    try:
        from afl.config import Cfg
        from afl.engine import Experiment
        jobs = build_jobs(plan, None, [SMOKE_SEED], out,
                          [f"fl.rounds={args.rounds}", "diag.enabled=false", *args.set])
        for j in jobs:
            t0 = time.time()
            try:
                Experiment(Cfg.from_dict(j["raw"])).run(verbose=False)
                print(f"ok      {j['cond']:20s} {time.time() - t0:5.1f}s", flush=True)
            except Exception as exc:                      # report every condition, then stop
                failed.append(j["cond"])
                print(f"FAILED  {j['cond']:20s} {type(exc).__name__}: {exc}", flush=True)
                if len(failed) == 1:
                    traceback.print_exc()
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print(f"smoke check: {len(jobs) - len(failed)}/{len(jobs)} conditions run"
          + (f"; FAILED: {failed} -- do not start the screen" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
