#!/usr/bin/env python3
"""Run the full experimental design of Sec. 3.23 and write one CSV per run.

Every condition in afl.conditions: Table 3.4 (aggregation-rule ablation plus
the heuristic control), the Table 3.5 baselines, and the Task 1/2 ablations.
For the multi-dataset sweep of docs/ADVANCED_EXPERIMENTS.md use scripts/sweep.py.

Runs are sequential and each one is independent, so this script is safe to stop
and restart: it skips any run whose CSV already exists. That matters on free
compute, where a session can be reclaimed mid-sweep.

    python scripts/run_experiments.py --config configs/mnist.yaml --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from afl.conditions import CONDITIONS, apply  # noqa: E402
from afl.config import Cfg            # noqa: E402
from afl.engine import Experiment     # noqa: E402
from afl.run import _load             # noqa: E402

# Every condition in the shared registry, in its declared order. Earlier
# versions of this script listed only Table 3.4 and four of the Table 3.5
# baselines; the ablations, FedCER-like and MIFA/MimiC-like runs had to be
# launched by hand. They are now part of the same sweep.
RUNS = list(CONDITIONS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mnist.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", help="restrict to these run tags")
    ap.add_argument("--force", action="store_true", help="re-run even if the CSV exists")
    ap.add_argument("--tag-prefix", default="",
                    help="prefix every run tag with this, e.g. 'n100_' for a "
                         "different client population -- RUNS tags are shared "
                         "across configs, so two sweeps at different scales "
                         "collide and silently skip each other without one")
    args = ap.parse_args()

    base = _load(args.config)
    todo = [r for r in RUNS if not args.only or r in args.only]

    for seed in args.seeds:
        for tag in todo:
            full = f"{args.tag_prefix}{tag}_s{seed}"
            path = os.path.join(args.out, f"{full}.csv")
            if os.path.exists(path) and not args.force:
                print(f"skip {full} (exists)")
                continue

            raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
            raw["tag"] = full
            raw["seed"] = seed
            raw["out"] = args.out
            raw.setdefault("data", {})["seed"] = seed
            apply(raw, CONDITIONS[tag])

            t0 = time.time()
            exp = Experiment(Cfg.from_dict(raw))
            exp.run(verbose=True)
            exp.save()
            print(f"done {full} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
