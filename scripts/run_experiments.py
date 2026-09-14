#!/usr/bin/env python3
"""Run the full experimental design of Sec. 3.23 and write one CSV per run.

Conditions 1-4 of Table 3.4 (aggregation-rule ablation, plus the heuristic
control) and the external baselines of Table 3.5 (Table 3.5's FedCER/MIFA rows
are not implemented here -- see docs/EXPERIMENT_PLAN.md).

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

from afl.config import Cfg            # noqa: E402
from afl.engine import Experiment     # noqa: E402
from afl.run import _load             # noqa: E402

# (tag, policy, aggregator) -- the tag is what report.py groups on.
RUNS = [
    ("c1_rl_fedavg",        "rl",            "fedavg"),
    ("c2_rl_fedprox",       "rl",            "fedprox"),
    ("c3_rl_fednova",       "rl",            "fednova"),
    ("c4_heuristic_fedprox", "roundrobin",   "fedprox"),
    ("b_random",            "random",        "fedprox"),
    ("b_fedcs",             "fedcs",         "fedprox"),
    ("b_powerofchoice",     "powerofchoice", "fedprox"),
    ("b_oort",              "oort",          "fedprox"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mnist.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", help="restrict to these run tags")
    ap.add_argument("--force", action="store_true", help="re-run even if the CSV exists")
    args = ap.parse_args()

    base = _load(args.config)
    todo = [r for r in RUNS if not args.only or r[0] in args.only]

    for seed in args.seeds:
        for tag, policy, agg in todo:
            full = f"{tag}_s{seed}"
            path = os.path.join(args.out, f"{full}.csv")
            if os.path.exists(path) and not args.force:
                print(f"skip {full} (exists)")
                continue

            raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
            raw["tag"] = full
            raw["seed"] = seed
            raw["out"] = args.out
            raw.setdefault("data", {})["seed"] = seed
            raw.setdefault("fl", {})["aggregator"] = agg
            raw.setdefault("policy", {})["name"] = policy

            t0 = time.time()
            exp = Experiment(Cfg.from_dict(raw))
            exp.run(verbose=True)
            exp.save()
            print(f"done {full} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
