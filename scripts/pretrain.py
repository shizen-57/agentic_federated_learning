#!/usr/bin/env python3
"""Pre-train a SAFE-PPO policy on practice episodes (docs/RL2_DESIGN.md, C9).

    python scripts/pretrain.py --group s2_femnist --condition ppo_full \
        --iterations 20 --workers 8 --out checkpoints/ppo_full_femnist.pt

Each iteration runs `--workers` practice episodes in parallel, each starting
from the current shared weights and learning online with PPO exactly as in a
real run; the episodes' final weights are averaged into the new shared
weights. Practice episodes:
- use practice seeds only (configs/seed_registry.yaml, 1000-1999), so their
  client partitions and device draws are never the evaluation ones;
- run with fl.practice: the reward uses the validation set, every test-set
  evaluation is skipped, and nothing is saved as a result;
- optionally randomise the device world (availability, its daily swing, the
  deadline) so the policy does not over-fit one setting.
The checkpoint records the simulator code hash, git commit, condition,
group, seeds, settings and `practice_only: true`; SafePPOPolicy refuses to
load a checkpoint from other code.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

RANDOMISE = {"system.availability_p": (0.75, 0.95), "system.diurnal_amplitude": (0.2, 0.5),
             "system.deadline": (0.85, 1.15)}            # deadline: x the group's value


def practice_config(plan_path: str, group: str, cond: str, seed: int, rounds: int,
                    randomise: bool) -> dict:
    import numpy as np
    from afl.run import _load
    from sweep import build_jobs
    plan = _load(plan_path)
    jobs = [j for j in build_jobs(plan, [group], [seed], "/tmp/afl_practice",
                                  ["fl.practice=true", f"fl.rounds={rounds}"]) if j["cond"] == cond]
    if not jobs:
        raise SystemExit(f"condition {cond} is not in group {group}")
    raw = jobs[0]["raw"]
    if randomise:
        rng = np.random.default_rng([seed, 4242])
        sysc = raw.setdefault("system", {})
        sysc["availability_p"] = float(rng.uniform(*RANDOMISE["system.availability_p"]))
        sysc["diurnal_amplitude"] = float(rng.uniform(*RANDOMISE["system.diurnal_amplitude"]))
        sysc["deadline"] = float(sysc.get("deadline", 1.0) * rng.uniform(*RANDOMISE["system.deadline"]))
    return raw


def episode(raw: dict, state_dict: dict | None, threads: int = 1) -> tuple[dict, dict]:
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    from afl.config import Cfg
    from afl.engine import Experiment
    exp = Experiment(Cfg.from_dict(raw))
    if state_dict is not None:
        exp.policy.net.load_state_dict(state_dict)
    rows = exp.run(verbose=False)
    stats = {"seed": raw["seed"], "successes": int(sum(r["n_success"] for r in rows)),
             "invites": int(sum(r["n_selected"] for r in rows)),
             "mean_reward": float(np.mean([r["reward"] for r in rows])),
             "val_acc_last": float(rows[-1]["val_accuracy"]),
             "cost": float(sum(r["cost"] for r in rows))}
    return {k: v.detach().clone() for k, v in exp.policy.net.state_dict().items()}, stats


def average(states: list[dict]) -> dict:
    import torch
    return {k: torch.stack([s[k].float() for s in states]).mean(0).to(states[0][k].dtype)
            for k in states[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    ap.add_argument("--group", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--no-randomise", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="extra override, e.g. fl.device=auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import seeds as seedreg
    from afl.provenance import code_hash

    seeds = [args.seed_start + k for k in range(args.iterations * args.workers)]
    seedreg.check(seeds, "practice")
    randomise = not args.no_randomise
    weights, log = None, []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
        for it in range(args.iterations):
            batch = seeds[it * args.workers:(it + 1) * args.workers]
            raws = [practice_config(args.plan, args.group, args.condition, s, args.rounds, randomise)
                    for s in batch]
            for raw in raws:
                for expr in args.set:
                    from afl.run import _apply_override
                    _apply_override(raw, expr)
            res = list(pool.map(episode, raws, [weights] * len(raws)))
            weights = average([r[0] for r in res])
            stats = [r[1] for r in res]
            log.append({"iteration": it, "seconds": round(time.time() - t0, 1), "episodes": stats})
            mean = lambda k: sum(s[k] for s in stats) / len(stats)  # noqa: E731
            print(f"[{it + 1}/{args.iterations}] success rate {mean('successes') / max(mean('invites'), 1):.3f} "
                  f"val acc {mean('val_acc_last'):.3f} reward {mean('mean_reward'):+.3f} "
                  f"cost {mean('cost'):.0f}  ({time.time() - t0:.0f}s)", flush=True)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip() or None
    meta = {"code_hash": code_hash(), "git_commit": commit, "condition": args.condition,
            "group": args.group, "plan": os.path.relpath(args.plan, ROOT), "seeds": [seeds[0], seeds[-1]],
            "iterations": args.iterations, "workers": args.workers, "rounds": args.rounds,
            "randomise": RANDOMISE if randomise else None, "overrides": args.set,
            "practice_only": True, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": weights, "meta": meta}, args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({"meta": meta, "log": log}, fh, indent=1)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
