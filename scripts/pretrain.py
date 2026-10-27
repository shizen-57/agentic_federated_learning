#!/usr/bin/env python3
"""Pre-train a SAFE-PPO policy on practice episodes (docs/RL2_DESIGN.md, C9).

    python scripts/pretrain.py --group s2_femnist --condition ppo_full \
        --iterations 20 --workers 8 --out checkpoints/ppo_full_femnist.pt

One shared starting network is built once and sent to every worker (each
checks its starting checksum). Mode `average`: each iteration runs
`--workers` practice episodes in parallel from the current shared weights,
each learning online with PPO exactly as in a real run, and their final
parameters are averaged (an algorithmic choice to be compared with
`sequential`: one policy trained episode after episode, no averaging; the
mode is recorded). The condition is applied to the group's shared settings,
so it need not be one of the group's own conditions. Practice episodes:
- use practice seeds only (configs/seed_registry.yaml, 1000-1999) and the
  persistent data split (data.split = persistent): rewards come from the
  fixed validation pool, FEMNIST clients from the practice writer pool, so
  no evaluation writer or test example is ever seen;
- run with fl.practice: the reward uses the validation set, every test-set
  evaluation is skipped, and nothing is saved as a result;
- optionally randomise the device world (availability, its daily swing, the
  deadline) so the policy does not over-fit one setting.
The checkpoint records the simulator code hash, git commit, condition,
group, seeds, settings, mode, starting checksum, split fingerprint, what the
practice itself cost (episodes, rounds, simulated cost, time) and
`practice_only: true`; a run refuses a checkpoint from other code or another
data split.
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
    """The group's shared settings with the requested condition applied (the
    condition need not be one of the group's own), as a practice episode on
    the persistent split's practice pool."""
    import copy

    import numpy as np
    from afl.conditions import CONDITIONS
    from afl.run import _load
    from sweep import build_jobs
    if cond not in CONDITIONS:
        raise SystemExit(f"unknown condition {cond}")
    plan = copy.deepcopy(_load(plan_path))
    if group not in plan["groups"]:
        raise SystemExit(f"unknown group {group}")
    plan["groups"][group]["conditions"] = [cond]
    raw = build_jobs(plan, [group], [seed], "/tmp/afl_practice",
                     ["fl.practice=true", "data.split=persistent", f"fl.rounds={rounds}"])[0]["raw"]
    if randomise:
        rng = np.random.default_rng([seed, 4242])
        sysc = raw.setdefault("system", {})
        sysc["availability_p"] = float(rng.uniform(*RANDOMISE["system.availability_p"]))
        sysc["diurnal_amplitude"] = float(rng.uniform(*RANDOMISE["system.diurnal_amplitude"]))
        sysc["deadline"] = float(sysc.get("deadline", 1.0) * rng.uniform(*RANDOMISE["system.deadline"]))
    return raw


def checksum(state_dict: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode() + state_dict[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def episode(raw: dict, state_dict: dict, threads: int = 1) -> tuple[dict, dict]:
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    from afl.config import Cfg
    from afl.engine import Experiment
    t0 = time.time()
    exp = Experiment(Cfg.from_dict(raw))
    exp.policy.net.load_state_dict(state_dict)
    start = checksum(exp.policy.net.state_dict())
    rows = exp.run(verbose=False)
    stats = {"seed": raw["seed"], "start_checksum": start, "split_fingerprint": exp.split_fp,
             "successes": int(sum(r["n_success"] for r in rows)),
             "invites": int(sum(r["n_selected"] for r in rows)),
             "mean_reward": float(np.mean([r["reward"] for r in rows])),
             "val_acc_last": float(rows[-1]["val_accuracy"]),
             "cost": float(sum(r["cost"] for r in rows)), "rounds": len(rows),
             "seconds": time.time() - t0}
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
    ap.add_argument("--mode", choices=("average", "sequential"), default="average",
                    help="average: W episodes in parallel from the shared weights, then their "
                         "parameters are averaged; sequential: one policy trained episode after "
                         "episode (no averaging)")
    ap.add_argument("--set", action="append", default=[], help="extra override, e.g. fl.device=auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import seeds as seedreg
    from afl.provenance import code_hash

    per_it = args.workers if args.mode == "average" else 1
    seeds = [args.seed_start + k for k in range(args.iterations * per_it)]
    seedreg.check(seeds, "practice")
    randomise = not args.no_randomise
    from afl.config import Cfg
    from afl.policies import build_policy
    from afl.run import _apply_override

    def config(s):
        raw = practice_config(args.plan, args.group, args.condition, s, args.rounds, randomise)
        for expr in args.set:
            _apply_override(raw, expr)
        return raw

    # ONE shared starting network for every worker (review finding 3): built
    # once from the first practice config, then broadcast; each worker checks
    # it starts from exactly these weights.
    torch.manual_seed(0)
    weights = {k: v.detach().clone() for k, v in
               build_policy(Cfg.from_dict(config(seeds[0]))).net.state_dict().items()}
    init_sum = checksum(weights)
    log, fps = [], set()
    totals = {"episodes": 0, "rounds": 0, "cost": 0.0, "invites": 0, "episode_seconds": 0.0}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(per_it, mp_context=ctx) as pool:
        for it in range(args.iterations):
            raws = [config(s) for s in seeds[it * per_it:(it + 1) * per_it]]
            expect = checksum(weights)
            res = list(pool.map(episode, raws, [weights] * len(raws)))
            if any(r[1]["start_checksum"] != expect for r in res):
                raise RuntimeError("a worker did not start from the shared weights")
            weights = average([r[0] for r in res]) if len(res) > 1 else res[0][0]
            stats = [r[1] for r in res]
            fps |= {s["split_fingerprint"] for s in stats}
            for s in stats:
                totals["episodes"] += 1
                totals["rounds"] += s["rounds"]
                totals["cost"] += s["cost"]
                totals["invites"] += s["invites"]
                totals["episode_seconds"] += s["seconds"]
            log.append({"iteration": it, "seconds": round(time.time() - t0, 1), "episodes": stats})
            mean = lambda k: sum(s[k] for s in stats) / len(stats)  # noqa: E731
            print(f"[{it + 1}/{args.iterations}] success rate {mean('successes') / max(mean('invites'), 1):.3f} "
                  f"val acc {mean('val_acc_last'):.3f} reward {mean('mean_reward'):+.3f} "
                  f"cost {mean('cost'):.0f}  ({time.time() - t0:.0f}s)", flush=True)

    if len(fps) != 1 or None in fps:
        raise RuntimeError(f"practice episodes did not share one persistent split: {fps}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip() or None
    meta = {"code_hash": code_hash(), "git_commit": commit, "condition": args.condition,
            "group": args.group, "plan": os.path.relpath(args.plan, ROOT), "seeds": [seeds[0], seeds[-1]],
            "iterations": args.iterations, "workers": args.workers, "rounds": args.rounds,
            "randomise": RANDOMISE if randomise else None, "overrides": args.set,
            "mode": args.mode, "init_checksum": init_sum, "split_fingerprint": fps.pop(),
            # what pre-training itself cost, to report next to deployment cost
            "practice_totals": {**totals, "wall_seconds": round(time.time() - t0, 1)},
            "practice_only": True, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": weights, "meta": meta}, args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({"meta": meta, "log": log}, fh, indent=1)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
