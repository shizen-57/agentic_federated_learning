#!/usr/bin/env python3
"""Run the advanced multi-dataset sweep described in configs/sweep_advanced.yaml.

Every (group, condition, seed) is one independent run writing
<out>/<group>/<condition>_s<seed>.csv. Like run_experiments.py, it skips any run
whose CSV already exists, so a reclaimed Colab/Kaggle session costs one run,
not the sweep.

    python scripts/sweep.py --list
    python scripts/sweep.py --groups 'weak_*' --jobs 4
    python scripts/sweep.py --groups main_cifar --shard 1/3     # machine 2 of 3
    python scripts/sweep.py --groups main_mnist --seeds 0 1 --set fl.rounds=5   # smoke

--jobs runs that many processes side by side. The models are small, so on a
CPU several single-threaded runs finish sooner than one run using every core;
on a GPU several processes share the card. --shard i/n keeps every n-th run
starting at i, so n machines can split one group with no coordination.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from afl.conditions import CONDITIONS, apply  # noqa: E402
from afl.run import _apply_override, _load     # noqa: E402


def build_jobs(plan: dict, groups: list[str] | None, seeds: list[int] | None,
               out: str, extra: list[str]) -> list[dict]:
    """Expand the plan into one fully-specified raw config per run."""
    sets = plan.get("sets", {})
    use_seeds = seeds if seeds is not None else plan["seeds"]
    jobs = []
    for gname, g in plan["groups"].items():
        if groups and not any(fnmatch.fnmatch(gname, pat) for pat in groups):
            continue
        conds = g["conditions"]
        conds = sets[conds] if isinstance(conds, str) else conds
        unknown = [c for c in conds if c not in CONDITIONS]
        if unknown:
            raise KeyError(f"group {gname}: unknown conditions {unknown}")
        base = _load(os.path.join(ROOT, g["config"]))
        for seed in use_seeds:
            for cond in conds:
                raw = _deepcopy(base)
                apply(raw, g.get("set", {}) or {})
                apply(raw, CONDITIONS[cond])
                for expr in extra:
                    _apply_override(raw, expr)
                raw["tag"] = f"{cond}_s{seed}"
                raw["seed"] = seed
                raw.setdefault("data", {})["seed"] = seed
                raw["out"] = os.path.join(out, gname)
                jobs.append({"group": gname, "cond": cond, "seed": seed, "raw": raw})
    return jobs


def _deepcopy(d):
    return {k: _deepcopy(v) if isinstance(v, dict) else v for k, v in d.items()}


def _run_one(raw: dict, threads: int, slot: int = 0) -> tuple[str, float]:
    import torch
    torch.set_num_threads(threads)
    from afl.config import Cfg
    from afl.engine import Experiment
    # With several GPUs (e.g. Kaggle's 2x T4), spread runs across them by slot.
    if raw.get("fl", {}).get("device") in ("auto", "cuda") and torch.cuda.device_count() > 1:
        raw["fl"]["device"] = f"cuda:{slot % torch.cuda.device_count()}"
    t0 = time.time()
    exp = Experiment(Cfg.from_dict(raw))
    exp.run(verbose=False)
    exp.save()
    return f"{os.path.basename(raw['out'])}/{raw['tag']}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    ap.add_argument("--groups", nargs="*",
                    help="group names or glob patterns, space- or comma-separated (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", help="override the plan's seeds")
    ap.add_argument("--out", default=os.path.join("results", "adv"))
    ap.add_argument("--set", action="append", default=[],
                    help="dotted override applied to every run, e.g. --set fl.device=cuda")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch threads per run (default: cores // jobs)")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--force", action="store_true", help="re-run even if the CSV exists")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="stop starting new runs after this many hours and let the running "
                         "ones finish (for sessions with a hard limit, e.g. Kaggle's 12 h)")
    args = ap.parse_args()

    plan = _load(args.plan)
    groups = [g for arg in (args.groups or []) for g in arg.split(",") if g] or None
    jobs = build_jobs(plan, groups, args.seeds, args.out, args.set)
    i, n = (int(v) for v in args.shard.split("/"))
    jobs = [j for k, j in enumerate(jobs) if k % n == i]

    def done(j):
        return os.path.exists(os.path.join(j["raw"]["out"], f"{j['raw']['tag']}.csv"))

    if args.list:
        by_group: dict[str, list] = {}
        for j in jobs:
            by_group.setdefault(j["group"], []).append(j)
        total = todo = 0
        for g, js in by_group.items():
            left = sum(not done(j) for j in js)
            d = js[0]["raw"]["data"]
            print(f"{g:22s} {d['dataset']:8s} N={d['num_clients']:<4d} "
                  f"rounds={js[0]['raw']['fl']['rounds']:<4d} runs={len(js):4d} todo={left}")
            total += len(js)
            todo += left
        print(f"{'TOTAL':22s} runs={total} todo={todo}")
        return 0

    pending = [j for j in jobs if args.force or not done(j)]
    print(f"{len(jobs)} runs selected, {len(jobs) - len(pending)} already done, "
          f"{len(pending)} to run with {args.jobs} job(s)", flush=True)
    if not pending:
        return 0

    # Download/prepare every dataset once, before parallel workers race on it.
    from afl.data import ensure
    for ds in sorted({j["raw"]["data"]["dataset"] for j in pending}):
        if ds != "synthetic":
            ensure(ds)

    threads = args.threads or max(1, (os.cpu_count() or 1) // max(args.jobs, 1))
    t_start = time.time()
    deadline = t_start + args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    finished = failed = 0
    queue = list(pending)

    def report(j, result=None, exc=None):
        nonlocal finished, failed
        if exc is None:
            finished += 1
            print(f"[{finished}/{len(pending)}] {result[0]} {result[1]:.0f}s", flush=True)
        else:  # keep going: one bad run must not end a sweep
            failed += 1
            print(f"FAILED {j['group']}/{j['raw']['tag']}: {exc!r}", flush=True)

    if args.jobs <= 1:
        while queue and time.time() < deadline:
            j = queue.pop(0)
            try:
                report(j, _run_one(j["raw"], threads))
            except Exception as exc:
                report(j, exc=exc)
    else:
        # Submit lazily, never more than --jobs at a time, so --max-hours can
        # stop new runs from starting; each run keeps its own GPU slot.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            running: dict = {}
            free_slots = list(range(args.jobs))
            while queue or running:
                while queue and free_slots and time.time() < deadline:
                    j, slot = queue.pop(0), free_slots.pop(0)
                    running[pool.submit(_run_one, j["raw"], threads, slot)] = (j, slot)
                if not running:
                    break
                done_set, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    j, slot = running.pop(fut)
                    free_slots.append(slot)
                    try:
                        report(j, fut.result())
                    except Exception as exc:
                        report(j, exc=exc)
    if queue:
        print(f"time budget of {args.max_hours} h reached: {len(queue)} runs not started "
              f"(run the same command again to continue)", flush=True)
    print(f"finished {finished}, failed {failed}, wall {time.time() - t_start:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
