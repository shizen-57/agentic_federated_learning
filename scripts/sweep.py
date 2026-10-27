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
import json
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
    jobs = []
    names = list(plan["groups"])
    if groups:
        # Run groups in the order they were asked for (by first matching
        # pattern), so a session that hits its time limit has done the most
        # important ones first.
        rank = {n: min((k for k, pat in enumerate(groups) if fnmatch.fnmatch(n, pat)),
                       default=None) for n in names}
        names = sorted((n for n in names if rank[n] is not None), key=lambda n: rank[n])
    for gname in names:
        g = plan["groups"][gname]
        # A group may fix its own seeds (e.g. fresh seeds for a confirmatory test).
        use_seeds = seeds if seeds is not None else g.get("seeds", plan["seeds"])
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


PROVENANCE_KEYS = ("init_checksum", "code_hash", "code_changed_during_run")


def expected_config(raw: dict) -> dict:
    """The full config a run of this job must have saved (device excluded:
    the GPU a run landed on is not part of the experiment)."""
    from afl.config import Cfg
    want = json.loads(json.dumps(Cfg.from_dict(_deepcopy(raw)).to_dict(), default=str))
    want.get("fl", {}).pop("device", None)
    return want


# Columns every run CSV has had since the first advanced sweep. Later code
# versions added the rest of afl.engine.LOG_FIELDS; a result made by older
# code (reused with --ignore-code) may lack those, but never these.
CORE_COLUMNS = (
    "round", "accuracy", "acc_worst10", "test_loss", "reward", "cost", "latency",
    "n_selected", "n_success", "fail_ratio", "arl_online", "arl_selected", "arl_oracle",
    "cv", "cv_selection", "jain", "ece", "mode_light", "mode_standard", "mode_priority",
    "weak_selected", "weak_success", "wall_s",
)


def _csv_status(path: str, rounds: int, exact_schema: bool) -> str:
    """'ok' or why the run CSV cannot be reused: wrong columns, a truncated or
    non-numeric row, a NaN/inf value, or rounds that are not exactly 1..rounds."""
    import csv as _csv
    import math
    from afl.engine import LOG_FIELDS
    try:
        with open(path, newline="") as fh:
            reader = _csv.reader(fh)
            header = next(reader, None)
            rows = list(reader)
    except (OSError, UnicodeDecodeError, _csv.Error):
        return "unreadable csv"
    if not header:
        return "empty csv"
    if len(set(header)) != len(header):
        return "duplicate columns"
    if [c for c in header if c not in LOG_FIELDS]:
        return "unexpected columns"
    if exact_schema and header != list(LOG_FIELDS):
        return "different columns"
    if [c for c in CORE_COLUMNS if c not in header]:
        return "missing columns"
    seq = []
    for row in rows:
        if len(row) != len(header):
            return "truncated row"
        try:
            vals = [float(v) for v in row]
        except ValueError:
            return "non-numeric value"
        if not all(math.isfinite(v) for v in vals):
            return "non-finite value"
        r = vals[header.index("round")]
        if not r.is_integer():
            return "fractional round"
        seq.append(int(r))
    if seq != list(range(1, rounds + 1)):
        return "incomplete"
    return "ok"


def run_status(raw: dict, current_code: str | None = None) -> str:
    """'ok' if this job's run exists, was made with exactly this config (and,
    if current_code is given, by this code), has every round in order with
    finite values in the expected columns, and was not affected by a source
    change mid-run; 'missing'; otherwise why the existing result cannot be
    used. Never raises on a damaged result: a reason means "run it again"."""
    base = os.path.join(raw["out"], raw["tag"])
    if not os.path.exists(base + ".csv"):
        return "missing"
    if not os.path.exists(base + ".config.json"):
        return "no saved config"
    try:
        with open(base + ".config.json") as fh:
            meta = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError):
        return "unreadable config"
    if not isinstance(meta, dict) or not isinstance(meta.get("fl", {}), dict):
        return "malformed config"
    if meta.get("code_changed_during_run"):
        return "source changed while it ran"
    want = expected_config(raw)
    got = {k: v for k, v in meta.items() if k not in PROVENANCE_KEYS}
    got.get("fl", {}).pop("device", None)
    if got != want:
        return "different config"
    if current_code is not None and meta.get("code_hash") != current_code:
        return "made by different code"
    return _csv_status(base + ".csv", want["fl"]["rounds"], exact_schema=current_code is not None)


def quarantine(raw: dict, why: str) -> str | None:
    """Move an existing, unusable result out of the way before it is re-run,
    so no evidence is overwritten. Returns the quarantine folder, or None."""
    base = os.path.join(raw["out"], raw["tag"])
    files = [base + ext for ext in (".csv", ".config.json") if os.path.exists(base + ext)]
    if not files:
        return None
    dest = os.path.join(raw["out"], "quarantine", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dest, exist_ok=True)
    for f in files:
        os.replace(f, os.path.join(dest, os.path.basename(f)))
    with open(os.path.join(dest, raw["tag"] + ".why.txt"), "w") as fh:
        fh.write(why + "\n")
    return dest


def frozen_evidence(out: str) -> bool:
    """A results folder holding an original evaluation (marked by a
    PROVENANCE.md next to its groups) must never be written to."""
    return os.path.exists(os.path.join(os.path.dirname(os.path.abspath(out)), "PROVENANCE.md"))


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
    ap.add_argument("--ignore-code", action="store_true",
                    help="reuse finished runs even if the simulator code has changed since "
                         "(config and round count are still checked)")
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

    from afl.provenance import code_hash
    current = code_hash()
    stale: dict[str, int] = {}

    def check(j) -> str:
        return run_status(j["raw"], None if args.ignore_code else current)

    def done(j):
        st = check(j)
        if st not in ("ok", "missing"):
            stale[st] = stale.get(st, 0) + 1
        return st == "ok"

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
        for why, n in stale.items():
            print(f"  {n} existing result(s) would be re-run: {why}")
        return 0

    pending = [j for j in jobs if args.force or not done(j)]
    print(f"{len(jobs)} runs selected, {len(jobs) - len(pending)} already done, "
          f"{len(pending)} to run with {args.jobs} job(s)", flush=True)
    for why, n in stale.items():
        print(f"  {n} existing result(s) will be re-run: {why}", flush=True)
    if not pending:
        return 0
    frozen = sorted({j["group"] for j in pending if frozen_evidence(j["raw"]["out"])})
    if frozen:
        print(f"refusing to write into an original evaluation (PROVENANCE.md): {frozen}; "
              f"use another --out", flush=True)
        return 1
    moved = 0
    for j in pending:
        if quarantine(j["raw"], "forced re-run" if args.force else check(j)):
            moved += 1
    if moved:
        print(f"  moved {moved} earlier result(s) to quarantine/ before re-running", flush=True)

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
