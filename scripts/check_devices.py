#!/usr/bin/env python3
"""Short real-data runs on this machine, to compare devices (fourth
external review, finding 7): CPU here, the RTX 3070 on the server.

    python scripts/check_devices.py --out results/devicecheck/cpu.json --set fl.device=cpu
    python scripts/check_devices.py --out runs/devicecheck/gpu.json --set fl.device=auto
    python scripts/check_devices.py --compare results/devicecheck/cpu.json runs/devicecheck/gpu.json

A fixed set: FEMNIST and CIFAR-10 x always-Light and the tuned method
(configs/rl2_tuned.json) x the GPU-smoke seed 900, 30 rounds, persistent
split, development pool. Bit-identical results are not expected (GPU
kernels round differently and the learner's choices then diverge); the
comparison flags material differences in accuracy, successes and cost.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

ROUNDS, SEED = 30, 900
GROUPS = {"femnist": "s2_femnist", "cifar": "s2_cifar"}
# material = worth investigating, not a failure by itself
LIMITS = {"acc10": 0.03, "successes": 0.15, "cost": 0.15}


def plan() -> dict:
    import rl2lib
    with open(os.path.join(ROOT, "configs", "rl2_tuned.json")) as fh:
        tuned = json.load(fh)
    groups = {}
    for d, g in GROUPS.items():
        src = rl2lib.source_group("configs/sweep_advanced.yaml", g)
        entry = rl2lib.group_entry(src, {"data.split": "persistent"}, ROUNDS, [tuned["base"]], [SEED])
        groups[f"{d}_tuned"] = {**entry, "override": rl2lib.resolve(entry, tuned["overrides"])}
        groups[f"{d}_light"] = rl2lib.group_entry(src, {"data.split": "persistent"}, ROUNDS, ["rl_light"], [SEED])
    return {"seeds": [SEED], "sets": {}, "groups": groups}


def run(out: str, sets: list[str]) -> int:
    import torch
    import rl2lib
    from afl.config import Cfg
    from afl.engine import Experiment
    from afl.provenance import code_hash
    from sweep import build_jobs
    if rl2lib.non_operational(sets):
        raise SystemExit("--set may only be operational")
    res = {"code_hash": code_hash(), "torch": torch.__version__, "python": platform.python_version(),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "runs": {}}
    for j in build_jobs(plan(), None, None, "/tmp/afl_devicecheck", ["diag.enabled=false", *sets]):
        t0 = time.time()
        rows = Experiment(Cfg.from_dict(j["raw"])).run(verbose=False)
        res["runs"][j["group"]] = {
            "acc10": st.mean(r["accuracy"] for r in rows[-10:]),
            "w10": st.mean(r["acc_client_worst10"] for r in rows[-10:]),
            "successes": int(sum(r["n_success"] for r in rows)),
            "invites": int(sum(r["n_selected"] for r in rows)),
            "cost": float(sum(r["cost"] for r in rows)),
            "heavy_share": sum(r["mode_standard"] + r["mode_priority"] for r in rows)
            / max(sum(r["n_selected"] for r in rows), 1),
            "seconds": round(time.time() - t0, 1)}
        print(f"{j['group']:15s} " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                              for k, v in res["runs"][j["group"]].items()), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"wrote {out}")
    return 0


def compare(a_path: str, b_path: str) -> int:
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    print(f"A: {a['device']} torch {a['torch']} code {a['code_hash']}\n"
          f"B: {b['device']} torch {b['torch']} code {b['code_hash']}")
    if a["code_hash"] != b["code_hash"]:
        print("note: different simulator code -- compare only like with like")
    print("\n| run | acc10 A / B | successes A / B | cost A / B | heavy share A / B | material |")
    print("|---|---|---|---|---|---|")
    flagged = 0
    for k in sorted(set(a["runs"]) & set(b["runs"])):
        x, y = a["runs"][k], b["runs"][k]
        why = []
        if abs(x["acc10"] - y["acc10"]) > LIMITS["acc10"]:
            why.append("accuracy")
        for m in ("successes", "cost"):
            if abs(x[m] - y[m]) > LIMITS[m] * max(abs(x[m]), 1e-9):
                why.append(m)
        flagged += bool(why)
        print(f"| {k} | {x['acc10']:.3f} / {y['acc10']:.3f} | {x['successes']} / {y['successes']} | "
              f"{x['cost']:.0f} / {y['cost']:.0f} | {x['heavy_share']:.2f} / {y['heavy_share']:.2f} | "
              f"{', '.join(why) or 'no'} |")
    print(f"\n{flagged} run(s) with a material difference (limits: accuracy {LIMITS['acc10']}, "
          f"successes and cost {LIMITS['cost']:.0%})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        raise SystemExit("give --out or --compare")
    return run(args.out, args.set)


if __name__ == "__main__":
    raise SystemExit(main())
