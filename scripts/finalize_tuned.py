#!/usr/bin/env python3
"""Carry the development verdicts into the tuned settings (gap review A3, A4).

    python scripts/finalize_tuned.py cost --spec configs/rl2_cost_check.yaml --out runs/rl2_cost_check
    python scripts/finalize_tuned.py arl  --spec configs/rl2_arl_screen.yaml --out runs/rl2_arl_screen

The order is fixed (docs/RL2_ARL_SCREEN.md, "Order"):
1. the cost check runs on M7's settings (configs/rl2_tuned.json); `cost`
   writes configs/rl2_tuned_cost.json = those settings plus the adopted
   cost variant (or unchanged);
2. the ARL screen runs on configs/rl2_tuned_cost.json; `arl` writes
   configs/rl2_tuned_final.json = those settings plus the ARL version that
   "adds value" (the larger mean equal-spend score gain if both do), or
   unchanged;
3. the sensitivity screen and screen C use configs/rl2_tuned_final.json.

Each step first re-checks the screen's evidence and re-applies its rule
(scripts/screen_rl2.py report), so only a complete, valid screen can change
the settings, and records the verdict's SHA-256 in the new file.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import yaml  # noqa: E402

import rl2lib  # noqa: E402

STEPS = {"cost": ("configs/rl2_tuned.json", "configs/rl2_tuned_cost.json"),
         "arl": ("configs/rl2_tuned_cost.json", "configs/rl2_tuned_final.json")}


def adopted(step: str, verdict: dict) -> tuple[str | None, dict]:
    """(name of the adopted arm or None, its override)."""
    if step == "cost":
        return verdict["adopt"], verdict["override"] if verdict["adopt"] else {}
    wins = {a: v for a, v in verdict["arms"].items() if v["verdict"] == "adds value"}
    if not wins:
        return None, {}
    best = max(sorted(wins), key=lambda a: wins[a]["mean_score_gain"])
    return best, wins[best]["override"]


def finalize(step: str, spec_path: str, verdict: dict, root: str = ROOT) -> dict:
    src, dst = STEPS[step]
    with open(spec_path) as fh:
        spec = yaml.safe_load(fh)
    if spec["tuned"] != src:
        raise SystemExit(f"step {step} needs a screen run on {src}; {spec_path} uses {spec['tuned']}")
    if verdict.get("kind") != step:
        raise SystemExit(f"the verdict is a {verdict.get('kind')} verdict, not {step}")
    if verdict.get("tuned_sha256") != rl2lib._sha(os.path.join(root, src)):
        raise SystemExit(f"{src} changed after the screen ran")
    if verdict.get("spec_sha256") != rl2lib._sha(spec_path):
        raise SystemExit(f"{spec_path} changed after the screen ran")
    with open(os.path.join(root, src)) as fh:
        tuned = json.load(fh)
    name, override = adopted(step, verdict)
    out = {**tuned, "overrides": {**tuned["overrides"], **override}}
    out["provenance"] = list(tuned.get("provenance", [])) + [{
        "step": step, "from": src, "from_sha256": rl2lib._sha(os.path.join(root, src)),
        "spec": os.path.relpath(spec_path, root), "verdict_sha256": verdict["_sha256"],
        "adopted": name, "override": override}]
    with open(os.path.join(root, dst), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=sorted(STEPS))
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True, help="the screen's run folder")
    args = ap.parse_args()
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "screen_rl2.py"), "report",
                         "--spec", args.spec, "--out", args.out]).returncode
    if rc:
        print(f"the screen's evidence does not check out (exit {rc}); nothing changed")
        return rc
    vpath = os.path.join(args.out, "VERDICT.json")
    with open(vpath) as fh:
        verdict = json.load(fh)
    verdict["_sha256"] = rl2lib._sha(vpath)
    out = finalize(args.step, args.spec, verdict)
    last = out["provenance"][-1]
    print(f"wrote {STEPS[args.step][1]}: {'adopted ' + last['adopted'] if last['adopted'] else 'unchanged'} "
          f"{last['override'] or ''}\ncommit it before the next step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
