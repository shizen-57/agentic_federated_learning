#!/usr/bin/env python3
"""Ground truth of the simulator, without any training (docs/REVIEW_5.md 2.1).

For every client group (slow / middle / fast devices, weak clients) and every
mode: how often would a job succeed if an online client were given it, and
why does it fail? Uses the round's fixed random numbers (ClientEnv.outcome),
with the battery as it is when nobody is invited (full) -- the most
favourable energy state. No model is trained and no result is evaluated, so
this reads nothing a policy could learn from.

    python scripts/sim_truth.py --dataset femnist --seeds 280 281 282 283 284
    python scripts/sim_truth.py --dataset femnist --set system.drop_model=exposure
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

GROUPS = ("slow", "middle", "fast", "weak")
REASONS = ("energy", "dropped", "deadline", "rejected")


def raw_config(dataset: str, seed: int, sets: list[str]) -> dict:
    import yaml
    from sweep import build_jobs
    with open(os.path.join(ROOT, "configs", "sweep_advanced.yaml")) as fh:
        g = dict(yaml.safe_load(fh)["groups"][f"s2_{dataset}"])
    g["set"] = {**(g.get("set") or {}), "data.split": "persistent"}
    g["conditions"] = ["rl_light_mimic"]
    g.pop("seeds", None)
    plan = {"groups": {"g": g}, "seeds": [seed], "sets": {}}
    return build_jobs(plan, None, [seed], "unused", list(sets))[0]["raw"]


def truth(dataset: str, seeds: list[int], sets: list[str], rounds: int = 200) -> dict:
    from afl.config import Cfg
    from afl.data import build as build_data
    from afl.sim import ClientEnv
    agg: dict = {}
    per: dict = {g: {"n": 0, "online": [], "p_light": [], "cost_light": [], "best": []} for g in GROUPS}
    modes = None
    for seed in seeds:
        cfg = Cfg.from_dict(raw_config(dataset, seed, sets))
        modes = list(cfg.system.modes)
        d = build_data(cfg)
        env = ClientEnv(cfg, d["n_i"], cfg.seed, sketch=d["sketch"])
        grp = np.where(env.weak, "weak", np.array(GROUPS[:3])[env.tier])
        ok = {m: np.zeros(env.N) for m in modes}
        anyok = np.zeros(env.N)
        online = np.zeros(env.N)
        for t in range(1, rounds + 1):
            env.new_round(t)                       # nobody spends: the battery stays full
            for i in np.where(env._draw["available"])[0]:
                online[i] += 1
                hit = 0
                for m in modes:
                    o = env.outcome(int(i), m)
                    a = agg.setdefault((grp[i], m), {"n": 0, "ok": 0, **{r: 0 for r in REASONS}})
                    a["n"] += 1
                    a[o["reason"]] += 1
                    ok[m][i] += o["success"]
                    hit |= o["success"]
                anyok[i] += hit
        for g in GROUPS:
            sel = grp == g
            p = per[g]
            p["n"] += int(sel.sum())
            p["online"] += list(online[sel] / rounds)
            p["p_light"] += list(ok[modes[0]][sel] / np.maximum(online[sel], 1))
            p["best"] += list(anyok[sel] / np.maximum(online[sel], 1))
            p["cost_light"] += [env.cost(int(i), modes[0]) for i in np.where(sel)[0]]
    out = {"dataset": dataset, "seeds": seeds, "set": sets, "rounds": rounds, "groups": {}}
    for g in GROUPS:
        p = per[g]
        pl = np.array(p["p_light"]) if p["p_light"] else np.zeros(1)
        out["groups"][g] = {
            "share": p["n"] / max(sum(per[x]["n"] for x in GROUPS), 1),
            "online": float(np.mean(p["online"])) if p["online"] else 0.0,
            "modes": {m: {k: agg[(g, m)][k] / agg[(g, m)]["n"] for k in ("ok", *REASONS)}
                      for m in modes if (g, m) in agg},
            "any_mode": float(np.mean(p["best"])) if p["best"] else 0.0,
            "light_below_0.2": float(np.mean(pl < 0.2)), "light_below_0.5": float(np.mean(pl < 0.5)),
            "light_median": float(np.median(pl)),
            "cost_light": float(np.mean(p["cost_light"])) if p["cost_light"] else 0.0,
        }
    return out


def render(res: dict) -> str:
    modes = list(next(iter(res["groups"].values()))["modes"])
    L = [f"Simulator ground truth: {res['dataset']}, seeds {res['seeds']}, {res['rounds']} rounds"
         + (f", with {res['set']}" if res["set"] else ""), "",
         "| group | share | online | " + " | ".join(f"{m} ok" for m in modes)
         + " | any mode | Light: drop / deadline / energy | clients with Light < 0.5 | Light cost |",
         "|---|---|---|" + "---|" * (len(modes) + 4)]
    for g, v in res["groups"].items():
        lm = v["modes"][modes[0]]
        L.append(f"| {g} | {v['share']:.0%} | {v['online']:.2f} | "
                 + " | ".join(f"{v['modes'][m]['ok']:.2f}" for m in modes)
                 + f" | {v['any_mode']:.2f} | {lm['dropped']:.2f} / {lm['deadline']:.2f} / {lm['energy']:.2f}"
                 f" | {v['light_below_0.5']:.0%} | {v['cost_light']:.2f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="femnist", choices=("femnist", "cifar"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[280, 281, 282, 283, 284])
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. system.drop_model=exposure")
    ap.add_argument("--json", help="also write the numbers here")
    args = ap.parse_args()
    res = truth(args.dataset, args.seeds, args.set, args.rounds)
    print(render(res))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
