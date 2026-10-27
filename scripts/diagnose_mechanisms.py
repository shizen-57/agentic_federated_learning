#!/usr/bin/env python3
"""Mechanism checks for the collapse: switch off ONE suspected link at a time.

    python scripts/diagnose_mechanisms.py --rounds 30 --jobs 4

ANALYSIS ONLY. These are not proposed fixes and not tests of any method. Each
intervention is applied in memory (monkeypatching one function) to the
s2_femnist settings on the already-inspected seeds, to see whether removing
that link prevents the lock-up. They break the policy's own likelihood where
they override its actions, so the learning in these runs is not meaningful
beyond the question asked. Results go to results/diagnostic_v1/mechanisms/.

Interventions (one per run):
  none             the original behaviour (reproduction)
  fallback_light   gate fill-in clients are run in Light instead of the
                   policy's sampled mode (is the learned mode on fill-ins the
                   link that makes the state absorbing?)
  random_fallback  fill-ins are drawn at random from the non-eligible
                   candidates instead of "highest predicted success" (is the
                   deterministic choice of the same, slowest clients the link?)
  prior_longer     the predictor keeps its cold-start prior blend until 200
                   observations instead of 20 (is the round-3 cliff the link?)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

INTERVENTIONS = ("none", "fallback_light", "random_fallback", "prior_longer")
CONDS = ("arl3_full", "c2_rl_fedprox")
SEEDS = (110, 113)


def _job(cond: str, seed: int, rounds: int, iv: str) -> dict:
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from afl.config import Cfg
    from afl.engine import Experiment
    from afl.run import _load
    from sweep import build_jobs
    import afl.policies.rl as rl
    import afl.predictor as pr

    src = _load(os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    raw = [j["raw"] for j in build_jobs(src, ["s2_femnist"], [seed], "/tmp/afl_mech",
                                        [f"fl.rounds={rounds}"]) if j["cond"] == cond][0]

    if iv == "random_fallback":
        rng = np.random.default_rng(seed + 9999)          # own stream; touches no run RNG
        def gate(cfg, state, cands, K):
            elig = [i for i in cands if state["eligible"][i]]
            if len(elig) >= K or not cfg.policy.strict_gate:
                return elig, []
            rest = [i for i in cands if not state["eligible"][i]]
            return elig, [int(i) for i in rng.permutation(rest)[:K - len(elig)]]
        rl.gate = gate
    if iv == "prior_longer":
        def predict(self, obs, hist, mode):
            z = float(self.features(obs, hist, mode) @ self.w + self.b)
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            return 0.5 * p + 0.5 * 0.7 if self.seen < 200 else p
        pr.SuccessPredictor.predict = predict

    exp = Experiment(Cfg.from_dict(raw))
    if iv == "fallback_light":
        act = exp.policy.act
        def act_light(state, t):
            sel, assign = act(state, t)
            for i in exp.policy.gate_fill:
                assign[i] = "light"
            return sel, assign
        exp.policy.act = act_light
    rows = exp.run(verbose=False)
    tail = rows[-10:]
    longest = cur = 0
    for r in rows:
        cur = cur + 1 if r["n_selected"] and not r["n_success"] else 0
        longest = max(longest, cur)
    return {"cond": cond, "seed": seed, "intervention": iv, "rounds": rounds,
            "successes": sum(r["n_success"] for r in rows),
            "invites": sum(r["n_selected"] for r in rows),
            "succ_last10": sum(r["n_success"] for r in tail),
            "fallback_share": sum(r["gate_filled"] for r in rows) / max(1, sum(r["n_selected"] for r in rows)),
            "longest_zero_streak": longest,
            "light/std/prio": "/".join(str(sum(r[f"mode_{m}"] for r in rows))
                                       for m in ("light", "standard", "priority")),
            "accuracy_fixed_last": rows[-1]["accuracy_fixed"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "diagnostic_v1", "mechanisms"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    work = [(c, s, args.rounds, iv) for iv in INTERVENTIONS for c in CONDS for s in SEEDS]
    with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn")) as pool:
        res = list(pool.map(_job, *zip(*work)))
    path = os.path.join(args.out, f"mechanisms_r{args.rounds}.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(res[0]))
        wr.writeheader()
        wr.writerows(res)
    print(f"ANALYSIS ONLY -- interventions on inspected seeds, {args.rounds} rounds ({path})\n")
    print("| cond | seed | intervention | successes/invites | last 10 rounds | fallback share "
          "| longest zero streak | modes L/S/P | acc (fixed set) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(res, key=lambda r: (r["cond"], r["seed"], INTERVENTIONS.index(r["intervention"]))):
        print(f"| {r['cond']} | {r['seed']} | {r['intervention']} | {r['successes']}/{r['invites']} | "
              f"{r['succ_last10']} | {r['fallback_share']:.2f} | {r['longest_zero_streak']} | "
              f"{r['light/std/prio']} | {r['accuracy_fixed_last']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
