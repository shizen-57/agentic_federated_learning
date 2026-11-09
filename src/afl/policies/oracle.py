"""Oracle schedulers -- analysis only, never a method (docs/REVIEW_5.md, F4).

They read state["truth_ok"], the true outcome of every (client, mode) this
round, which the engine provides to these policies alone. No real server has
it. They answer one question: how much could ANY scheduler gain in this
simulator if it never wasted a job? If an oracle is no better than a simple
control, the test bed has no room for a real method to show a gain there.

    oracle_light   Light only; K random clients among those that will succeed.
    oracle_work    every client gets the heaviest job it will finish; K random
                   clients among those that can finish something.
    oracle_fair    Light only; among the clients that will succeed, the K with
                   the fewest contributions so far (the fairness ceiling).
    oracle_cov     Light only; among the clients that will succeed, K picked
                   greedily by the class coverage they add (weighted by how
                   under-covered each class has been) -- the ceiling of
                   contribution-aware selection.
"""
from __future__ import annotations

import numpy as np

from .base import Policy, pool

KINDS = ("oracle_light", "oracle_work", "oracle_fair", "oracle_cov")


class OraclePolicy(Policy):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.name not in KINDS:
            raise ValueError(f"unknown oracle {cfg.policy.name}")
        if cfg.policy.mode_rule != "own":
            raise ValueError("an oracle chooses its own modes (policy.mode_rule=own)")
        self.kind = cfg.policy.name
        self.N = cfg.data.num_clients
        self.modes = list(cfg.system.modes)            # light -> heavy
        self.rng = np.random.default_rng(cfg.seed + 8100)
        self.count = np.zeros(self.N)                  # successful contributions so far

    def act(self, state, t):
        K = min(self.cfg.fl.clients_per_round, self.N)
        truth = state["truth_ok"]
        cands = pool(state, self.N)
        if self.kind == "oracle_work":
            best = {i: next((m for m in reversed(self.modes) if truth[i][m]), None) for i in cands}
            ok = [i for i in cands if best[i] is not None]
        else:
            best = {i: self.modes[0] for i in cands}
            ok = [i for i in cands if truth[i][self.modes[0]]]
        if not ok:
            return [], {}
        order = [ok[k] for k in self.rng.permutation(len(ok))]     # random order, random ties
        if self.kind == "oracle_fair":
            order = sorted(order, key=lambda i: self.count[i])      # stable: ties stay random
            sel = order[:K]
        elif self.kind == "oracle_cov":
            sk, w = state["sketch"], state["cov_deficit"] + 0.1
            sel, covered, left = [], np.zeros(sk.shape[1]), list(order)
            while left and len(sel) < K:
                gain = [float(np.maximum(sk[i] - covered, 0.0) @ w) for i in left]
                i = left.pop(int(np.argmax(gain)))
                sel.append(i)
                covered = np.maximum(covered, sk[i])
        else:
            sel = order[:K]
        return [int(i) for i in sel], {int(i): best[i] for i in sel}

    def observe_outcome(self, state, action, reward, info):
        for i, ok in (info or {}).get("success", {}).items():
            self.count[i] += ok
