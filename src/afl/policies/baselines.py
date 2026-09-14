"""External baselines of Table 3.5.

All four select K clients per round and assign a mode, under the same
aggregation rule and simulation settings as the proposed system, so any
difference in the reported metrics is attributable to the decision rule alone.
"""
from __future__ import annotations

import numpy as np

from .base import Policy


def _default_mode(cfg) -> str:
    return "standard"


class RandomPolicy(Policy):
    """Uniform random sampling: the lower-bound reference."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg.seed + 7)

    def act(self, state, t):
        K = min(self.cfg.fl.clients_per_round, self.cfg.data.num_clients)
        sel = list(self.rng.choice(self.cfg.data.num_clients, size=K, replace=False))
        return sel, {int(i): _default_mode(self.cfg) for i in sel}


class FedCSPolicy(Policy):
    """Deadline-feasibility greedy selection.

    Picks the K clients with the highest predicted success probability under the
    cheapest mode -- feasibility gating with no fairness or value term, which is
    exactly the competitor the thesis wants to separate itself from.
    """

    def act(self, state, t):
        K = min(self.cfg.fl.clients_per_round, self.cfg.data.num_clients)
        p = np.array([state["p_succ"][i]["light"] for i in range(self.cfg.data.num_clients)])
        sel = list(np.argsort(-p)[:K])
        return sel, {int(i): "light" for i in sel}


class PowerOfChoicePolicy(Policy):
    """Biased selection toward clients with the highest local loss.

    Purely utility-driven and fairness-blind: it should reach good accuracy fast
    and produce a visibly worse CV_t than the proposed policy.
    """

    def __init__(self, cfg, d: int = 3):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg.seed + 11)
        self.d = d

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        cand = self.rng.choice(N, size=min(self.d * K, N), replace=False)
        loss = np.array([state["local_loss"][i] for i in cand])
        sel = [int(cand[j]) for j in np.argsort(-loss)[:K]]
        return sel, {i: _default_mode(self.cfg) for i in sel}


class OortPolicy(Policy):
    """Statistical utility x system utility, with the usual exploration bonus.

    The strongest published utility+system-aware competitor. Utility is
    loss-weighted by sample count and penalised when the client's expected time
    overruns the deadline.
    """

    def __init__(self, cfg, eps: float = 0.2, alpha: float = 2.0):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg.seed + 13)
        self.eps = eps
        self.alpha = alpha

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        n_i = state["n_i"]
        util = np.zeros(N)
        for i in range(N):
            stat = n_i[i] * state["local_loss"][i]
            t_exp = max(state["t_expected"][i]["standard"], 1e-6)
            pen = 1.0 if t_exp <= self.cfg.system.deadline else (
                self.cfg.system.deadline / t_exp) ** self.alpha
            util[i] = stat * pen

        n_explore = max(1, int(self.eps * K))
        order = list(np.argsort(-util))
        sel = [int(i) for i in order[:K - n_explore]]
        rest = [i for i in range(N) if i not in sel]
        sel += [int(i) for i in self.rng.choice(rest, size=min(n_explore, len(rest)),
                                                replace=False)]
        return sel, {i: _default_mode(self.cfg) for i in sel}
