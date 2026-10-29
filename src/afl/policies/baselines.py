"""External baselines of Table 3.5.

All four select K clients per round and assign a mode, under the same
aggregation rule and simulation settings as the proposed system, so any
difference in the reported metrics is attributable to the decision rule alone.
"""
from __future__ import annotations

import numpy as np

from .. import metrics as M
from .base import Policy, best_work_mode, est_time, gate, pool


def _default_mode(cfg) -> str:
    return "standard"


class FedCERPolicy(Policy):
    """Submodular-diverse selection + greedy per-client epoch configuration.

    A simplified but faithful stand-in for FedCER [6]/FedCG [27] -- the
    closest published joint selection-and-configuration competitor named in
    Table 3.5. It has no learning and no fairness or success-prediction
    mechanism, so any advantage the proposed RL system shows over it is
    attributable to those two things specifically, not to "does joint
    selection+configuration help at all" (which this baseline already does).

    Selection: greedy submodular maximization over the same class-coverage
    function as Sec. 3.11 (Eq. 3.25) -- iteratively add whichever remaining
    client gives the largest marginal coverage gain, the standard DivFL/FedCER
    construction, in place of the linear-programming relaxation those papers
    use for the same greedy-submodular selection problem.
    Configuration: assign each selected client the richest mode (Priority >
    Standard > Light) whose predicted time fits its deadline given observed
    speed/bandwidth -- a greedy proxy for FedCER's per-client epoch LP that
    needs no gradient information, matching what a real deployed heuristic
    could compute from the same descriptors the proposed system observes.
    """

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        sketch = state["sketch"]

        remaining = pool(state, N)
        selected: list[int] = []
        for _ in range(min(K, len(remaining))):
            vals = M.per_client_representation_value(sketch, selected)
            best = max(remaining, key=lambda i: vals[i])
            selected.append(best)
            remaining.remove(best)

        assign = {}
        for i in selected:
            obs = state["obs"][i]
            for mode in ("priority", "standard", "light"):
                epochs = self.cfg.system.local_epochs[mode]
                frac = self.cfg.system.upload_frac[mode]
                if self.cfg.policy.shared_time_estimate:
                    t_est = est_time(self.cfg, state, i, mode)
                else:
                    # Original formula: leaves out the client's workload.
                    t_est = (epochs / max(obs["speed_hat"], 1e-3)
                             + 0.25 * frac / max(obs["bw_hat"], 1e-3))
                if t_est <= self.cfg.system.deadline:
                    assign[i] = mode
                    break
            else:
                assign[i] = "light"
        return selected, assign


class RoundRobinPolicy(Policy):
    """Deterministic rotation, fixed Standard mode: Condition 4 of Table 3.4.

    No utility signal, no learning, no adaptivity in either selection or
    configuration -- as close to "no intelligence at all" as a working
    scheduler can be. This is deliberately kept distinct from every Table 3.5
    external baseline (all of which use some utility or system signal), so
    that Condition 4 isolates the benefit of RL-based orchestration itself
    (Table 3.4's stated purpose) rather than duplicating one of the Table 3.5
    comparisons under a different name.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._cursor = 0

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        c = state.get("candidates")
        if c is None:
            sel = [(self._cursor + j) % N for j in range(K)]
            self._cursor = (self._cursor + K) % N
        else:
            # Continue the rotation, skipping clients that did not check in.
            cs, sel, j = set(c), [], 0
            while len(sel) < min(K, len(cs)) and j < N:
                if (self._cursor + j) % N in cs:
                    sel.append((self._cursor + j) % N)
                j += 1
            self._cursor = (self._cursor + j) % N
        return sel, {i: _default_mode(self.cfg) for i in sel}


class RandomPolicy(Policy):
    """Uniform random sampling: the lower-bound reference."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg.seed + 7)

    def act(self, state, t):
        K = min(self.cfg.fl.clients_per_round, self.cfg.data.num_clients)
        c = state.get("candidates")
        if c is None:
            sel = [int(i) for i in self.rng.choice(self.cfg.data.num_clients, size=K, replace=False)]
        else:
            sel = [int(i) for i in self.rng.choice(np.array(c, dtype=int), size=min(K, len(c)),
                                                   replace=False)] if c else []
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
        c = state.get("candidates")
        if c is None:
            sel = [int(i) for i in np.argsort(-p)[:K]]
        else:
            sel = [int(i) for i in sorted(c, key=lambda i: -p[i])[:K]]
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
        c = state.get("candidates")
        if c is None:
            cand = self.rng.choice(N, size=min(self.d * K, N), replace=False)
        elif not c:
            return [], {}
        else:
            cand = self.rng.choice(np.array(c, dtype=int), size=min(self.d * K, len(c)), replace=False)
        loss = np.array([state["local_loss"][i] for i in cand])
        sel = [int(cand[j]) for j in np.argsort(-loss)[:K]]
        return sel, {i: _default_mode(self.cfg) for i in sel}


class OortPolicy(Policy):
    """Statistical utility x system utility, with the usual exploration bonus.

    The strongest published utility+system-aware competitor. Utility is
    loss-weighted by sample count and penalised when the client's expected time
    overruns the deadline.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg.seed + 13)
        self.eps = cfg.policy.oort_eps
        self.alpha = cfg.policy.oort_alpha

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        n_i = state["n_i"]
        util = np.zeros(N)
        for i in range(N):
            stat = n_i[i] * state["local_loss"][i]
            # The original runs used this round's true time (a leak: a real
            # server cannot know it before the round); the fix uses the
            # estimate from observed speed and bandwidth.
            times = state["t_est"] if self.cfg.policy.oort_estimated_time else state["t_expected"]
            t_exp = max(times[i]["standard"], 1e-6)
            pen = 1.0 if t_exp <= self.cfg.system.deadline else (
                self.cfg.system.deadline / t_exp) ** self.alpha
            util[i] = stat * pen

        c = state.get("candidates")
        if c is not None:
            cs = set(c)
            util = np.where([i in cs for i in range(N)], util, -np.inf)
            K = min(K, len(cs))
            if K == 0:
                return [], {}
        n_explore = max(1, int(self.eps * K))
        order = list(np.argsort(-util))
        sel = [int(i) for i in order[:K - n_explore]]
        rest = [i for i in (range(N) if c is None else c) if i not in sel]
        sel += [int(i) for i in self.rng.choice(rest, size=min(n_explore, len(rest)),
                                                replace=False)]
        return sel, {i: _default_mode(self.cfg) for i in sel}


class FedCGPolicy(Policy):
    """FedCG (Li et al., "Adaptive Control of Client Selection and Gradient
    Compression for Efficient Federated Learning", arXiv 2212.09483).

    Implements Alg. 2 of the paper, mapped onto this simulator:

    - Selection (Sec. V-B): greedy maximisation of the facility-location
      objective, i.e. choose K clients minimising
      Q(M) = sum_{n in N} min_{i in M} ||g_n - g_i||  (Eq. 14), where g_n is
      the client's latest known update ("historical gradient information").
      The engine supplies these as 1024-d count-sketch projections, which
      approximately preserve the distances.
    - Compression ratio (Sec. V-C, Eq. 17): each selected client gets the
      largest ratio theta in (0, 1] whose estimated round time
      H*T_cmp + theta*R/C still fits the round deadline -- the one-shot
      solution of the paper's LP, since its objective sum (1-theta_n) H^2 G^2
      only decreases as theta grows. Estimates use the observed (noisy) speed
      and bandwidth only.
    - Alg. 2 loop: K iterations; each picks a diverse set from the remaining
      candidates, keeps it if its total compression ratio is the best so far,
      then removes that set's most-compressed client from the candidates.
    - Local training uses a fixed number of epochs (the paper's fixed H); the
      engine applies top-k sparsification with error feedback at ratio theta
      and aggregates with an equal-weight mean (aggregator "fedcg").

    Like the paper, it does not model availability or failure: it assumes the
    selected clients report back.
    """

    THETA_MIN = 0.01

    def __init__(self, cfg):
        super().__init__(cfg)
        self.upload_frac: dict[int, float] = {}

    def _theta(self, state, i):
        sysc = self.cfg.system
        obs = state["obs"][i]
        t_cmp = est_time(self.cfg, state, i, "standard", frac=0.0)   # compute part only
        budget = sysc.deadline * self.cfg.policy.fedcg_margin
        theta = (budget - t_cmp) * max(obs["bw_hat"], 1e-3) / 0.25
        return float(np.clip(theta, self.THETA_MIN, 1.0))

    @staticmethod
    def _greedy(dist, cands, K):
        sel: list[int] = []
        cur = np.full(dist.shape[0], np.inf)
        for _ in range(K):
            pool = [c for c in cands if c not in sel]
            vals = np.minimum(cur[:, None], dist[:, pool]).sum(axis=0)
            best = pool[int(np.argmin(vals))]
            sel.append(best)
            cur = np.minimum(cur, dist[:, best])
        return sel

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        g = state["update_sketch"]
        sq = (g ** 2).sum(axis=1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * g @ g.T, 0.0))
        cands = pool(state, N)
        K = min(K, len(cands))
        if K == 0:
            return [], {}
        best_sel, best_theta, best_sum = None, None, -1.0
        for _ in range(K):
            if len(cands) < K:
                break
            sel = self._greedy(dist, cands, K)
            theta = {i: self._theta(state, i) for i in sel}
            if sum(theta.values()) > best_sum:
                best_sel, best_theta, best_sum = sel, theta, sum(theta.values())
            cands.remove(min(sel, key=lambda i: theta[i]))
        self.upload_frac = dict(best_theta)
        return list(best_sel), {i: "standard" for i in best_sel}


class GreedyCoveragePolicy(Policy):
    """A simple, non-learning rule for the whole joint decision.

    Uses the same information as the RL policy: each client's predicted
    success per mode, its class shares, and how under-covered each class has
    been lately. Each client gets the expected-work mode (base.best_work_mode);
    then K clients are picked greedily, each time the eligible client with the
    largest expected coverage gain,
        P(success | its mode) x sum_k max(0, c_ik - covered_k) x (u_k + 0.1),
    covered_k the best share of class k among clients already picked and u_k
    the class's recent coverage deficit (0.1 keeps covered classes from
    counting zero). The comparison "does learning beat a rule?" is RL vs this.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg.seed + 31)

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        sk = state["sketch"]
        w = state["cov_deficit"] + 0.1
        mode = {i: best_work_mode(self.cfg, state["p_succ"][i]) for i in range(N)}
        q = np.array([state["p_succ"][i][mode[i]] for i in range(N)])
        cands = pool(state, N)
        elig, fill = gate(self.cfg, state, cands, K)
        self.gate_filled = len(fill)
        self.gate_fill = list(fill)
        if self.cfg.policy.strict_gate:
            ok = elig
        else:
            ok = elig if len(elig) >= K else cands
        tie = self.rng.random(N) * 1e-9                  # random order among exact ties
        cur = np.zeros(sk.shape[1])
        sel: list[int] = []
        for _ in range(min(K, len(ok))):
            gain = q * (np.maximum(sk - cur, 0.0) @ w) + tie
            best = max((i for i in ok if i not in sel), key=lambda i: gain[i])
            sel.append(best)
            cur = np.maximum(cur, sk[best])
        sel += list(fill)
        return sel, {i: mode[i] for i in sel}
