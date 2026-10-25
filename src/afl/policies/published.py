"""Published client-selection algorithms, implemented from the papers.

FAVOR  Wang, Kaplan, Niu, Li. "Optimizing Federated Learning on Non-IID Data
       with Reinforcement Learning." IEEE INFOCOM 2020.
DELTA  Wang, Guo, Lin, Tang. "DELTA: Diverse Client Sampling for Fasting
       Federated Learning." NeurIPS 2023 (practical version, FedPracDELTA).
REFL   Abdelmoniem, Sahu, Canini, Fahmy. "REFL: Resource-Efficient Federated
       Learning." EuroSys 2023.

None of the three chooses a participation mode, so each gives every selected
client Standard (as Random and Oort do) unless policy.mode_rule says otherwise.
Where a paper leaves a setting open, or this simulator forces a change, the
class docstring says what was chosen.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .base import Policy


class FAVORPolicy(Policy):
    """FAVOR: Double-DQN client selection on PCA-compressed model weights.

    As in the paper (Sec. III):
    - State: the global model and every client's latest local model, each
      reduced by PCA to `favor_pca` dimensions. The PCA is fitted once, on the
      local models of the initial round in which every client trains one epoch
      (Step 2; the engine runs it, uncharged), and reused afterwards.
    - Q-network: two-layer MLP, 512 hidden units, one Q-value per client.
    - Action: the K clients with the top Q-values.
    - Reward: r_t = 64^(acc_t - Omega) - 1, Omega the target accuracy
      (paper: 99% MNIST, 55% CIFAR-10; FEMNIST is not in the paper, 80% here).
    - Learning: Double DQN, target network frozen between syncs.

    Changes forced by this setting, applied to every learning method alike:
    the paper trains the agent over ~200 full FL runs before using it; here,
    like the proposed RL policy, it learns online within one run. So each
    round's K picks are stored as K transitions sharing the round's reward,
    exploration is epsilon-greedy (1.0 -> 0.05 over the first 30% of rounds),
    and the network takes a few minibatch steps per round. The paper's
    discount factor, replay size and sync interval are not stated; gamma 0.9,
    batch 32, 4 steps per round and a sync every 10 steps are used.
    PCA coordinates are divided by the spread of the initial ones so the
    network's inputs are O(1).
    """

    needs_probe = True
    TARGETS = {"mnist": 0.99, "cifar10": 0.55, "femnist": 0.80, "synthetic": 0.90}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.N = cfg.data.num_clients
        self.q = max(1, min(cfg.policy.favor_pca, self.N - 1))
        self.target = cfg.policy.favor_target or self.TARGETS.get(cfg.data.dataset, 0.9)
        self.rng = np.random.default_rng(cfg.seed + 19)
        h = cfg.policy.favor_hidden
        # Own seed, without touching the global generator the FL model is
        # initialised from.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 19)
            self.net = nn.Sequential(nn.Linear((self.N + 1) * self.q, h), nn.ReLU(),
                                     nn.Linear(h, self.N))
        self.tgt = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.buffer: list[tuple] = []       # (state, action, reward, next_state)
        self.pending = None                 # (state, actions, reward) awaiting s'
        self.n_steps = 0
        self.V = None

    # ------------------------------------------------------------- state

    def init_probe(self, updates: dict, global_flat: torch.Tensor) -> None:
        dev = global_flat.device
        self.dev = dev
        W = torch.stack([global_flat + updates.get(i, torch.zeros_like(global_flat))
                         for i in range(self.N)])
        self.mu = W.mean(dim=0)
        X = W - self.mu
        _, _, V = torch.pca_lowrank(X, q=self.q, center=False)
        self.V = V[:, :self.q]
        proj = X @ self.V
        self.scale = float(proj.std()) + 1e-8
        self.client = proj / self.scale
        self.net.to(dev)
        self.tgt.to(dev)

    def _proj(self, w: torch.Tensor) -> torch.Tensor:
        return ((w - self.mu) @ self.V) / self.scale

    def _state_vec(self, global_flat: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._proj(global_flat), self.client.reshape(-1)]).detach()

    # ------------------------------------------------------------- act

    def act(self, state, t):
        K = min(self.cfg.fl.clients_per_round, self.N)
        s = self._state_vec(state["global_flat"])
        if self.pending is not None:
            s_prev, acts, r = self.pending
            for a in acts:
                self.buffer.append((s_prev, a, r, s))
            self._train()
            self.pending = None

        T = max(self.cfg.fl.rounds, 1)
        eps = max(0.05, 1.0 - t / (0.3 * T))
        with torch.no_grad():
            q = self.net(s).cpu().numpy()
        order = list(np.argsort(-q))
        sel: list[int] = []
        for _ in range(K):
            rest = [i for i in range(self.N) if i not in sel]
            if self.rng.random() < eps:
                sel.append(int(self.rng.choice(rest)))
            else:
                sel.append(int(next(i for i in order if i not in sel)))
        self.pending = (s, sel, None)
        return sel, {i: "standard" for i in sel}

    def observe_outcome(self, state, action, reward, info):
        r = 64.0 ** (info["acc"] - self.target) - 1.0
        if self.pending is not None:
            s, acts, _ = self.pending
            self.pending = (s, acts, r)
        base = info["base_flat"]
        for i, u in info["updates"].items():
            self.client[i] = self._proj(base + u)

    def _train(self, steps: int = 4, batch: int = 32, sync: int = 10) -> None:
        if len(self.buffer) < batch:
            return
        g = self.cfg.policy.favor_gamma
        for _ in range(steps):
            idx = self.rng.choice(len(self.buffer), size=batch, replace=False)
            s = torch.stack([self.buffer[k][0] for k in idx])
            a = torch.tensor([self.buffer[k][1] for k in idx], device=s.device)
            r = torch.tensor([self.buffer[k][2] for k in idx], dtype=torch.float32, device=s.device)
            s2 = torch.stack([self.buffer[k][3] for k in idx])
            with torch.no_grad():
                a2 = self.net(s2).argmax(dim=1, keepdim=True)
                y = r + g * self.tgt(s2).gather(1, a2).squeeze(1)
            qsa = self.net(s).gather(1, a[:, None]).squeeze(1)
            loss = ((y - qsa) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.n_steps += 1
            if self.n_steps % sync == 0:
                self.tgt.load_state_dict(self.net.state_dict())


class DELTAPolicy(Policy):
    """FedPracDELTA (Alg. 1 and Eq. 16 of the paper).

    Sampling probabilities start uniform. After each round, for the set S_t
    of clients whose updates arrived, the probability mass S_t held is
    redistributed among them in proportion to
        sqrt(alpha_1 zeta_i^2 + alpha_2 sigma_i^2),
    zeta_i = ||g_i - mean_{S_t} g|| (gradient diversity, g_i the client's
    summed local gradients = -update / lr) and sigma_i^2 the variance of its
    per-batch gradients (local variance). alpha_1 = alpha_2 = 0.5, the
    paper's default. K clients are drawn without replacement with these
    probabilities and aggregated with the unbiased weights n_i / p_i
    (the engine applies `agg_scale`).

    Change forced by this setting: the paper assumes every sampled client
    reports; here clients can fail, so the weights are normalised over the
    updates that arrived (self-normalised importance weighting), as every
    other aggregator here normalises.
    """

    wants_grad_stats = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.N = cfg.data.num_clients
        self.p = np.full(self.N, 1.0 / self.N)
        self.p_used = self.p.copy()
        self.rng = np.random.default_rng(cfg.seed + 23)

    def act(self, state, t):
        K = min(self.cfg.fl.clients_per_round, self.N)
        self.p_used = self.p.copy()
        sel = [int(i) for i in self.rng.choice(self.N, size=K, replace=False, p=self.p)]
        return sel, {i: "standard" for i in sel}

    def agg_scale(self, i: int) -> float:
        return 1.0 / max(float(self.p_used[i]), 1e-12)

    def observe_outcome(self, state, action, reward, info):
        fresh = [i for i in info["recv_ids"]
                 if info["stale"].get(i, 0) == 0 and i in info["grad_var"]]
        if not fresh:
            return
        lr = self.cfg.fl.lr
        g = {i: -info["updates"][i] / lr for i in fresh}
        mean = sum(g.values()) / len(g)
        a = self.cfg.policy.delta_alpha
        score = np.array([np.sqrt(a * float(((g[i] - mean) ** 2).sum()) + a * info["grad_var"][i])
                          for i in fresh])
        if not np.isfinite(score).all() or score.sum() <= 0:
            return
        mass = float(self.p[fresh].sum())
        self.p[fresh] = score / score.sum() * mass
        self.p = np.maximum(self.p, 1e-8)
        self.p /= self.p.sum()


class REFLPolicy(Policy):
    """REFL: Intelligent Participant Selection + Adaptive Participant Target.

    IPS (Alg. 1): among clients that check in (online this round, i.e. the
    observed availability signal), each reports the probability that it will
    be available in the next slot; the server sorts ascending, shuffles ties,
    and takes the N_t *least* available ones. Clients that just submitted an
    update sit out `refl_hold` rounds (5, as in the paper). The forecast is
    right with probability 0.9 (the paper's assumed predictor accuracy;
    engine state "avail_forecast").
    APT: N_t = max(1, K - B_t), B_t the stragglers from last round whose
    late updates arrive this round.
    The staleness-aware aggregation (SAA, Eq. 5) is the engine's "refl"
    aggregator; run with fl.async_late so late updates exist.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.N = cfg.data.num_clients
        self.rng = np.random.default_rng(cfg.seed + 29)
        self.last_submit = np.full(self.N, -10**9)

    def act(self, state, t):
        K = min(self.cfg.fl.clients_per_round, self.N)
        n_t = max(1, K - int(state["n_late"]))
        online = [i for i in range(self.N) if state["obs"][i]["avail_hint"] > 0.5]
        pool = [i for i in online if t - self.last_submit[i] > self.cfg.policy.refl_hold]
        if len(pool) < n_t:
            pool = online
        if len(pool) < n_t:
            pool = list(range(self.N))
        pool = list(self.rng.permutation(pool))          # random order among ties
        fc = state["avail_forecast"]
        pool.sort(key=lambda i: fc[i])                   # stable: least available first
        sel = [int(i) for i in pool[:n_t]]
        return sel, {i: "standard" for i in sel}

    def observe_outcome(self, state, action, reward, info):
        for i in info["recv_ids"]:
            self.last_submit[i] = info["t"]
