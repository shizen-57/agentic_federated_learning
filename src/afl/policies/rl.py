"""The proposed RL orchestration policy (Sec. 3.14, 3.19).

Design choices that matter, and why:

*Per-client scoring instead of a joint action space.*  Scoring every client with
one shared network and taking the Top-K keeps the parameter count independent of
N and sidesteps the C(N,K) subset explosion that objective (ii) in Sec. 1.0.2
calls out. The same shared trunk feeds a 3-way mode head, so selection and
configuration are produced by one policy rather than two stacked heuristics.

*Plackett-Luce sampling for the gradient.*  A hard Top-K is not differentiable
and gives no exploration. Sampling K clients without replacement from the score
softmax yields an exact sequential log-probability, so plain REINFORCE applies.
Evaluation still uses the deterministic Top-K.

*REINFORCE with a moving-average baseline, not PPO.*  A 60-round run is 60
policy-gradient steps. At that sample budget the variance reduction from a
learned critic does not pay for the extra hyperparameters, and a simple,
inspectable update is worth more for a thesis than a marginally better optimiser.

*One-round or multi-round credit (policy.horizon).*  With horizon 1 (the
original runs) each round's action is credited with that round's reward only.
With horizon H > 1 the policy collects H rounds and credits each action with
its discounted return over the rest of the window,
G_t = sum_{k>=0} gamma^k r_{t+k} (reward.gamma), so a choice that pays off a few
rounds later -- e.g. keeping a weak client alive -- can be learned.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import Policy

FEATURES = [
    "repr_value", "n_i_norm", "p_light", "p_standard", "p_priority",
    "speed_hat", "bw_hat", "energy_hat", "success_rate",
    "rounds_since_success", "selected_frac", "local_loss",
]


class ScoreNet(nn.Module):
    def __init__(self, dim: int, hidden: int, n_modes: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.score = nn.Linear(hidden, 1)      # selection logit per client
        self.mode = nn.Linear(hidden, n_modes)  # configuration logits per client

    def forward(self, x):
        h = self.trunk(x)
        return self.score(h).squeeze(-1), self.mode(h)


class RLPolicy(Policy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.modes = list(cfg.system.modes)
        # The fairness-debt feature exists only when the debt module is on, so
        # the original network (and its results) are unchanged otherwise.
        self.use_debt = cfg.reward.lambda_debt > 0
        n_feat = len(FEATURES) + (1 if self.use_debt else 0)
        self.net = ScoreNet(n_feat, cfg.policy.rl_hidden, len(self.modes))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.policy.rl_lr)
        self.baseline = 0.0
        self.baseline_n = 0
        self.gen = torch.Generator().manual_seed(cfg.seed + 17)
        self._pending = None
        self._window: list = []           # (logp, entropy, reward) when horizon > 1
        self.train_mode = True

    # ------------------------------------------------------------- features

    def _feature_matrix(self, state) -> torch.Tensor:
        N = self.cfg.data.num_clients
        rows = []
        for i in range(N):
            obs = state["obs"][i]
            hist = state["hist"][i]
            rows.append([
                state["repr_value"][i],
                state["n_i"][i] / max(state["n_i"].max(), 1.0),
                state["p_succ"][i]["light"],
                state["p_succ"][i]["standard"],
                state["p_succ"][i]["priority"],
                np.log1p(obs["speed_hat"]),
                np.log1p(obs["bw_hat"]),
                obs["energy_hat"],
                hist["success_rate"],
                np.tanh(hist["rounds_since_success"] / 10.0),
                hist["selected_frac"],
                state["local_loss"][i],
            ] + ([state["debt"][i]] if self.use_debt else []))
        return torch.tensor(rows, dtype=torch.float32)

    # ----------------------------------------------------------------- act

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        x = self._feature_matrix(state)
        scores, mode_logits = self.net(x)

        # Eligibility gate (Sec. 3.10): a client with no feasible configuration
        # is masked out entirely rather than merely scored low, so the policy
        # never spends one of its K slots on a client that cannot succeed.
        elig = torch.tensor([state["eligible"][i] for i in range(N)], dtype=torch.bool)
        if elig.sum() < K:
            elig = torch.ones(N, dtype=torch.bool)
        masked = scores.masked_fill(~elig, -1e9)

        sel, logp_sel, step_logp = self._sample_topk(masked, K)

        logp_mode = 0.0
        entropy = 0.0
        assign = {}
        # Per-client log-probabilities (pick + mode), for per-decision credit.
        logp_client = dict(step_logp)
        for i in sel:
            dist = torch.distributions.Categorical(logits=mode_logits[i])
            m = dist.sample() if self.train_mode else mode_logits[i].argmax()
            lp = dist.log_prob(m)
            logp_mode = logp_mode + lp
            logp_client[int(i)] = logp_client[int(i)] + lp
            entropy = entropy + dist.entropy()
            assign[int(i)] = self.modes[int(m)]

        sel_entropy = torch.distributions.Categorical(logits=masked).entropy()
        self._pending = {"logp": logp_sel + logp_mode, "entropy": entropy + sel_entropy,
                         "logp_client": logp_client}
        return [int(i) for i in sel], assign

    def _sample_topk(self, logits: torch.Tensor, K: int):
        """Sequential softmax sampling without replacement (Plackett-Luce)."""
        avail = logits.clone()
        chosen, logp, steps = [], 0.0, {}
        for _ in range(K):
            dist = torch.distributions.Categorical(logits=avail)
            if self.train_mode:
                i = dist.sample()
            else:
                i = avail.argmax()
            lp = dist.log_prob(i)
            logp = logp + lp
            steps[int(i)] = lp
            chosen.append(int(i))
            avail = avail.clone()
            avail[int(i)] = -1e9
        return chosen, logp, steps

    # -------------------------------------------------------------- learn

    def observe_outcome(self, state, action, reward, info):
        if self._pending is None or not self.train_mode:
            return
        if self.cfg.policy.horizon > 1:
            self._window.append((self._pending["logp"], self._pending["entropy"], reward))
            self._pending = None
            if len(self._window) >= self.cfg.policy.horizon:
                self._update_window()
            return
        self.baseline_n += 1
        self.baseline += (reward - self.baseline) / self.baseline_n
        advantage = reward - self.baseline

        loss = -(self._pending["logp"] * advantage) \
               - self.cfg.policy.entropy_coef * self._pending["entropy"]
        # Per-decision credit (ARL-v2): each invited client's own advantage
        # goes only to the log-probability of that client's pick and mode.
        client_adv = (info or {}).get("client_adv")
        if client_adv:
            for i, a in client_adv.items():
                if i in self._pending["logp_client"]:
                    loss = loss - self._pending["logp_client"][i] * a
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        self._pending = None

    def _update_window(self):
        """REINFORCE on discounted returns within the window (horizon > 1)."""
        gamma = self.cfg.reward.gamma
        rewards = [r for _, _, r in self._window]
        returns, g = [], 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.append(g)
        returns.reverse()
        loss = 0.0
        for (logp, ent, _), g in zip(self._window, returns):
            self.baseline_n += 1
            self.baseline += (g - self.baseline) / self.baseline_n
            loss = loss - logp * (g - self.baseline) - self.cfg.policy.entropy_coef * ent
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        self._window = []
