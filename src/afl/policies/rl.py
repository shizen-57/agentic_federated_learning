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

from .base import Policy, gate, pool

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
        self.use_cov = cfg.policy.coverage_features
        n_feat = len(FEATURES) + (1 if self.use_debt else 0) + (2 if self.use_cov else 0)
        self.clean = cfg.fl.clean_rng
        if cfg.policy.credit_total and cfg.policy.credit_split:
            raise ValueError("policy.credit_total and policy.credit_split are alternatives")
        if cfg.policy.horizon > 1 and cfg.policy.per_client_credit and not cfg.policy.credit_total:
            # The multi-round update only sees the shared reward, and with
            # per-client credit the damage term has been moved out of it, so
            # this combination would silently drop the ARL signal.
            raise ValueError("policy.horizon > 1 with policy.per_client_credit is not supported")
        self.gate_filled = 0
        if self.clean:
            # Own seed for the network; sampling below uses self.gen. Neither
            # touches PyTorch's global generator.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed + 17)
                self.net = ScoreNet(n_feat, cfg.policy.rl_hidden, len(self.modes))
        else:
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
            ] + ([state["debt"][i]] if self.use_debt else [])
              + ([0.0, 0.0] if self.use_cov else []))     # filled in per pick
        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _coverage_inputs(state, picked: list[int]):
        """Class coverage each client would add to the clients picked so far:
        plain, and weighted by how under-covered each class has been lately."""
        sk = state["sketch"]
        cur = sk[picked].max(axis=0) if picked else np.zeros(sk.shape[1])
        gain = np.maximum(sk - cur, 0.0)
        return (torch.tensor(gain.sum(axis=1), dtype=torch.float32),
                torch.tensor(gain @ state["cov_deficit"], dtype=torch.float32))

    def _draw(self, logits: torch.Tensor):
        """Sample one index (or take the argmax when not training)."""
        if not self.train_mode:
            i = int(logits.argmax())
        elif self.clean:
            i = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=self.gen))
        else:
            i = int(torch.distributions.Categorical(logits=logits).sample())
        return i, torch.log_softmax(logits, dim=-1)[i]

    # ----------------------------------------------------------------- act

    def act(self, state, t):
        N = self.cfg.data.num_clients
        K = min(self.cfg.fl.clients_per_round, N)
        x = self._feature_matrix(state)
        allowed, k_s, fill = self._allowed(state, N, K)
        if k_s == 0 and not fill:
            self._pending = None
            return [], {}
        if self.use_cov:
            return self._act_sequential(state, x, allowed, k_s, fill)
        scores, mode_logits = self.net(x)

        # Eligibility gate (Sec. 3.10): a client with no feasible configuration
        # is masked out entirely rather than merely scored low, so the policy
        # never spends one of its K slots on a client that cannot succeed.
        masked = scores.masked_fill(~allowed, -1e9)

        sel, logp_sel, step_logp = self._sample_topk(masked, k_s)
        sel = sel + fill

        logp_mode = 0.0
        entropy = 0.0
        assign = {}
        mode_lp: dict[int, torch.Tensor] = {}
        # Per-client log-probabilities (pick + mode), for per-decision credit.
        logp_client = dict(step_logp)
        # Under a fixed mode rule (policy.mode_rule) the engine assigns the
        # modes, so the mode head is not sampled and gets no gradient.
        fixed_modes = self.cfg.policy.mode_rule != "own"
        for i in sel:
            if fixed_modes:
                assign[int(i)] = self.modes[0]
                continue
            dist = torch.distributions.Categorical(logits=mode_logits[i])
            if self.clean and self.train_mode:
                m = torch.multinomial(torch.softmax(mode_logits[i], -1), 1, generator=self.gen)[0]
            else:
                m = dist.sample() if self.train_mode else mode_logits[i].argmax()
            lp = dist.log_prob(m)
            logp_mode = logp_mode + lp
            logp_client[int(i)] = logp_client.get(int(i), 0.0) + lp
            mode_lp[int(i)] = lp
            entropy = entropy + dist.entropy()
            assign[int(i)] = self.modes[int(m)]

        sel_entropy = torch.distributions.Categorical(logits=masked).entropy()
        self._pending = {"logp": logp_sel + logp_mode, "entropy": entropy + sel_entropy,
                         "logp_client": logp_client, "logp_sel": logp_sel, "logp_mode": mode_lp}
        return [int(i) for i in sel], assign

    def _allowed(self, state, N: int, K: int):
        """Mask of clients the policy may sample, how many to sample, and the
        gate's fill-ins (base.gate). Without the protocol switches this is the
        original rule: the gated clients, or everyone if fewer than K pass."""
        cands = pool(state, N)
        elig, fill = gate(self.cfg, state, cands, K)
        if self.cfg.policy.strict_gate:
            ok = elig
        else:
            ok = elig if len(elig) >= K else cands
        mask = torch.zeros(N, dtype=torch.bool)
        mask[ok] = True
        self.gate_filled = len(fill)
        return mask, min(K, len(ok)), fill

    def _act_sequential(self, state, x, allowed, K, fill):
        """Pick K clients one at a time, recomputing the coverage inputs (and so
        every score) after each pick, so the policy can choose clients that
        complement the ones already chosen. The log-probability is still the
        exact sequential one, so REINFORCE is unchanged."""
        open_ = allowed.clone()
        picked, steps, mode_rows = [], {}, {}
        logp, entropy = 0.0, 0.0
        if K == 0:
            g, gu = self._coverage_inputs(state, [])
            xs = x.clone()
            xs[:, -2], xs[:, -1] = g, gu
            _, mode_logits = self.net(xs)
        for k in range(K):
            g, gu = self._coverage_inputs(state, picked)
            xs = x.clone()
            xs[:, -2], xs[:, -1] = g, gu
            scores, mode_logits = self.net(xs)
            masked = scores.masked_fill(~open_, -1e9)
            if k == 0:
                entropy = torch.distributions.Categorical(logits=masked).entropy()
            i, lp = self._draw(masked)
            logp = logp + lp
            steps[i] = lp
            mode_rows[i] = mode_logits[i]
            picked.append(i)
            open_[i] = False
        for i in fill:                       # gate fill-ins: not sampled, modes still are
            mode_rows[i] = mode_logits[i]
        picked = picked + list(fill)
        assign, logp_client = {}, dict(steps)
        logp_sel, mode_lp = logp, {}
        fixed_modes = self.cfg.policy.mode_rule != "own"
        for i in picked:
            if fixed_modes:
                assign[i] = self.modes[0]
                continue
            m, lp = self._draw(mode_rows[i])
            logp = logp + lp
            logp_client[i] = logp_client.get(i, 0.0) + lp
            mode_lp[i] = lp
            entropy = entropy + torch.distributions.Categorical(logits=mode_rows[i]).entropy()
            assign[i] = self.modes[m]
        self._pending = {"logp": logp, "entropy": entropy, "logp_client": logp_client,
                         "logp_sel": logp_sel, "logp_mode": mode_lp}
        return picked, assign

    def _sample_topk(self, logits: torch.Tensor, K: int):
        """Sequential softmax sampling without replacement (Plackett-Luce)."""
        avail = logits.clone()
        chosen, logp, steps = [], 0.0, {}
        for _ in range(K):
            dist = torch.distributions.Categorical(logits=avail)
            if self.train_mode and self.clean:
                i = torch.multinomial(torch.softmax(avail, -1), 1, generator=self.gen)[0]
            elif self.train_mode:
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
        if not torch.is_tensor(self._pending["logp"]):
            self._pending = None               # nothing was sampled this round
            return
        if self.cfg.policy.horizon > 1:
            self._window.append((self._pending["logp"], self._pending["entropy"], reward))
            self._pending = None
            if len(self._window) >= self.cfg.policy.horizon:
                self._update_window()
            return
        if self.cfg.policy.prior_baseline:
            # Baseline from previous rounds only, so it does not depend on this
            # round's action; then fold this round in.
            advantage = reward - self.baseline
            self.baseline_n += 1
            self.baseline += (reward - self.baseline) / self.baseline_n
        else:
            self.baseline_n += 1
            self.baseline += (reward - self.baseline) / self.baseline_n
            advantage = reward - self.baseline

        loss = -(self._pending["logp"] * advantage) \
               - self.cfg.policy.entropy_coef * self._pending["entropy"]
        # Per-decision credit (ARL-v2): each invited client's own advantage
        # goes only to the log-probability of that client's pick and mode.
        client_adv = (info or {}).get("client_adv")
        set_adv = (info or {}).get("set_adv")
        if set_adv is not None:
            # policy.credit_split: the round's total ARL term on the picks, each
            # client's own term on its mode only (see config.PolicyCfg).
            if torch.is_tensor(self._pending["logp_sel"]):
                loss = loss - self._pending["logp_sel"] * set_adv
            for i, a in (client_adv or {}).items():
                if i in self._pending["logp_mode"]:
                    loss = loss - self._pending["logp_mode"][i] * a
        elif client_adv:
            for i, a in client_adv.items():
                if i in self._pending["logp_client"]:
                    loss = loss - self._pending["logp_client"][i] * a
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        self._pending = None

    def finish_run(self) -> None:
        """Learn from a last, unfinished multi-round window instead of dropping it."""
        if self.train_mode and self.cfg.policy.horizon > 1 and self._window:
            self._update_window()

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
            if self.cfg.policy.prior_baseline:
                adv = g - self.baseline
                self.baseline_n += 1
                self.baseline += (g - self.baseline) / self.baseline_n
            else:
                self.baseline_n += 1
                self.baseline += (g - self.baseline) / self.baseline_n
                adv = g - self.baseline
            loss = loss - logp * adv - self.cfg.policy.entropy_coef * ent
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        self._window = []
