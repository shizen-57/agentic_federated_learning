"""Contribution-success predictor p^succ_{i,t}(E) (Sec. 3.9).

An online logistic model, one shared weight vector over
(client features x mode features). It is trained incrementally on every
(selected client, assigned mode) -> observed outcome pair, so it starts
uninformative and calibrates as the run proceeds, which is the realistic setting:
a deployed server has no labelled failure data before it starts scheduling.

Logistic regression is chosen over a neural predictor on purpose. It gives
usable probabilities from a few hundred observations, and since ARL' is only as
good as this model's calibration, a low-variance estimator matters more here
than capacity.
"""
from __future__ import annotations

import numpy as np


class SuccessPredictor:
    def __init__(self, cfg, dim: int = 9, lr: float = 0.08, l2: float = 1e-4):
        self.cfg = cfg
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.seen = 0
        self.mode_idx = {m: k for k, m in enumerate(cfg.system.modes)}
        self.norm = cfg.policy.predictor_norm
        if self.norm not in ("none", "round"):
            raise ValueError(f"unknown policy.predictor_norm {self.norm}")
        self.ref_mean = None
        self.ref_std = None

    @staticmethod
    def client_features(obs: dict, hist: dict) -> np.ndarray:
        return np.array([
            np.log1p(obs["speed_hat"]),
            np.log1p(obs["bw_hat"]),
            obs["energy_hat"],
            obs["avail_hint"],
            hist["success_rate"],
            np.tanh(hist["rounds_since_success"] / 10.0),
        ])

    def set_reference(self, rows: np.ndarray) -> None:
        """policy.predictor_norm = round: this round's population statistics
        of the client features (one row per client), used to standardise them.
        The raw features are all non-negative; with uncentred inputs a model
        trained on failures pushes every weight negative, so the slowest,
        emptiest clients score highest. Centred inputs leave failures to the
        intercept and keep the ranking tied to real differences."""
        self.ref_mean = rows.mean(axis=0)
        self.ref_std = np.maximum(rows.std(axis=0), 1e-3)

    def features(self, obs: dict, hist: dict, mode: str) -> np.ndarray:
        m = np.zeros(3)
        m[self.mode_idx[mode]] = 1.0
        c = self.client_features(obs, hist)
        if self.norm == "round":
            if self.ref_mean is None:
                raise RuntimeError("policy.predictor_norm=round needs set_reference() each round")
            c = (c - self.ref_mean) / self.ref_std
        return np.concatenate([c, m])

    def predict(self, obs: dict, hist: dict, mode: str) -> float:
        z = float(self.features(obs, hist, mode) @ self.w + self.b)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        if self.seen < 20:
            # Before any real evidence, fall back to the prior rather than
            # reporting a confident 0.5 that the eligibility gate would trust.
            return 0.5 * p + 0.5 * 0.7
        return p

    def update(self, obs: dict, hist: dict, mode: str, y: int) -> None:
        f = self.features(obs, hist, mode)
        z = float(f @ self.w + self.b)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        self.w -= self.lr * (g * f + self.l2 * self.w)
        self.b -= self.lr * g
        self.seen += 1


class SuccessPredictorV2(SuccessPredictor):
    """Predictor v2 (docs/RL2_DESIGN.md, C1): the shared logistic model on
    standardised features, corrected per client and mode by the client's own
    record, with an uncertainty estimate.

    - The shared model is blended with the cold-start prior smoothly,
      seen / (seen + c), instead of switching it off at 20 observations (the
      round-3 cliff in docs/DIAGNOSTIC_FINDINGS.md).
    - Per (client, mode), successes s and failures f give a Beta posterior
      with the shared prediction p0 as its mean and strength k:
      a = s + k*p0, b = f + k*(1 - p0), p = a / (a + b). A client's own
      record outweighs the shared model only as evidence builds up.
    - sd is the posterior standard deviation; the gate and the safety layer
      use it so uncertain predictions are not trusted as if certain.
    """

    def __init__(self, cfg, dim: int = 9, lr: float = 0.08, l2: float = 1e-4,
                 k: float = 5.0, prior: float = 0.7, prior_strength: float = 20.0):
        super().__init__(cfg, dim, lr, l2)
        self.norm = "round"                 # v2 always standardises client features
        self.k, self.prior, self.c = k, prior, prior_strength
        self.s: dict[tuple[int, str], float] = {}
        self.f: dict[tuple[int, str], float] = {}
        # policy.pred_decay: each round, a client's own evidence counts this
        # much less, so a battery or availability change is not hidden by old
        # outcomes and the uncertainty grows again while evidence ages.
        self.decay = cfg.policy.pred_decay
        self.t = 0                          # current round (set by the engine)
        self.t_last: dict[tuple[int, str], int] = {}

    def shared(self, obs: dict, hist: dict, mode: str) -> float:
        z = float(self.features(obs, hist, mode) @ self.w + self.b)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return (self.seen * p + self.c * self.prior) / (self.seen + self.c)

    def _aged(self, key) -> float:
        if self.decay >= 1.0 or key not in self.t_last:
            return 1.0
        return self.decay ** (self.t - self.t_last[key])

    def posterior(self, i: int, obs: dict, hist: dict, mode: str) -> tuple[float, float]:
        p0 = self.shared(obs, hist, mode)
        w = self._aged((i, mode))
        a = w * self.s.get((i, mode), 0.0) + self.k * p0
        b = w * self.f.get((i, mode), 0.0) + self.k * (1.0 - p0)
        return a, b

    def predict_client(self, i: int, obs: dict, hist: dict, mode: str) -> tuple[float, float]:
        a, b = self.posterior(i, obs, hist, mode)
        n = a + b
        return a / n, float(np.sqrt(a * b / (n * n * (n + 1.0))))

    def update_client(self, i: int, obs: dict, hist: dict, mode: str, y: int) -> None:
        self.update(obs, hist, mode, y)
        key = (i, mode)
        w = self._aged(key)                 # age the stored counts to now, then add
        self.s[key], self.f[key] = w * self.s.get(key, 0.0), w * self.f.get(key, 0.0)
        d = self.s if y else self.f
        d[key] += 1.0
        self.t_last[key] = self.t
