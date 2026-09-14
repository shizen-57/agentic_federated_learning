"""Representation value, ARL, and successful-contribution fairness.

This module is where the thesis's contribution lives. Everything else in the
package is standard federated learning machinery; these three quantities are
what distinguish contribution-aware orchestration from ordinary client selection.
"""
from __future__ import annotations

import numpy as np


def coverage(sketch_subset: np.ndarray) -> float:
    """Facility-location class coverage Cov(S) of Sec. 3.11.

    Cov(S) = sum_k max_{i in S} c_{i,k}: a class is covered as well as the best
    client covers it. Submodular, so marginal gains are diminishing, which is
    exactly the property that makes a second client holding the same rare class
    worth less than the first.
    """
    if len(sketch_subset) == 0:
        return 0.0
    return float(sketch_subset.max(axis=0).sum())


def representation_value(sketch: np.ndarray, selected: list[int], i: int) -> float:
    """V^repr_{i,t}: client i's marginal coverage gain over the selected set.

    Computed against the set selected *so far*, so a rare-class holder scores
    high while a redundant client scores near zero.
    """
    base = coverage(sketch[selected]) if selected else 0.0
    with_i = coverage(sketch[selected + [i]])
    return max(0.0, with_i - base)


def standalone_value(sketch: np.ndarray) -> np.ndarray:
    """Value of each client evaluated on its own, used as a cheap static prior."""
    return sketch.sum(axis=1) * (sketch > 0.05).sum(axis=1) / sketch.shape[1]


def arl_online(values: dict[int, float], eligible: dict[int, int],
               success: dict[int, int]) -> float:
    """ARL'_t (Eq. 3.27), the causal online estimate.

    Sums the representation value of every client that the *predictor* believed
    could have succeeded but did not. Uses predicted eligibility only, so it is
    computable by a deployed policy and is safe to put in the reward.
    """
    return float(sum(values.get(i, 0.0)
                     for i in values
                     if eligible.get(i, 0) == 1 and success.get(i, 0) == 0))


def arl_oracle(values: dict[int, float], env, success: dict[int, int]) -> float:
    """ARL*_t (Eq. 3.29), the oracle-relative diagnostic.

    Replaces predicted eligibility with the simulator's ground-truth answer to
    "did *any* configuration work for this client this round?". Requires
    counterfactual queries under common random numbers, so it is evaluation-only
    and never enters the reward -- reporting it is how the gap between what the
    policy believed it was optimising and what a perfectly-informed scheduler
    could have achieved gets measured instead of assumed away.
    """
    return float(sum(v for i, v in values.items()
                     if env.oracle_eligible(i) == 1 and success.get(i, 0) == 0))


def cv_fairness(x_succ: np.ndarray) -> float:
    """CV_t of Eq. (3.32) over cumulative successful contributions X^succ_{i,t}."""
    mu = float(x_succ.mean())
    if mu <= 0:
        return 0.0
    return float(x_succ.std() / mu)


def jain(x_succ: np.ndarray) -> float:
    """Jain's index, reported alongside CV so results compare to prior FL work."""
    s = float(x_succ.sum())
    if s <= 0:
        return 1.0
    return s ** 2 / (len(x_succ) * float((x_succ ** 2).sum()))


def calibration_error(probs: list[float], outcomes: list[int], bins: int = 10) -> float:
    """Expected calibration error of p^succ, reported per Sec. 3.23.

    ARL' is only as trustworthy as the predictor feeding it, so the thesis needs
    this number next to every ARL' number it reports.
    """
    if not probs:
        return 0.0
    p = np.asarray(probs)
    o = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        err += m.mean() * abs(o[m].mean() - p[m].mean())
    return float(err)


class Normalizer:
    """Running standardiser for the reward terms (Eq. 3.57).

    Accuracy gain, cost, latency and ARL live on wildly different scales, so the
    reward coefficients would be uninterpretable without this.
    """

    def __init__(self, eps: float = 1e-6):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = eps

    def __call__(self, x: float) -> float:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if self.n < 2:
            return 0.0
        std = (self.m2 / (self.n - 1)) ** 0.5
        return float((x - self.mean) / (std + self.eps))
