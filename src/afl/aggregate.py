"""Aggregation rules (Sec. 3.17): FedAvg, FedProx, FedNova.

FedAvg and FedProx share the same server-side rule -- they differ only in the
client objective -- so both land in `_weighted`. FedNova additionally normalises
each update by the number of local steps it took, which is what removes the
objective-inconsistency bias when clients are assigned different modes.
"""
from __future__ import annotations

import torch


def _weighted(updates, weights):
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    return sum(u * wi for u, wi in zip(updates, w))


def aggregate(rule: str, updates, n_i, steps, success_probs=None):
    """Return the flat delta to add to the global model.

    `success_probs` enables the optional importance reweighting of Eq. (3.54),
    which corrects the bias from aggregating only over clients that succeeded.
    Pass None for the core formulation.
    """
    if not updates:
        return None

    weights = list(n_i)
    if success_probs is not None:
        weights = [n / max(p, 1e-3) for n, p in zip(n_i, success_probs)]

    if rule in ("fedavg", "fedprox"):
        return _weighted(updates, weights)

    if rule == "fednova":
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        tau_eff = float(sum(wi * s for wi, s in zip(w, steps)))
        normed = [u / max(s, 1) for u, s in zip(updates, steps)]
        return _weighted(normed, weights) * tau_eff

    raise ValueError(f"unknown aggregator {rule}")
