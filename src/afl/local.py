"""Local training with the FedProx proximal term (Sec. 3.17)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import build_model, get_flat, set_flat


def local_train(cfg, global_flat: torch.Tensor, x: np.ndarray, y: np.ndarray,
                epochs: int, dim: int, num_classes: int, gen: torch.Generator):
    """Run `epochs` local epochs from the current global model.

    Returns (update, steps) where `update` is the flat parameter delta and
    `steps` is the number of SGD steps taken -- FedNova needs the step count, so
    it is tracked here rather than reconstructed by the aggregator.
    """
    model = build_model(cfg, dim, num_classes)
    set_flat(model, global_flat)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.fl.lr)

    xb_all = torch.from_numpy(x)
    yb_all = torch.from_numpy(y)
    n = len(y)
    bs = min(cfg.fl.batch_size, n)
    steps = 0

    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_all[idx]), yb_all[idx])
            if cfg.fl.aggregator == "fedprox" and cfg.fl.mu > 0:
                prox = 0.0
                off = 0
                for p in model.parameters():
                    k = p.numel()
                    prox = prox + ((p - global_flat[off:off + k].view_as(p)) ** 2).sum()
                    off += k
                loss = loss + 0.5 * cfg.fl.mu * prox
            loss.backward()
            opt.step()
            steps += 1

    return get_flat(model) - global_flat, steps


def accepted(update: torch.Tensor, issued_round: int, t: int, cfg) -> bool:
    """Update-acceptance rule V_{i,t} of Eq. (3.18): staleness AND norm."""
    if t - issued_round > cfg.system.staleness_max:
        return False
    nrm = float(torch.linalg.norm(update))
    return np.isfinite(nrm) and nrm <= cfg.system.norm_clip
