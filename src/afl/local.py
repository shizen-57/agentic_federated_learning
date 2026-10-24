"""Local training with the FedProx proximal term (Sec. 3.17)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import build_model, get_flat, set_flat


def local_train(cfg, global_flat: torch.Tensor, x, y, epochs: int, dim: int,
                num_classes: int, gen: torch.Generator):
    """Run `epochs` local epochs from the current global model.

    `x`/`y` may be numpy arrays or tensors; training runs on `global_flat`'s
    device. Returns (update, steps) where `update` is the flat parameter delta
    and `steps` is the number of SGD steps taken -- FedNova needs the step
    count, so it is tracked here rather than reconstructed by the aggregator.
    """
    dev = global_flat.device
    model = build_model(cfg, dim, num_classes).to(dev)
    set_flat(model, global_flat)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.fl.lr)

    xb_all = torch.as_tensor(x, device=dev)
    yb_all = torch.as_tensor(y, device=dev)
    n = len(yb_all)
    bs = min(cfg.fl.batch_size, n)
    steps = 0

    use_prox = cfg.fl.aggregator in ("fedprox", "fedprox_fednova") and cfg.fl.mu > 0
    if use_prox:
        anchors, off = [], 0
        for p in model.parameters():
            k = p.numel()
            anchors.append(global_flat[off:off + k].view_as(p))
            off += k

    for _ in range(epochs):
        # The permutation is drawn on the CPU generator so a run's batch order
        # does not depend on which device it trains on.
        perm = torch.randperm(n, generator=gen).to(dev)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_all[idx]), yb_all[idx])
            if use_prox:
                prox = 0.0
                for p, g in zip(model.parameters(), anchors):
                    prox = prox + ((p - g) ** 2).sum()
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
