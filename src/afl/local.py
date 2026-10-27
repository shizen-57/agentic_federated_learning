"""Local training with the FedProx proximal term (Sec. 3.17)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import build_model, get_flat, set_flat


def local_train(cfg, global_flat: torch.Tensor, x, y, epochs: int, dim: int,
                num_classes: int, gen: torch.Generator, grad_stats: dict | None = None):
    """Run `epochs` local epochs from the current global model.

    `x`/`y` may be numpy arrays or tensors; training runs on `global_flat`'s
    device. Returns (update, steps) where `update` is the flat parameter delta
    and `steps` is the number of SGD steps taken -- FedNova needs the step
    count, so it is tracked here rather than reconstructed by the aggregator.

    If `grad_stats` is a dict, the variance of the per-batch gradients over
    the local batches is stored in grad_stats["var"] (DELTA's sigma_L^2).
    """
    dev = global_flat.device
    if cfg.fl.clean_rng:
        # The fresh model's random init is overwritten below; building it must
        # not move PyTorch's global generator, which other parts of the run use.
        with torch.random.fork_rng(devices=[]):
            model = build_model(cfg, dim, num_classes).to(dev)
    else:
        model = build_model(cfg, dim, num_classes).to(dev)
    set_flat(model, global_flat)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.fl.lr)

    xb_all = torch.as_tensor(x, device=dev)
    yb_all = torch.as_tensor(y, device=dev)
    n = len(yb_all)
    bs = min(cfg.fl.batch_size, n)
    steps = 0

    # REFL changes only the server side, so it keeps the FedProx local
    # objective every other baseline is run with.
    use_prox = cfg.fl.aggregator in ("fedprox", "fedprox_fednova", "refl") and cfg.fl.mu > 0
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
            if grad_stats is not None:
                g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
                g_sum = g if steps == 0 else g_sum + g
                sq_sum = float(g @ g) + (0.0 if steps == 0 else sq_sum)
            opt.step()
            steps += 1

    if grad_stats is not None and steps > 0:
        mean = g_sum / steps
        grad_stats["var"] = max(0.0, sq_sum / steps - float(mean @ mean))
    return get_flat(model) - global_flat, steps


def accepted(update: torch.Tensor, issued_round: int, t: int, cfg) -> bool:
    """Update-acceptance rule V_{i,t} of Eq. (3.18): staleness AND norm."""
    if t - issued_round > cfg.system.staleness_max:
        return False
    nrm = float(torch.linalg.norm(update))
    return np.isfinite(nrm) and nrm <= cfg.system.norm_clip
