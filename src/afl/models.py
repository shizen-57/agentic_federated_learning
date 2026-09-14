"""Deliberately small global models.

Nothing here is larger than ~100k parameters. That is a research constraint, not
a shortcut: the thesis studies the orchestration policy, and a small global model
keeps a 60-round, 30-client experiment inside a free Colab session while leaving
the non-IID dynamics the policy has to cope with fully intact.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """For MNIST-shaped input, fed as a flat 784-vector and reshaped here."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        return self.fc(self.conv(x).flatten(1))


def build_model(cfg, dim: int, num_classes: int) -> nn.Module:
    if cfg.fl.model == "cnn":
        return SmallCNN(num_classes)
    return MLP(dim, num_classes)


def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat(model: nn.Module, flat: torch.Tensor) -> None:
    off = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[off:off + n].view_as(p))
            off += n
