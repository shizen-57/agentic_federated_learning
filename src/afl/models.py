"""Global models, small on purpose.

The thesis studies the orchestration policy, not the architecture, so every
model here stays small enough for a free Colab/Kaggle session:

``mlp``          ~1k params, synthetic data only.
``cnn``          ~20k params, MNIST (the original Chapter 5 model).
``cnn_femnist``  ~220k params, FEMNIST's 62 classes (28x28 grayscale).
``cnn_cifar``    ~190k params, CIFAR-10 (3x32x32 colour); the classic
                 three-conv-block CNN, enough to separate policies on a task
                 that is not saturated after a few rounds.
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


class FemnistCNN(nn.Module):
    """For FEMNIST: same 28x28 input as SmallCNN, wider, with a hidden FC layer."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
                                nn.Linear(128, num_classes))

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        return self.fc(self.conv(x).flatten(1))


class CifarCNN(nn.Module):
    """For CIFAR-10, fed as a flat 3072-vector in CHW order."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
                                nn.Linear(128, num_classes))

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        return self.fc(self.conv(x).flatten(1))


def build_model(cfg, dim: int, num_classes: int) -> nn.Module:
    if cfg.fl.model == "cnn":
        return SmallCNN(num_classes)
    if cfg.fl.model == "cnn_femnist":
        return FemnistCNN(num_classes)
    if cfg.fl.model == "cnn_cifar":
        return CifarCNN(num_classes)
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
