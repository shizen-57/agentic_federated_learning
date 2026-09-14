"""Experiment configuration.

Every number here maps to a symbol in Chapter 3 of the thesis; the comment on
each field names the section it comes from so results can be traced back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Any


@dataclass
class DataCfg:
    dataset: str = "synthetic"          # synthetic | mnist
    num_clients: int = 30               # N          (Sec. 3.1)
    dirichlet_alpha: float = 0.3        # non-IID severity (Sec. 3.5)
    min_samples: int = 40
    max_samples: int = 600
    num_classes: int = 10
    seed: int = 0


@dataclass
class SystemCfg:
    """Systems heterogeneity and the participation-mode cost model (Sec. 3.6, 3.15)."""
    modes: tuple = ("light", "standard", "priority")
    local_epochs: dict = field(default_factory=lambda: {"light": 1, "standard": 2, "priority": 4})
    upload_frac: dict = field(default_factory=lambda: {"light": 0.25, "standard": 0.5, "priority": 1.0})
    deadline: float = 1.0               # tau_{i,t}, normalised round deadline
    cost_max: float = 1e9               # C_max  (Eq. 3.64); effectively off by default
    latency_max: float = 1e9            # L_max  (Eq. 3.65)
    availability_p: float = 0.85        # P(A_{i,t} = 1)
    staleness_max: int = 1              # acceptance rule (Eq. 3.18)
    norm_clip: float = 10.0             # acceptance rule (Eq. 3.18)


@dataclass
class FLCfg:
    rounds: int = 60                    # T
    clients_per_round: int = 5          # K  (Eq. 3.63)
    lr: float = 0.05
    batch_size: int = 32
    mu: float = 0.01                    # FedProx proximal weight (Sec. 3.17)
    aggregator: str = "fedprox"         # fedavg | fedprox | fednova
    model: str = "mlp"                  # mlp | cnn


@dataclass
class RewardCfg:
    """Coefficients of Eq. (3.55): R = a*dAcc - b*Cost - d*Lat - lA*ARL' - lF*CV."""
    alpha: float = 1.0
    beta: float = 0.1
    delta: float = 0.1
    lambda_arl: float = 0.5
    lambda_fair: float = 0.3
    gamma: float = 0.95


@dataclass
class PolicyCfg:
    name: str = "rl"                    # random | fedcs | powerofchoice | oort | rl
    elig_threshold: float = 0.5         # e^elig_{i,t} gate (Sec. 3.10)
    rl_lr: float = 1e-2
    rl_hidden: int = 32
    entropy_coef: float = 0.01


@dataclass
class Cfg:
    data: DataCfg = field(default_factory=DataCfg)
    system: SystemCfg = field(default_factory=SystemCfg)
    fl: FLCfg = field(default_factory=FLCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    seed: int = 0
    out: str = "results"
    tag: str = "run"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Cfg":
        cfg = Cfg()
        for f in fields(Cfg):
            if f.name not in d:
                continue
            val = d[f.name]
            cur = getattr(cfg, f.name)
            if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
                for k, v in val.items():
                    if not hasattr(cur, k):
                        raise KeyError(f"unknown config key {f.name}.{k}")
                    setattr(cur, k, v)
            else:
                setattr(cfg, f.name, val)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
