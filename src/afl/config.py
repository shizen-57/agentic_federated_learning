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
    dataset: str = "synthetic"          # synthetic | mnist | cifar10 | femnist
                                        # (femnist ignores dirichlet_alpha: one
                                        # client per real writer)
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
    availability_p: float = 0.85        # mean of P(A_{i,t}=1) over a full diurnal cycle
    diurnal_amplitude: float = 0.35     # swing of availability around the mean across
                                        # the day (Hard et al. 2018, arXiv:1812.02903:
                                        # real device availability is concentrated
                                        # overnight while charging/idle/on WiFi, and
                                        # "round completion around noon is nearly flat")
    rounds_per_day: int = 96            # modeling assumption: ~15 min/round -> 96
                                        # rounds/day, so a 100-round run spans ~1 day
    staleness_max: int = 1              # acceptance rule (Eq. 3.18)
    norm_clip: float = 10.0             # acceptance rule (Eq. 3.18)
    work_ref_samples: float = 0.0       # if > 0, rescale every client's simulated
                                        # workload so the mean client looks like it
                                        # holds this many samples. Keeps the
                                        # Light/Standard/Priority deadline separation
                                        # the same across datasets whose per-client
                                        # sample counts differ (FEMNIST writers vs
                                        # a Dirichlet split). 0 = off (original runs).
    # Persistent-unreliability regime (docs/ADVANCED_EXPERIMENTS.md). Off by
    # default. The original simulator draws failures independently per round, so
    # "valuable client fails" and "any client fails" mostly coincide and ARL has
    # nothing to add over a raw failure count; this regime makes a fixed subset of
    # clients persistently weak, and optionally makes that subset the rare-data
    # holders, which is the setting the thesis argues ARL is for.
    weak_frac: float = 0.0              # fraction of clients that are persistently weak
    weak_select: str = "rarity"         # rarity: the clients holding the rarest
                                        # classes are the weak ones; random: a random
                                        # subset (control -- weakness without value)
    weak_reliability: float = 0.45      # link reliability of weak clients
    weak_avail_drop: float = 0.15       # subtracted from weak clients' availability
    avail_persistence: float = 0.0      # P(availability carries over from last round)


@dataclass
class FLCfg:
    rounds: int = 60                    # T
    clients_per_round: int = 5          # K  (Eq. 3.63)
    lr: float = 0.05
    batch_size: int = 32
    mu: float = 0.01                    # FedProx proximal weight (Sec. 3.17)
    aggregator: str = "fedprox"         # fedavg | fedprox | fednova | fedprox_fednova
    model: str = "mlp"                  # mlp | cnn | cnn_femnist | cnn_cifar
    device: str = "cpu"                 # cpu | cuda | auto
    eval_subset: int = 0                # if > 0, per-round accuracy uses a fixed
                                        # random subset of this many test samples;
                                        # the final round always uses the full set
    impute_missing: bool = False        # MIFA/MimiC-style post-hoc substitution
                                        # of a failed client's update with its
                                        # last known successful delta (Table 3.5);
                                        # tests prevention vs. imputation (Sec. 3.11)


@dataclass
class RewardCfg:
    """Coefficients of Eq. (3.55): R = a*dAcc - b*Cost - d*Lat - lA*ARL' - lF*CV."""
    alpha: float = 1.0
    beta: float = 0.1
    delta: float = 0.1
    lambda_arl: float = 0.5
    lambda_fair: float = 0.3
    gamma: float = 0.95
    # Ablation toggles, added to test whether the thesis's two specific claims
    # (value-weighted counterfactual damage; capability-conditioned fairness)
    # actually change behaviour versus the naive alternative each replaces.
    damage_metric: str = "arl"          # "arl" | "raw_fail" -- Task 1 ablation
    fairness_metric: str = "cv_success"  # "cv_success" | "cv_selection" -- Task 2 ablation


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
