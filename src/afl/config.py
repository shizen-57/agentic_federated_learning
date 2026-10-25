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
    cost_max: float = 1e9               # C_max  (Eq. 3.64): per-round cost budget,
                                        # enforced by Experiment._enforce_budget; 1e9 = off
    latency_max: float = 1e9            # L_max  (Eq. 3.65): per-client expected time limit
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
    weak_speed: float = 0.8             # weak clients' device speed (mid tier = 1.0)
    weak_bandwidth: float = 0.6         # weak clients' bandwidth (mid tier = 1.0)
    weak_reliability: float = 0.6       # link reliability of weak clients (mid tier ~0.7)
    weak_avail_drop: float = 0.1        # subtracted from weak clients' availability
    avail_persistence: float = 0.0      # P(availability carries over from last round)
    churn_after: int = 0                # client churn: a client with limited patience
                                        # leaves for good after this many consecutive
                                        # invitations that it attempted and failed
                                        # (deadline, energy, link drop, rejection).
                                        # 0 = off. Offline rounds do not count.
    churn_prob: float = 0.5             # share of clients with limited patience (fixed
                                        # per client at the start, same for every policy)
    charge_actual: bool = False         # protocol fix: charge cost and battery for what a
                                        # client actually did -- nothing if it was offline
                                        # or could not start for lack of battery. Off =
                                        # the original runs, where every invitation was
                                        # charged the mode's full cost and energy
    checkin_pool: bool = False          # protocol fix: every policy selects only from
                                        # clients that checked in (are online) this
                                        # round, as real FL servers do. Off = the
                                        # original runs, where only policies that read
                                        # the availability signal avoided offline clients
    device_trace: str = "tiers"         # tiers: the three guessed device tiers;
                                        # fedscale: real per-device speed and
                                        # bandwidth from FedScale's trace
                                        # (src/afl/traces/README.md)


@dataclass
class FLCfg:
    rounds: int = 60                    # T
    clients_per_round: int = 5          # K  (Eq. 3.63)
    lr: float = 0.05
    batch_size: int = 32
    mu: float = 0.01                    # FedProx proximal weight (Sec. 3.17)
    aggregator: str = "fedprox"         # fedavg | fedprox | fednova | fedprox_fednova
                                        # | mifa | mimic | fedcg (published baselines)
    model: str = "mlp"                  # mlp | cnn | cnn_femnist | cnn_cifar
    device: str = "cpu"                 # cpu | cuda | auto
    deterministic: bool = False         # ask PyTorch for deterministic kernels so a GPU
                                        # run repeats exactly (slower; untested on GPU
                                        # here). CPU runs are already exactly repeatable.
    eval_subset: int = 0                # if > 0, per-round accuracy uses a fixed
                                        # random subset of this many test samples;
                                        # the final round always uses the full set
    impute_missing: bool = False        # MIFA/MimiC-style post-hoc substitution
                                        # of a failed client's update with its
                                        # last known successful delta (Table 3.5);
                                        # tests prevention vs. imputation (Sec. 3.11)
    importance_reweight: bool = False   # Eq. (3.54): weight each received update by
                                        # n_i / p_succ to undo the bias of averaging
                                        # only clients that succeeded
    reweight_floor: float = 0.1         # lower bound on p_succ in that weight, so
                                        # one unlikely success cannot get >10x weight
    async_late: bool = False            # semi-asynchronous: a client that misses the
                                        # deadline but finishes within staleness_max
                                        # rounds delivers late instead of failing,
                                        # so the staleness rule of Eq. (3.18) applies
    val_frac: float = 0.0               # protocol fix: hold out this share of the test
                                        # pool as a server validation set. The reward
                                        # (and every learning baseline) sees only it;
                                        # reported accuracy uses only the rest. 0 = the
                                        # original runs, where the reward was computed
                                        # on test images that the final score also used
    clean_rng: bool = False             # protocol fix: the FL model is initialised from
                                        # its own seed (identical for every method with
                                        # the same seed), and policy init, policy
                                        # sampling and local training each use their own
                                        # random stream instead of sharing PyTorch's
    absence_counter: bool = False       # protocol fix: rounds_since_success counts every
                                        # round without a contribution, not only rounds
                                        # in which the client was invited and failed
    compress_upload: bool = False       # simulator fix: a mode's upload_frac is what the
                                        # client really sends -- its update is top-k
                                        # sparsified to that fraction (with error
                                        # feedback) for every aggregator. Off = the
                                        # original runs, where Light and Standard were
                                        # charged for a partial upload but the server
                                        # still received the full update


@dataclass
class RewardCfg:
    """Coefficients of Eq. (3.55): R = a*dAcc - b*Cost - d*Lat - lA*ARL' - lF*CV."""
    alpha: float = 1.0
    beta: float = 0.1
    delta: float = 0.1
    lambda_arl: float = 0.5
    lambda_fair: float = 0.3
    lambda_debt: float = 0.0            # fairness debt (Sec. 3.13), off by default
    gamma: float = 0.95                 # discount, used when policy.horizon > 1
    # Ablation toggles, added to test whether the thesis's two specific claims
    # (value-weighted counterfactual damage; capability-conditioned fairness)
    # actually change behaviour versus the naive alternative each replaces.
    damage_metric: str = "arl"          # "arl" | "raw_fail" -- Task 1 ablation
                                        # | "arl_selected": ARL' restricted to the
                                        # clients actually invited. Eq. (3.27) sums
                                        # over all clients, and at N=100/K=10 ~90%
                                        # of it comes from clients never invited,
                                        # so the value-weighting of the policy's own
                                        # failures is a small part of the signal.
                                        # | "arl_v2": loss the policy caused -- invited
                                        # clients that failed, weighted by the success
                                        # probability their assigned mode gave up, by
                                        # V^repr, and by how under-covered their classes
                                        # have been lately (see engine.run_round)
                                        # | "arl_v3": ARL-v2 fixed -- expected useful
                                        # work (success prob x local epochs) the chosen
                                        # mode gave up, plus credit (lambda_cov) for
                                        # useful work delivered on under-covered data
    lambda_cov: float = 0.5             # ARL-v3 success-credit weight
    arl_v2_ema: float = 0.1             # rate of the class-coverage moving average that
                                        # ARL-v2 uses to tell short gaps from long ones
    batch_client_norm: bool = False     # protocol fix: per-client credit (ARL-v2/v3) is
                                        # standardised with the statistics from before
                                        # the round, then the statistics are updated with
                                        # the whole round -- so a client's credit does not
                                        # depend on the order clients are processed in
    fairness_metric: str = "cv_success"  # "cv_success" | "cv_selection" -- Task 2 ablation


@dataclass
class PolicyCfg:
    name: str = "rl"                    # random | fedcs | powerofchoice | oort | rl
    elig_threshold: float = 0.5         # e^elig_{i,t} gate (Sec. 3.10)
    strict_gate: bool = False           # protocol fix: when fewer than K clients pass the
                                        # gate, keep the ones that pass and fill the rest
                                        # with the highest predicted success (instead of
                                        # dropping the gate for everyone)
    rl_lr: float = 1e-2
    rl_hidden: int = 32
    entropy_coef: float = 0.01
    value_feature: str = "standalone"   # the RL policy's "value" input. standalone: the
                                        # original static prior (number of classes a
                                        # client holds) -- which turns out to be strongly
                                        # anti-correlated (about -0.7) with the value ARL
                                        # rewards; marginal: expected marginal coverage,
                                        # i.e. E[V^repr] (metrics.expected_marginal_value)
    fedcg_margin: float = 1.0           # FedCG fits each client's compression to
                                        # margin x deadline (1.0 = the paper's exact fit;
                                        # <1 leaves slack for timing jitter)
    per_client_credit: bool = False     # ARL-v2: each invited client's own loss is
                                        # credited to that client's pick and mode only,
                                        # instead of the shared round reward
    horizon: int = 1                    # rounds per policy update. 1 = each round's own
                                        # reward (the original runs); H > 1 = discounted
                                        # returns over H rounds with reward.gamma
    coverage_features: bool = False     # give the RL policy two more inputs, recomputed
                                        # after each pick within a round: how much class
                                        # coverage client i would add to the clients
                                        # picked so far, plain and weighted by how
                                        # under-covered each class has been lately
    mode_rule: str = "own"              # own: the policy picks modes itself (original);
                                        # expected_work: every selected client gets the
                                        # mode with the most expected training (success
                                        # prob x local epochs) among modes with success
                                        # prob >= elig_threshold -- gives selection-only
                                        # baselines the same mode choice as the RL
                                        # policy; light: always Light (control)
    oort_estimated_time: bool = False   # simulator fix: Oort's deadline penalty uses the
                                        # round time estimated from observed speed and
                                        # bandwidth, not this round's true time
    # FAVOR (Wang et al., INFOCOM 2020)
    favor_target: float = 0.0           # target accuracy Omega; 0 = the paper's value
                                        # for the dataset (see FAVORPolicy)
    favor_pca: int = 100                # PCA dimensions per model
    favor_hidden: int = 512
    favor_gamma: float = 0.9
    # DELTA (Wang et al., NeurIPS 2023)
    delta_alpha: float = 0.5            # alpha_1 = alpha_2 (the paper's default)
    # REFL (Abdelmoniem et al., EuroSys 2023)
    refl_beta: float = 0.35             # weight of the boosting term in Eq. (5)
    refl_hold: int = 5                  # rounds a client sits out after submitting
    refl_pred_acc: float = 0.9          # accuracy of the availability forecast


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
