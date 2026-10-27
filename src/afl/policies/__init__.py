from .baselines import (RandomPolicy, FedCSPolicy, PowerOfChoicePolicy, OortPolicy,
                        RoundRobinPolicy, FedCERPolicy, FedCGPolicy,
                        GreedyCoveragePolicy)
from .published import FAVORPolicy, DELTAPolicy, REFLPolicy
from .rl import RLPolicy
from .safe_rl import SafePPOPolicy

REGISTRY = {
    "random": RandomPolicy,
    "fedcs": FedCSPolicy,
    "powerofchoice": PowerOfChoicePolicy,
    "oort": OortPolicy,
    "roundrobin": RoundRobinPolicy,
    "fedcer": FedCERPolicy,
    "fedcg": FedCGPolicy,
    "greedycov": GreedyCoveragePolicy,
    "favor": FAVORPolicy,
    "delta": DELTAPolicy,
    "refl": REFLPolicy,
    "rl": RLPolicy,
    "safe_ppo": SafePPOPolicy,
}


def build_policy(cfg):
    name = cfg.policy.name
    if name not in REGISTRY:
        raise ValueError(f"unknown policy {name}; have {sorted(REGISTRY)}")
    return REGISTRY[name](cfg)
