from .baselines import (RandomPolicy, FedCSPolicy, PowerOfChoicePolicy, OortPolicy,
                        RoundRobinPolicy, FedCERPolicy)
from .rl import RLPolicy

REGISTRY = {
    "random": RandomPolicy,
    "fedcs": FedCSPolicy,
    "powerofchoice": PowerOfChoicePolicy,
    "oort": OortPolicy,
    "roundrobin": RoundRobinPolicy,
    "fedcer": FedCERPolicy,
    "rl": RLPolicy,
}


def build_policy(cfg):
    name = cfg.policy.name
    if name not in REGISTRY:
        raise ValueError(f"unknown policy {name}; have {sorted(REGISTRY)}")
    return REGISTRY[name](cfg)
