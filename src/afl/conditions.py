"""Every experimental condition, as config overrides on top of a base config.

One registry so that the Table 3.4 conditions, the Table 3.5 baselines and the
Task 1/2 ablations are all run the same way by every sweep script. The tag is
what the report and stats scripts group on.
"""
from __future__ import annotations

CONDITIONS: dict[str, dict] = {
    # Table 3.4: aggregation-rule ablation under the proposed RL policy, plus
    # the no-intelligence heuristic control.
    "c1_rl_fedavg":          {"policy.name": "rl", "fl.aggregator": "fedavg"},
    "c2_rl_fedprox":         {"policy.name": "rl", "fl.aggregator": "fedprox"},
    "c3_rl_fednova":         {"policy.name": "rl", "fl.aggregator": "fednova"},
    "c4_heuristic_fedprox":  {"policy.name": "roundrobin", "fl.aggregator": "fedprox"},
    # Beyond the original four: FedProx's local proximal term and FedNova's
    # server-side step normalisation act on different parts of the pipeline.
    "c5_rl_fedprox_fednova": {"policy.name": "rl", "fl.aggregator": "fedprox_fednova"},
    # Table 3.5 external baselines, all under FedProx.
    "b_random":              {"policy.name": "random", "fl.aggregator": "fedprox"},
    "b_fedcs":               {"policy.name": "fedcs", "fl.aggregator": "fedprox"},
    "b_powerofchoice":       {"policy.name": "powerofchoice", "fl.aggregator": "fedprox"},
    "b_oort":                {"policy.name": "oort", "fl.aggregator": "fedprox"},
    "b_fedcer":              {"policy.name": "fedcer", "fl.aggregator": "fedprox"},
    # MIFA/MimiC-like: Oort selection plus post-hoc substitution of a failed
    # client's last good update (prevention vs. imputation, Sec. 3.11).
    "b_mimic_impute":        {"policy.name": "oort", "fl.aggregator": "fedprox",
                              "fl.impute_missing": True},
    # Task 1/2 ablations: c2 with one of the thesis's two claims swapped for
    # the naive alternative.
    "ablation_rawfail":      {"policy.name": "rl", "fl.aggregator": "fedprox",
                              "reward.damage_metric": "raw_fail"},
    "ablation_cvsel":        {"policy.name": "rl", "fl.aggregator": "fedprox",
                              "reward.fairness_metric": "cv_selection"},
}


def apply(raw: dict, overrides: dict) -> dict:
    """Apply dotted-key overrides to a nested config dict, in place."""
    for key, val in overrides.items():
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return raw
