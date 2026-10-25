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
    # c2 with ARL' counted only over the invited clients: separates "value-
    # weight the policy's own failures" from Eq. (3.27)'s never-invited term,
    # which is ~90% of ARL' at N=100/K=10 (docs/ADVANCED_EXPERIMENTS.md).
    "ablation_arlsel":       {"policy.name": "rl", "fl.aggregator": "fedprox",
                              "reward.damage_metric": "arl_selected"},

    # --- Parts of Chapter 3 that the original runs left switched off ---------
    # Multi-round credit: discounted returns over 10-round windows (gamma 0.95).
    "c2_gamma":              {"policy.name": "rl", "fl.aggregator": "fedprox", "policy.horizon": 10},
    # Eq. (3.54): reweight received updates by n_i / p_succ (floor 0.1).
    "c2_reweight":           {"policy.name": "rl", "fl.aggregator": "fedprox", "fl.importance_reweight": True},
    # Semi-asynchronous: late updates within the staleness bound still count.
    "c2_async":              {"policy.name": "rl", "fl.aggregator": "fedprox", "fl.async_late": True},
    # Fairness debt (Sec. 3.13) as an extra reward term and policy input.
    "c2_debt":               {"policy.name": "rl", "fl.aggregator": "fedprox", "reward.lambda_debt": 0.3},
    # Eq. (3.64)-(3.65) budgets, applied identically to every policy compared.
    "c2_budget":             {"policy.name": "rl", "fl.aggregator": "fedprox", "system.cost_max": 10.0, "system.latency_max": 1.0},
    "b_random_budget":       {"policy.name": "random", "fl.aggregator": "fedprox", "system.cost_max": 10.0, "system.latency_max": 1.0},
    "b_oort_budget":         {"policy.name": "oort", "fl.aggregator": "fedprox", "system.cost_max": 10.0, "system.latency_max": 1.0},

    # --- Published baselines implemented from the papers' algorithms ---------
    # MIFA (Gu et al., NeurIPS 2021) and MimiC (Sun et al., IEEE TMC 2024) are
    # aggregation rules for unavailable clients; both papers sample clients
    # without a scheduler, so they run under random selection here.
    "b_mifa":                {"policy.name": "random", "fl.aggregator": "mifa"},
    "b_mimic":               {"policy.name": "random", "fl.aggregator": "mimic"},
    # FedCG (arXiv 2212.09483): diverse selection + per-client compression.
    "b_fedcg":               {"policy.name": "fedcg", "fl.aggregator": "fedcg"},
    # Follow-ups after the 1,190-run sweep (docs/ADVANCED_RESULTS.md):
    # prevention AND correction together: RL selection + MimiC aggregation.
    "c2_mimic":              {"policy.name": "rl", "fl.aggregator": "mimic"},
    # FedCG fitted to 70% of the deadline, leaving slack for timing jitter.
    "b_fedcg_m07":           {"policy.name": "fedcg", "fl.aggregator": "fedcg",
                              "policy.fedcg_margin": 0.7},

    # --- Fair re-test of ARL: the policy sees the value ARL rewards ---------
    # The original value input is anti-correlated with ARL's value, so no ARL
    # reward could teach the policy whom to protect. Same four rewards with
    # the corrected input (policy.value_feature = marginal).
    "v2_c2":                 {"policy.name": "rl", "fl.aggregator": "fedprox", "policy.value_feature": "marginal"},
    "v2_rawfail":            {"policy.name": "rl", "fl.aggregator": "fedprox", "policy.value_feature": "marginal", "reward.damage_metric": "raw_fail"},
    "v2_arlsel":             {"policy.name": "rl", "fl.aggregator": "fedprox", "policy.value_feature": "marginal", "reward.damage_metric": "arl_selected"},
    "v2_arl0":               {"policy.name": "rl", "fl.aggregator": "fedprox", "policy.value_feature": "marginal", "reward.lambda_arl": 0.0},

    # --- Reward-weight sensitivity (Sec. 3.18); c2 uses lambda_arl 0.5, lambda_fair 0.3
    "sens_arl0":             {"policy.name": "rl", "fl.aggregator": "fedprox", "reward.lambda_arl": 0.0},
    "sens_arl025":           {"policy.name": "rl", "fl.aggregator": "fedprox", "reward.lambda_arl": 0.25},
    "sens_arl1":             {"policy.name": "rl", "fl.aggregator": "fedprox", "reward.lambda_arl": 1.0},
    "sens_arl2":             {"policy.name": "rl", "fl.aggregator": "fedprox", "reward.lambda_arl": 2.0},
    "sens_fair0":            {"policy.name": "rl", "fl.aggregator": "fedprox", "reward.lambda_fair": 0.0},
    "sens_fair01":           {"policy.name": "rl", "fl.aggregator": "fedprox", "reward.lambda_fair": 0.1},
    "sens_fair1":            {"policy.name": "rl", "fl.aggregator": "fedprox", "reward.lambda_fair": 1.0},
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
