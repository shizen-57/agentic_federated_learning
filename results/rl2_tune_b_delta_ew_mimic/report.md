# Matched tuning of b_delta_ew_mimic

Evidence complete in both stages (every run checked).

## Stage 1 (seeds 220-224)

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|
| c06 | 5 | 0.6002 | 0.725 | 0.476 | 0.753 | 0 |
| c17 | 5 | 0.5962 | 0.727 | 0.466 | 0.747 | 0 |
| c16 | 5 | 0.5943 | 0.717 | 0.472 | 0.756 | 0 |
| c09 | 5 | 0.5920 | 0.714 | 0.470 | 0.749 | 0 |
| c15 | 5 | 0.5914 | 0.720 | 0.463 | 0.750 | 0 |
| c12 | 5 | 0.5869 | 0.725 | 0.449 | 0.751 | 0 |
| c10 | 5 | 0.5866 | 0.720 | 0.453 | 0.749 | 0 |
| c11 | 5 | 0.5860 | 0.719 | 0.452 | 0.751 | 0 |
| c05 | 5 | 0.5858 | 0.714 | 0.458 | 0.746 | 0 |
| c00 | 5 | 0.5846 | 0.712 | 0.457 | 0.745 | 0 |
| c04 | 5 | 0.5845 | 0.718 | 0.451 | 0.745 | 0 |
| c18 | 5 | 0.5836 | 0.718 | 0.449 | 0.752 | 0 |
| c02 | 5 | 0.5829 | 0.703 | 0.463 | 0.753 | 0 |
| c13 | 5 | 0.5808 | 0.693 | 0.468 | 0.749 | 0 |
| c01 | 5 | 0.5806 | 0.695 | 0.466 | 0.748 | 0 |
| c08 | 5 | 0.5796 | 0.704 | 0.455 | 0.752 | 0 |
| c03 | 5 | 0.5772 | 0.704 | 0.450 | 0.748 | 0 |
| c07 | 5 | 0.5769 | 0.693 | 0.461 | 0.747 | 0 |
| c14 | 5 | 0.5741 | 0.703 | 0.446 | 0.745 | 0 |

References:

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|

Top 5 (qualified): c06, c17, c16, c09, c15

## Stage 2 (seeds 225-239)

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|
| c06 | 15 | 0.6099 | 0.736 | 0.483 | 0.762 | 0 |
| c16 | 15 | 0.6037 | 0.724 | 0.483 | 0.762 | 0 |
| c17 | 15 | 0.6022 | 0.732 | 0.472 | 0.755 | 0 |
| c00 | 15 | 0.5999 | 0.719 | 0.481 | 0.758 | 0 |
| c09 | 15 | 0.5980 | 0.718 | 0.478 | 0.756 | 0 |
| c15 | 15 | 0.5933 | 0.715 | 0.471 | 0.755 | 0 |

References:

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|

**Selected: c06**

Settings: `{"policy.delta_alpha": 0.25, "policy.elig_threshold": 0.6, "policy.gate_mode": "thompson", "policy.pred_decay": 0.98, "policy.predictor": "v2"}`

## Exploratory (no decision): stage-1 mean score by option, c01-c18

- policy.delta_alpha: 0.25: 0.5852 (n=6); 0.5: 0.5847 (n=6); 0.75: 0.5867 (n=6)
- policy.elig_threshold: 0.4: 0.5791 (n=6); 0.5: 0.5877 (n=6); 0.6: 0.5898 (n=6)
- predictor: gate_mode=top, predictor=v1: 0.5852 (n=9); gate_mode=thompson, pred_decay=0.98, predictor=v2: 0.5858 (n=9)
