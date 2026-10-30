# Matched tuning of b_cov_ew_mimic

Evidence complete in both stages (every run checked).

## Stage 1 (seeds 220-224)

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|
| c00 | 5 | 0.6152 | 0.749 | 0.481 | 0.755 | 0 |
| c03 | 5 | 0.6124 | 0.748 | 0.477 | 0.756 | 0 |
| c04 | 5 | 0.6097 | 0.740 | 0.480 | 0.756 | 0 |
| c01 | 5 | 0.6097 | 0.733 | 0.486 | 0.756 | 0 |
| c02 | 5 | 0.6076 | 0.740 | 0.475 | 0.764 | 0 |
| c06 | 5 | 0.5948 | 0.734 | 0.456 | 0.740 | 0 |
| c05 | 5 | 0.5909 | 0.724 | 0.457 | 0.729 | 0 |

References:

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|

Top 5 (qualified): c00, c03, c04, c01, c02

## Stage 2 (seeds 225-239)

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|
| c01 | 15 | 0.6157 | 0.745 | 0.487 | 0.754 | 0 |
| c02 | 15 | 0.6150 | 0.739 | 0.491 | 0.769 | 0 |
| c04 | 15 | 0.6098 | 0.739 | 0.480 | 0.765 | 0 |
| c03 | 15 | 0.6038 | 0.721 | 0.486 | 0.740 | 0 |
| c00 | 15 | 0.6004 | 0.717 | 0.484 | 0.736 | 0 |

References:

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|

**Selected: c01**

Settings: `{"policy.elig_threshold": 0.4, "policy.gate_mode": "top", "policy.predictor": "v1"}`

## Exploratory (no decision): stage-1 mean score by option, c01-c06

- policy.elig_threshold: 0.4: 0.6086 (n=2); 0.5: 0.6111 (n=2); 0.6: 0.5928 (n=2)
- predictor: gate_mode=top, predictor=v1: 0.6043 (n=3); gate_mode=thompson, pred_decay=0.98, predictor=v2: 0.6040 (n=3)
