# RL2 M7 tuning (random search, then the best 5 on fresh seeds)

Evidence complete in both stages (every run checked).

## Stage 1 (seeds 220-224)

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|
| c17 | 5 | 0.6486 | 0.767 | 0.530 | 0.784 | 0 |
| c33 | 5 | 0.6448 | 0.771 | 0.519 | 0.774 | 0 |
| c34 | 5 | 0.6420 | 0.766 | 0.518 | 0.776 | 0 |
| c20 | 5 | 0.6416 | 0.765 | 0.518 | 0.776 | 0 |
| c32 | 5 | 0.6412 | 0.766 | 0.516 | 0.779 | 0 |
| c11 | 5 | 0.6392 | 0.757 | 0.521 | 0.771 | 0 |
| c06 | 5 | 0.6382 | 0.763 | 0.513 | 0.777 | 0 |
| c31 | 5 | 0.6372 | 0.759 | 0.515 | 0.771 | 0 |
| c05 | 5 | 0.6371 | 0.760 | 0.514 | 0.776 | 0 |
| c26 | 5 | 0.6370 | 0.760 | 0.513 | 0.768 | 0 |
| c25 | 5 | 0.6369 | 0.754 | 0.519 | 0.771 | 0 |
| c16 | 5 | 0.6358 | 0.755 | 0.517 | 0.775 | 0 |
| c15 | 5 | 0.6357 | 0.758 | 0.513 | 0.775 | 0 |
| c37 | 5 | 0.6353 | 0.768 | 0.503 | 0.758 | 0 |
| c00 | 5 | 0.6348 | 0.752 | 0.518 | 0.773 | 0 |
| c29 | 5 | 0.6344 | 0.754 | 0.515 | 0.768 | 0 |
| c12 | 5 | 0.6339 | 0.759 | 0.509 | 0.773 | 0 |
| c38 | 5 | 0.6317 | 0.765 | 0.499 | 0.777 | 0 |
| c36 | 5 | 0.6316 | 0.751 | 0.513 | 0.769 | 0 |
| c24 | 5 | 0.6307 | 0.753 | 0.509 | 0.772 | 0 |
| c30 | 5 | 0.6302 | 0.749 | 0.512 | 0.766 | 0 |
| c22 | 5 | 0.6302 | 0.759 | 0.501 | 0.766 | 0 |
| c10 | 5 | 0.6301 | 0.755 | 0.506 | 0.771 | 0 |
| c13 | 5 | 0.6300 | 0.745 | 0.515 | 0.764 | 0 |
| c23 | 5 | 0.6296 | 0.760 | 0.500 | 0.773 | 0 |
| c19 | 5 | 0.6291 | 0.747 | 0.511 | 0.755 | 0 |
| c28 | 5 | 0.6289 | 0.750 | 0.508 | 0.764 | 0 |
| c14 | 5 | 0.6283 | 0.747 | 0.510 | 0.761 | 0 |
| c35 | 5 | 0.6274 | 0.745 | 0.509 | 0.769 | 0 |
| c08 | 5 | 0.6257 | 0.739 | 0.512 | 0.757 | 0 |
| c21 | 5 | 0.6239 | 0.740 | 0.508 | 0.769 | 0 |
| c27 | 5 | 0.6233 | 0.743 | 0.504 | 0.763 | 0 |
| c09 | 5 | 0.6223 | 0.755 | 0.490 | 0.751 | 0 |
| c01 | 5 | 0.6218 | 0.746 | 0.498 | 0.757 | 0 |
| c39 | 5 | 0.6213 | 0.729 | 0.513 | 0.765 | 0 |
| c18 | 5 | 0.6205 | 0.748 | 0.493 | 0.758 | 0 |
| c04 | 5 | 0.6194 | 0.745 | 0.494 | 0.744 | 0 |
| c07 | 5 | 0.6185 | 0.743 | 0.494 | 0.753 | 0 |
| c03 | 5 | 0.6130 | 0.740 | 0.486 | 0.753 | 0 |
| c02 | 5 | 0.5106 | 0.612 | 0.409 | 0.634 | 1 |

References:

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|
| rl_light | 5 | 0.5857 | 0.722 | 0.449 | 0.726 | 0 |

Top 5 (qualified): c17, c33, c34, c20, c32

## Stage 2 (seeds 225-239)

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|
| c34 | 15 | 0.6378 | 0.760 | 0.516 | 0.780 | 0 |
| c33 | 15 | 0.6365 | 0.768 | 0.505 | 0.774 | 0 |
| c17 | 15 | 0.6365 | 0.765 | 0.508 | 0.777 | 0 |
| c20 | 15 | 0.6362 | 0.762 | 0.510 | 0.772 | 0 |
| c32 | 15 | 0.6357 | 0.762 | 0.510 | 0.780 | 0 |
| c00 | 15 | 0.6311 | 0.759 | 0.503 | 0.772 | 0 |

References:

| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) | starved seeds |
|---|---|---|---|---|---|---|
| rl_light | 15 | 0.5866 | 0.724 | 0.449 | 0.724 | 0 |

**Selected: c34**

Settings: `{"policy.cost_target_mult": 1.0, "policy.curriculum": false, "policy.encoder": "attention", "policy.gate_z": 1.5, "policy.hard_budget_mult": 3.0, "policy.mode_mask_z": 0.25, "policy.per_client_credit": false, "policy.ppo_batch": 8, "policy.ppo_clip": 0.3, "policy.ppo_ent_start": 0.003, "policy.ppo_epochs": 4, "policy.ppo_gamma": 0.9, "policy.ppo_lr": 0.005, "policy.pred_decay": 0.98, "reward.arl4_mu": 2.0, "reward.beta": 0.0, "reward.damage_metric": "raw_fail", "reward.lambda_arl": 0.5, "reward.lambda_cov": 1.0, "reward.protect_weight": 1.0}`

## Exploratory (no decision): stage-1 mean score by option, c01-c39

- cost: cost_target_mult=1.0, beta=0.0: 0.6335 (n=13); cost_target_mult=1.25, beta=0.0: 0.6134 (n=9); cost_target_mult=1.5, beta=0.0: 0.6314 (n=12); cost_target_mult=0.0, beta=__group__: 0.6327 (n=5)
- damage: per_client_credit=True, damage_metric=arl_v4: 0.6193 (n=11); per_client_credit=True, damage_metric=arl_v3: 0.6316 (n=19); per_client_credit=False, damage_metric=raw_fail: 0.6316 (n=9)
- policy.curriculum: True: 0.6265 (n=21); False: 0.6300 (n=18)
- policy.encoder: attention: 0.6301 (n=21); mlp: 0.6258 (n=18)
- policy.gate_z: 0.5: 0.6282 (n=16); 1.0: 0.6313 (n=13); 1.5: 0.6239 (n=10)
- policy.hard_budget_mult: 0.0: 0.6353 (n=7); 1.5: 0.6335 (n=7); 2.0: 0.6301 (n=15); 3.0: 0.6163 (n=10)
- policy.mode_mask_z: 0.25: 0.6247 (n=15); 0.5: 0.6303 (n=18); 1.0: 0.6301 (n=6)
- policy.ppo_batch: 4: 0.6275 (n=9); 8: 0.6236 (n=17); 16: 0.6345 (n=13)
- policy.ppo_clip: 0.1: 0.6286 (n=9); 0.2: 0.6312 (n=20); 0.3: 0.6216 (n=10)
- policy.ppo_ent_start: 0.003: 0.6239 (n=15); 0.01: 0.6293 (n=13); 0.03: 0.6325 (n=11)
- policy.ppo_epochs: 2: 0.6155 (n=10); 4: 0.6348 (n=14); 8: 0.6304 (n=15)
- policy.ppo_gamma: 0.8: 0.6316 (n=11); 0.9: 0.6297 (n=18); 0.95: 0.6214 (n=10)
- policy.ppo_lr: 0.001: 0.6377 (n=7); 0.002: 0.6351 (n=6); 0.003: 0.6309 (n=10); 0.005: 0.6279 (n=7); 0.01: 0.6132 (n=9)
- policy.pred_decay: 0.95: 0.6218 (n=10); 0.98: 0.6369 (n=11); 0.99: 0.6267 (n=8); 1.0: 0.6259 (n=10)
- reward.arl4_mu: 0.5: 0.6159 (n=10); 1.0: 0.6333 (n=14); 2.0: 0.6314 (n=15)
- reward.lambda_arl: 0.25: 0.6313 (n=11); 0.5: 0.6302 (n=18); 1.0: 0.6209 (n=10)
- reward.lambda_cov: 0.25: 0.6179 (n=12); 0.5: 0.6300 (n=9); 1.0: 0.6341 (n=18)
- reward.protect_weight: 0.0: 0.6298 (n=11); 0.25: 0.6296 (n=11); 0.5: 0.6321 (n=3); 1.0: 0.6248 (n=14)
