# Exploratory (after the rule; not a decision)

| condition | median last-10 acc | median last-10 worst-10% | median wobble (sd, last 20) |
|---|---|---|---|
| arl3_full | 0.715 | 0.428 | 0.027 |
| arl3_full_pg | 0.783 | 0.513 | 0.011 |
| ppo_full | 0.777 | 0.513 | 0.015 |
| ppo_full_arl3 | 0.775 | 0.523 | 0.014 |
| ppo_full_fixcost | 0.768 | 0.509 | 0.016 |
| ppo_full_mimic | 0.785 | 0.554 | 0.012 |
| ppo_full_mlp | 0.776 | 0.508 | 0.015 |
| ppo_full_nocap | 0.777 | 0.531 | 0.014 |
| ppo_full_nocurr | 0.771 | 0.525 | 0.021 |
| ppo_full_nodecay | 0.770 | 0.539 | 0.016 |
| ppo_full_nomask | 0.761 | 0.488 | 0.023 |
| ppo_full_nomem | 0.777 | 0.501 | 0.019 |
| ppo_full_noprot | 0.765 | 0.501 | 0.021 |
| ppo_full_pv1 | 0.152 | 0.004 | 0.000 |
| ppo_full_rawfail | 0.750 | 0.505 | 0.013 |
| ppo_full_top | 0.739 | 0.489 | 0.023 |
| ppo_v3 | 0.723 | 0.459 | 0.016 |
| rl_light | 0.767 | 0.508 | 0.013 |
| rl_light_pg | 0.720 | 0.482 | 0.020 |
| v2_rawfail | 0.721 | 0.463 | 0.015 |

## ppo_full_mimic minus rl_light (seeds 215-219)

| measure | per seed | median | wins | sign-flip p |
|---|---|---|---|---|
| equal-spend acc | +0.002 -0.013 +0.020 +0.013 -0.021 | +0.002 | 3/5 | 0.938 |
| final acc | +0.056 +0.025 +0.059 +0.004 -0.306 | +0.025 | 4/5 | 1.000 |
| final worst-10% | +0.040 +0.050 +0.083 +0.077 -0.396 | +0.050 | 4/5 | 1.000 |
| last-10 mean acc | +0.029 +0.030 +0.044 +0.028 -0.026 | +0.029 | 4/5 | 0.125 |
| last-10 mean worst-10% | +0.020 +0.052 +0.069 +0.054 -0.036 | +0.052 | 4/5 | 0.188 |

## ppo_full_mimic minus ppo_full (seeds 215-219)

| measure | per seed | median | wins | sign-flip p |
|---|---|---|---|---|
| equal-spend acc | +0.047 +0.003 +0.013 +0.016 +0.004 | +0.013 | 5/5 | 0.062 |
| final acc | +0.033 +0.021 +0.030 +0.018 -0.214 | +0.021 | 4/5 | 1.000 |
| final worst-10% | -0.001 +0.078 +0.034 +0.110 -0.175 | +0.034 | 3/5 | 0.875 |
| last-10 mean acc | +0.020 +0.016 +0.017 +0.018 -0.018 | +0.017 | 4/5 | 0.250 |
| last-10 mean worst-10% | +0.012 +0.051 +0.017 +0.092 -0.033 | +0.017 | 4/5 | 0.312 |

## ppo_full_mimic minus arl3_full_pg (seeds 215-219)

| measure | per seed | median | wins | sign-flip p |
|---|---|---|---|---|
| equal-spend acc | +0.014 +0.046 -0.003 +0.045 +0.049 | +0.045 | 4/5 | 0.125 |
| final acc | +0.002 +0.035 +0.028 +0.010 -0.305 | +0.010 | 4/5 | 1.000 |
| final worst-10% | -0.004 +0.060 +0.048 +0.111 -0.353 | +0.048 | 3/5 | 0.938 |
| last-10 mean acc | +0.004 +0.023 +0.010 +0.011 -0.020 | +0.010 | 4/5 | 0.438 |
| last-10 mean worst-10% | +0.042 +0.048 +0.017 +0.072 -0.022 | +0.042 | 4/5 | 0.188 |
