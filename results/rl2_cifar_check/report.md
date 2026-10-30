# RL2 CIFAR-10 check of the FEMNIST tuning

Evidence complete: 30 runs, one code version.

| dataset | setting | arm | eq5 | w10b | w10 (whole run) | acc10 | spend | starved |
|---|---|---|---|---|---|---|---|---|
| cifar | - | c00 | 0.521 | 0.378 | 0.410 | 0.564 | 1878 | 0/5 |
| cifar | - | c17 | 0.531 | 0.366 | 0.418 | 0.558 | 1810 | 0/5 |
| cifar | - | c20 | 0.532 | 0.378 | 0.413 | 0.559 | 1766 | 0/5 |
| cifar | - | c32 | 0.534 | 0.365 | 0.398 | 0.559 | 1714 | 0/5 |
| cifar | - | c33 | 0.527 | 0.374 | 0.405 | 0.554 | 1644 | 0/5 |
| cifar | - | c34 | 0.522 | 0.345 | 0.413 | 0.568 | 1720 | 0/5 |

## Do M7's settings (tuned on FEMNIST) hold on cifar?

| candidate | M7 score | score_eq | eq5 | w10b |
|---|---|---|---|---|
| c34 (selected) | 0.4717 | 0.4423 | 0.522 | 0.345 |
| c20 | 0.4712 | 0.4504 | 0.532 | 0.378 |
| c17 | 0.4690 | 0.4461 | 0.531 | 0.366 |
| c32 | 0.4680 | 0.4483 | 0.534 | 0.365 |
| c33 | 0.4656 | 0.4528 | 0.527 | 0.374 |
| c00 | 0.4645 | 0.4430 | 0.521 | 0.378 |

Selected minus untuned base, paired: +0.0072 (3/5 seeds); rank 1 of 6.

**Verdict: the tuning holds on cifar** (rule: the selected settings are at least as good as the untuned base). Nothing is re-selected: the baselines are also tuned on FEMNIST only, so both sides are treated alike.
