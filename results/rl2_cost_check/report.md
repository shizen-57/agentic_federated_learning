# RL2 cost check (does the spend target bind?)

Evidence complete: 15 runs, one code version.

| dataset | setting | arm | eq5 | w10b | w10 (whole run) | acc10 | spend | starved |
|---|---|---|---|---|---|---|---|---|
| femnist | - | base | 0.768 | 0.484 | 0.503 | 0.784 | 1680 | 0/5 |
| femnist | - | fast_dual | 0.764 | 0.482 | 0.522 | 0.782 | 2292 | 0/5 |
| femnist | - | cap_target | 0.776 | 0.509 | 0.514 | 0.771 | 1422 | 0/5 |

## Spend against the soft target (1.0 x K x mean Light cost), second half of the run

| arm | median spend / target | meets target (<= 1.05) | eq5 | w10b | score_eq | score_eq minus base (paired mean) |
|---|---|---|---|---|---|---|
| base | 1.007 | yes | 0.768 | 0.484 | 0.6308 | +0.0000 |
| fast_dual | 1.599 | no | 0.764 | 0.482 | 0.6217 | -0.0092 |
| cap_target | 0.949 | yes | 0.776 | 0.509 | 0.6354 | +0.0045 |

**Verdict: the base already meets its target; it is kept.**
