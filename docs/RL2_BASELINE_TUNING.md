# Matched tuning of the comparators (gap review B1, A7)

Written before any run. Specs: `configs/rl2_tune_<name>.yaml`. Runner:
`scripts/tune_rl2.py` (the same one that tuned our method in M7). GPU:
`bash scripts/dept_gpu.sh tunebase <name>` or `tunebase all`.

## Why

Our method got a 40-candidate search (M7). The comparators got none. A win
could then come from tuning effort, not from the method. Also, every
`_ew` baseline used the **original** success predictor, which the
diagnosis found faulty (docs/DIAGNOSTIC_FINDINGS.md), while our method uses
the repaired one (predictor v2, Thompson gate, evidence ageing).

## What is tuned

Every comparator that screen C may choose for a primary test, in the
version that aggregates like our method (MimiC; REFL keeps its own
aggregation, which is part of the published method):

| comparator | what its search may change |
|---|---|
| `rl_light_mimic` (always Light, the control) | learning rate, entropy, network size, cost / fairness / damage weights, predictor, threshold |
| `arl3_full_pg_mimic` (the repaired original RL) | the same, plus coverage weight, evidence ageing, gate width |
| `b_oort_ew_mimic` (Oort) | exploration share, deadline penalty, predictor, threshold |
| `b_refl_ew` (REFL) | boosting weight, hold-out rounds, predictor, threshold |
| `b_favor_ew_mimic` (FAVOR) | learning rate, discount, target accuracy, PCA size, predictor, threshold |
| `b_delta_ew_mimic` (DELTA) | its alpha, predictor, threshold (18 combinations: all tried) |
| `b_cov_ew_mimic` (greedy coverage) | predictor, threshold (6 combinations: all tried) |
| `b_fairq_ew_mimic` (fairness queues, after RBCS-F) | promised rate, efficiency weight, predictor, threshold |

"Predictor" means either the original predictor or the repaired one
(gap review A7). "Threshold" is the eligibility threshold that the gate and
the shared mode rule use.

## Rule (the same as M7, for parity)

- Stage 1: the untouched comparator (c00) and up to 39 random candidates
  (every combination when there are fewer), seeds 220-224.
- Stage 2: the best 5 (no starved seed) and c00 on seeds 225-239.
- Ranked by M7's score, (eq5 + w10) / 2, on FEMNIST only, exactly as our
  method was. The winner is the best stage-2 candidate among the top 5.
- Its settings are committed as `configs/rl2_tuned_<name>.json` before
  screen C runs; screen C refuses to run without them.

Parity notes, stated in the thesis:
- Our method was tuned on the old code (whole evaluation pool); the
  comparators are tuned on the new code (development pool). Both are
  development evidence and use the same seeds.
- Both sides are tuned on FEMNIST only and use the same settings on
  CIFAR-10 (`docs/RL2_COST_AND_CIFAR_CHECKS.md` checks how well that holds
  for our method).

## Cost

About 2,050 FEMNIST runs in all (six searches of 290 runs, plus 185 for
DELTA and 125 for greedy coverage). At about 4 runs a minute on the RTX 3070
that is roughly 8-9 hours. The searches are independent and resumable, so
they can be spread over several sessions.

## Outcome (run 2026-10-30 on the RTX 3070, commit 2812614, re-checked here)

All 8 searches: evidence complete (every run present and valid, one code
version, clean checkout); the recorded stage-1 rankings match a fresh
re-ranking; the winners and their numbers are identical to the server's.
Settings committed as `configs/rl2_tuned_<name>.json`. Stage 2 (seeds
225-239, FEMNIST), M7's score:

| comparator | untuned (c00) | tuned | tuning gain | tuned eq5 | tuned w10 | tuned w10b | spend | predictor chosen |
|---|---|---|---|---|---|---|---|---|
| fairness queues `b_fairq_ew_mimic` | 0.6290 | **0.6532** | +0.024 | 0.769 | 0.537 | 0.508 | 2,078 | repaired (v2) |
| always-Light `rl_light_mimic` (control) | 0.6251 | **0.6519** | +0.027 | 0.784 | 0.520 | 0.513 | 1,468 | original (v1) |
| REFL `b_refl_ew` | 0.6425 | 0.6491 | +0.007 | 0.778 | 0.520 | 0.499 | 1,815 | repaired |
| Oort `b_oort_ew_mimic` | 0.6323 | 0.6430 | +0.011 | 0.757 | 0.529 | 0.493 | 2,244 | repaired |
| FAVOR `b_favor_ew_mimic` | 0.6186 | 0.6334 | +0.015 | 0.750 | 0.517 | 0.494 | 2,201 | repaired |
| original RL `arl3_full_pg_mimic` | 0.6216 | 0.6324 | +0.011 | 0.749 | 0.515 | 0.481 | 2,161 | repaired (fixed) |
| greedy coverage `b_cov_ew_mimic` | 0.6004 | 0.6157 | +0.015 | 0.745 | 0.487 | 0.470 | 2,121 | original |
| DELTA `b_delta_ew_mimic` | 0.5999 | 0.6099 | +0.010 | 0.736 | 0.483 | 0.438 | 1,979 | repaired |

**What this suggests (exploratory, development evidence).** Our tuned method
c34 scored 0.6378 on these same seeds in M7. With the same aggregation and
the same tuning effort, four comparators score higher, including the
always-Light control (0.652, at the lowest spend). This is not yet a fair
head-to-head: c34's number comes from the old code and the old data pool,
and every winner is the best of six on these seeds. Screen C makes the
paired comparison on fresh seeds and the same code. But it suggests that
most of c34's earlier lead over always-Light (+0.036 equal-spend accuracy in
M7) came from MimiC and the extra tuning -- the two gaps the review found.
