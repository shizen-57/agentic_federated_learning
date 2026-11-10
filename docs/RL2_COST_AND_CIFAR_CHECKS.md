# Two small checks before the ARL screen (gap review A3, B2)

Written before any run. Runner: `scripts/screen_rl2.py`.

## Cost check (`configs/rl2_cost_check.yaml`)

**Why.** The tuned method's soft cost constraint does not reach its target.
In M7 stage 2 it spent 9.4 units per round in the second half, against a
target of 7.5 (K x mean Light cost). Always-Light spent 6.9. The spend
stays above the target all through the run, so this is not a warm-up effect.
The likely cause: the price on cost rises slowly (`lagrange_lr` 0.05), and
PPO's advantage normalisation removes most of a near-constant penalty.

**Design.** FEMNIST, seeds 280-284, development pool, 200 rounds:
- `base`: the tuned method as it is;
- `fast_dual`: the price adjusts 10x faster (`policy.lagrange_lr` 0.5);
- `cap_target`: a hard per-round cap at the target (`policy.hard_budget_mult` 1.0).

**Rule.**
- An arm meets the target if its median spend per round in the second half
  is at most 1.05 x the target.
- If the base meets it, nothing changes.
- Otherwise, among the variants that meet it and lose at most 0.01 of the
  equal-spend score against the base (paired mean), the one with the highest
  score is adopted.
- If none qualifies, the base is kept, and the thesis states that the soft
  constraint does not bind (the equal-spend endpoints keep the comparisons
  fair either way).

`python scripts/finalize_tuned.py cost --spec configs/rl2_cost_check.yaml
--out <run folder>` re-checks the evidence and writes
`configs/rl2_tuned_cost.json`, which the ARL screen uses.

## CIFAR-10 check of the tuning (`configs/rl2_cifar_check.yaml`)

**Why.** M7 tuned on FEMNIST only; CIFAR-10 gets the same settings.

**Design.** M7's top 5 candidates and the untuned base c00, on CIFAR-10,
seeds 285-289, development pool.

**Rule.** The tuning "holds" on CIFAR-10 if the selected candidate (c34)
scores at least as well as the untuned base (M7's score, paired mean).
Nothing is re-selected, because the comparators are also tuned on FEMNIST
only: both sides are treated alike. The verdict, and c34's rank among the 6,
go into the thesis.

## Order

1. Cost check, then `finalize_tuned.py cost` (commit the new file).
2. ARL screen on `configs/rl2_tuned_cost.json`, then `finalize_tuned.py arl`
   (commit `configs/rl2_tuned_final.json`).
3. Sensitivity screen and screen C on `configs/rl2_tuned_final.json`.

The CIFAR-10 check and the comparator tuning
(`docs/RL2_BASELINE_TUNING.md`) do not depend on these and can run at any
time. The sensitivity screen and screen C need the tuned comparators.

## Outcome (run 2026-10-30 on the RTX 3070, commit 2812614, re-checked here)

Evidence complete in both checks: every run present and valid, one code
version (a1be0f1e34704d16), clean checkout. My re-run of each report gave
the same verdict as the server's.

**Cost check (`results/rl2_cost_check`).** Spend per round in the second half,
as a multiple of the soft target, per seed (280-284):

| arm | per seed | median | meets (<= 1.05) | total spend (median) | score_eq vs base |
|---|---|---|---|---|---|
| base | 1.01, 0.98, 1.35, 1.95, 1.00 | 1.007 | yes | 1,680 | - |
| fast_dual | 0.97, 1.18, 1.60, 1.65, 1.61 | 1.599 | no | 2,292 | -0.009 |
| cap_target | 0.95, 0.87, 0.95, 0.95, 0.95 | 0.949 | yes | 1,422 | +0.005 |

- **Verdict, by the rule: the base already meets its target; it is kept.**
  `configs/rl2_tuned_cost.json` = M7's settings, unchanged (with provenance).
- Stated plainly, because the median hides it: on 2 of 5 seeds the soft
  constraint fails badly (seed 283 spends 1.95 x the target). Only the hard
  cap at the target held it on every seed, and it also scored slightly
  higher. The rule looked at the median and was fixed before the runs, so
  it is applied as written; switching after seeing the results would be the
  after-the-fact choice the rules exist to prevent. The thesis must not
  claim that the soft constraint holds spending; the equal-spend endpoints
  keep the comparisons fair either way.
- A 10x faster price adjustment made spending worse, not better (likely
  overshooting and oscillation of the multiplier).

**CIFAR-10 check (`results/rl2_cifar_check`).** Seeds 285-289:

| candidate | M7 score | score_eq | eq5 | w10b |
|---|---|---|---|---|
| c34 (selected) | 0.4717 | 0.4423 | 0.522 | 0.345 |
| c20 | 0.4712 | 0.4504 | 0.532 | 0.378 |
| c17 | 0.4690 | 0.4461 | 0.531 | 0.366 |
| c32 | 0.4680 | 0.4483 | 0.534 | 0.365 |
| c33 | 0.4656 | 0.4528 | 0.527 | 0.374 |
| c00 (untuned) | 0.4645 | 0.4430 | 0.521 | 0.378 |

- **Verdict, by the rule: the tuning holds on CIFAR-10.** c34 ranks first of
  six on M7's score and beats the untuned base by +0.007 (3 of 5 seeds).
- A caution for the confirmation: on the equal-spend fairness endpoint `w10b`,
  c34 is the weakest of the six on CIFAR-10 (0.345 vs 0.378 untuned). Five
  seeds and small differences, so a warning, not a finding.
