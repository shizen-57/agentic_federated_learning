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
