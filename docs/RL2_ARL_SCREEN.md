# RL2 ARL screen: does the contribution-aware reward add value?

Written before any run (fourth external review, finding 3). Spec:
`configs/rl2_arl_screen.yaml`. Runner: `scripts/screen_rl2.py`.

## Why

The thesis is about a contribution-aware reward (ARL). But the method that M7
selected, c34, uses **no ARL**: its damage term is the plain failure count.
So the central claim is open. This screen answers it in the simplest fair way:
take the tuned method and change **only** the damage term.

## Design

- Base: the tuned method (`configs/rl2_tuned.json`, c34), from scratch (no
  pre-training, as in M7), on the persistent split and the **development**
  pool (`data.eval_pool: dev`). The confirmation pool is never touched.
- Three arms, everything else identical:
  - `plain`: the failure count (`raw_fail`, no per-client credit) = c34 itself;
  - `arl3`: ARL-v3 (lost work, Shapley share) with per-client credit;
  - `arl4`: ARL-v4 (set-level, within the cost constraint) with per-client credit.
- Datasets: FEMNIST and CIFAR-10. Seeds 250-269 (20, development, never used).
  200 rounds. 120 runs.
- Same seeds for every arm (paired comparison; common random numbers).

**One caveat, stated in advance.** c34 was tuned with ARL switched off, so
its ARL weights (`reward.lambda_arl` 0.5, `reward.arl4_mu` 2.0) were drawn by
the random search but never used. In the ARL arms they become active at those
values. This screen therefore asks "does switching ARL on help the tuned
method?", not "does a separately tuned ARL method beat a separately tuned plain
one?". The second question is answered below from M7 with a matched tuning
budget.

## Rule (fixed before the runs)

- Evidence first: every run present, one code version, clean checkout,
  recorded hashes (`rl2lib.check_evidence`). Otherwise no decision.
- Per dataset, each ARL arm minus `plain`, paired by seed, on the two
  development endpoints `eq5` (accuracy at equal spend) and `w10` (worst-10%
  clients). Two-sided paired sign-flip test; Holm over all 8 tests
  (2 arms x 2 endpoints x 2 datasets) at alpha 0.05; 95% bootstrap interval of
  the mean difference.
- Verdict per ARL version:
  - **adds value**: Holm-significantly better than plain on at least one
    endpoint and dataset, and significantly worse on none;
  - **makes it worse**: significantly worse somewhere, better nowhere;
  - **mixed**: both;
  - **no evidence that it adds value**: neither.
- What follows:
  - If an ARL version adds value, it stays in the method and screen C's
    final method is re-examined with it before any freeze.
  - Otherwise ARL is **not** part of the final method. The thesis then reports
    honestly that the contribution-aware reward did not improve the tuned
    scheduler in this setting, and the claim rests on the other components.
  - With 20 seeds, a true difference of about 0.02 to 0.025 in `eq5` or
    `w10` is found with 80% power even at Holm's strictest step (paired
    spread 0.022-0.030 between M7 stage-2 candidates). A smaller effect may
    be missed; "no evidence" is not "no effect", and the report says so.

## Matched tuning budget (M7, exploratory, development evidence)

M7's random search drew the damage term at random, so the three options got
different numbers of candidates: ARL-v3 19, ARL-v4 10, failure count 9 (an
11th ARL-v4 candidate, c02, starved a seed and is out by the M7 rule). More
candidates means a better best score by luck alone. To compare fairly, the
expected best stage-1 score from **9** candidates (the smallest count) is
computed for each: the exact mean of the best of 9 candidates drawn without
replacement from that option's candidates:

| damage term | candidates | best | median | expected best of 9 |
|---|---|---|---|---|
| ARL-v3 | 19 | 0.6416 | 0.6307 | 0.6404 |
| ARL-v4 | 10 | 0.6448 | 0.6321 | 0.6441 |
| failure count | 9 | 0.6486 | 0.6291 | 0.6486 |

With equal tuning effort, the failure count is still ahead, by 0.005 (v4) and
0.008 (v3) in score. These are 5-seed stage-1 scores, and the differences
are within the spread of the top candidates, so this is weak evidence in either
direction. That is why the controlled screen above is needed.

## Outcome

Not run yet. To run on the department GPU:
`bash scripts/dept_gpu.sh screen2 configs/rl2_arl_screen.yaml`.
