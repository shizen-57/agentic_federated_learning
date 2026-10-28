# RL2 screen C (M8): pre-trained or not, and against what

Written and committed before any run. Development step; exploratory.
Spec: `configs/rl2_screen_c.yaml`. Runner: `scripts/screen_c_rl2.py`
(`bash scripts/dept_gpu.sh screen-c-pretrain`, then `screen-c`).

**Re-run on the development pool (added 2026-10-28, before any screen C
result was read).** The first screen C ran on the old code (a442febc), which
drew FEMNIST clients from the whole evaluation pool and evaluated CIFAR-10 on
every non-validation test example, so it read data that is now reserved for
the confirmation (`docs/RL2_CONFIRMATION.md`, "History"). Its results are
exploratory only. Screen C is run again, unchanged (same spec, same seeds),
on the new code, which evaluates on the development pool; only that run's
`DECISION.json` can feed the freeze (`scripts/freeze_rl2.py` checks it). It
runs after the ARL screen, so that screen's verdict is known first.

**Methods** (each = M7's tuned settings, `configs/rl2_tuned.json`, with the
following on top; a method whose full config equals an earlier one is run
once):

- `tuned`: the tuned settings as they are;
- `tuned_arl4`: with ARL-v4 as the damage term;
- `tuned_noarl`: with the plain failure count instead (no ARL).

**Pre-training** (`screen-c-pretrain`): every method gets its own checkpoint
per dataset. That is 20 iterations x 8 practice episodes of 200 rounds from the
same weights, parameters averaged after each iteration. To fit the 8 GB GPU,
FEMNIST runs 4 and CIFAR-10 3 of the 8 at a time (8 at once ran out of memory
on the first attempt); the result is identical. Pre-training is resumable after
an interruption, again with an identical result. Practice seeds only (FEMNIST
1000-1159, CIFAR-10 1200-1359), on the persistent split's practice pool,
rewards from the validation pool, and the test set never evaluated. A
checkpoint is accepted only if it was made by this simulator code, from
practice episodes, for exactly that method, condition and data group.
Practice cost is recorded and reported next to deployment cost.

**Runs** (both datasets, s2 settings, persistent split, 200 rounds, dev seeds
240-249):

- each method pre-trained (`<method>_pt`) and `tuned` from scratch
  (`tuned_scratch`);
- references: `rl_light`, `rl_light_mimic` (always Light with MimiC, so a
  MimiC gain is not credited to RL), `arl3_full_pg` (the repaired original
  RL), `v2_rawfail`, `b_oort_ew`, `b_refl_ew`, `b_favor_ew`, `b_delta_ew`,
  `b_cov_ew`.

**Measures:** as in M7 (`eq5`, `w10`, score = their mean, starvation).

**Rule:**

1. Evidence first: every run and checkpoint checked, as in M7, plus every
   pre-trained run must have started from its method's checkpoint (by
   SHA-256). Anything missing or invalid: INCOMPLETE, no decision.
2. Final method: `tuned` pre-trained if, over all dataset-seed pairs, the
   median paired score difference (pre-trained minus scratch) is >= 0;
   otherwise `tuned` from scratch. Pre-training must not make it worse.
3. Best baseline for P1: the reference other than `rl_light` with the
   highest median `eq5` (mean of the two datasets' medians).
4. The confirmation's primary tests are then fixed. Per dataset:
   - P1 (`eq5`): final vs `rl_light`, and final vs the best baseline;
   - P2 (`w10`): final vs `rl_light`, and final vs `arl3_full_pg`.

   Screen C only estimates their mean paired difference and spread, for the
   power analysis (M9). It runs no test.
5. Reported, no decision: everything against everything (median paired), the
   ARL contrast (with minus without ARL-v4, pre-trained), the MimiC control,
   and pre-training cost.

Writes `DECISION.json` (used by `scripts/freeze_rl2.py`) and `report.md`.
