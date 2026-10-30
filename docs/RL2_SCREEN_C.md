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

**Version 2 (gap review, 2026-10-29, before any new-code run).** The
rule below replaces version 1 (which is in the git history):
- the method is M7's c34 **plus the cost-check and ARL-screen verdicts**
  (`configs/rl2_tuned_final.json`; A3, A4);
- every comparator aggregates like the method (**MimiC**; REFL keeps its own
  published aggregation) and has been **tuned with the same budget** as the
  method (`docs/RL2_BASELINE_TUNING.md`; A1, A7, B1);
- both endpoints are at **equal spend** (`eq5`, `w10b`; A2);
- P2's second comparator is the best tuned comparator **on fairness**, as
  P1's is on accuracy (A6).

**Methods** (each = the tuned settings in `configs/rl2_tuned_final.json`,
with the following on top; a method whose full config equals an earlier one
is run once):

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
Practice cost is recorded and reported next to deployment cost. Pre-training
does not need the tuned comparators, so it can run while they are tuned.

**Runs** (both datasets, s2 settings, persistent split, development pool,
200 rounds, dev seeds 240-249):

- each method pre-trained (`<method>_pt`) and `tuned` from scratch
  (`tuned_scratch`);
- the control `rl_light_mimic_tuned` (always Light, MimiC, tuned);
- the tuned comparators: `arl3_full_pg_mimic_tuned` (the repaired original
  RL), `b_oort_ew_mimic_tuned`, `b_refl_ew_tuned`, `b_favor_ew_mimic_tuned`,
  `b_delta_ew_mimic_tuned`, `b_cov_ew_mimic_tuned`, `b_fairq_ew_mimic_tuned`
  (fairness queues, after RBCS-F);
- `rl_light` (untuned, FedProx), to show how much of a gain is MimiC's own.

**Measures:** `eq5`, `w10b` (worst-10% client accuracy over the last 10
rounds within the same 1,400 budget), score_eq = their mean, starvation; also
reported: `w10` over the whole run, spend, clients who never contributed,
Jain's index.

**Rule:**

1. Evidence first: every run and checkpoint checked, as in M7, plus every
   pre-trained run must have started from its method's checkpoint (by
   SHA-256), and every tuned comparator's settings file must match its
   recorded hash. Anything missing or invalid: INCOMPLETE, no decision.
2. Final method: `tuned` pre-trained if, over all dataset-seed pairs, the
   median paired score_eq difference (pre-trained minus scratch) is >= 0;
   otherwise `tuned` from scratch. Pre-training must not make it worse.
3. P1's second comparator: the tuned comparator (not the control) with the
   highest median `eq5` (mean of the two datasets' medians). P2's second
   comparator: the same on `w10b`.
4. The confirmation's primary tests are then fixed. Per dataset:
   - P1 (`eq5`): final vs the control, and final vs P1's comparator;
   - P2 (`w10b`): final vs the control, and final vs P2's comparator.

   Screen C only estimates their mean paired difference and spread, for the
   power analysis (M9). It runs no test.
5. Reported, no decision: everything against everything (median paired), the
   ARL contrast (with minus without ARL-v4, pre-trained), MimiC's own share
   (`rl_light`), and pre-training cost.

Writes `DECISION.json` (used by `scripts/freeze_rl2.py`) and `report.md`.
