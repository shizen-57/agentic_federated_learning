# RL2 screen B: the SAFE-PPO stack, one component at a time

Written and committed before any run. Development gate; exploratory.

**Version 2.** Version 1 (a ladder that added several components per step)
was revised after the second external review, before any of its runs: its
steps bundled components, it lacked the "same stack without ARL-v4" and
"repaired simple scheduler" comparisons, and it used per-seed data splits.
No version-1 run exists.

**Conditions** (`configs/rl2_screen_b.yaml`; FEMNIST, s2 settings, 200 rounds,
dev seeds 215-219, from scratch -- no pre-training; every condition on the
persistent data split, `data.split: persistent`):

- references: `arl3_full`, `arl3_full_pg` (screen A's winner), `v2_rawfail`,
  `rl_light` (always Light), `rl_light_pg` (always Light + the same predictor
  v2, Thompson gate and evidence ageing as the full stack: the repaired simple
  scheduler);
- `ppo_v3`: the PPO learner alone (ARL-v3 reward, no other upgrade);
- `ppo_full`: everything (attention, memory, predictor v2 with ageing 0.98,
  Thompson gate, feasible mask z 0.5, cost constraint 1.25, hard cap 2.0,
  curriculum, protection 0.5, ARL-v4 with the set-level objective, beta 0);
- `ppo_full_mimic`: the same with MimiC aggregation;
- leave-one-out, each `ppo_full` minus ONE component: `ppo_full_mlp`
  (attention), `ppo_full_nomem` (memory), `ppo_full_pv1` (predictor v2 +
  Thompson gate, which needs v2), `ppo_full_top` (Thompson gate),
  `ppo_full_nodecay` (evidence ageing), `ppo_full_nomask` (feasible mask),
  `ppo_full_fixcost` (cost constraint; fixed cost weight instead),
  `ppo_full_nocap` (hard cap), `ppo_full_nocurr` (curriculum),
  `ppo_full_noprot` (protection reward), `ppo_full_arl3` (ARL-v4 -> ARL-v3),
  `ppo_full_rawfail` (ARL-v4 -> plain failure count: the same stack without ARL).

20 conditions x 5 seeds = 100 runs. Run on the department RTX 3070
(`bash scripts/dept_gpu.sh screen configs/rl2_screen_b.yaml`) or CPU; the device
is recorded in MANIFEST.json. Run it from a clean checkout and do not edit the
code while it runs.

**Measures:** as in docs/DEV_SCREEN_PREDICTOR_V1.md, plus the final worst-10%
client accuracy (`acc_client_worst10`, the P2 measure). Starvation (changed
before any run, third external review): the longest run of rounds with NO
successful contribution -- all invitations failed, or nobody was invited
(the safety layer may abstain). A round with no client online (`n_online` 0)
is skipped: it neither extends nor breaks a run. The failed-invitation and
empty-round runs, the number of empty rounds and of offline rounds are
reported per seed as well.

**Evidence check (before any decision).** `scripts/report_dev_screen.py --spec`
rebuilds all 100 expected runs from the spec and checks each: the exact
expected config, rounds 1-200 all present with finite values, no source change
during the run; all runs from one code version, made from a clean checkout,
and the report run with that same code. The spec file, the source plan and the
generated plan.yaml must match the hashes recorded in MANIFEST.json, and the
runner may only have been given operational overrides (`fl.device`): no
override can replace a spec setting such as the data split. If anything is
missing or invalid it prints "INCOMPLETE -- no decision" and nothing below is
applied.

**Rule** (also in the spec's `rule` block):

1. *Gate*, for `ppo_full` and `ppo_full_mimic`: no seed starves (50+ rounds
   in a row with no successful contribution, as defined above); median accuracy at equal spend
   (B = 1,400) >= median of `rl_light` - 0.02; median final accuracy >= median
   of `arl3_full` - 0.02.
2. *Carried forward* to pre-training (M6) and tuning (M7): the passing one
   with the higher median equal-spend accuracy. If neither passes, `ppo_full`
   still goes forward: pre-training is part of the design and from-scratch
   learning over 200 rounds is the known weak point (W7); screen C judges the
   pre-trained version on new dev seeds (240-259).
3. *Comparisons* (reported, no decision): median over seeds of the paired
   difference (condition - `ppo_full`) in equal-spend accuracy, final accuracy
   and worst-10% accuracy, for every reference, `ppo_v3` and `ppo_full_mimic`.
4. *Leave-one-out*: for each removed component, the median paired difference
   (minus-one - `ppo_full`) in equal-spend accuracy and in worst-10% accuracy.
   A component is a *drop candidate* if its removal starves no seed and raises
   one of the two by >= 0.01 while lowering neither by more than 0.01. Drop
   candidates are decided in tuning (M7) on seeds 220-239, not here; every
   other component is kept.

Results are reported for every seed either way; nothing is tuned on seeds
215-219.

Report: `python scripts/report_dev_screen.py <out>/full/s2_femnist --spec
configs/rl2_screen_b.yaml` (the GPU script writes it to `<out>/full/report.md`).

## Run log

- **Attempt 1 (2026-10-28, dept RTX 3070, commit ec4866f): void.** 65 of 100
  runs crashed at their first round -- every condition with the feasible-mode
  mask -- because the server's torch (2.5.1, the newest build for its CUDA
  12.1 driver) cannot build a tensor from a list of numpy bools; newer torch
  can, so the tests here passed. The 35 runs that finished were not looked
  at. Fix: the masks are built from plain bools; a test now fails if numpy
  bools reach `torch.tensor`; the GPU script runs every condition for two
  rounds (`scripts/smoke_screen.py`, seed 900) before a screen starts; the
  suite is also run on torch 2.5.1. The whole screen is re-run from scratch
  with the fixed code (one code version for all 100 runs, as the evidence
  check requires); attempt 1's folder is kept aside, unused.
- **Attempt 2 (2026-10-28, dept RTX 3070, commit 61eff94): all 100 runs
  finished, none failed.** The report then said INCOMPLETE for every run
  ("different config"): the runner had saved each run's output folder as a
  relative path (`runs/rl2_screen_b/full/s2_femnist`), the report rebuilt it
  as an absolute one, and compared the two strings. Every other setting
  matched (reproduced here end to end). Fix: the report leaves the output
  folder out of the config comparison -- the files are found at the expected
  place, and a moved or unpacked results folder is the same run. The fix is
  in scripts/ only, so the simulator code (and its hash) is unchanged and the
  runs stand; only the report is run again. A new end-to-end test runs the
  real runner with a relative folder, then the report.

## Outcome (attempt 2; rule applied as written, report re-run with the fix)

Evidence complete: all 100 runs present, valid, one code version.

- Gate: `ppo_full` PASS, `ppo_full_mimic` PASS (no starvation; equal spend
  and final accuracy within the margins). **Carried forward:
  `ppo_full_mimic`** (higher median equal-spend accuracy).
- Paired vs `ppo_full` (median over seeds, condition - full; equal-spend /
  final / worst-10%): `rl_light` +0.016 / -0.004 / +0.028; `rl_light_pg`
  -0.025 / -0.049 / -0.062; `arl3_full` -0.029 / -0.023 / -0.067;
  `arl3_full_pg` -0.029 / +0.008 / +0.004; `v2_rawfail` -0.047 / -0.026 /
  -0.047; `ppo_v3` -0.031 / -0.043 / -0.051; `ppo_full_mimic` +0.013 /
  +0.021 / +0.034.
- Leave-one-out (removed: equal-spend / worst-10%): predictor v2 + Thompson
  gate -0.587 / -0.459, 3/5 seeds starve (kept); Thompson gate -0.026 /
  -0.016 (kept); mask -0.019 / -0.016 (kept); memory +0.008 / -0.024 (kept).
  Drop candidates (8): attention +0.017 / +0.020; evidence ageing -0.007 /
  +0.031; cost constraint +0.001 / +0.017; hard cap +0.009 / +0.024;
  curriculum +0.009 / +0.010; protection reward -0.009 / +0.024; ARL-v4 ->
  ARL-v3 +0.014 / -0.004; ARL-v4 -> failure count -0.001 / +0.013.
- Reading: exploratory, 5 seeds, from scratch. The predictor/gate repair is
  essential (without it the collapse returns). Always-Light is still a
  little ahead of `ppo_full` on equal spend and worst-10%; the MimiC version
  closes most of that gap. ARL-v4 shows no benefit over ARL-v3 or the plain
  failure count here. Drop candidates are decided in tuning (M7) on seeds
  220-239, as pre-registered; nothing is tuned on 215-219.

**Results and re-validation.** The runs are in `results/rl2_screen_b/full`
(copied from the server; the report re-run here on them gives the same
decision, word for word). MANIFEST: commit 61eff94, clean checkout, torch
2.5.1+cu121 on the RTX 3070.

**Exploratory, after the rule** (`results/rl2_screen_b/EXPLORATORY.md`,
`scripts/explore_screen.py`; not a decision):

- `ppo_full_mimic` vs `rl_light`, seed by seed: equal spend +0.002 (3/5
  seeds ahead, a tie); last-10-round mean accuracy +0.029 and worst-10%
  +0.052 (4/5 seeds ahead each); sign-flip p 0.125 and 0.188 (5 seeds cannot
  go below 0.0625). The RL reaches its higher accuracy partly by spending
  more (median 1,836 vs 1,413).
- Seed 219: `ppo_full_mimic` stood at 0.78-0.80 for rounds 189-199 and fell to
  0.49 in round 200 alone (`ppo_full` 0.81 -> 0.71). Last-round "final"
  measures are decided by such single dips; the median round-to-round wobble
  is no larger for the RL than for always-Light (0.012 vs 0.013).
- `rl_light_pg` (always Light + predictor v2 + Thompson gate) is clearly
  WORSE than plain `rl_light` (last-10 mean 0.720 vs 0.767) despite more
  successes: the repaired gate helps a learner escape the stuck state, but
  a fixed Light-only rule gains nothing from it and loses coverage.
- Consequence for the next protocols (M7 on): "final" accuracy and worst-10%
  are the mean of the last 10 rounds (last round kept as secondary). This is
  decided now, after seeing screen B, and before any M7 run; it is stated as
  such.
