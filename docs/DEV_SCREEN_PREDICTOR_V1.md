# Development screen v1: standardised predictor features

Written and committed **before** any run of this screen. Exploratory: it
decides whether the revision is worth a confirmation study, nothing more.

## What is tested

The revision `arl3_full_pn` is ARL-v3 (`arl3_full`) with one switch,
`policy.predictor_norm: round`: the success predictor's six client features
are centred and scaled by the current round's population before the shared
logistic model. It targets the root cause found in
`docs/DIAGNOSTIC_FINDINGS.md` (failure-only training inverts the ranking, so
the gate's fallback picks the slowest, lowest-battery devices). Nothing else
changes: gate, fallback, modes, reward and credit are as in s2.

## Design (fixed)

- Plan: `configs/dev_predictor_v1.yaml`, expanded from the s2_femnist group
  (FEMNIST, weak-client regime, every s2 protocol switch), 200 rounds,
  diagnostics on.
- Conditions: `arl3_full` (unchanged, for attribution), `arl3_full_pn`
  (revision), `v2_rawfail` (plain failure count), `rl_light` (always Light).
- Seeds 200-204 (never used before). No seed is dropped or added.
- CPU, `--jobs 4`. Results: `results/dev_predictor_v1/full/`.

## Measures (all reported for every seed)

- **Starvation**: a run with 50 or more consecutive rounds that had
  invitations and zero successes.
- Cumulative successes; successes per invitation.
- Share of invitations that were gate fallbacks; mode shares.
- Final accuracy (full test set, round 200).
- Total actual spend (sum of `cost`), and **accuracy at equal spend**: the
  fixed-set accuracy at the last round whose cumulative cost is at most
  **B = 1,400** simulated cost units. B is fixed now from earlier
  information only (always-Light's mean total cost in s2_femnist, 1,394);
  a run that never reaches B is reported at its last round, with its spend.

## Decision rule (fixed)

The revision passes this screen only if all three hold:

1. no `arl3_full_pn` seed starves;
2. median cumulative successes of `arl3_full_pn` >= median of `arl3_full`;
3. median final accuracy of `arl3_full_pn` is not more than 0.02 below
   `arl3_full`.

Passing means "worth a confirmation study on fresh seeds (300+)", not
"better". Its comparison with `rl_light` and `v2_rawfail` is reported but is
not part of the rule. If it fails, that is reported as it is and the
revision is not tuned on these seeds.

## Outcome (added after the runs; rule applied as written)

Full table: `results/dev_predictor_v1/full/screen.md`
(`python scripts/report_dev_screen.py results/dev_predictor_v1/full/s2_femnist`, the report
version of that time; the current one takes `--spec` and validates the runs first).
20/20 runs completed on CPU in 1,537 s.

| condition | median successes | median final acc | median acc at equal spend | starved seeds |
|---|---|---|---|---|
| arl3_full | 1303 | 0.744 | 0.744 | 0/5 |
| arl3_full_pn | 1363 | 0.701 | 0.676 | 0/5 |
| rl_light | 1353 | 0.781 | 0.749 | 0/5 |
| v2_rawfail | 1337 | 0.756 | 0.761 | 0/5 |

Rule: (1) no revision seed starves: pass; (2) median successes >= unchanged
ARL-v3: pass; (3) median final accuracy within 0.02 of unchanged ARL-v3:
**fail** (0.701 vs 0.744). **The screen failed.** As fixed above, the
revision is not tuned on these seeds and goes to no confirmation study.

What the numbers show (descriptive, not part of the rule):
- No method starved on these five seeds, unchanged ARL-v3 included, so this
  screen could not show the revision removing starvation; in s2 only 2 of
  20 ARL-v3 FEMNIST seeds starved.
- With the revision, the policy moves to heavy modes (Light 1-20% of
  invitations vs 4-81% unchanged) and spends far more (2,626-4,095 vs
  1,147-3,915 cost units). Accuracy at equal spend falls (0.676 vs 0.744).
  Fixing the predictor's ranking makes more clients look able to do heavy
  work, and ARL-v3's work-based loss then pushes for it.
- Always-Light and the raw failure count remain the strongest of the four
  on accuracy at equal spend.
