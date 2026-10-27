# Development screen: feasible-mode mask (revision 2)

Written and committed **before** any run of this screen. Exploratory.

## What is tested

`arl3_full_fm` is ARL-v3 (`arl3_full`) with one switch,
`policy.mode_mask: feasible`: the RL policy may only draw a mode whose
predicted success is at least `elig_threshold` (0.5); if no mode clears it,
only the most likely mode. The log-probability is that of the masked draw.
It targets the absorbing link in `docs/DIAGNOSTIC_FINDINGS.md` (slow fallback
clients sent Standard/Priority jobs their battery cannot cover) and the
heavy-mode drift seen in `docs/DEV_SCREEN_PREDICTOR_V1.md`. It is applied
to unchanged ARL-v3, **without** the predictor revision (one change at a time).

## Design, measures and decision rule

Identical to `docs/DEV_SCREEN_PREDICTOR_V1.md` (same measures, starvation
definition, equal-spend budget B = 1,400, and three-part rule with
`arl3_full_fm` in place of `arl3_full_pn`), except:

- Plan `configs/dev_modemask_v1.yaml`; seeds **205-209** (fresh; 200-204 were
  inspected in the first screen). Results: `results/dev_modemask_v1/full/`.
- Report: `python scripts/report_dev_screen.py results/dev_modemask_v1/full/s2_femnist --revision arl3_full_fm`.

## Mechanism check (reported, not part of the rule)

`configs/dev_modemask_v1_lockcheck.yaml`: `arl3_full_fm` on the inspected
seeds 110 and 113, where unchanged ARL-v3 (s113) and c2 (both) locked up.
Reported: whether either run starves. These seeds are not evidence for the rule.

## Outcome (added after the runs; rule applied as written)

Full table: `results/dev_modemask_v1/full/screen.md`. 20/20 runs on CPU.

| condition | median successes | median final acc | median acc at equal spend | starved seeds |
|---|---|---|---|---|
| arl3_full | 1184 | 0.700 | 0.697 | 0/5 |
| arl3_full_fm | 943 | 0.738 | 0.700 | 0/5 |
| rl_light | 1340 | 0.707 | 0.737 | 0/5 |
| v2_rawfail | 1191 | 0.715 | 0.708 | 0/5 |

Rule: (1) no starvation: pass; (2) median successes >= unchanged ARL-v3:
**fail** (943 vs 1,184); (3) final accuracy within 0.02: pass (+0.038).
**The screen failed.** Not tuned on these seeds; no confirmation study.

Descriptive (not part of the rule):
- The mask moved the policy back to Light (64-77% on four of five seeds) and
  made spend far steadier (1,687-2,254 vs 1,175-4,477); median final
  accuracy rose by 0.038, equal-spend accuracy was level (0.700 vs 0.697).
- It delivered fewer successes, with a larger share of fallback invitations
  on three seeds (0.30-0.37): restricting modes to "predicted feasible"
  with the still-uncorrected predictor keeps more rounds in the gate's
  fallback.
- Again no method starved on these seeds.

Mechanism check (inspected seeds, CPU; `results/dev_modemask_v1_lockcheck/full/lockcheck.md`):
the mask escapes both lock-ups -- s110 904 successes, acc 0.733; s113 689,
acc 0.759 (unchanged ARL-v3 s113 on CPU: 4 successes, 0.098).
