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
