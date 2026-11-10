# RL2 sensitivity screen: does the result depend on one chosen world?

Written before any run (fourth external review, finding 4). Spec:
`configs/rl2_sensitivity.yaml`. Runner: `scripts/screen_rl2.py`.

**Version 2 (gap review, 2026-10-29, still before any run):** the reference
is the **tuned** always-Light with **MimiC**, aggregated like the tuned
method (A1, B1); the tuned method is the final one (cost and ARL verdicts
carried in); both endpoints at equal spend (A2); and new settings: the
measured FedScale device trace (B5) and the other weak-client scenarios
(B4).

## Why

Many world settings are our own choice, not measured (see
`docs/SIMULATOR_MODEL.md`, "What is not calibrated"). If "the tuned scheduler
beats always-Light" held only at those exact values, the claim would be weak.
So each of four world settings is moved down and up, **one at a time**, with
everything else at its usual value; the invented device tiers are swapped for
measured ones; and the scenario changes (who is weak, nobody weak, churn).

## Design

- Arms: the final tuned method (`configs/rl2_tuned_final.json`, from scratch)
  and the control: always Light with MimiC, with the settings of its matched
  tuning (`configs/rl2_tuned_rl_light_mimic.json`).
- Dataset: FEMNIST, persistent split, development pool. Seeds 270-279
  (10, development, never used). 200 rounds.
- Settings (13): the usual world (`base`) and, one at a time:

| factor | low | usual | high | what it changes |
|---|---|---|---|---|
| deadline | 0.8 | 1.0 | 1.2 | `system.deadline`: time allowed per round |
| availability | 0.7 | 0.85 | 0.95 | `system.availability_p`: mean chance a client is online |
| battery | 0.5x | 1x | 2x | `system.energy_scale`: every battery capacity |
| bandwidth | 0.5x | 1x | 2x | `system.bandwidth_scale`: every upload bandwidth |
| devices | tiers | - | FedScale trace | `system.device_trace: fedscale`: measured phone speeds and bandwidths instead of our tiers |
| scenario | weak = rarest data | random weak, no weak, churn | | who is weak (`system.weak_select: random`), no weak devices (`system.weak_frac: 0`), clients that leave for good (`system.churn_after: 3`) |

- 13 settings x 2 arms x 10 seeds = 260 runs. Same seeds in every cell.

## Rule (fixed before the runs)

- Evidence first, as for every screen (`rl2lib.check_evidence`).
- Per setting: tuned minus the control, paired by seed, on `eq5` and
  `w10b` (both at equal spend). Reported with wins out of 10
  and an unadjusted two-sided sign-flip p (descriptive; 10 seeds per cell is
  enough to see a reversal, not to prove small differences).
- A factor is **robust** if the mean difference stays positive on both
  endpoints at every one of its levels. Otherwise it is **NOT robust**, and
  every reversal is listed.
- What follows:
  - Robust factors: the thesis states that the conclusion holds across that
    range.
  - A reversal: the thesis states the condition under which always-Light
    does as well or better (for example "when deadlines are tight").
  - "No weak devices" is expected to shrink the fairness gain: the method is
    built for that problem. A reversal there is reported as such, not as a
    failure of the scheduler.
  - Nothing is tuned on these seeds or settings, and the confirmation
    endpoints and settings do not change because of this screen.

## Outcome

Not run yet. It needs `configs/rl2_tuned_final.json` (after the ARL screen)
and the tuned control (`tunebase rl_light_mimic`). Then:
`bash scripts/dept_gpu.sh screen2 configs/rl2_sensitivity.yaml`.
