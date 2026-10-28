# RL2 sensitivity screen: does the result depend on one chosen world?

Written before any run (fourth external review, finding 4). Spec:
`configs/rl2_sensitivity.yaml`. Runner: `scripts/screen_rl2.py`.

## Why

Many world settings are our own choice, not measured (see
`docs/SIMULATOR_MODEL.md`, "What is not calibrated"). If "the tuned scheduler
beats always-Light" held only at those exact values, the claim would be weak.
So each of four world settings is moved down and up, **one at a time**, with
everything else at its usual value.

## Design

- Arms: the tuned method (`configs/rl2_tuned.json`, from scratch) and the
  control `rl_light` (the RL scheduler that always picks Light).
- Dataset: FEMNIST, persistent split, development pool. Seeds 270-279
  (10, development, never used). 200 rounds.
- Settings (9): the usual world (`base`) and, one at a time:

| factor | low | usual | high | what it changes |
|---|---|---|---|---|
| deadline | 0.8 | 1.0 | 1.2 | `system.deadline`: time allowed per round |
| availability | 0.7 | 0.85 | 0.95 | `system.availability_p`: mean chance a client is online |
| battery | 0.5x | 1x | 2x | `system.energy_scale`: every battery capacity |
| bandwidth | 0.5x | 1x | 2x | `system.bandwidth_scale`: every upload bandwidth |

- 9 settings x 2 arms x 10 seeds = 180 runs. Same seeds in every cell.

## Rule (fixed before the runs)

- Evidence first, as for every screen (`rl2lib.check_evidence`).
- Per setting: tuned minus `rl_light`, paired by seed, on `eq5` (accuracy at
  equal spend) and `w10` (worst-10% clients). Reported with wins out of 10
  and an unadjusted two-sided sign-flip p (descriptive; 10 seeds per cell is
  enough to see a reversal, not to prove small differences).
- A factor is **robust** if the mean difference stays positive on both
  endpoints at all three of its levels. Otherwise it is **NOT robust**, and
  every reversal is listed.
- What follows:
  - Robust factors: the thesis states that the conclusion holds across that
    range.
  - A reversal: the thesis states the condition under which always-Light
    does as well or better (for example "when deadlines are tight").
  - Nothing is tuned on these seeds or settings, and the confirmation
    endpoints and settings do not change because of this screen.

## Outcome

Not run yet. To run on the department GPU:
`bash scripts/dept_gpu.sh screen2 configs/rl2_sensitivity.yaml`.
