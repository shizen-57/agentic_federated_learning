# Validation: can the simulator and the scheduler be trusted?

The checks asked for by the fourth external review, what covers each one,
and what they found. Unit tests run with `python -m pytest tests -q` on
CPU, here (torch 2.14) and with the server's torch 2.5.1. The model of the
simulated world is in `docs/SIMULATOR_MODEL.md`.

## 1. The repairs, checked together (finding 1)

`tests/test_validation_matrix.py::test_select_project_learn_hold_together`
runs the full SAFE-PPO scheduler (attention, predictor v2, Thompson gate,
mask, cost constraint, hard cap, PPO) for many rounds, with a cap tight
enough that the budget step often changes or drops jobs. It watches every
step: the policy's choice, the budget step, every outcome, every charge and
every local training. Acceptance criteria, all checked every round:

- every executed job is one the safety mask allowed, and the round's
  executed jobs fit the cap;
- the predictor is calibrated against the mode that actually ran (not the
  one first chosen);
- a dropped job runs nothing, costs nothing and trains nothing;
- PPO keeps a transition for every round, empty rounds included.

## 2. Development data kept apart from confirmation data (finding 2)

| Check | Covered by |
|---|---|
| FEMNIST practice / development / confirmation writers are disjoint by ID | `tests/test_persistent_split.py::test_femnist_practice_development_and_confirmation_are_disjoint_by_id` |
| CIFAR-10 development and confirmation test examples are disjoint | `...::test_pooled_development_and_confirmation_test_examples_are_disjoint` |
| the confirmation pool is never practice and never a per-seed split | `...::test_the_confirmation_pool_is_never_practice_or_per_seed` |
| the recorded ID lists (sizes, SHA-256) match the code | `...::test_recorded_data_roles_match_the_code`, `scripts/data_roles.py` |
| development runners refuse the confirmation pool | `tests/test_rl2_pipeline.py::test_development_runs_cannot_touch_the_confirmation_pool` |
| the freeze refuses a screen C made before the pools existed | `tests/test_rl2_pipeline.py::test_freeze_refuses_a_screen_c_from_before_the_reserved_pools` |

The history (earlier screens drew from the whole evaluation pool) and what
is done about it: `docs/RL2_CONFIRMATION.md`, "History".

## 3. Does ARL help? (finding 3)

A controlled screen, everything else fixed, with the rule written first:
`docs/RL2_ARL_SCREEN.md`. Also there: M7 compared at a matched tuning budget.

## 4. Is the simulated world reasonable? (finding 4)

- **Equations, units, values, sources**: `docs/SIMULATOR_MODEL.md`. Energy and
  cost are simulated units.
- **The code follows the documented equations**:
  `tests/test_validation_matrix.py::test_cost_time_and_energy_follow_the_documented_equations`
  (cost linear in examples, epochs and upload fraction; time and energy as
  written) and `...::test_sensitivity_scales_multiply_battery_and_bandwidth`.
- **Predicted vs measured time**, `scripts/timing_check.py`,
  `results/simulator_checks/timing.md`. Real local training on this machine,
  100-800 examples x 1, 2, 4 epochs:
  - FEMNIST: time = 4.0e-4 x (n x E) - 0.005 s, R^2 0.997, largest error 10%;
  - CIFAR-10: time = 9.8e-4 x (n x E) + 0.007 s, R^2 0.999, largest error 24%
    (the smallest job, where fixed overhead matters).

  So real training time is proportional to examples x epochs, as the
  simulator assumes. Real upload bytes are proportional to the upload
  fraction: FEMNIST 222 / 444 / 888 KB for Light / Standard / Priority. The
  simulator's seconds are not real seconds; only the shape is claimed.
- **One setting at a time**: the sensitivity screen moves deadline,
  availability, battery and bandwidth down and up (`docs/RL2_SENSITIVITY.md`).

## 5. The predictor and the safety rules under change (finding 5)

`scripts/predictor_stress.py`, `results/predictor_stress/report.md`. The
tuned method on real FEMNIST, 3 seeds, 200 rounds, in a stable world and a
shocked one. In the shocked world, at round 80 half the devices lose 98% of
their bandwidth and 95% of their battery; at round 140 they are restored.
Every executed job's predicted success is compared with what happened.

Results: added when the 200-round run finishes (`results/predictor_stress`).

## 6. Resources compared precisely (finding 6)

- The development endpoint stays `eq5`: mean accuracy over the last 5 rounds
  within a cumulative cost of 1,400. The confirmation's endpoints are fixed
  in the freeze and never change after its results are seen.
- **Actual spend is disclosed.** `scripts/cost_curves.py` gives accuracy at
  common budget checkpoints (500 to 2,000) with the actual spend at each, and
  draws accuracy against cumulative cost:
  - M7 stage 2: `results/rl2_tune/cost_curves.{md,png}`;
  - screen B: `results/rl2_screen_b/cost_curves.{md,png}`.
- What the curves show (M7 stage 2, 15 seeds):
  - at every checkpoint, the methods' actual spend is within about 2% of
    each other, so `eq5` compares like with like;
  - always-Light is ahead at a budget of 500 (0.684 vs 0.666 for the tuned
    method); the tuned method leads from about 800 on (0.730 vs 0.708) and at
    1,400 (0.772 vs 0.729);
  - always-Light spends only about 1,371 in 200 rounds. So at budgets above
    that, it has simply stopped spending; a comparison there is not "equal
    spend", and the report says so.
- Pre-training cost is reported separately (screen C's `pretrain_cost`;
  `docs/RL2_CONFIRMATION.md`), never inside the 1,400.

## 7. The fixed validation matrix (finding 7)

Run before any further confirmation sweep. Each row names its test.

| Scenario | Acceptance | Test |
|---|---|---|
| nobody online | no job, no cost, finite values; logged as offline, not counted as starvation; rounds kept for PPO | `test_validation_matrix.py::test_nobody_online` |
| everyone abstains (safety layer rules everyone out) | rounds run empty but are kept for PPO and counted as no-contribution rounds; once the rule lifts, clients are invited again | `...::test_everyone_abstains_then_recovers` |
| every update fails / every update is rejected | nothing is aggregated, the global model is unchanged, values stay finite, the starvation measure sees it | `...::test_every_update_fails_or_is_rejected` |
| a cap too tight for any job | every job dropped, no work, no cost, empty round kept | `...::test_a_cap_too_tight_for_any_job` |
| tight budgets with mode changes | executed jobs fit the cap and the mask; calibration by executed mode | `...::test_select_project_learn_hold_together` |
| sudden bandwidth and battery drop, then recovery | the hit clients' success falls; after recovery most of them are invited again (real jobs only, not the oracle's what-ifs) and succeed again | `...::test_sudden_bandwidth_and_battery_drop_and_recovery` |
| interrupted run, resumed | a damaged or partial result is re-run, a valid one is kept, an original is never overwritten | `test_resume_validation.py` |
| interrupted pre-training, resumed | the same checkpoint as without the interruption | `test_pretrain.py::test_an_interrupted_pretraining_resumes_to_the_same_checkpoint` |
| persistent splits | same roles for every seed; disjoint pools; one fingerprint for all pools | `test_persistent_split.py` |
| checkpoint compatibility | a checkpoint from other code, data or method is refused | `test_pretrain.py::test_foreign_checkpoints_are_refused`, `scripts/screen_c_rl2.py` (`checkpoint_problem`) |

**Short real-data checks on CPU and GPU** (`scripts/check_devices.py`):
FEMNIST and CIFAR-10, the tuned method and always-Light, seed 900, 30 rounds,
on the development pool. Records last-10 accuracy, worst-10%, successes,
invitations, cost and the share of heavy jobs. `--compare` flags a material
difference: accuracy by more than 0.03, or successes or cost by more than
15%. Small differences are expected (GPU arithmetic is not bit-identical to
CPU). The CPU result is `results/devicecheck/cpu.json`; the GPU one comes from
`bash scripts/dept_gpu.sh devicecheck`.
