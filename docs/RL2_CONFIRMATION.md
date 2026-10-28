# RL2 confirmation (M9 freeze, M10 run)

The protocol, written before screen C. The freeze fills in the method,
comparators and seed count from screen C's decision, by the rules below, and
records them in `configs/rl2_frozen.json`.

**Freeze (M9, `scripts/freeze_rl2.py`, run on a clean checkout):**

- Inputs: screen C's `DECISION.json` (committed), the same simulator code
  (its checkpoints are tied to it), the unchanged spec and tuned settings.
- Fixed: the final method (settings, pre-trained or not, checkpoints by
  SHA-256), the comparators, the 8 primary tests, the secondary comparisons,
  the endpoints, the test and its multiplicity correction, the seeds, the
  full run plan, and the SHA-256 of every input (plans, configs, tuned
  settings, checkpoints).
- Seeds: confirmation seeds from 300 upward, never run before. Their number
  comes from a power analysis on screen C's paired spread: n = ((z_{1-0.05/8}
  + z_0.8) x sd / delta)^2 per primary test, with delta = 0.01 for P1 and
  0.02 for P2. The largest n is used, kept between 20 and 100. If a test would
  need more than 100 seeds, its power is lower and the report says so.
- `configs/rl2_frozen.json`'s SHA-256 and the code hash are written into
  `scripts/prespec_rl2.py` together. A freeze is never redone to admit new
  code or results (the same rule as s2).

**Run (M10, `scripts/confirm_rl2.py`):** only with `--go` (the user's
explicit go-ahead: `bash scripts/dept_gpu.sh confirm go`), and only if
- the frozen manifest matches its pins;
- the simulator code is the frozen code and the checkout is clean;
- every pinned input and checkpoint is byte-identical;
- the seeds are confirmation seeds run from the frozen manifest.

Then a 2-round pre-flight of every condition (seed 900), the runs
(resumable), and the tests.

**Tests (`scripts/prespec_rl2.py`):**

- Primary: 8 tests. Per dataset (FEMNIST, CIFAR-10):
  - P1, accuracy at equal spend (`eq5`): final method vs always-Light, and vs
    screen C's best baseline;
  - P2, weakest clients (`w10`): final method vs always-Light, and vs the
    repaired original RL (`arl3_full_pg`).
- Each test is a paired sign-flip test on the per-seed difference, one-sided
  (final better): exact for n <= 20, otherwise Monte Carlo with 200,000 draws
  and a fixed seed.
- Holm-adjusted together at alpha 0.05. A test supports superiority only if
  its Holm-adjusted p < 0.05.
- Secondary, descriptive with unadjusted two-sided p:
  - the final method vs the other RL2 versions and `rl_light_mimic`;
  - the ARL contrast (with minus without ARL-v4).
- No verdict unless the frozen manifest matches its pins and every run is
  complete and valid. Every result is reported, supported or not.

**Endpoints:**

- `eq5`: mean fixed-set accuracy over the last 5 rounds whose cumulative cost
  is <= 1,400;
- `w10`: mean worst-10% client accuracy over the last 10 rounds.

Pre-training cost is reported next to deployment cost.
