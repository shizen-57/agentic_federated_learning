# RL2 confirmation (M9 freeze, M10 run)

The protocol, written before screen C. The freeze fills in the method,
comparators and seed count from screen C's decision, by the rules below, and
records them in `configs/rl2_frozen.json`.

**Freeze (M9, `scripts/freeze_rl2.py`, run on a clean checkout):**

- Inputs: screen C's `DECISION.json` (committed), the unchanged spec and
  tuned settings, and the recorded data roles (`configs/rl2_data_roles.json`).
  Screen C's results are **development evidence**: they choose the method;
  they never count as confirmation. Screen C's code hash is recorded; its
  checkpoints are not reused (see "Run").
- Fixed: the final method (settings, pre-trained or not, and the
  pre-training recipe), the comparators, the 8 primary tests, the secondary
  comparisons, the endpoints, the test and its multiplicity correction, the
  seeds, the full run plan, and the SHA-256 of every input (plans, configs,
  tuned settings, data roles).
- Data (fourth review, finding 2): the confirmation evaluates on the
  **reserved confirmation pool** (`data.eval_pool: confirm`), which no
  development, tuning or screening run has ever read:
  - FEMNIST: 900 writers kept apart from the 1,798 practice writers
    (pre-training) and the 899 development writers (every screen so far);
  - CIFAR-10: 3,506 test examples kept apart from the 3,006 validation and
    the 3,488 development examples.
  The roles come from the fixed `data.split_seed`; every ID list's size and
  SHA-256, and their disjointness, are pinned in `configs/rl2_data_roles.json`
  (`scripts/data_roles.py` checks them). The freeze refuses if the roles have
  changed, and every development runner refuses the confirmation pool
  (`rl2lib.check_pool`).
- **History, stated plainly.** The pools were split on 2026-10-28. Earlier
  RL2 development runs used the old code, which drew clients from the whole
  evaluation pool:
  - FEMNIST: screen B (seeds 215-219), M7 (220-239) and the first screen C
    (240-249) between them used 715 of the 799 confirmation writers that have
    enough data (screen B 234, M7 552, screen C 385);
  - CIFAR-10: the first screen C evaluated on every non-validation test
    example, so on all 3,506 confirmation examples.

  So the confirmation pool is reserved **from the split on**, not since the
  start of the project. What this means and what is done about it:
  - M7's choice (c34) stays **development evidence**. It chose global
    settings, and the confirmation uses fresh seeds (new client sets, devices
    and training randomness), but it was not made on data the confirmation
    never saw, and the thesis says so.
  - Every choice that feeds the freeze is made again on the development pool
    only: the ARL screen (`docs/RL2_ARL_SCREEN.md`), the sensitivity screen
    (`docs/RL2_SENSITIVITY.md`) and **screen C, re-run on the new code**.
    The first screen C (old code) is exploratory only.
  - The freeze enforces this: it refuses a screen C decision that does not
    record the development pool of the pinned data roles
    (`eval_pool: dev`, `data_roles_sha256`), which no old-code screen C has.
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
- every pinned input (including the data roles) is byte-identical;
- the seeds are confirmation seeds run from the frozen manifest.

Then:
1. pre-training of every pre-trained method afresh, under the frozen code, on
   practice episodes only (seeds 1000-1999, practice writers), into a new
   checkpoint folder (`.../confirm`); resumable;
2. a 2-round pre-flight of every condition (seed 900);
3. the run manifest, with the SHA-256 of every checkpoint;
4. the runs (resumable);
5. the tests.

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

The endpoints are fixed here and never change after confirmation results are
seen. Next to them, the report discloses the actual spend of each method and
accuracy at common budget checkpoints (`scripts/cost_curves.py`; fourth review,
finding 6). Pre-training cost is reported separately, next to deployment cost,
not inside the 1,400 budget.
