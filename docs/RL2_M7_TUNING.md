# RL2 M7: tuning the carried-forward method

Written and committed before any run. Development step; exploratory.
Spec: `configs/rl2_tune.yaml`. Runner: `scripts/tune_rl2.py`
(`bash scripts/dept_gpu.sh tune`).

**What is tuned:** `ppo_full_mimic`, carried forward by screen B. Every
candidate is that condition with its own settings on top (FEMNIST, s2
settings, persistent split, 200 rounds, from scratch).

**Search space** (all in the spec): the eight drop candidates of screen B as
switches -- attention encoder or not, evidence ageing (0.95 / 0.98 / 0.99 /
none), cost constraint (1.0 / 1.25 / 1.5 x, or a fixed cost weight instead),
hard round cap (none / 1.5 / 2 / 3 x), curriculum on/off, protection reward
(0 / 0.25 / 0.5 / 1), damage term (ARL-v4 / ARL-v3 / plain failure count) --
and the safety and learning settings: mask margin, gate z, PPO learning rate,
epochs, batch, clip, entropy, discount, and the reward weights. Memory, the
predictor v2 + Thompson gate repair and the feasible mask stay on (screen B:
removing them hurt).

**Procedure:**

1. 40 candidates: c00 = `ppo_full_mimic` unchanged; c01-c39 drawn at random
   from the space with a fixed seed (20261028), one option per setting.
2. Stage 1: every candidate on seeds 220-224 (+ `rl_light` for reference).
3. Stage 2: the best 5 of stage 1 (and c00 for comparison) on fresh seeds
   225-239 (+ `rl_light`).
4. The winner is the best of those 5 on stage 2.

**Measures** (changed after screen B showed that one last-round dip can
decide a "final" value; decided before any M7 run):

- P1, equal spend: mean fixed-set accuracy over the last 5 rounds whose
  cumulative cost is <= 1,400 (`eq5`);
- P2, weakest clients: mean worst-10% client accuracy over the last 10 rounds
  (`w10`);
- score = (eq5 + w10) / 2 per seed; a candidate's score is its mean over seeds;
- starved = 50+ rounds in a row with no successful contribution.

**Rule:**

- Evidence first: every run of a stage is checked (exact config, all rounds,
  finite values, one code version from a clean checkout, recorded hashes).
  Anything missing or invalid: INCOMPLETE, nothing is ranked.
- A candidate with a starved seed is disqualified.
- Stage 1: the 5 qualified candidates with the highest score go on (ties: lower
  id).
- Stage 2: the winner is the qualified top-5 candidate with the highest
  stage-2 score. If none qualifies, c00 is kept.
- The winner's settings become `configs/rl2_tuned.json`, committed before
  screen C. Which drop candidates stay is whatever the winner uses. The
  per-option averages in the report are exploratory.

About 310 runs, around 1.5 hours on the RTX 3070 (screen B: 100 runs in
26 minutes). Nothing is tuned on any other seed.
