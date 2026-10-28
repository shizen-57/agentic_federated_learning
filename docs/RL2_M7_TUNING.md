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

## Outcome (rule applied as written)

Run on the department RTX 3070 (commit 1a1662e, torch 2.5.1). Evidence
complete in both stages. Re-checked here independently: same winner, same
settings. (The re-check now compares the recorded stage-1 ranking instead of
rewriting it: Python 3.11 on the server and 3.13 here differ in the 16th
digit of float sums.) Results: `results/rl2_tune`; settings:
`configs/rl2_tuned.json`.

- Stage 1 top 5: c17, c33, c34, c20, c32 (scores 0.649-0.641; base c00 lower).
- Stage 2 (15 seeds): **c34 0.6378**, c33 0.6365, c17 0.6365, c20 0.6362,
  c32 0.6357; base c00 0.6311; `rl_light` 0.5866. The top five are within
  0.002 of each other, so the pick among them is close to a coin toss; all
  of them beat the untuned base.
- **Selected c34**: attention kept; curriculum off; cost constraint 1.0x
  (beta 0); hard cap 3.0x; protection 1.0; evidence ageing 0.98; mask
  margin 0.25; gate z 1.5; PPO lr 0.005, 4 epochs, batch 8, clip 0.3,
  entropy 0.003, gamma 0.9; **damage term: the plain failure count (no ARL)**.
- So of screen B's drop candidates, tuning dropped the curriculum and ARL-v4
  (and ARL-v3), and kept attention, ageing, the cost constraint, the cap and
  protection. Stage-1 averages by option (exploratory): ARL-v4 0.6193 vs
  ARL-v3 0.6316 vs failure count 0.6316.

Exploratory, stage-2 seeds (paired, two-sided sign-flip; the winner was
chosen on these seeds, so its own numbers are slightly flattering):

- c34 vs `rl_light`: equal spend +0.036 (13/15 seeds, p 0.004); worst-10%
  +0.067 (14/15, p 0.001); accuracy +0.056 (15/15, p 0.0001). The untuned
  base shows a similar gap, so it is not just the selection.
- c34 vs untuned c00: equal spend +0.001 (7/15); worst-10% +0.012 (10/15,
  p 0.11); accuracy +0.008 (13/15, p 0.06). Tuning helped a little.
- Screen B's seeds were good ones for `rl_light` (eq5 0.74-0.79 vs 0.66-0.77
  here): on them the gap was +0.005 / +0.032. Always-Light varies much more
  from seed to seed than the RL, which is why the confirmation needs many
  seeds.
