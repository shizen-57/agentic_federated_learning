# RL2 M1 screen: predictor v2 + Thompson gate

Written and committed before any run. Development gate for milestone M1 of
docs/RL2_DESIGN.md; exploratory.

**Tested:** `arl3_full_pg` = ARL-v3 + `policy.predictor: v2` + `policy.gate_mode:
thompson` (C1 + C2), against unchanged `arl3_full`, `v2_rawfail`, `rl_light`.
FEMNIST, s2 settings, 200 rounds, CPU, dev seeds 210-214 (`configs/rl2_m1_screen.yaml`);
plus the inspected lock-up seeds 110 and 113 (`configs/rl2_m1_lockcheck.yaml`;
on CPU unchanged ARL-v3 s113 had 4 successes, c2 s110/s113 5 and 8).

**Measures:** as in docs/DEV_SCREEN_PREDICTOR_V1.md (starvation = 50+ rounds
with invitations and no success; equal-spend budget B = 1,400).

**Rule (M1 passes only if all hold):**
1. no `arl3_full_pg` run starves, on dev seeds or on 110/113;
2. median final accuracy of `arl3_full_pg` >= median of `arl3_full` - 0.02;
3. median accuracy at equal spend of `arl3_full_pg` >= median of `arl3_full` - 0.02.

Why not "median successes" as in the two earlier screens: RL2's endpoints are
accuracy at equal spend and worst-client accuracy (docs/RL2_DESIGN.md), and M1's
job is to remove the stuck state without costing accuracy. Success counts are
still reported for every seed.

If M1 fails, it is reported and not tuned on these seeds; later milestones
then build on the original predictor/gate and M1 is revisited only with new
evidence on new dev seeds.

## Outcome (added after the runs; rule applied as written)

CPU, 20/20 dev runs + 2 lock-check runs, from a frozen worktree of commit
c6f18b1 (a first attempt was stopped and discarded because the simulator
source changed while it ran). Tables: `results/rl2_m1_screen/full/screen.md`,
`results/rl2_m1_lockcheck/full/lockcheck.md`.

| condition | median successes | median final acc | median acc at equal spend | median worst-10% client acc | starved |
|---|---|---|---|---|---|
| arl3_full | 803 | 0.748 | 0.738 | 0.483 | 1/5 (s211) |
| **arl3_full_pg** | 1175 | **0.768** | **0.769** | **0.529** | **0/5** |
| rl_light | 1498 | 0.754 | 0.754 | 0.460 | 0/5 |
| v2_rawfail | 523 | 0.744 | 0.744 | 0.511 | 1/5 (s210) |

Lock-check (inspected seeds): s110 1,224 successes, acc 0.754; s113 1,249,
acc 0.766 (unchanged ARL-v3 s113 on CPU: 4 successes, 0.098).

Rule: (1) no starvation (dev and lock-check): pass; (2) final accuracy
0.768 >= 0.748 - 0.02: pass; (3) equal-spend accuracy 0.769 >= 0.738 - 0.02:
pass. **M1 passed.** Gate fallbacks fell to 0.00-0.00 of invitations.

Caveat recorded later (external review): these runs used the per-seed
validation/test split of s2. That is clean within each run, so this gate
stands; RL2 runs from here on use persistent splits (docs/RL2_DESIGN.md).
