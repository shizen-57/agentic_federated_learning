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
