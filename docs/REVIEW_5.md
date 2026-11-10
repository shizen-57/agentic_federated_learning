# Review 5: what is not working, and the plan to fix it

Written 2026-11-09, from a line-by-line read of the simulator and the method
(`src/afl/sim.py`, `engine.py`, `predictor.py`, `policies/safe_rl.py`,
`local.py`, `aggregate.py`) and from the saved runs in `results/`. The earlier
reviews (`docs/GAP_REVIEW.md`) looked at the experiment design. This one looks
at the simulator and the code, and at why the method does not win.

Everything here is development evidence. No confirmation seed or confirmation
data was touched.

## 1. Short answer

The method does not beat a simple tuned control, and the main reason is not a
single bug. It is that **the test bed gives the method almost nothing to
win**, and three parts of the method work against its own goal:

1. In this simulator a heavier job almost never pays off (section 2.1, 2.5).
2. One client in four can hardly ever finish any job (2.2).
3. The biggest cause of failure is a coin flip no scheduler can avoid (2.3).
4. The "safe job" rule shuts out most weak clients -- the clients the thesis
   wants to protect (3.1).
5. The success predictor does not know how much work a client has (3.2).
6. The scheduler learns very little during one run (3.3).
7. Training is unstable: a round in which only a few updates arrive often
   wrecks the model (3.7). This hurts every method and blurs the end points.

The pilot (6.3) puts a number on the first point: a scheduler that never
wastes a job would gain at most about +0.02 in equal-spend accuracy over the
tuned control. No real method can win clearly on that end point here.

## 2. Simulator weaknesses

### 2.1 Mode choice has little room (measured)

Ground truth of the simulator, FEMNIST setting, seeds 280-284, 200 rounds,
battery full: how often would a job succeed if a client that is online were
given it? (Script: section 7.)

| client group | share | Light | Standard | Priority |
|---|---|---|---|---|
| slow devices (tier 0) | 26% | 0.17 | 0.00 | 0.00 |
| middle devices (tier 1) | 25% | 0.69 | 0.50 | 0.06 |
| fast devices (tier 2) | 19% | 0.84 | 0.83 | 0.65 |
| weak clients (hold the rare classes) | 30% | 0.41 | 0.04 | 0.00 |

- Priority is only usable by fast devices (19% of clients).
- Standard is only usable by fast and middle devices (44%).
- For 56% of the clients (slow and weak) there is no choice: Light or nothing.

So "choose the job size" is a real decision for fewer than half the clients.

### 2.2 A quarter of the clients can hardly ever contribute

Slow devices succeed 17% of the time even with Light (45% link drop, 38%
deadline). 58% of them succeed in fewer than 1 round in 5. Whatever the
scheduler does, their data almost never reaches the model. There is no job
smaller than Light and no way to give a slow client more time. This puts a
ceiling on fairness that no method can pass.

### 2.3 The biggest failure cause cannot be avoided by any job choice

"Link drop" is a coin flip per client and round. It does not depend on the
job: a 4-epoch job with a full upload drops exactly as often as a 1-epoch job
with a quarter upload. Its rate is high: 14% (fast), 29% (middle), 41% (weak),
45% (slow). About one job in three fails this way.

Result: every good method fails about 40% of its jobs (tuned always-Light
40%, our method 40%, fairness queues 48%). In real systems a longer job is
more likely to be interrupted, and reported drop-out rates are lower
(about 6-10% per round in Google's system, Bonawitz et al. 2019).

### 2.4 The eligibility line sits above the weak clients

A client counts as "eligible" when its predicted success is at least 0.5
(thesis Eq. 3.26). Weak clients truly succeed 41% of the time with their
best job, and 83% of them are truly below 0.5. So a *perfectly accurate*
predictor, used with this rule, declares most weak clients not eligible.

### 2.5 The cost model makes Light the best buy by construction

Cost = epochs x data / 1000 + upload share. Light, Standard and Priority cost
1 : 2 : 4, for 1 : 2 : 4 epochs and 1 : 2 : 4 upload. There is no fixed cost
per job (every invited client must first download the whole model, whatever
its job). So Light always gives the most training per unit of cost, and two
Light jobs on two clients beat one Standard job on one client. Under the
"equal spend" endpoint, heavier modes cannot win.

Also: weak clients hold more data, so a Light job costs them 0.95 against
0.66 for the others, and it fails 59% of the time. A successful weak
contribution costs about 3 times a fast client's. The equal-spend accuracy
endpoint (P1) therefore rewards leaving weak clients out, while the
worst-10% endpoint (P2) rewards bringing them in.

### 2.6 Smaller inconsistencies

- Energy use does not depend on the amount of data; time and cost do.
- A job that misses the deadline reports its full run time. A real server
  stops waiting at the deadline. The latency number and the reward's latency
  penalty are inflated by stragglers.
- There is no clock: all rounds count the same, so "time to reach an
  accuracy" (a standard measure) is not reported.

### 2.7 All good methods end at the same height

Mean accuracy (5-round average), comparator tuning stage 2, seeds 225-239:

| method (tuned) | round 25 | 50 | 100 | 150 | 200 | cost to reach 0.70 | to 0.75 |
|---|---|---|---|---|---|---|---|
| always-Light + MimiC | 0.40 | 0.60 | 0.73 | 0.76 | 0.78 | 528 | 782 |
| REFL | 0.56 | 0.69 | 0.75 | 0.77 | 0.79 | 502 | 797 |
| Oort | 0.57 | 0.67 | 0.75 | 0.77 | 0.78 | 703 | 1,173 |
| fairness queues | 0.55 | 0.68 | 0.75 | 0.75 | 0.79 | 635 | 966 |
| our method c34 (new code, seeds 280-284) | 0.52 | 0.67 | 0.74 | 0.77 | 0.78 | 539 | 871 |

Methods differ a lot early (0.40 to 0.57 at round 25) and end within 0.01 of
each other. Both primary endpoints are measured near the end, where the curve
is flat. They have little power to tell methods apart.

### 2.8 A device never changes (added after the user's remark, 2026-11-10)

The user's point: the simulator is too simple to show what ARL could do in
real life. The strongest version of that point is this. In the simulator a
client's device is fixed: a weak client is weak in every round and can only
ever finish a Light job. So there is never a moment where a better decision
(another job, or another time) would have saved its contribution -- and
"avoidable loss" is exactly what ARL measures.

A real phone changes during the day: at night it is plugged in, idle and on
Wi-Fi; in the afternoon it runs on battery and mobile data while in use.

Option built (off by default): `system.state_model: diurnal`. Each round a
client is in a good or a poor state, on its own day/night cycle, sticky like
availability. Speed x1.25 / x0.6, bandwidth x1.5 / x0.4, and the battery only
recharges in the good state. The server sees the current (noisy) speed and
bandwidth, as before. Values assumed.

Ground truth with this option and the four of 2.3-2.6 ("life" setting),
FEMNIST, seeds 403-405:

| group and state | share of online rounds | Light | Standard | Priority |
|---|---|---|---|---|
| slow, good state | 58% | 0.29 | 0.01 | 0.00 |
| slow, poor state | 42% | 0.00 | 0.00 | 0.00 |
| middle, good state | 57% | 0.86 | 0.68 | 0.17 |
| middle, poor state | 43% | 0.48 | 0.05 | 0.00 |
| fast, good state | 56% | 0.95 | 0.93 | 0.82 |
| fast, poor state | 44% | 0.91 | 0.69 | 0.11 |
| weak, good state | 58% | 0.58 | 0.10 | 0.00 |
| weak, poor state | 42% | 0.02 | 0.00 | 0.00 |

Job rules in this setting (`results/r5_predictor/femnist_life`; epochs of
training delivered per unit of cost): always Light 0.28, the baselines'
expected-work rule 0.37, the oracle 0.85. Knowing which job a device can
finish *right now* is worth far more here than in the current world.

Now the right decision depends on the moment: a weak client is worth
inviting in its good state (0.58, above the eligibility line) and a waste in
its poor state (0.02); a middle device can take Standard at night and only
Light by day. Whom to invite, when, and with which job are all real
decisions here. Whether ARL then helps is an open question that only runs
can answer (Phase 3).

## 3. Method and code weaknesses

### 3.1 The safety layer works against inclusion

- Our method allows a job only when its predicted success, minus a margin,
  is at least 0.5. In the cost-check runs **22 to 28 of about 75 online
  clients were ruled out every round**. Weak clients picked per round fall
  from 3.3 to 2.5 during a run (fair share: 3.0).
- The tuned always-Light control chose a lower line (0.4) and no fairness
  penalty. It picks 2.8 to 3.2 weak clients per round. The simple control is
  *more* inclusive than our method.
- The method's own tuning (M7) could not change the 0.5 line. Every
  comparator's tuning could (three chose 0.4, three chose 0.6). The tuning
  was not symmetric -- this time against our method.
- The reward's damage term is the share of failed jobs. Weak clients fail
  most, so this term teaches the scheduler to avoid them.

### 3.2 The success predictor is missing its most useful input

*Checked (6.1): adding the input does not help on FEMNIST and helps only a
little on CIFAR-10. The gap is real, but it is not what holds the method
back.*

The predictor sees device speed, bandwidth, battery and history, plus which
mode. It does not see the client's amount of data, and it has no "this mode
on this device" term. But run time = epochs x data / speed. So the shared
model cannot learn "Priority only works on fast devices"; only each client's
own record can, slowly. Effect: Priority is over-predicted (FEMNIST 0.72
predicted vs 0.64 real; CIFAR-10 0.69 vs 0.38). Every `_ew` baseline uses the
same predictor for its mode rule, so they waste spend too (fairness queues
start with 29% Priority jobs).

### 3.3 The scheduler learns very little inside one run

A run has 200 rounds = 25 updates of 8 rounds each. Share of Light jobs in
our method: 76% in rounds 1-25, 80% in rounds 101-200 (old code); 80% to 88%
(new code). The choice of job barely moves. What the method does is mostly
fixed by its masks and its starting weights. Only pre-training can change
this (it is built, and screen C tests it).

### 3.4 The soft cost limit does not hold

Known from the cost check: 2 of 5 seeds overspend (1.35x, 1.95x). The price
on cost rises slowly, and the advantage normalisation removes most of a
steady penalty. A hard cap at the target held on every seed.

### 3.5 MimiC mixes updates of different length

*Checked (6.4): scaling the updates by their length does not help; it is
slightly worse. Not pursued.*

MimiC averages the received updates and keeps one correction per client.
With mixed modes an update is 1, 2 or 4 epochs long (and clients differ in
data), so updates of very different size are averaged without scaling, and a
client's stored correction has the wrong size when its mode changes. Always-
Light does not have the mode part of this problem.

### 3.6 No ARL version has helped

ARL', v2, v3 and v4 all failed their tests or were not selected. Given 2.1
to 2.5, this is expected: the reward term cannot help where the mode is not a
real choice, weak clients are gated out before the reward matters, and the
endpoints sit on the flat part of the curve.

### 3.7 Training is unstable: a round with few updates often wrecks the model

Found while reading the pilot runs, then checked on the saved tuning runs.
In every method and with every aggregation, accuracy often falls sharply
from one round to the next. Rounds 41-200, share of rounds that lose more
than 0.05 accuracy, by how many updates arrived in that round:

| runs | 1 update | 2 | 3 | 4 | 5 | 6 or more |
|---|---|---|---|---|---|---|
| tuned always-Light + MimiC (15 seeds) | 42% | 19% | 17% | 9% | 5% | 2% |
| always-Light + FedProx (M7 reference, 15 seeds) | 12% | 25% | 3% | 11% | 7% | 2% |
| tuned REFL (15 seeds; 0 fresh updates: 34%) | 12% | 9% | 4% | 3% | 3% | 2% |
| our method c34 (cost check, 5 seeds) | - | 18% | 19% | 7% | 1% | 2% |

- For the tuned control the mean change of accuracy in a round is -0.050
  with one update, -0.039 with two, -0.022 with three, -0.010 with four,
  0.000 with five and +0.008 with six or more. **Rounds with four or fewer
  updates (19% of all rounds) make the model worse on average.**
- A run has about 3 drops of more than 0.08 after round 40. One pilot run
  lost 0.66 in a single round (accuracy 0.05) and needed 30 rounds to
  recover.
- Cause: the server moves by the *mean* of whatever arrived, at full size.
  With one or two clients' updates on non-IID data, that mean is one
  writer's direction.
- Why it matters twice: it costs accuracy, and it makes the end points
  noisy (a dip inside the last rounds within the budget costs a seed 0.1 or
  more). Averaging the last 5 or 10 rounds (done since M7) only hides it.
- Fix (opt-in): `fl.min_full_step = m` -- with fewer than m updates the step
  is scaled by updates / m. Tested in the second pilot (6.5).

## 4. Plan

All new behaviour is opt-in (default off, old runs unchanged). Nothing is
re-pinned. Confirmation seeds and data stay untouched.

**Phase 1 -- code (here)**

| id | fix | for |
|---|---|---|
| F1 | Predictor v3: add estimated run time (uses data amount) and energy need per mode | 3.2 |
| F2 | Simulator options: link drop that grows with job time; fixed cost and time per job (model download); energy that follows the amount of work; latency capped at the deadline | 2.3, 2.5, 2.6 |
| F3 | MimiC with step scaling (`mimic_nova`) | 3.5 |
| F4 | Oracle schedulers (analysis only, they see the truth): upper bounds on what any scheduler could gain | 2.1, 2.7 |
| F5 | Report: cost to reach an accuracy, area under the accuracy-cost curve, time to accuracy | 2.6, 2.7 |
| F6 | Method: eligibility line and a hard cap at the target become tunable; a damage term that counts wasted cost instead of failed jobs | 3.1, 3.4 |
| F7 | A second block of development seeds (400-599) | -- |

**Phase 2 -- small checks on CPU (here, exploratory)**

| id | check | question |
|---|---|---|
| P1 | ground-truth table (done, section 2.1) | where can a job succeed? |
| P2 | oracle head-room | how much could a perfect scheduler gain over tuned always-Light, in the current world and in the more realistic one? |
| P3 | predictor check (no training needed) | is v3 better calibrated per mode than v1 and v2? |
| P4 | aggregation pilot | does step scaling help MimiC with mixed modes? |

**Phase 3 -- server (the user decides after Phase 2)**

Re-tune the method with the wider search space and the fixed parts; re-tune
the comparators with the same new options; then the screens as before.
Whether the more realistic world becomes the main setting is the user's
decision; the old world stays as a reported setting either way.

## 5. Status

(Updated as the work proceeds; see section 6 for results.)

| id | status |
|---|---|
| F1-F7 | done (commit 7d637a8; 23 new tests; default runs row-identical on 7 conditions) |
| P1 | done (2.1) |
| P3 | done (6.1, 6.2): predictor v3 does not matter; the job-rule table confirms 2.5 |
| P2 | done (6.3): the ceiling is close; heavier jobs lose even with perfect knowledge in the current world |
| P4 | done (6.4): step-scaled MimiC does not help; the lower eligibility line helps a little |
| 3.7 | found during the pilot; fix built (`fl.min_full_step`); second pilot done (6.5): with 5, a third to a half fewer dips and no loss at the end points -- a partial fix |
| 2.8 | built after the user's remark: devices that change over the day (`system.state_model: diurnal`); third pilot running (`configs/r5_life.yaml`) |

## 6. Results of the checks

All exploratory: a few development seeds, no test, no decision.

### 6.1 P3 -- predictor check (`scripts/predictor_check.py`, seeds 403-405)

No training is needed: only the simulator and the predictor run. Results in
`results/r5_predictor/`.

FEMNIST, jobs chosen by each version's own expected-work rule:

| predictor | Brier (lower is better) | Priority predicted / real | weak clients predicted / real |
|---|---|---|---|
| v1 (original) | 0.217 | 0.68 / 0.62 | 0.41 / 0.37 |
| v2 (repaired) | 0.209 | 0.72 / 0.69 | 0.40 / 0.37 |
| v3 (knows the workload) | 0.208 | 0.70 / 0.65 | 0.39 / 0.38 |

**My guess in 3.2 was mostly wrong.** v3 is not better than v2 on FEMNIST
(Brier 0.208 vs 0.209; the mode rule delivers the same training per cost).
v2 already learns each client's own record, and a client's speed and data
do not change, so the record carries the same information. On CIFAR-10 v3
is a little better (training delivered per unit cost by the mode rule:
v1 0.70, v2 0.74, v3 0.78). v3 stays as an option; it is not a fix that
matters. The over-prediction of Priority seen inside real runs comes from
the scheduler picking exactly the jobs the predictor is too hopeful about
(a selection effect), not from a missing input.

### 6.2 Which job-choice rule is the best buy? (same script)

Every online client, every round, is given the job a rule would choose, and
the round's true outcome is looked up. FEMNIST, predictor v2:

| rule | world | Light / Standard / Priority | jobs that succeed | epochs delivered per unit cost | successful jobs per unit cost |
|---|---|---|---|---|---|
| always Light | current | 100 / 0 / 0 | 50% | 0.67 | 0.67 |
| expected work (the baselines' rule) | current | 71 / 14 / 15 | 46% | 0.74 | 0.40 |
| expected work per unit cost | current | 98 / 2 / 0 | 50% | 0.66 | 0.66 |
| oracle: heaviest job that finishes | current | 40 / 33 / 26 | 100% | 1.47 | 0.69 |
| always Light | more realistic | 100 / 0 / 0 | 54% | 0.31 | 0.31 |
| expected work (the baselines' rule) | more realistic | 69 / 16 / 15 | 46% | 0.40 | 0.21 |
| expected work per unit cost | more realistic | 68 / 21 / 11 | 49% | 0.40 | 0.23 |
| oracle: heaviest job that finishes | more realistic | 41 / 34 / 24 | 100% | 0.86 | 0.42 |

- **Current world: the cost-best rule is always Light** (98% Light). This
  confirms 2.5 with numbers: the mode decision is settled by the cost model
  before any scheduler acts. Heavier jobs buy 10% more epochs per cost but
  40% fewer successful contributions per cost.
- **More realistic world: heavier jobs are worth it.** Choosing the job by
  expected work per cost delivers 29% more training per unit of cost than
  always Light (0.40 vs 0.31). Here the job size is a real decision.
- The oracle shows how much is lost to jobs that fail: about half of all
  spend, in both worlds.
- CIFAR-10 gives the same picture (`results/r5_predictor/cifar*`): the
  cost-best rule is 98% Light in the current world (0.67 against 0.68 for
  always Light), and delivers 30% more training per cost than always Light
  in the more realistic one (0.43 against 0.33).

### 6.3 P2 -- head-room: how much could any scheduler gain? (`results/r5_pilot`)

60 runs, FEMNIST, seeds 400-402, evidence complete. The reference is the
tuned always-Light control. Oracles see each job's true outcome; they are
upper bounds, not methods. Differences are paired by seed.

Current world (budget 1,400):

| arm | eq5 | w10b | area under accuracy-spend curve | cost to reach 0.75 | failed jobs |
|---|---|---|---|---|---|
| tuned always-Light (reference) | 0.774 | 0.490 | 0.648 | 893 | 43% |
| oracle: Light, no wasted job | +0.022 | +0.049 | +0.030 | 781 | 0% |
| oracle: + contribution-aware picks | +0.025 | +0.047 | +0.031 | 695 | 0% |
| oracle: + most equal contributions | +0.011 | +0.041 | +0.029 | 701 | 0% |
| oracle: heaviest job that finishes | -0.017 | -0.009 | +0.001 | 1,006 | 0% |

More realistic world (budget 3,267):

| arm | eq5 | w10b | area | cost to reach 0.75 | failed jobs |
|---|---|---|---|---|---|
| tuned always-Light (reference) | 0.782 | 0.502 | 0.636 | 2,242 | 34% |
| oracle: Light, no wasted job | +0.008 | +0.015 | +0.039 | 1,531 | 0% |
| oracle: + contribution-aware picks | +0.015 | +0.011 | +0.045 | 1,554 | 0% |
| oracle: + most equal contributions | +0.006 | +0.021 | +0.038 | 1,635 | 0% |
| oracle: heaviest job that finishes | -0.005 | +0.001 | +0.043 | 1,869 | 0% |

What this says:

1. **The ceiling is close.** A scheduler that never wastes a single job
   gains at most about +0.02 in equal-spend accuracy (eq5) and +0.05 in
   worst-10% accuracy (w10b) over the tuned control. A real scheduler cannot
   see the future, so it can only get part of that. The spread between seeds
   is about 0.02. So on eq5 a clear win is close to impossible in this test
   bed, whatever the method. w10b has more room.
2. **Heavier jobs lose at equal spend even with perfect knowledge** in the
   current world (-0.017 eq5 for 2 times the spend). This is the strongest
   evidence for 2.5: the mode half of the thesis's decision cannot pay off
   here.
3. **Heavier jobs help early and hurt late.** In the more realistic world
   the heavy-job oracle reaches 0.70 sooner (1,052 against 1,136 for the
   Light oracle) but 0.75 later (1,869 against 1,531). More local epochs
   mean faster early progress and more drift near the end. The best job size
   changes during training -- something a fixed rule cannot use and a
   learned scheduler could.
4. **The whole curve has room.** Area +0.03 to +0.045, cost to reach 0.75
   down by 13-31%. A measure of the whole curve can tell methods apart; the
   end points hardly can.
5. Contribution-aware picking (coverage) is the best oracle in both worlds,
   but only by a little over random picking among clients that succeed.

### 6.4 P4 and the method variants (same pilot)

Differences to the tuned always-Light control (eq5 / w10b / area):

| arm | current world | more realistic world |
|---|---|---|
| our method c34 (as tuned in M7) | -0.010 / -0.004 / -0.011 | -0.017 / -0.020 / +0.012 |
| c34 + predictor v3 | -0.022 / -0.011 / -0.011 | -0.084 / -0.074 / +0.018 (one run collapsed, see 3.7) |
| c34 + v3 + eligibility line 0.4 | -0.008 / +0.002 / -0.013 | 0.000 / +0.016 / +0.022 |
| always-Light with step-scaled MimiC | -0.014 / -0.001 / -0.011 | -0.021 / -0.013 / -0.014 |
| c34 with step-scaled MimiC | -0.015 / -0.001 / -0.029 | -0.011 / +0.004 / +0.004 |

- **Step-scaled MimiC (3.5) does not help.** It is worse for the control in
  both worlds. The concern was real on paper and does not matter in
  practice. Not pursued.
- **Predictor v3 alone does not help** (agrees with 6.1).
- **The lower eligibility line helps a little** (c34 + v3 + line 0.4 against
  c34 + v3: +0.014 eq5 and +0.013 w10b in the current world). It brings the
  method level with the control, not above it.
- Our method as tuned stays slightly below the tuned control on fresh
  seeds, as the tuning results suggested.

Three seeds: these are directions, not results.

### 6.5 Second pilot -- a smaller step when few updates arrive (`results/r5_stability`)

33 runs, FEMNIST, current world, seeds 400-402, evidence complete. Each
method as tuned, and with `fl.min_full_step = 5` (half a step with fewer
than 5 updates, scaled in proportion). For the control also 10 (the plain
"a missing update counts as zero" rule).

| method | dips per run: as tuned -> with 5 | eq5 change | w10b change | area change |
|---|---|---|---|---|
| tuned always-Light | 6.3 -> 4.0 | +0.002 | +0.020 | -0.005 |
| our method c34 | 5.7 -> 3.0 | +0.011 | 0.000 | -0.004 |
| c34 + v3 + line 0.4 | 6.0 -> 4.7 | -0.001 | +0.001 | -0.002 |
| REFL | 2.3 -> 1.0 | +0.001 | -0.006 | -0.007 |
| fairness queues | 4.7 -> 3.0 | +0.003 | -0.002 | +0.002 |
| always-Light with 10 | 6.3 -> 0.7 | -0.026 | -0.021 | -0.055 |

(A dip = a one-round loss of more than 0.08 accuracy after round 40.)

- **With 5: about a third to a half fewer dips for every method, and no
  loss at the end points.** Learning is a little slower early (area -0.002
  to -0.007).
- **With 10: almost no dips, but learning is clearly slower** (-0.026 eq5).
  Too strong.
- It is a partial fix. Dips also happen in rounds with many updates, so
  some remain. Three seeds: a direction, not a result.
- If adopted, it must be a shared setting for every method (it is a server
  rule, not part of any scheduler).

## 7. How the numbers in this review were made

- Section 2.1-2.3: the simulator alone, no training: for every online client
  and every mode, `ClientEnv.outcome` with the round's fixed random numbers
  (`scripts/sim_truth.py`).
- Section 6.1, 6.2: `scripts/predictor_check.py` (simulator and predictor
  only; random clients each round).
- Section 2.7, 3.1, 3.3: the saved runs in `results/rl2_tune*` and
  `results/rl2_cost_check` (per-round CSV columns `mode_*`, `n_abstain`,
  `weak_selected`, `fail_ratio`, `cost`, `accuracy_fixed`).
