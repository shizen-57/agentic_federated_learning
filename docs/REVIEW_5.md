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
| P2, P4 | running (`configs/r5_pilot.yaml`, 60 runs) |

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

## 7. How the numbers in this review were made

- Section 2.1-2.3: the simulator alone, no training: for every online client
  and every mode, `ClientEnv.outcome` with the round's fixed random numbers
  (`scripts/sim_truth.py`).
- Section 6.1, 6.2: `scripts/predictor_check.py` (simulator and predictor
  only; random clients each round).
- Section 2.7, 3.1, 3.3: the saved runs in `results/rl2_tune*` and
  `results/rl2_cost_check` (per-round CSV columns `mode_*`, `n_abstain`,
  `weak_selected`, `fail_ratio`, `cost`, `accuracy_fixed`).
