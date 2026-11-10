# The simulated world: equations, units, values, sources

What every simulated quantity is, how it is computed (`src/afl/sim.py`,
`src/afl/engine.py`), the values the s2 / RL2 experiments use, and where each
value comes from. "Assumed" marks a value chosen by us, not measured. Costs
and energy are **simulated units**, not joules, seconds or currency; only
ratios between them are meaningful. Companion documents:
`docs/REALISM_CALIBRATION.md` (what is calibrated against which source) and
`docs/VALIDATION.md` (the checks).

In the terms of FLGo (Wang et al., IJCAI 2023), which names four kinds of
system heterogeneity, the simulator models all four:
- **availability** (diurnal, sticky, per device);
- **responsiveness** (time to finish against a deadline);
- **completeness** (failures: battery, deadline, dropped link, rejected update);
- **connectivity** (clients that drop out; "churn", off in the main settings).

## Data and clients

| Quantity | Value (s2 / RL2) | Source |
|---|---|---|
| FEMNIST | LEAF's FEMNIST, one writer = one client; 100 clients per run, 150-400 examples each | LEAF benchmark (Caldas et al. 2018), Flower's copy |
| CIFAR-10 | 100 clients, Dirichlet(alpha 0.3) label split, 50-600 examples | standard non-IID protocol (Hsu et al. 2019) |
| Data roles (persistent split) | FEMNIST writers: practice 1,798 / development 899 / confirmation 900; CIFAR-10 test examples: validation 3,006 / development 3,488 / confirmation 3,506 | fixed by `data.split_seed`; ids pinned in `configs/rl2_data_roles.json` |
| Clients per round K | 10 | assumed (typical) |
| Rounds | 200 | assumed |

**Per-client test examples (gap review B7).** FEMNIST holds out 10% of each
chosen writer's examples (15-40, median 18) and 70% of those are test
examples: a median of about 13 per client. A client's accuracy is therefore
coarse (steps of about 8 points), and the worst-10% measure (the mean of the
10 lowest of 100 clients) is noisy. Comparisons are paired (same clients,
same examples, per seed), which removes most of that noise from the
differences. CIFAR-10 clients have no held-out examples of their own; their
accuracy is the per-class accuracy weighted by their label mix
(`engine._client_metrics`).

**How far apart the data roles are (gap review B9).** FEMNIST's practice,
development and confirmation clients are different people (writers). On
CIFAR-10 there are no people: every run draws its 100 clients' data from the
same 50,000 training images (a Dirichlet split per seed), and the reward's
validation images are shared by every run; only the test images are split
into development and confirmation. So on CIFAR-10 the separation is weaker:
the scheduler has trained on episodes made of the same training images it
later meets. The scheduler sees device, availability and coverage
features, not images, so what it could carry over is small, but the thesis
states it.

## Devices

Three tiers (slow / mid / fast) with probabilities 0.35 / 0.40 / 0.25.

Each device gets its own values:
- speed s_i = {0.35, 1.0, 2.2} x U(0.8, 1.2);
- bandwidth b_i = {0.3, 1.0, 2.0} x U(0.7, 1.3);
- link reliability r_i = clip(0.55 + 0.15 x tier + N(0, 0.08), 0.3, 0.98);
- battery capacity e_i = {0.4, 1.0, 2.5} x U(0.8, 1.2).

The heavy-tailed shape follows FedScale and Oort, which measured real
phones; the tier values themselves are assumed. Option `system.device_trace:
fedscale` replaces speed and bandwidth with FedScale's measured device
sample (`src/afl/traces`).

**Weak clients:** 30%, chosen as the holders of the rarest classes. They have:
- speed 0.8 x U(0.9, 1.1);
- bandwidth 0.6 x U(0.85, 1.15);
- reliability 0.6;
- availability 0.1 lower than other clients.

These values are assumed: chosen so that Light usually works for them,
Standard rarely, Priority almost never.

**Sensitivity scales:** `system.energy_scale` and `system.bandwidth_scale`
multiply every battery capacity and bandwidth (1.0 = unchanged).

## Availability

P(online_i,t) = clip(0.85 + 0.35 x a_i x cos(2 pi (h_t - p_i) / 24) + shift_i, 0.03, 0.99),

where:
- h_t = 24 (t mod 96) / 96, so one round is about 15 minutes;
- the peak hour is p_i ~ N(3 am, 3 h);
- the amplitude scale is a_i ~ clip(N(1, 0.25), 0.3, 1.6).

A client keeps last round's state with probability 0.8 (sticky availability).

Source: the diurnal shape follows Hard et al. 2018 (Gboard). The 0.85 mean,
0.35 swing and 0.8 stickiness are assumed.

## One job: time, energy, outcome

Modes m in {Light, Standard, Priority}:
- epochs E_m = 1 / 2 / 4;
- upload fraction f_m = 0.25 / 0.5 / 1.0.

n_i = the client's examples, rescaled so their mean is 500 (`work_ref_samples`).

- **Training time:** t_train = E_m x n_i / 1200 / s_i x J, with jitter
  J ~ LogNormal(0, 0.35).
  - Checked: real training time is proportional to n x E, with R^2 0.997
    (FEMNIST) and 0.999 (CIFAR-10) (`results/simulator_checks/timing.md`).
  - The constants 1200 and 0.35 are assumed.
- **Upload time:** t_comm = 0.25 x f_m / b_i x C, with channel noise
  C ~ LogNormal(0, 0.45).
  - Upload bytes really are proportional to f_m: 222 / 444 / 888 KB for
    FEMNIST's model.
  - The constants are assumed.
- **Energy:** 0.08 x E_m / s_i + 0.05 x f_m (simulated units, assumed).
  - Batteries recharge by 0.25 x capacity every round.
  - A job is charged energy and cost only if it actually ran (`charge_actual`).
- **Outcome:** checked in this order, with all random numbers drawn before
  any decision (common random numbers, so "what if another mode" compares
  like with like):
  - offline -> fail;
  - energy > battery at round start -> fail (energy);
  - link roll > r_i -> fail (dropped);
  - t_train + t_comm > deadline 1.0 -> fail (deadline);
  - a 2% residual rejection, plus the trainer's norm check -> fail (rejected);
  - otherwise success.
- **Cost:** C = E_m x n_i / 1000 + f_m (simulated units), linear in examples,
  epochs and upload size by construction.
  - The budget step (hard cap, `cost_max`, `latency_max`) changes or drops
    a job before it runs.
  - A dropped job runs nothing, costs nothing and trains nothing
    (`tests/test_validation_matrix.py`).

## What the scheduler knows (gap review B11)

Every policy sees each device's speed and bandwidth with only 15% noise
(`speed_hat`, `bw_hat`), the battery level, an availability hint, and its
history. The server-side time estimate (`policies/base.est_time`) uses the
simulator's own time formula, without the hidden per-round jitter and
channel noise. A real server would know much less about the time function.
Feasibility therefore looks easier here than in practice. Every method gets
the same information, so the comparison stays fair, but absolute success
rates are likely optimistic.

**Communication time (gap review D4).** Upload bytes per mode were checked
against the real model (`results/simulator_checks/timing.md`). Upload *time*
is modelled (0.25 x f / b x channel noise) and not measured: it depends on
real networks, which this project does not have. With
`system.device_trace: fedscale` the bandwidths come from FedScale's measured
phones; the sensitivity screen and the confirmation's secondary worlds use
that setting.

## What is not calibrated (stated in the thesis)

- The energy model and the battery values.
- The 2% rejection rate.
- The tier values (as opposed to their shape).
- The deadline and the time constants.
- What a round means in clock time.

The sensitivity screen (`configs/rl2_sensitivity.yaml`) varies deadline,
availability, battery and bandwidth one at a time, swaps the device tiers for
the measured FedScale trace, and changes the scenario (random weak clients,
no weak clients, churn), so that no conclusion rests on one chosen setting.
The confirmation repeats the last two as secondary worlds.

## Built-in limits, and more realistic options (review 5)

`python scripts/sim_truth.py` measures the simulator's ground truth without
any training: how often a job would succeed for each kind of client. In the
main setting (FEMNIST, seeds 280-284):

| client group | share | Light | Standard | Priority | Light fails by: drop / deadline |
|---|---|---|---|---|---|
| slow devices | 26% | 0.17 | 0.00 | 0.00 | 0.45 / 0.38 |
| middle devices | 25% | 0.69 | 0.50 | 0.06 | 0.29 / 0.01 |
| fast devices | 19% | 0.84 | 0.83 | 0.65 | 0.14 / 0.00 |
| weak clients | 30% | 0.41 | 0.04 | 0.00 | 0.41 / 0.18 |

What this means, and what the thesis must say (docs/REVIEW_5.md, section 2):

- The job size is a real choice for fewer than half of the clients.
- Slow devices can hardly ever contribute: a fairness ceiling no scheduler
  can pass.
- The link drop is one coin flip per client and round, whatever the job, and
  it is the largest cause of failure. No job choice can avoid it.
- Weak clients succeed 41% of the time with their best job, below the 0.5
  eligibility line.
- Modes cost 1 : 2 : 4 with no fixed cost per job, so Light always buys the
  most training per unit of cost.

Options that remove the unrealistic parts (all off by default; earlier runs
are unchanged):

| Option | What it does | Why |
|---|---|---|
| `system.drop_model: exposure`, `system.drop_ref_time` | the link holds with probability reliability ^ (job time / reference time), time capped at the deadline; the same roll decides every mode | a longer job is more likely to be interrupted (REFL and FedScale model drop-out through availability windows; Bonawitz et al. 2019 report 6-10% drop-out per round) |
| `system.job_overhead`, `system.job_overhead_time` | a fixed cost and time for every job that starts: the download of the whole model | every invited client receives the full model, whatever its job; with 1.0 the download costs as much as a full upload (the same bytes) |
| `system.energy_by_work` | the training part of the energy follows epochs x data | time and cost already do; before, a client with three times the data used the same energy |
| `system.cap_latency` | a job that misses the deadline counts as one deadline of waiting | the server stops waiting at the deadline |

| `system.state_model: diurnal` (+ `state_good_p`, `state_good_speed`, `state_poor_speed`, `state_good_bandwidth`, `state_poor_bandwidth`, `state_poor_charge`) | each round a client is in a good state (plugged in, Wi-Fi, idle) or a poor one (battery, mobile data, in use), on its own day/night cycle and sticky like availability; speed x1.25 / x0.6, bandwidth x1.5 / x0.4, recharge only in the good state | a real device is not the same device all day (Hard et al. 2018: eligibility = charging, idle, unmetered network); without it a weak client is weak in every round and no decision can ever save its contribution |

The "real" setting of `configs/r5_pilot.yaml` turns on the first four
(`drop_ref_time` 1.0, `job_overhead` 1.0, `job_overhead_time` 0.05). Its
ground truth: slow 0.09, middle 0.82 / 0.50 / 0.05, fast 0.94 / 0.90 / 0.66,
weak 0.38 / 0.02 / 0.00 (Light / Standard / Priority). A Light job then
costs 1.67 instead of 0.67, Standard 1.4 times a Light job and Priority 2.2
times, so a heavier job on a reliable device becomes worth its cost.

The "life" setting adds the device states to "real". Its ground truth is
in docs/REVIEW_5.md, section 2.8: a weak client succeeds 58% of the time
with Light in its good state and 2% in its poor state, so when to invite
whom becomes a real decision.

These values are assumed, like the rest of the cost model. They are a
second world to report next to the first, not a calibration.
