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

## What is not calibrated (stated in the thesis)

- The energy model and the battery values.
- The 2% rejection rate.
- The tier values (as opposed to their shape).
- The deadline and the time constants.
- What a round means in clock time.

The sensitivity screen (`configs/rl2_sensitivity.yaml`) varies deadline,
availability, battery and bandwidth one at a time, so that no conclusion
rests on one chosen setting.
