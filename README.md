# Contribution-Aware RL for Fair Client Orchestration in Federated Learning

Implementation and experiment harness for the undergraduate thesis
*"Contribution-Aware Reinforcement Learning for Fair Client Orchestration in
Heterogeneous and Non-IID Federated Learning"* (Department of CSE, University of
Dhaka).

Chapter 3 of the thesis specifies the formulation. This repository is Chapters 4
and 5: the simulator, the policy, the baselines, and the scripts that produce the
numbers.

## Why this runs on nothing

The thesis is simulation-only by design -- Section 3.11's oracle ARL* is
*only* computable in simulation, since it requires counterfactual queries under
common random numbers. So there are no devices to buy and no cloud to rent. One
process simulates all clients sequentially.

Measured on 4 CPU cores, no GPU:

| Run | Time |
|---|---|
| Smoke test (synthetic, 20 clients, 15 rounds) | ~5 s |
| One MNIST run (30 clients, 100 rounds, CNN) | ~60 s |
| **Full Chapter 5 sweep (8 configs x 5 seeds)** | **~45 min** |

The entire experimental result set for the thesis is an overnight laptop job.
See [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for the free-compute
strategy and the scale justification.

## Quick start

```bash
pip install -r requirements.txt

# smoke test: synthetic data, finishes in seconds
PYTHONPATH=src python -m afl.run --config configs/quick.yaml

# a single real run
PYTHONPATH=src python -m afl.run --config configs/mnist.yaml --seed 0

# the full experimental design of Sec. 3.23
python scripts/run_experiments.py --config configs/mnist.yaml --seeds 0 1 2 3 4

# summary table and Chapter 5 figures
python scripts/report.py --dir results
python scripts/plot.py --dir results --out figures

python -m pytest tests -q
```

`run_experiments.py` skips any run whose CSV already exists, so a reclaimed
Colab session costs you one run rather than the whole sweep.

## Advanced sweep (FEMNIST, CIFAR-10, 100-200 clients, weak-client regime)

```bash
pip install pyarrow pillow && python scripts/prepare_femnist.py   # once
python scripts/sweep.py --list                                    # 22 groups, 1,580 runs
python scripts/sweep.py --groups 'weak_*' --jobs 4                # resumable, shardable
python scripts/stats.py --dir results/adv/weak_mnist --ref c2_rl_fedprox
```

To run it on Kaggle's free GPUs, import [notebooks/run_kaggle.ipynb](notebooks/run_kaggle.ipynb)
into Kaggle; its first cell has the setup steps and a session plan.

Design, rationale, compute costs and how to read the results:
[docs/ADVANCED_EXPERIMENTS.md](docs/ADVANCED_EXPERIMENTS.md).

## What maps to what

| Thesis | Code |
|---|---|
| Non-IID partition (Sec. 3.5) | `src/afl/data.py` -- Dirichlet split, class sketches `c_{i,k}` |
| Systems heterogeneity, uncertain participation (Sec. 3.6-3.7) | `src/afl/sim.py` -- device tiers, availability, common random numbers |
| Observed state `o_t` (Sec. 3.8) | `Experiment._state` in `src/afl/engine.py` |
| Success predictor `p^succ_{i,t}(E)` (Sec. 3.9) | `src/afl/predictor.py` |
| Eligibility `e^elig_{i,t}` (Sec. 3.10) | `Experiment._state`, gate applied in `policies/rl.py` |
| Representation value, ARL', ARL* (Sec. 3.11, Eq. 3.27/3.29) | `src/afl/metrics.py` |
| Fairness `CV_t` (Eq. 3.32) | `metrics.cv_fairness` |
| Modes Light/Standard/Priority (Sec. 3.15) | `SystemCfg` in `src/afl/config.py`, `sim.workload` |
| Acceptance rule `V_{i,t}` (Eq. 3.18) | `local.accepted` |
| FedAvg / FedProx / FedNova (Sec. 3.17) | `src/afl/aggregate.py`, prox term in `src/afl/local.py` |
| Eq. (3.54) importance reweighting | `aggregate(..., success_probs=...)` |
| Reward (Eq. 3.55) | `Experiment.run_round` |
| Policy `pi: o_t -> (S_t, E_t)` (Sec. 3.14) | `src/afl/policies/rl.py` |
| Baselines (Table 3.5) | `src/afl/policies/baselines.py` |
| Experimental conditions (Table 3.4, 3.5, ablations) | `CONDITIONS` in `src/afl/conditions.py` |
| Persistent weak-client regime | `ClientEnv.__init__` / `new_round` in `src/afl/sim.py` |
| Paired significance tests | `scripts/stats.py` |

## Metrics logged every round

`accuracy`, `acc_worst10` (mean accuracy of the worst 10% of classes), `test_loss`, `reward`, `cost`, `latency`, `n_selected`, `n_success`,
`fail_ratio` (`D^fail_t`), `arl_online` (`ARL'_t`), `arl_oracle` (`ARL*_t`),
`cv` (`CV_t`), `jain`, `ece` (predictor calibration error), per-mode assignment
counts, `weak_selected`/`weak_success` (weak-client regime), and wall-clock seconds.

`ARL'`, `ARL*` and `ece` are logged together on purpose: Section 3.23 asks for
all three, and the gap between ARL' and ARL* is only interpretable next to the
calibration of the predictor that produced ARL'.

## Status of Chapter 3 and Table 3.5 in code

Everything below is implemented, off by default (so earlier results are
reproduced exactly), and has its own experiment group in
`configs/sweep_advanced.yaml`:

| Thesis | Code switch | Group |
|---|---|---|
| Discount `gamma` (multi-round credit) | `policy.horizon > 1` | `ext_*` (`c2_gamma`) |
| Budgets `C_max`, `L_max` (Eq. 3.64-3.65) | `system.cost_max`, `system.latency_max` | `ext_*` (`*_budget`) |
| Importance reweighting (Eq. 3.54) | `fl.importance_reweight` | `ext_*` (`c2_reweight`) |
| Staleness rule (Eq. 3.18), semi-asynchronous | `fl.async_late` | `ext_*` (`c2_async`) |
| Fairness debt (Sec. 3.13) | `reward.lambda_debt` | `ext_*` (`c2_debt`) |
| Reward-weight sensitivity (Sec. 3.18) | `sens_*` conditions | `sens_cifar` |
| MIFA, MimiC, FedCG (Table 3.5), from the papers' algorithms | `fl.aggregator: mifa / mimic`, `policy.name: fedcg` | `ext_*` |
| Real device heterogeneity (FedScale trace) | `system.device_trace: fedscale` | `trace_*` |
| Simulator fix: real upload compression, Oort without the time leak | `fl.compress_upload`, `policy.oort_estimated_time` | `s2_*` |
| ARL-v3 (work-based loss + credit for delivered rare data) | `reward.damage_metric: arl_v3` | `s2_*` |
| Mode-fair baselines (shared expected-work mode rule) | `policy.mode_rule: expected_work` | `s2_*` (`*_ew`) |
| FAVOR, DELTA, REFL, from the papers' algorithms | `policy.name: favor / delta / refl` | `s2_*` |

## Known gaps

- **FedCER** is still the simplified `b_fedcer` ("FedCER-like"): its paper
  could not be identified from the name alone. Add the thesis's reference [6]
  and it can be implemented like the others.
- **Fairness debt** follows the standard virtual-queue form (each eligible
  client accrues K/N per round and pays it off with a successful
  contribution). Check it against the thesis's own Sec. 3.13 definition.
- **Asynchrony** is semi-asynchronous only: a late client may deliver up to
  `staleness_max` rounds late. A fully asynchronous server is not modelled.
- **Before the simulator fix** (every group except `s2_*`), Oort saw each
  round's true times and Light/Standard uploads were not really compressed;
  see "Simulator fix" in [docs/ADVANCED_RESULTS.md](docs/ADVANCED_RESULTS.md).
- **FAVOR** learns online within one run; the paper pre-trains its agent over
  many runs first.
- Everything is simulated; see [docs/REALISM_CALIBRATION.md](docs/REALISM_CALIBRATION.md).

## Authors

Hasibul Islam Sifat, Md. Nuruzzaman. Supervisor: Shabab Murshed, Lecturer,
Department of CSE, University of Dhaka.
