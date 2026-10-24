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
python scripts/sweep.py --list                                    # 16 groups, 1,060 runs
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

## Known gaps

- The FedCER/FedCG and MIFA/MimiC baselines of Table 3.5 are implemented as
  simplified "-like" versions (`b_fedcer`, `b_mimic_impute`), not the
  published algorithms; FedCG itself is not implemented.
- Defined in Chapter 3 but not used by the code: the discount factor `gamma`
  (the policy update is one-step REINFORCE), the cost/latency budgets
  `C_max`/`L_max` (Eq. 3.64-3.65), and the Eq. (3.54) importance reweighting
  (supported by `aggregate()` but never switched on). The staleness check
  never fires because rounds are synchronous.
- No reward-coefficient sensitivity sweep (Sec. 3.18) and no fairness-debt
  module (Sec. 3.13, optional).
- Everything is simulated; see [docs/REALISM_CALIBRATION.md](docs/REALISM_CALIBRATION.md).

## Authors

Hasibul Islam Sifat, Md. Nuruzzaman. Supervisor: Shabab Murshed, Lecturer,
Department of CSE, University of Dhaka.
