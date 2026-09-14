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
| Experimental conditions (Table 3.4) | `RUNS` in `scripts/run_experiments.py` |

## Metrics logged every round

`accuracy`, `test_loss`, `reward`, `cost`, `latency`, `n_selected`, `n_success`,
`fail_ratio` (`D^fail_t`), `arl_online` (`ARL'_t`), `arl_oracle` (`ARL*_t`),
`cv` (`CV_t`), `jain`, `ece` (predictor calibration error), per-mode assignment
counts, and wall-clock seconds.

`ARL'`, `ARL*` and `ece` are logged together on purpose: Section 3.23 asks for
all three, and the gap between ARL' and ARL* is only interpretable next to the
calibration of the predictor that produced ARL'.

## Known gaps

The FedCER/FedCG and MIFA/MimiC baselines of Table 3.5 are **not** implemented.
They are the two most important missing pieces and are the first items in
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md). Until they exist, Section
3.22's "closest prior work" claim and Section 3.11's prevention-versus-imputation
argument rest on argument rather than measurement.

## Authors

Hasibul Islam Sifat, Md. Nuruzzaman. Supervisor: Shabab Murshed, Lecturer,
Department of CSE, University of Dhaka.
