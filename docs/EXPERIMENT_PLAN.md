# Experiment plan on zero budget

This document records what the experiments in Chapter 5 will actually be, given
that the project has no funding, no cluster, and no paid cloud account. It is
written down rather than improvised because the scale decisions below are
defensible only if they are deliberate, stated, and consistent.

## The one fact that makes this project affordable

The thesis is **simulation-only**. Section 3.21 already assumes it, and
Section 3.11 requires it -- the oracle ARL* of Equation (3.29) is computable
*only* in simulation, because it needs counterfactual queries under common
random numbers. There are no real devices to buy, no network to rent, and no
deployment to maintain. Everything runs as one process on one machine that
simulates N clients sequentially.

That means the cost driver is not "federated learning". It is
`rounds x clients_per_round x local_epochs x samples_per_client` forward and
backward passes on a small model. That number is fully under our control, and
the configurations below keep it inside a free Colab session.

## Scale, and why it is enough

| Knob | Value | Reason |
|---|---|---|
| Clients `N` | 30 (100 for the scalability ablation) | Non-IID heterogeneity and starvation dynamics appear at 30. Nothing in Chapter 3 needs 1000. |
| Selected `K` | 5 | Standard 10-20% participation rate. |
| Rounds `T` | 100 | Enough for the accuracy curves to separate and for CV_t to stabilise. |
| Model | ~11k-param CNN | The thesis studies the *policy*, not the architecture. |
| Dataset | MNIST, Dirichlet(0.3) | Free, 11 MB, universally comparable. |
| Seeds | 5 | Non-negotiable -- see below. |

Full sweep: 8 configurations x 5 seeds = 40 runs. Measured on 4 CPU cores with
no GPU, one 100-round MNIST run takes about 60 seconds, so the entire Chapter 5
result set is roughly 45 minutes of laptop time. This is not a corner we are
cutting under protest -- it is the actual cost of the experiment.

**The scale is a limitation to state, not to hide.** Write it in Section 3.21 as
an explicit scope assumption and repeat it in the Chapter 6 limitations: results
are reported for a 30-client simulated population, and the scalability ablation
at N=100 is what speaks to larger deployments.

## Free compute, ranked

1. **Your own laptop.** The synthetic config runs a full experiment in under a
   minute. Do all development here. Never debug on a remote GPU.
2. **Google Colab free tier.** T4 GPU, sessions capped around 12 hours and
   reclaimed when idle. This is the workhorse. `scripts/run_experiments.py`
   skips runs whose CSV already exists, so a reclaimed session costs you one
   run, not the sweep -- mount Drive and point `--out` at it.
3. **Kaggle Notebooks.** ~30 GPU-hours per week, and sessions are *scheduled*
   rather than idle-killed, which makes them better than Colab for the long
   final sweep.
4. **Your university lab machines.** Ask the supervisor. A single shared
   workstation left running overnight beats every cloud free tier.
5. **Academic credit programmes.** GitHub Student Developer Pack, Google
   Research Credits, and AWS/Azure educational credits are worth one afternoon
   of applications. Do not block on them.

Do **not** pay for anything. Nothing in this thesis needs it.

## Rules that protect the results

**Five seeds, always, and report the standard deviation.** With K=5 sampled from
N=30, the round-to-round variance is large. A single-seed run will show a 2-3
point "improvement" that is pure noise, and a committee member who asks for the
variance will find nothing there. `scripts/report.py` always prints the std for
this reason.

**Change one thing at a time.** Table 3.4's four conditions exist to attribute
any gain to either the RL orchestration or the aggregation rule. If the RL
hyperparameters differ between conditions, that attribution is void.

**Log every round, plot afterwards.** Every run writes a per-round CSV. Chapter 5
plots are produced from those CSVs, never from a re-run. Re-running to make a
plot is how a thesis ends up with figures that do not match its tables.

**Report ARL' next to ARL* next to the calibration error.** Section 3.23 asks for
all three together, and they only mean something as a set: ARL' is what the
policy believed it was optimising, ARL* is what a perfectly-informed scheduler
could have achieved, and the ECE says how much to trust the gap between them.

## What is implemented, and what is not

Implemented: the full round loop, the three participation modes, the
contribution-success predictor, online ARL' and oracle ARL*, CV_t and Jain
fairness, FedAvg/FedProx/FedNova, the acceptance rule of Equation (3.18), the
importance-reweighting correction of Equation (3.54), and the Random / FedCS /
Power-of-Choice / Oort baselines.

> **Update:** items 1, 2 and 4 below have since been done -- FedCER-like and
> MIFA/MimiC-like baselines in `src/afl/policies/baselines.py` and
> `fl.impute_missing`, and CIFAR-10, FEMNIST and alpha in {0.1, 0.5} in the
> advanced sweep ([ADVANCED_EXPERIMENTS.md](ADVANCED_EXPERIMENTS.md)). The
> list is kept as originally written.

Not implemented, in the order they are worth adding:

1. **FedCER / FedCG** (Table 3.5) -- the closest published joint
   selection-and-configuration competitor, and therefore the most important
   missing baseline. Without it the "closest prior work" claim in Section 3.22
   rests on argument rather than measurement.
2. **MIFA / MimiC** (Table 3.5) -- needed for the prevention-versus-imputation
   argument of Section 3.11. That argument is currently untested.
3. **The fairness-debt module** (Section 3.13) -- explicitly optional in the
   thesis, and it should stay optional until the core results are in.
4. **CIFAR-10 and a second non-IID regime** (alpha in {0.1, 0.5}) -- a
   robustness check, not a core result. Do it only if time allows.

Items 1 and 2 are the honest gap. If they do not get implemented, say so in
Chapter 6 rather than letting Table 3.5 imply they were run.

## Order of work

1. Get Condition 2 (RL + FedProx) and the Random baseline running on MNIST for
   5 seeds. That is the minimum viable result: if RL does not beat random, stop
   and debug before building anything else.
2. Add Conditions 1, 3, 4 -- the aggregation ablation.
3. Add FedCS, Power-of-Choice, Oort.
4. Run the reward-coefficient sensitivity sweep that Section 3.18 requires.
5. Only then consider FedCER and the second dataset.
