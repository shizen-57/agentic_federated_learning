# Advanced experiments: bigger, harder, and aimed at the claims

The Chapter 5 pilot (MNIST, 30 clients, 5 per round, 5 seeds) answered "does
the system work?" It could not answer "does each of the two claimed mechanisms
matter?" -- both ablations came out null (see CONTRIBUTION_VALIDATION.md), and
with 5 seeds no paired difference could ever be significant. This document is
the design that replaces it.

## What changed, in one table

| | Pilot (Chapter 5) | Advanced sweep |
|---|---|---|
| Datasets | MNIST | MNIST, **FEMNIST**, **CIFAR-10** |
| Non-IID split | Dirichlet α=0.3 | Dirichlet α ∈ {0.1, 0.3, 0.5}; FEMNIST uses its **natural split -- one client per real writer** |
| Clients / per round | 30 / 5 | **100 / 10** (main); 30/5 and 200/20 (scale) |
| Rounds | 100 | 100 (MNIST), 200 (FEMNIST, CIFAR-10) |
| Model | CNN, 11k params | 11k (MNIST), 222k (FEMNIST), 189k (CIFAR-10) |
| Failure model | independent every round | adds a **persistent weak-client regime** (below) |
| Conditions | 13, four run by hand | 14, all in one registry (`src/afl/conditions.py`) |
| Seeds | 5 | **10** |
| Statistics | mean ± std | + **paired sign-flip tests, bootstrap CIs, Holm correction** |
| New metric | -- | `acc_worst10`: accuracy on the worst 10% of classes |

Note: earlier versions of the docs described the MNIST model as "~20k
parameters". It has 11,274; `tests/test_advanced.py` now checks every model's
size. Chapter 5's text should use the corrected number.

## The key addition: persistent weak clients

**Why.** The pilot's null result had a likely cause: failures were drawn
independently every round, so "a valuable client failed" and "some client
failed" were almost the same event. ARL weights failures by data value, and
that can only differ from a raw failure count if failures are concentrated on
particular clients.

**What it does** (`system.weak_frac`, `weak_select`, `weak_reliability`,
`weak_avail_drop`, `avail_persistence`; all off by default):

- A fixed 30% of clients are *weak* for the whole run: device speed 0.8 and
  bandwidth 0.6 (the mid tier is 1.0), link reliability ~0.6 (mid tier
  ~0.7), availability 0.1 lower.
- They are calibrated to be **mode-sensitive, not dead**. Measured success
  rates under Light / Standard / Priority: weak 0.33 / 0.13 / 0.04, all other
  clients 0.43 / 0.28 / 0.14. A first version put weak clients on the low
  device tier; they then succeeded 9% of the time under Light and never under
  heavier modes, so no scheduling decision could rescue them and the regime
  tested nothing. `test_weak_clients_are_mode_sensitive_not_dead` guards this.
  (The same measurement shows the *original* simulator's low tier -- 35% of
  clients -- succeeds only ~16% of the time even under Light, despite the
  `sim.py` docstring saying Light is "feasible for almost everyone".)
- Availability is *sticky*: with probability 0.8 a client keeps last round's
  on/off state, so offline stretches last several rounds.
- `weak_rare`: the weak clients are the ones holding the **rarest classes**
  (highest `sum_k c_ik log(1/q_k)`). This is the setting the thesis argues ARL
  exists for -- the clients you most need are the ones most likely to fail,
  and a heavy mode breaks them.
- `weak_random` (control): same weakness, random 30%. If ARL beats the raw
  failure-count reward under `weak_rare` but not under `weak_random`, the
  effect is about *value*, which is exactly the claim.

Common random numbers survive: every draw is still made before the policy
acts, so the ARL* oracle stays valid (`test_sticky_availability_...`).

**Be honest about what this is.** It is a stress regime designed to test a
claim, not a measured property of real fleets. The thesis should present it as
"the regime in which the mechanism is predicted to matter", and report the
`weak_random` control and the original independent-failure results beside it.

## A second finding: most of ARL' is about clients never invited

Eq. (3.27) sums ARL' over **all** clients, so an eligible client that was
simply not invited counts as a loss. Instrumenting the round loop shows how
much of the signal that is:

| Setting | Mean ARL'_t | From invited clients that failed | From eligible clients never invited |
|---|---|---|---|
| N=30, K=5 (pilot) | 3.75 | 0.54 (14%) | 3.22 (86%) |
| N=100, K=10 | 6.87 | 0.65 (9%) | 6.22 (91%) |
| N=100, K=10, weak_rare | 7.05 | 0.73 (10%) | 6.32 (90%) |

So the part of ARL that rewards *protecting a valuable client from a bad
configuration* -- the part the prevention argument rests on -- is about a
tenth of the term. The rest rewards choosing a high-coverage set (which
shrinks every outsider's marginal value). This is a second candidate
explanation for the pilot's null ablation, alongside independent failures.

To separate the two, the sweep adds `ablation_arlsel`: the same policy with
ARL' counted only over invited clients (`reward.damage_metric:
arl_selected`). Every run also logs `arl_selected`. The thesis's own formula
stays the default for `c2_rl_fedprox`; which one the thesis should use is a
decision for the authors once the results are in.

## The design: 22 groups, 1,580 runs

`configs/sweep_advanced.yaml` (`python scripts/sweep.py --list`):

| Family | Groups | Conditions | Question |
|---|---|---|---|
| Main | `main_{mnist,femnist,cifar}` | all 14 | Does the full comparison hold on harder data and at N=100? |
| Weak, rare-data | `weak_{mnist,femnist,cifar}` | 8 (claims + baselines) | **Does ARL beat a raw failure count when valuable clients are the weak ones?** |
| Weak, random | `weakrand_{mnist,femnist}` | 4 (c2 + three ablations) | Control: is any effect about value? |
| Scale | `scale_n{30,200}_{mnist,femnist}` | 4 | Does it hold from 30 to 200 clients? |
| Non-IID | `alpha{01,05}_{mnist,cifar}` | 4 | Does it hold for very skewed and milder splits? |
| Confirmation | `confirm_cifar` | 2, fresh seeds 10-39 | Pre-specified test of ARL vs the plain failure count (ADVANCED_RESULTS.md) |
| Chapter 3 extensions | `ext_{mnist,cifar}` | 11 (`ext`) | Do gamma, Eq. 3.54, semi-async staleness, fairness debt and budgets change the picture? How do the published MIFA, MimiC and FedCG compare? |
| Sensitivity | `sens_cifar` | 8 (`sens`) | Does the result depend on the reward weights (Sec. 3.18), and does the ARL term matter at all (`lambda_arl` = 0)? |
| Real devices | `trace_{mnist,cifar}` | 8 (`focus`) | Does it hold with FedScale's measured phones instead of guessed tiers? |

Condition sets: `all` = the 14 core conditions; `focus` = c2, the three
ablations, Random, Oort, FedCS, FedCER-like; `core` = c2, raw-failure
ablation, Random, Oort; `claims` = c2 and the three ablations; `ext` = c2,
c2_gamma, c2_reweight, c2_async, c2_debt, c2/Random/Oort under budgets
(C_max = 10, L_max = 1.0), MIFA, MimiC, FedCG; `sens` = c2 and lambda_arl in
{0, 0.25, 1, 2}, lambda_fair in {0, 0.1, 1}.

## The remaining pieces of Chapter 3, and the published baselines

Each is off by default; re-running `weak_mnist` seed 0 (all 8 conditions)
with them in the code reproduced the committed CSVs exactly in every column
of every round.

- **gamma** (`policy.horizon`): with H > 1 the policy is updated once per
  H-round window on discounted returns G_t = sum_k gamma^k r_{t+k}, so a choice
  that pays off a few rounds later can be credited. `c2_gamma` uses H = 10.
- **C_max / L_max** (`system.cost_max`, `system.latency_max`): applied to every
  policy's chosen action from observable estimates -- a client whose
  estimated time exceeds L_max moves to a lighter mode or is dropped; while
  the round's cost exceeds C_max the lowest-priority client moves to a
  lighter mode or is dropped. Same rule for RL, Random and Oort.
- **Eq. 3.54** (`fl.importance_reweight`): updates weighted by n_i / p_succ,
  with p_succ floored at 0.1 (`fl.reweight_floor`) -- without a floor one
  lucky success predicted at 1% would get 100x weight.
- **Staleness** (`fl.async_late`): a client that misses the deadline but would
  finish within `staleness_max` rounds trains on the round-t model and
  delivers next round; Eq. 3.18's staleness check decides whether it counts.
  Semi-asynchronous only.
- **Fairness debt** (`reward.lambda_debt`): d_i <- max(0, d_i + (K/N) e_i - s_i)
  with e_i the eligibility gate and s_i a successful contribution; the mean
  debt is a reward penalty and each client's debt a policy input. This is
  the standard virtual-queue form; check it against the thesis's Sec. 3.13.
- **MIFA** (Gu et al., NeurIPS 2021, Alg. 1): the server keeps every client's
  latest update (zero before it first reports) and steps by their mean over
  all N clients.
- **MimiC** (Sun et al., IEEE TMC 2024, Alg. 1): each received update g_i is
  corrected with the client's correction variable from its last active
  round, v_i = g_i + c_i; the server steps by the mean v and sets c_i = v - g_i.
- **FedCG** (Li et al., arXiv 2212.09483, Alg. 1-2): greedy facility-location
  selection on clients' latest updates (count-sketch projected to 1,024
  dimensions; clients never seen get one probe epoch), a per-client
  compression ratio set to the largest value whose estimated time fits the
  deadline, top-k sparsification with error feedback, equal-weight mean.
  Like the paper it does not model availability.
- **FedScale device trace** (`system.device_trace: fedscale`): per-client
  compute speed and bandwidth drawn from a 20,000-device sample of FedScale's
  measured `client_device_capacity` trace (Apache-2.0; `src/afl/traces/`),
  relative to the trace median. Reliability and battery still come from the
  speed tier, since the trace does not record them.

After the simulator fix (`s2_*` groups; why and how in
docs/ADVANCED_RESULTS.md, "Simulator fix"):

- **Real upload compression** (`fl.compress_upload`): a client's update is
  top-k sparsified to its mode's upload fraction with error feedback, for
  every aggregator. **Oort** uses estimated, not true, round times
  (`policy.oort_estimated_time`).
- **Mode rule** (`policy.mode_rule`): `expected_work` gives each selected
  client the mode with the most P(success) x local epochs among modes with
  P >= 0.5; `light` puts everyone on Light. Lets selection-only baselines
  choose modes like the RL policy (`*_ew`), and gives two RL controls
  (`rl_ew`, `rl_light`).
- **ARL-v3** (`reward.damage_metric: arl_v3`, `reward.lambda_cov`).
- **FAVOR** (Wang et al., INFOCOM 2020), **DELTA** (Wang et al., NeurIPS
  2023, practical version) and **REFL** (Abdelmoniem et al., EuroSys 2023),
  in `src/afl/policies/published.py`.

MIFA and MimiC are aggregation rules for absent clients; their papers sample
clients without a scheduler, so they run under random selection. FedCER is
still the simplified `b_fedcer`: its paper could not be identified from the
name alone.

## How to run it

```bash
pip install -r requirements.txt
pip install pyarrow pillow          # only for the one-off FEMNIST conversion
python scripts/prepare_femnist.py   # ~200 MB download, once

python scripts/sweep.py --list                        # plan and what's left
python scripts/sweep.py --groups 'weak_mnist' --jobs 4

# summaries and paired tests for one group
python scripts/report.py --dir results/adv/weak_mnist
python scripts/stats.py  --dir results/adv/weak_mnist --ref c2_rl_fedprox
```

- `--jobs N` runs N processes side by side (on a CPU, several single-thread
  runs beat one run on every core; on a GPU they share the card).
- `--shard i/n` lets n machines (e.g. several free Kaggle sessions) split one
  group with no coordination: machine i takes every n-th run.
- Runs are resumable: a run whose CSV exists is skipped, and CSVs are written
  atomically, so a killed session never leaves a truncated file.
- `fl.device: auto` uses a GPU when one exists, otherwise the CPU.

## Compute

Measured on this project's 4-core cloud CPU (no GPU), one run per core:

| Dataset | Per round | One run | Runs in plan | Whole dataset, 4 runs at a time |
|---|---|---|---|---|
| MNIST, N=100 | ~2 s | ~3-4 min | 420 | **~6 h** |
| FEMNIST, N=100 | ~1.2 s | ~4-5 min | 340 | **~6-7 h** |
| CIFAR-10, N=100 | ~7 s | ~20-25 min | 300 | **~25 h** |

MNIST and FEMNIST are fine on any 4-core machine overnight. CIFAR-10 is the
one to put on a GPU.

**On Kaggle** use `notebooks/run_kaggle.ipynb`: it reads a GitHub token from
Kaggle Secrets (the repository is private), runs any group in the
background with *Save & Run All*, stops starting new runs before Kaggle's
12-hour limit (`--max-hours`), spreads runs over both GPUs of a T4 x2
machine, and continues an unfinished group in the next session from the
previous version's output. Its first cell lists a suggested 8-session plan.
`notebooks/run_sweep.ipynb` is the simpler Colab version.

**CIFAR-10 on Kaggle: attach a dataset.** The official download server
(Toronto) can take 20+ minutes for 170 MB. In the notebook's right panel use
*Add Input -> Datasets*, search "cifar10 python" and add one containing
`data_batch_1`...`data_batch_5` and `test_batch`; `afl.data` finds it under
`/kaggle/input` and builds the same arrays as the official download (checked:
bit-identical), so results are comparable whichever source is used.
GPU speed-ups were not measured (no GPU was available while building this);
expect most of the gain on CIFAR-10, since the simulator, predictor and policy
run on the CPU either way.

Measured update norms are 0.2-1.6 for every model, far below the acceptance
rule's `norm_clip = 10`, so the larger models are not spuriously rejected.

## Reading the results

- **The claim test** is `stats.py --dir results/adv/weak_<dataset> --ref
  c2_rl_fedprox --vs ablation_rawfail ablation_cvsel ablation_arlsel`, and
  the same with `--ref ablation_arlsel --vs ablation_rawfail`. Look at
  `final_acc_worst10` and `mean_arl_oracle` first: those are what
  representation loss should move. Overall accuracy can hide it.
- With 10 seeds the smallest possible exact p-value is 2/1024 ≈ 0.002, so a
  consistent effect can reach significance even after Holm correction across
  six or more comparisons. With the pilot's 5 seeds the floor was 0.0625.
- Report the null results too. If ARL still does not beat the raw failure
  count under `weak_rare`, that is a strong, publishable negative: the
  mechanism does not matter even in the regime built for it.
