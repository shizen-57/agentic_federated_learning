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
| Conditions | 13, four run by hand | 13, all in one registry (`src/afl/conditions.py`) |
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

## The design: 16 groups, 980 runs

`configs/sweep_advanced.yaml` (`python scripts/sweep.py --list`):

| Family | Groups | Conditions | Question |
|---|---|---|---|
| Main | `main_{mnist,femnist,cifar}` | all 13 | Does the full comparison hold on harder data and at N=100? |
| Weak, rare-data | `weak_{mnist,femnist,cifar}` | 7 (claims + baselines) | **Does ARL beat a raw failure count when valuable clients are the weak ones?** |
| Weak, random | `weakrand_{mnist,femnist}` | 3 (c2 + both ablations) | Control: is any effect about value? |
| Scale | `scale_n{30,200}_{mnist,femnist}` | 4 | Does it hold from 30 to 200 clients? |
| Non-IID | `alpha{01,05}_{mnist,cifar}` | 4 | Does it hold for very skewed and milder splits? |

Conditions sets: `all` = the 13 of `afl.conditions`; `focus` = c2, both
ablations, Random, Oort, FedCS, FedCER-like; `core` = c2, raw-failure
ablation, Random, Oort; `claims` = c2 and both ablations.

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

TIMINGS_PLACEHOLDER

## Reading the results

- **The claim test** is `stats.py --dir results/adv/weak_<dataset> --ref
  c2_rl_fedprox --vs ablation_rawfail ablation_cvsel`. Look at
  `final_acc_worst10` and `mean_arl_oracle` first: those are what
  representation loss should move. Overall accuracy can hide it.
- With 10 seeds the smallest possible exact p-value is 2/1024 ≈ 0.002, so a
  consistent effect can reach significance even after Holm correction across
  six or more comparisons. With the pilot's 5 seeds the floor was 0.0625.
- Report the null results too. If ARL still does not beat the raw failure
  count under `weak_rare`, that is a strong, publishable negative: the
  mechanism does not matter even in the regime built for it.
