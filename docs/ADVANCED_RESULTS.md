# Advanced sweep: results so far

Groups run so far (10 seeds each, N=100 clients, K=10):
`weak_mnist` (80 runs, 100 rounds) and `weakrand_mnist` (40 runs), on this
project's cloud CPU; `weak_cifar` (80 runs, 200 rounds) on Kaggle 2x T4 in
35 minutes -- every run's saved config was checked against the plan before
the results were committed. The main, FEMNIST, scale and non-IID groups are
still to run (see ADVANCED_EXPERIMENTS.md). Every number below is a paired comparison
across the 10 seeds; "significant" means Holm-corrected p < 0.05 within each
metric (`scripts/stats.py`). Full tables: `results/adv/*/stats_*.csv`.

## The question

Does ARL (value-weighted, avoidable failures) beat a plain failure count when
the clients that keep failing are also the ones holding the rarest data?

## Result: no significant effect on either dataset

`weak_mnist` (weak clients hold the rarest classes), c2 = the thesis's reward:

| c2 (ARL) vs | final accuracy | worst-10%-class accuracy | CV | failure ratio |
|---|---|---|---|---|
| plain failure-count reward | -0.0004, n.s. | +0.016, n.s. | +0.07, n.s. | **0.507 vs 0.454, plain is better (p=0.039)** |
| ARL over invited clients only | -0.0015, n.s. | +0.006, n.s. | -0.002, n.s. | n.s. |
| CV over selection counts | -0.0005, n.s. | +0.002, n.s. | -0.03, n.s. | n.s. |

(n.s. = not significant.) None of the three mechanisms moves any measured
outcome, and the simple failure count has the lower failure ratio. The
random-weak control (`weakrand_mnist`) gives the same pattern: nothing
significant. So the null result from the pilot survives the regime built to
test it.

`weak_cifar` (same regime, CIFAR-10, 200 rounds):

| c2 (ARL) vs | final accuracy | worst-10%-class accuracy | CV | failure ratio |
|---|---|---|---|---|
| plain failure-count reward | 0.535 vs 0.538, n.s. | 0.178 vs 0.127, n.s. (p=0.13) | 1.31 vs 1.46, n.s. | 0.449 vs 0.429, n.s. |
| ARL over invited clients only | 0.535 vs 0.535, n.s. | 0.178 vs 0.202, n.s. | 1.31 vs 1.53, n.s. | n.s. |
| CV over selection counts | 0.535 vs 0.520, n.s. | 0.178 vs 0.135, n.s. | 1.31 vs 1.40, n.s. | n.s. |

Again nothing is significant. (Invited-only ARL vs plain failure count on
worst-10%-class accuracy: 0.202 vs 0.127, p=0.088, also n.s.)

**A direction worth testing, not a finding.** On worst-10%-class accuracy --
the outcome representation loss is supposed to move -- the ARL reward is
ahead of the plain failure count on both datasets (MNIST +0.016, CIFAR-10
+0.051), but neither is significant, and noticing it after the fact and then
pooling the two would be fishing. So it is written down here, before the
FEMNIST results exist, as the one pre-specified test for `weak_femnist`:

> **Pre-specified for `weak_femnist`:** paired sign-flip test, 10 seeds,
> `final_acc_worst10`, `c2_rl_fedprox` minus `ablation_rawfail`, one-sided
> (ARL better), alpha = 0.05, no other comparison counted as evidence for
> ARL. If it is not significant, the thesis reports ARL as not shown to
> matter.

## Where the RL policy does help

All RL variants -- including the plain failure-count one -- behave alike, and
differ from the baselines:

| | RL (c2) | Random | Oort | FedCER-like | FedCS |
|---|---|---|---|---|---|
| Final accuracy | 0.967 | 0.954 (n.s.) | 0.959 (n.s.) | 0.925 (**better**) | 0.748 (**better**) |
| Worst-10%-class accuracy | 0.928 | 0.860 (n.s., p=0.07) | 0.898 (**better**, p=0.049) | 0.751 (**better**) | 0.597 (**better**) |
| Failure ratio | 0.507 | 0.759 (**better**) | 0.735 (**better**) | 0.638 (**better**) | 0.536 (n.s.) |
| CV fairness (lower better) | 1.38 | 1.26 (n.s.) | 1.18 (n.s.) | 3.64 (**better**) | 3.16 (**better**) |

How the weak clients fared (share of invitations they received / their
success rate when invited): RL 0.20 / 0.34 (plain failure-count: 0.18 / 0.37),
Random 0.30 / 0.17, Oort 0.39 / 0.16. The RL policies invite weak clients
less and succeed twice as often with them -- but the plain failure-count
variant does just as well, so this comes from learning to *choose the
configuration*, not from ARL's value weighting.

CIFAR-10 (`weak_cifar`), c2 vs baselines:

| | RL (c2) | Random | Oort | FedCER-like | FedCS |
|---|---|---|---|---|---|
| Final accuracy | 0.535 | 0.500 (n.s.) | 0.466 (n.s., p=0.045 before correction) | 0.379 (**better**) | 0.493 (n.s.) |
| Accuracy averaged over all rounds | 0.413 | 0.370 (**better**) | 0.420 (n.s.) | 0.292 (**better**) | 0.369 (n.s.) |
| Worst-10%-class accuracy | 0.178 | 0.120 (n.s.) | 0.104 (n.s.) | 0.056 (**better**) | 0.160 (n.s.) |
| Failure ratio | 0.449 | 0.762 (**better**) | 0.735 (**better**) | 0.662 (**better**) | 0.292 (n.s.; FedCS only picks safe clients) |
| CV fairness (lower better) | 1.31 | 1.18 (n.s.) | 1.44 (n.s.) | 3.72 (**better**) | 2.37 (**better**) |
| Cost | 12.0 | 15.0 (n.s.) | 17.7 (**better**) | 8.0 (worse) | 7.6 (worse) |

Weak clients on CIFAR-10 (invitation share / success when invited): RL 0.19
/ 0.39, plain failure-count 0.17 / 0.38, Random 0.30 / 0.16, Oort 0.37 /
0.13, FedCS 0.04 / 0.27 -- FedCS gets its low failure ratio by almost never
inviting the weak, rare-data clients. Same pattern as MNIST.

## One thing to be careful about: ARL* rewards coverage, not safety

FedCER-like has by far the lowest oracle ARL* (MNIST 0.54 vs 5.48 for RL;
CIFAR-10 0.38 vs 5.50) while
failing 64% of its invitations and having the worst accuracy of the
non-trivial methods. Because ARL sums over *all* clients (Eq. 3.27/3.29) and
~90% of it comes from clients never invited (ADVANCED_EXPERIMENTS.md), a
policy that greedily maximises class coverage drives every outsider's value
towards zero and scores well on ARL regardless of how many of its
invitations fail. That is consistent with ARL measuring coverage of the
selected set more than avoided failure, and it means ARL* should not be
reported alone as evidence of a better scheduler.

## What this supports for the thesis

- Supported: a learned joint selection-and-configuration policy fails far
  less than Random/Oort (0.51 vs 0.74-0.76), protects weak and rare-data
  clients better, and clearly beats FedCER-like and FedCS on accuracy and
  fairness. Its accuracy edge over Random and Oort is small and not
  significant at 10 seeds.
- Not supported: that ARL's value weighting, or the success-based fairness
  term, adds anything measurable beyond simpler alternatives -- in the
  independent-failure pilot, and now in a persistent-failure regime too.

## Caveats

- Two datasets so far (MNIST, CIFAR-10), one weak-client calibration, one reward weighting
  (`lambda_arl = 0.5`, never swept; Sec. 3.18 requires a sensitivity sweep).
  A null here is not a proof that no setting works.
- The regime was designed before seeing results and was not retuned
  afterwards; tuning it until ARL wins would invalidate the test.
- FEMNIST, the main comparison, scale and non-IID groups are pending.
