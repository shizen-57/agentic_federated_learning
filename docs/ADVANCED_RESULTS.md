# Advanced sweep: results so far

Only the MNIST weak-client groups have been run (`weak_mnist`: 80 runs,
`weakrand_mnist`: 40 runs; 10 seeds each, N=100 clients, K=10, 100 rounds).
The CIFAR-10 and FEMNIST groups and the scale / non-IID groups are still to
run (see ADVANCED_EXPERIMENTS.md). Every number below is a paired comparison
across the 10 seeds; "significant" means Holm-corrected p < 0.05 within each
metric (`scripts/stats.py`). Full tables: `results/adv/*/stats_*.csv`.

## The question

Does ARL (value-weighted, avoidable failures) beat a plain failure count when
the clients that keep failing are also the ones holding the rarest data?

## Result: no

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

## One thing to be careful about: ARL* rewards coverage, not safety

FedCER-like has by far the lowest oracle ARL* (0.54 vs 5.48 for RL) while
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

- One dataset (MNIST), one weak-client calibration, one reward weighting
  (`lambda_arl = 0.5`, never swept; Sec. 3.18 requires a sensitivity sweep).
  A null here is not a proof that no setting works.
- The regime was designed before seeing results and was not retuned
  afterwards; tuning it until ARL wins would invalidate the test.
- CIFAR-10 and FEMNIST results are pending.
