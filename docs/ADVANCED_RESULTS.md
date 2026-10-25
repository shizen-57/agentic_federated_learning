# Advanced sweep: results

**Headline** (details and numbers below):

1. **ARL is not shown to matter.** Both pre-specified tests (CIFAR-10 with 30
   fresh seeds; FEMNIST) found no effect on worst-class accuracy
   (+0.007 and +0.007, p = 0.36 each). Setting the ARL weight to **0** -- no
   damage term at all -- is statistically indistinguishable from the thesis's
   0.5 and is nominally better on every measure.
2. **ARL's reward produces *more* failed invitations than a plain failure
   count** in 17 of 18 groups (4 points higher on average), with no accuracy
   gain. Exploratory, but very consistent.
3. **The learned joint policy's real benefit is efficiency, not accuracy:**
   about 30-45% fewer failed invitations than Random/Oort in every group,
   ~20% lower cost, and (on CIFAR-10) higher accuracy than Random. On MNIST
   and FEMNIST its accuracy is level with Random, Oort and imputation.
4. **Prevention does not beat a good correction on accuracy.** MimiC
   (server-side correction, random selection) ends **more accurate** than the
   RL policy on CIFAR-10 in the weak-client regime (10 of 10 seeds,
   +0.030), while failing 76% of invitations against 46%.
5. **GPU runs are not bit-reproducible** (below). CPU runs are.
6. **The fair re-test also failed.** With the policy shown the value ARL
   rewards (`value_cifar`, fresh seeds), neither ARL' nor invited-only ARL
   beat the plain failure count (p = 0.53 and 0.36).
7. **MimiC vs RL was chance**: on fresh seeds the two are level on accuracy
   (0.545 vs 0.529, p = 0.25), so finding 4 does not hold up.
8. **RL selection + MimiC aggregation is the best combination tested**:
   significantly more accurate than RL alone (0.565 vs 0.529) with the same
   failure ratio, and faster-learning than MimiC alone with far fewer
   failures. Found as a secondary result; a pre-specified confirmation is
   set up below.

Groups: 1,190 runs on Kaggle 2x T4 (17 groups, 10 seeds each; `confirm_cifar`
30 fresh seeds), plus 120 weak-client MNIST runs on this project's CPU. Every
saved config was checked against `configs/sweep_advanced.yaml` before the
results were committed (0 mismatches, 0 missing, all rounds complete). Every
number is a paired comparison across seeds; "significant" means
Holm-corrected p < 0.05 within each metric (`scripts/stats.py`). Full tables:
`results/adv/*/stats_*.csv` and `summary.csv`. `ext_mnist` and `trace_mnist`
were still running on the CPU when this was written.

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

### Confirmatory CIFAR-10 test (`confirm_cifar`), also fixed before running

The thesis argues ARL's effect is real but small. 10 seeds cannot tell a
small effect from none, so the right move is a fresh, adequately sized test
-- not adding seeds to `weak_cifar` until it crosses p < 0.05, which would
inflate the false-positive rate.

Effect sizes estimated from seeds 0-9 (`c2_rl_fedprox` minus
`ablation_rawfail`, final worst-10%-class accuracy): CIFAR-10 +0.051
(sd 0.095, d = 0.53 -> ~22 seeds for 80% power, one-sided); MNIST +0.016
(d = 0.16 -> ~257 seeds, impractical: MNIST is near its ceiling). Estimates
from 10 seeds tend to be optimistic, so the test uses 30 seeds.

> **Pre-specified:** group `confirm_cifar` = the `weak_cifar` setting with
> **fresh seeds 10-39 only** (never pooled with seeds 0-9), conditions
> `c2_rl_fedprox` and `ablation_rawfail`. One test: paired sign-flip,
> `final_acc_worst10`, one-sided (ARL better), alpha = 0.05:
> `python scripts/stats.py --dir results/adv/confirm_cifar --ref c2_rl_fedprox --vs ablation_rawfail --one-sided`
> (read the `final_acc_worst10` row only). Significant -> the thesis can
> claim ARL improves the worst-represented classes on CIFAR-10 in this
> regime, with the effect size and its CI. Not significant -> ARL is reported
> as not shown to matter. Other metrics in that output are descriptive only.

### Outcome of both pre-specified tests: not significant

| Test | n | ARL | plain failure count | diff | 95% CI | one-sided p |
|---|---|---|---|---|---|---|
| `confirm_cifar`, final worst-10%-class accuracy | 30 fresh seeds | 0.201 | 0.194 | **+0.007** | [-0.031, +0.046] | **0.361** |
| `weak_femnist`, final worst-10%-class accuracy | 10 | 0.094 | 0.087 | **+0.007** | [-0.031, +0.042] | **0.356** |

The +0.051 seen in `weak_cifar` seeds 0-9 shrank to +0.007 on fresh seeds
(ARL ahead in 17 of 30): it was chance, the usual regression of an effect
picked out of noise. Per the rule written down in advance, **ARL is reported
as not shown to matter**. FEMNIST has 62 classes, so "worst 10%" is the worst
six; `scale_n30_femnist` reports 0.000 for every method because with 30
writers some classes are never seen, so that column is not informative there.

### Reward-weight sensitivity (`sens_cifar`, Sec. 3.18)

Final accuracy / worst-10%-class accuracy / failure ratio, weak-client regime,
10 seeds (the thesis's setting is `lambda_arl = 0.5`, `lambda_fair = 0.3`):

| setting | accuracy | worst-10% | failures | CV |
|---|---|---|---|---|
| `lambda_arl = 0` (no ARL term) | 0.544 | 0.186 | 0.457 | 1.42 |
| 0.25 | 0.532 | 0.182 | 0.465 | 1.43 |
| **0.5 (thesis)** | 0.529 | 0.149 | 0.484 | 1.32 |
| 1.0 | 0.538 | 0.182 | 0.477 | 1.31 |
| 2.0 | 0.511 | 0.156 | 0.515 | 1.20 |
| `lambda_fair = 0` | 0.496 | 0.173 | 0.534 | 1.26 |
| 0.1 | 0.542 | 0.192 | 0.470 | 1.32 |
| 1.0 | 0.548 | 0.189 | 0.485 | 1.27 |

No setting differs significantly from the thesis's on any measure. Dropping
the ARL term entirely is nominally better on all three outcomes, so "the
weight was too low" is not the explanation: the term simply does not help.
Raising it nudges the policy towards lower CV (fairer) and *more* failures
(none of this is significant) -- in line with the finding below that ARL
rewards coverage over safety.

### ARL vs the plain failure count, in every group (exploratory)

Across all 18 groups where both ran, the plain failure-count reward has
**fewer failed invitations than ARL in 17** (11 individually p < 0.05,
uncorrected; typical gap 0.03-0.07, e.g. `weak_femnist` 0.376 vs 0.446,
p = 0.002), with no accuracy gain for ARL (ahead in 10 of 18 groups, mean
+0.005) and no worst-class gain (ahead in 6 of 18, mean -0.002). This was not a
pre-specified test, and the groups share seeds, so treat it as a consistent
descriptive pattern, not a p-value. The mechanism fits the earlier
decomposition: ~90% of ARL' comes from never-invited clients, so the term
rewards covering many classes, not avoiding failure.

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

## The main comparison (`main_*`, no weak clients, 10 seeds, N=100, K=10)

c2 (the thesis's RL policy) vs everything else, final accuracy / failure ratio;
\* = significant after correction:

| | MNIST | FEMNIST | CIFAR-10 |
|---|---|---|---|
| RL (c2) | 0.971 / 0.47 | 0.771 / 0.43 | 0.572 / 0.40 |
| Random | 0.969 / 0.73\* | 0.756 / 0.72\* | 0.501\* / 0.72\* |
| Oort | 0.962 / 0.67\* | 0.770 / 0.64\* | 0.537 / 0.65\* |
| imputation (Oort + last update) | 0.971 / 0.61\* | 0.770 / 0.59\* | 0.564 / 0.58\* |
| Power-of-Choice | 0.951\* / 0.80\* | 0.698\* / 0.86\* | 0.413\* / 0.83\* |
| FedCS | 0.794\* / 0.52 | 0.673\* / 0.41 | 0.421\* / 0.42 |
| FedCER-like | 0.897\* / 0.68\* | 0.587\* / 0.62\* | 0.398\* / 0.65\* |
| round-robin heuristic | 0.951 / 0.72\* | 0.760 / 0.72\* | 0.564 / 0.72\* |

- Fewer failed invitations than every selection baseline except FedCS
  (which only picks safe clients), in every dataset.
- Accuracy: significantly better than FedCER-like, FedCS and Power-of-Choice
  everywhere; better than Random only on CIFAR-10; level with Random, Oort and
  imputation on MNIST and FEMNIST.
- Aggregation (FedAvg, FedProx, FedNova, FedProx+FedNova): no significant
  difference in any dataset.
- Fairness (CV, lower is fairer): RL 1.07 / 1.01 / 1.06 vs Random 1.20 /
  1.10 / 1.10 -- better than Random only on MNIST, level on the others.
- Cost per round: RL 10.7-12.4 vs Random 15.0 and Oort 16-18 (about 20-30%
  lower), because it assigns lighter modes.

Robustness: with FedScale's real device speeds (`trace_cifar`), at N=30 and
N=200 (`scale_*`), and at Dirichlet alpha 0.1 and 0.5 (`alpha*`), RL still
fails far less than Random and Oort (always significant) and its accuracy is
level with or better than theirs (significantly better than Random in
`alpha01_mnist`, `alpha05_cifar`, `scale_n30_femnist`, `scale_n200_femnist`).

## Chapter 3 pieces and published baselines (`ext_cifar`, weak regime)

Final accuracy / worst-10% / failure ratio vs c2 (0.538 / 0.203 / 0.457):

| | accuracy | worst-10% | failures |
|---|---|---|---|
| c2 + discounted multi-round returns | 0.536 | 0.166 | 0.516 |
| c2 + Eq. 3.54 reweighting | 0.541 | 0.194 | 0.481 |
| c2 + semi-async staleness | 0.563 | 0.194 | 0.467 |
| c2 + fairness debt | 0.535 | 0.150 | 0.458 |
| c2 under budgets C_max/L_max | 0.545 | 0.139 | 0.423 |
| Random under budgets | 0.523 | 0.167 | 0.557\* |
| Oort under budgets | 0.561 | 0.216 | 0.481 |
| MIFA (random selection) | 0.488 | 0.189 | 0.762\* |
| **MimiC (random selection)** | **0.568\*** | 0.264 | 0.762\* |
| FedCG | 0.269\* | 0.002\* | 0.941\* |

- None of the five Chapter 3 pieces changes the picture significantly
  (budgets cut cost by 31%, 12.5 to 8.6 per round, and failures slightly).
- **MimiC beats RL on accuracy**: 10 of 10 seeds, +0.030 final (0.568 vs
  0.538), +0.010 averaged over all rounds, at 20% higher cost and 66% more
  failed invitations. A correction that makes absent clients harmless does as
  well as avoiding the absence. The thesis's prevention-vs-imputation claim
  therefore has to be stated as an *efficiency* claim (same accuracy for far
  fewer wasted invitations), not an accuracy claim. Its own simplified
  imputation baseline (`b_mimic_impute`) did not show this (0.564 vs RL 0.572
  in `main_cifar`), so the effect needs the proper algorithm. The natural next
  test is both together (RL selection + MimiC aggregation), which the code
  already supports.
- **FedCG's result should not be read as "FedCG is bad".** The paper's setting
  has no hard deadline and every selected client reports; its compression
  ratio is the largest that exactly fits the time budget, which with this
  simulator's timing jitter leaves no slack, so about half the clients miss
  the deadline even before availability failures (94% fail overall). A fair
  comparison needs a safety margin on the deadline; that variant is not yet
  run.

## Follow-up, pre-specified before running (`follow_cifar`)

"MimiC beats RL on CIFAR-10" came from looking at ten comparisons in
`ext_cifar`, so it needs confirming on fresh data before it goes in a thesis.
Group `follow_cifar` reruns the weak-client CIFAR-10 setting on **fresh seeds
10-29 only** (never pooled with seeds 0-9) with five conditions: `c2_rl_fedprox`,
`b_mimic`, `c2_mimic` (RL selection + MimiC aggregation, i.e. prevention and
correction together), `b_fedcg` and `b_fedcg_m07` (FedCG fitted to 70% of the
deadline, so it is not handicapped by a hard deadline it was never designed for).

> **Pre-specified primary test:** paired sign-flip, two-sided, alpha = 0.05,
> `final_accuracy`, `b_mimic` vs `c2_rl_fedprox`, 20 seeds:
> `python scripts/stats.py --dir results/adv/follow_cifar --ref c2_rl_fedprox --vs b_mimic`
> (read the `final_accuracy` row only). Significant with MimiC higher -> the
> thesis states that a correction method matches or beats the RL policy on
> accuracy and the RL policy's advantage is efficiency. Not significant -> the
> `ext_cifar` result is treated as chance and the two are reported as level on
> accuracy. **Secondary, descriptive only:** `c2_mimic` vs `b_mimic` and vs
> `c2_rl_fedprox` (does prevention add to correction?), failure ratio, cost,
> and `b_fedcg_m07` vs `b_fedcg`.

Run it with `GROUPS = "follow_cifar"` (100 runs, about 45 minutes on a T4 x2).

## Why ARL showed no gain: diagnosis

After both pre-specified tests came out null, the question was whether the
simulation itself prevents ARL from mattering. Instrumented runs on the
weak-client MNIST setting (CPU, exactly repeatable) checked each suspect:

| Suspect | Measured | Verdict |
|---|---|---|
| The eligibility gate hides weak clients' failures from ARL (they succeed ~33% under Light, below the 0.5 threshold) | 83 of 98 and 121 of 148 weak-client failures were counted in ARL' | not the cause |
| Clients are worth about the same, so value-weighting = counting | Weak (rare-data) clients' V^repr is 1.7-1.9x the others'; CV of failed clients' values ~0.9-1.0 | not the cause |
| The policy does not learn | With learning switched off (`policy.rl_lr = 0`) failures rise from 0.43-0.58 to 0.58-0.62 | it learns, modestly |
| **The policy cannot see the value ARL rewards** | The RL policy's "value" input (`metrics.standalone_value`, roughly the number of classes a client holds) has correlation **-0.71** with V^repr and does not separate weak rare-data clients from the rest (0.37 vs 0.39) | **a real flaw** |
| **ARL' does not track the policy's own mistakes** | corr(ARL'_t, failure ratio) = 0.18; corr(ARL'_t, its invited-only part) = 0.46 -- ~90% comes from never-invited clients | **a real weakness of Eq. 3.27 as a reward** |

The simulator's ingredients behave as designed (configs match the plan,
weak clients are mode-sensitive, failures are counted, values vary). The
problem is in how ARL is wired into the policy: ARL rewards protecting
rare-data clients, but the policy is shown a value input that points the
other way, and the reward term moves mostly with clients it never chose.
No reward can teach a policy to protect clients it cannot identify, so the
null results so far test "ARL reward + misleading value input", not ARL.

Fix (opt-in, defaults unchanged): `policy.value_feature: marginal` gives the
policy each client's expected marginal coverage, E[V^repr] over random
K-client sets (`metrics.expected_marginal_value`) -- the same quantity ARL
weights failures by, computed once from the class sketches.

### Fair re-test, pre-specified before running (`value_cifar`)

Weak-client CIFAR-10, corrected value input for every condition, **fresh
seeds 40-59** (no overlap with any earlier group): `v2_c2` (ARL', Eq. 3.27),
`v2_arlsel` (ARL over invited clients only), `v2_rawfail` (plain failure
count), `v2_arl0` (no damage term).

> **Pre-specified primary tests (two, Holm-corrected):** paired sign-flip,
> one-sided (ARL better), `final_acc_worst10`, 20 seeds:
> p1 from `python scripts/stats.py --dir results/adv/value_cifar --ref v2_c2 --vs v2_rawfail --one-sided`,
> p2 from `python scripts/stats.py --dir results/adv/value_cifar --ref v2_arlsel --vs v2_rawfail --one-sided`
> (the `final_acc_worst10` row of each). Holm: the smaller p must be
> <= 0.025, then the other <= 0.05. Significant -> ARL (in that form) helps
> once the policy can see value. Neither significant -> ARL is reported as
> not shown to matter even with a value-aware policy. **Descriptive only:**
> weak-client invitation share and success rate, failure ratio, accuracy,
> and `v2_arl0` vs `v2_c2`.

Remaining limitation either way: one scalar reward per round for about 20
decisions (10 clients and their modes), and 100-200 policy updates per run,
is a weak learning signal; a policy-gradient method with per-client credit
or more rounds might extract more from any reward term.

### Outcome of the fair re-test and the MimiC follow-up

**`value_cifar` (pre-specified, 20 fresh seeds):** final worst-10%-class
accuracy, one-sided, vs the plain failure count:

| | ARL reward | plain failure count | diff | 95% CI | p |
|---|---|---|---|---|---|
| ARL' (Eq. 3.27), `v2_c2` | 0.187 | 0.189 | -0.002 | [-0.055, +0.051] | 0.53 |
| ARL over invited clients, `v2_arlsel` | 0.198 | 0.189 | +0.009 | [-0.039, +0.061] | 0.36 |

Neither passes (Holm needed p <= 0.025). Descriptively the four rewards
behave almost identically (weak-client share of invitations 0.16-0.21,
success when invited 0.37-0.40, failures 0.45-0.48). Per the rule written
down in advance: **ARL is not shown to matter, even when the policy can see
the value ARL rewards.** The diagnosis above therefore explains the
mechanism's weakness, but fixing the input does not turn it into a gain;
the remaining limits are ARL' being dominated by never-invited clients and
the weak per-round learning signal.

**`follow_cifar` (pre-specified primary test, 20 fresh seeds):** final
accuracy, MimiC vs RL, two-sided: 0.545 vs 0.529, diff -0.016, 95% CI
[-0.042, +0.010], **p = 0.25, not significant**. The `ext_cifar` result was
chance; MimiC and the RL policy are level on accuracy, and the RL policy
fails far less (0.48 vs 0.75).

Secondary results (descriptive per the pre-specification, so to be
confirmed before being claimed):

| `follow_cifar`, 20 seeds | final acc | mean acc (speed) | worst-10% | failures | cost |
|---|---|---|---|---|---|
| RL (c2) | 0.529 | 0.413 | 0.197 | 0.476 | 14.2 |
| MimiC, random selection | 0.545 | 0.420 | 0.247 | 0.753 | 15.0 |
| **RL selection + MimiC aggregation** | **0.565** | **0.444** | 0.257 | 0.481 | 13.4 |
| FedCG | 0.305 | 0.250 | 0.007 | 0.924 | 11.7 |
| FedCG, 70% deadline margin | 0.296 | 0.238 | 0.002 | 0.921 | 10.8 |

- RL + MimiC vs RL: significantly more accurate (+0.036, Holm-corrected)
  at the same failure ratio. RL + MimiC vs MimiC: faster learning (mean
  accuracy +0.024, p = 0.002) and far fewer failures (0.48 vs 0.75); final
  accuracy +0.020, not significant. Prevention and correction are
  complementary: the RL policy wastes fewer invitations, MimiC makes the
  remaining absences less harmful.
- FedCG with a deadline margin is no better. Its problem is not the
  deadline fit: its greedy diverse selection is deterministic, so it picks
  the same clients round after round (CV of selection counts 1.9-2.2 vs 1.0
  for RL and 0.2 for Random), drains their batteries and loses them. That is
  faithful to the paper, which assumes clients are always available.

`ext_mnist` and `trace_mnist` (now complete, 10 seeds) agree with the
CIFAR-10 groups: no Chapter 3 extension improves on c2 except that budgets
cut failures (0.42 vs 0.53, significant) at equal accuracy; MIFA and FedCG
are less accurate; with real device traces RL fails significantly less than
Random and Oort and matches or beats their accuracy.

### Confirmation of RL + MimiC, pre-specified before running (`confirm_mimic`)

> **Pre-specified primary tests (two, Holm-corrected):** paired sign-flip,
> one-sided (RL + MimiC better), `final_accuracy`, `c2_mimic` vs
> `c2_rl_fedprox`, **fresh seeds 60-79**, weak-client regime, on CIFAR-10
> (`confirm_mimic_cifar`) and FEMNIST (`confirm_mimic_femnist`):
> `python scripts/stats.py --dir results/adv/confirm_mimic_<cifar|femnist> --ref c2_mimic --vs c2_rl_fedprox --one-sided`
> (the `final_accuracy` row). The smaller p must be <= 0.025, then the
> other <= 0.05. Passing -> the thesis can claim that combining the RL
> scheduler with server-side correction improves accuracy over the
> scheduler alone. **Descriptive only:** `b_mimic` in the same groups,
> failures, cost, worst-class accuracy.

## GPU runs are not bit-reproducible

The same configuration and seed gives different results across the three
CIFAR groups that contain `c2_rl_fedprox` on seeds 0-9: the runs agree in
round 1, accuracy diverges by round 3, failure counts by round 15, and final
accuracy differs by 0.07 on average (up to 0.17). GPU arithmetic (cuDNN,
atomics) is not exactly repeatable, and training noise is then amplified by
the RL policy. CPU runs are exactly repeatable (re-running `weak_mnist` seed
0 reproduced all 22 columns of all 100 rounds, 8 of 8 conditions).

Consequences: (a) the paired tests stay valid -- the simulated world (data
split, devices, availability) is shared by seed, but the run-to-run noise is
real and is already part of every standard deviation and p-value reported;
(b) exact numbers cannot be reproduced on a GPU, only the statistics; (c) the
three copies of c2 are an independent measure of that noise, and the spread
between them (0.07) is larger than most of the differences tested here --
which is why small effects need many seeds.

An option, `fl.deterministic: true`, asks PyTorch for deterministic GPU kernels
so a run repeats exactly. It is not used for the results above, it is slower,
and it has not been tested on a GPU here (only that it leaves CPU results
unchanged), so try one short run on Kaggle before relying on it.

## What this supports for the thesis

- Supported (3 datasets, 10 seeds, robust to scale, skew and real device
  speeds): a learned joint selection-and-configuration policy wastes far fewer
  invitations than Random/Oort (about 0.45 vs 0.65-0.73), costs ~20-30% less,
  protects weak and rare-data clients better, and clearly beats FedCER-like,
  FedCS and Power-of-Choice. On CIFAR-10 it is also more accurate than Random.
  On MNIST and FEMNIST its accuracy is level with Random, Oort and imputation.
- Not supported: that ARL's value weighting adds anything measurable beyond a
  plain failure count. It did not in the independent-failure pilot, in the
  persistent-failure regime, in the 30-seed confirmation, in FEMNIST, or in
  the weight sweep (including weight 0), and it produces slightly more
  failures. The success-based fairness term showed no effect either.
- Needs restating: "prevention beats imputation". Against a faithful
  correction method (MimiC) the RL policy matches or loses on accuracy and
  wins on efficiency.
- Suggested framing: the contribution is the learned joint selection and
  configuration of participation modes, evidenced by efficiency (failures,
  cost) and by beating the closest selection baselines; ARL is a proposed
  metric whose benefit was tested thoroughly and not found.

## Caveats

- One weak-client calibration; the reward-weight sweep is CIFAR-10 only and
  one-at-a-time (no joint grid). A null is not a proof that no setting works.
- The regime was designed before seeing results and was not retuned
  afterwards; tuning it until ARL wins would invalidate the test.
- `ext_mnist` and `trace_mnist` (CPU) were still running; MNIST and FEMNIST
  versions of the extension group do not exist (`ext_cifar` only).
- FedCER is still the simplified "-like" version; FedCG is run without a
  deadline margin (see above).
- Oort's system utility uses each round's true (hidden) client times
  (`t_expected` in `engine._state`), which a real server would not know. Only
  Oort uses it, so wins over Oort understate the proposed policy's margin.
