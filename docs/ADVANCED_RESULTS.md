# Advanced sweep: results

**Headline** (details and numbers below):

1. **ARL is not shown to matter.** Both pre-specified tests (CIFAR-10 with 30
   fresh seeds; FEMNIST) found no effect on worst-class accuracy
   (+0.007 and +0.007, p = 0.36 each). Setting the ARL weight to **0** -- no
   damage term at all -- shows no significant difference from the thesis's
   0.5 and is nominally better on every measure.
2. **ARL's reward produces *more* failed invitations than a plain failure
   count** in 17 of 18 groups (4 points higher on average), with no accuracy
   gain. Exploratory, but very consistent.
3. **The learned joint policy's real benefit is efficiency, not accuracy:**
   about 30-45% fewer failed invitations than Random/Oort in every group
   (but see item 12: the baselines also invited offline clients, which the
   RL policy could see and avoid),
   ~20% lower cost, and (on CIFAR-10) higher accuracy than Random. On MNIST
   and FEMNIST its accuracy shows no significant difference from Random, Oort
   and imputation (which is not the same as shown equal).
4. **Prevention does not beat a good correction on accuracy.** MimiC
   (server-side correction, random selection) ends **more accurate** than the
   RL policy on CIFAR-10 in the weak-client regime (10 of 10 seeds,
   +0.030), while failing 76% of invitations against 46%.
5. **GPU runs are not bit-reproducible** (below). CPU runs are.
6. **The fair re-test also failed.** With the policy shown the value ARL
   rewards (`value_cifar`, fresh seeds), neither ARL' nor invited-only ARL
   beat the plain failure count (p = 0.53 and 0.36).
7. **MimiC vs RL was chance**: on fresh seeds the accuracy difference is not
   significant (0.545 vs 0.529, p = 0.25), so finding 4 does not hold up.
8. **RL selection + MimiC aggregation is the best combination tested**:
   significantly more accurate than RL alone (0.565 vs 0.529) with the same
   failure ratio, and faster-learning than MimiC alone with far fewer
   failures. Found as a secondary result; a pre-specified confirmation is
   set up below.
9. **That confirmation narrowly failed.** RL + MimiC beat RL alone on
   CIFAR-10 again (+0.016, exact one-sided p = 0.0252) but missed the
   pre-specified Holm threshold of 0.025, and showed nothing on FEMNIST
   (p = 0.49). It is suggestive on CIFAR-10, not confirmed.
10. **ARL-v2 (the redesign) failed its pre-specified test** in all four
    settings (worst-class accuracy, p = 0.50-0.98; nominally worse in all
    four). Exploratory but very consistent: it makes the scheduler markedly
    more economical -- 5-7 points fewer failures, 2-3.7 lower cost per round
    and 6-10 fewer clients lost to churn (p <= 0.004 in every setting) --
    with no significant accuracy difference on CIFAR-10 and slightly lower
    accuracy on FEMNIST.

11. **The simulator had two flaws, now fixed** (section "Simulator fix"):
    Light's partial upload was charged but not actually compressed, and Oort
    saw the true round times. ARL-v2 had exploited the first by putting
    ~93% of clients on Light. ARL-v3, mode-fair baselines and the published
    FAVOR, DELTA and REFL are set up with pre-specified tests on fresh seeds
    100-119 (`s2_*` groups).
12. **An external review found that the reward was computed on test
    images** (the scheduler learned from data the final score also used),
    that methods with the same seed did not start from the same model, and
    that ARL's per-client value can hide the joint loss of two clients
    holding the same rare class. All confirmed and fixed for the `s2_*`
    groups ("Protocol fix"); every accuracy comparison above was made
    before the fix and needs the `s2_*` confirmation. A second review found
    more (order-dependent credit, offline clients charged and invited by
    the baselines, a gate that switched itself off, a biased DELTA weighting,
    reports that could hide failures or reuse stale runs), also fixed; the
    failure-ratio advantage over Random/Oort also needs re-confirming.

13. **The final, pre-specified `s2_*` tests (1,540 runs, every protocol
    fix on) give clear answers** (section "Outcome of the pre-specified
    `s2_*` tests"): ARL-v3 does not help (0/8 tests pass); ARL-v2 is not
    more efficient than always using Light; the proposed RL system beats
    DELTA on CIFAR-10 but is significantly worse than six of nine
    mode-fair baselines on FEMNIST, partly because RL policies that learn
    their own modes sometimes lock into all-fail states that ARL cannot
    see; a simple mode rule does as well as learned modes on accuracy.

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
  everywhere; better than Random only on CIFAR-10; no significant difference from Random,
  Oort and imputation on MNIST and FEMNIST.
- Aggregation (FedAvg, FedProx, FedNova, FedProx+FedNova): no significant
  difference in any dataset.
- Fairness (CV, lower is fairer): RL 1.07 / 1.01 / 1.06 vs Random 1.20 /
  1.10 / 1.10 -- better than Random only on MNIST, level on the others.
- Cost per round: RL 10.7-12.4 vs Random 15.0 and Oort 16-18 (about 20-30%
  lower), because it assigns lighter modes.

Robustness: with FedScale's real device speeds (`trace_cifar`), at N=30 and
N=200 (`scale_*`), and at Dirichlet alpha 0.1 and 0.5 (`alpha*`), RL still
fails far less than Random and Oort (always significant) and its accuracy is
not significantly worse than theirs (significantly better than Random in
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
  therefore has to be stated as an *efficiency* claim (no significant accuracy
  difference, far fewer wasted invitations), not an accuracy claim. Its own simplified
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
chance; MimiC and the RL policy show no significant accuracy difference, and the
RL policy fails far less (0.48 vs 0.75).

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
  with no significant difference in failure ratio. RL + MimiC vs MimiC: faster learning (mean
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
cut failures (0.42 vs 0.53, significant) with no significant accuracy
difference; MIFA and FedCG
are less accurate; with real device traces RL fails significantly less than
Random and Oort, and its accuracy is not significantly lower.

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

## ARL-v2: a redesign built from the diagnosis (frozen before testing)

The diagnosis found four reasons the original ARL could not matter. ARL-v2
changes one thing per reason. The design below was fixed **before** any
ARL-v2 run; the MNIST development runs (`dev_mnist`, `dev_churn_mnist`,
seeds 0-2) are used only to catch bugs, not to tune or choose.

| Why ARL' failed | ARL-v2 |
|---|---|
| ~90% of ARL' is about clients never invited | only **invited** clients that **failed** count |
| the eligibility gate already protects clients, so "failed but predicted eligible" adds little | each failure is weighted by **gap_i = max_m p_succ(i, m) - p_succ(i, assigned mode)**: the success probability the scheduler's own mode choice gave up. Zero if it already chose the safest mode. This is the thesis's distinctive idea -- feasibility is partly caused by the configuration -- made into the loss itself |
| missing one round is temporary, so the damage ARL' counts mostly does not last | weighted by **a_i = 1 + sum_k c_ik u_k**, where u_k = 1 - q_k / max(q) is how under-covered class k has been lately (q = moving average of per-round class coverage by successful clients, rate 0.1): losing a client whose classes have gone uncovered for a while costs up to 2x |
| one reward per round for ~20 decisions | **per-decision credit** (`policy.per_client_credit`): each invited client's loss d_i is normalised and credited only to the log-probability of that client's pick and mode; it leaves the shared reward |

Per round: d_i = gap_i x V^repr_i x a_i for each invited client that failed,
ARL-v2_t = sum d_i (logged as `arl_v2`). The policy also gets the corrected
value input (`policy.value_feature: marginal`). Conditions: `arl2_full`
(all four changes), `arl2_shared` (same loss in the shared reward, no
per-decision credit), compared with `v2_rawfail` (plain failure count) and
`v2_c2` (original ARL'), all with the corrected value input, plus Random.

**Client churn** (`system.churn_after: 3`, `churn_prob: 0.5`): half the
clients, fixed per seed and the same for every policy, leave for good after
3 consecutive invitations they attempted and failed (deadline, battery,
link drop, rejection; offline rounds do not count). This is the setting
where a scheduler's mistakes cause permanent loss, which is what ARL is
meant to prevent. It is reported next to the standard weak-client regime,
not instead of it.

> **Pre-specified primary tests (four, Holm-corrected):** paired sign-flip,
> one-sided (ARL-v2 better), `final_acc_worst10`, `arl2_full` vs
> `v2_rawfail`, **fresh seeds 80-99**, in each of `arl2_cifar`,
> `arl2_churn_cifar`, `arl2_femnist`, `arl2_churn_femnist`:
> `python scripts/stats.py --dir results/adv/<group> --ref arl2_full --vs v2_rawfail --one-sided`
> (the `final_acc_worst10` row). Holm across the four p-values: sorted
> ascending, they must pass 0.0125, 0.0167, 0.025, 0.05 in turn. A group
> passes -> ARL-v2 protects the worst-represented classes better than a plain
> failure count in that setting. None passes -> ARL-v2 is reported as not
> shown to matter either. **Descriptive only:** clients departed (and how
> many were weak rare-data clients), failure ratio, accuracy, weak-client
> success, `arl2_shared` vs `arl2_full` (does per-decision credit matter?),
> `v2_c2` and Random.

**Development check (MNIST, seeds 0-2, bug check only; nothing was changed
afterwards):** no errors or NaNs; the full ARL-v2 policy drives its own loss
down (mean ARL-v2 0.012 vs 0.058 for the plain failure count and 0.285 for
Random), so the per-decision credit is wired correctly; without churn no
client leaves, with churn 9-40 of 100 leave (Random 32-40, RL variants
9-30), so the scenario is neither trivial nor degenerate.

### Outcome of the RL + MimiC confirmation and the ARL-v2 test

**`confirm_mimic_*` (pre-specified, fresh seeds 60-79):** final accuracy,
RL + MimiC vs RL alone, one-sided, exact sign-flip:

| | RL + MimiC | RL | diff | 95% CI | exact p | Holm threshold |
|---|---|---|---|---|---|---|
| CIFAR-10 | 0.564 | 0.548 | +0.016 | [+0.002, +0.031] | 0.0252 | 0.025 |
| FEMNIST | 0.751 | 0.748 | +0.003 | [-0.041, +0.030] | 0.485 | 0.05 |

Neither passes under the rule fixed in advance. CIFAR-10 misses by 0.0002 --
the Monte Carlo estimate the script used for n > 16 (0.0249) would have
passed, so `stats.py` now enumerates all sign patterns up to n = 22. Taken
with `follow_cifar` (+0.036), the CIFAR-10 direction is consistent and
favourable but not confirmed; FEMNIST shows no effect. Descriptively RL +
MimiC learns faster on both (mean accuracy 0.441 vs 0.417; 0.651 vs 0.645)
with no significant difference in failure ratio. The thesis can report it as a promising, not
established, combination.

**`arl2_*` (pre-specified, fresh seeds 80-99):** final worst-10%-class
accuracy, ARL-v2 (full) vs the plain failure count, one-sided, exact:

| setting | ARL-v2 | plain count | diff | exact p |
|---|---|---|---|---|
| CIFAR-10 | 0.175 | 0.211 | -0.036 | 0.900 |
| CIFAR-10 + churn | 0.147 | 0.148 | -0.000 | 0.501 |
| FEMNIST | 0.083 | 0.093 | -0.010 | 0.753 |
| FEMNIST + churn | 0.077 | 0.096 | -0.019 | 0.979 |

None passes; ARL-v2 is not shown to protect the worst-represented classes,
even where losses are permanent. **Exploratory** (not pre-specified, two-sided
exact tests vs the plain failure count, 20 seeds each):

| setting | failures | cost / round | clients lost | final accuracy |
|---|---|---|---|---|
| CIFAR-10 | -0.068 (p < 0.001) | -3.7 (p < 0.001) | -- | -0.006 (p = 0.60) |
| CIFAR-10 + churn | -0.072 (p = 0.001) | -3.5 (p < 0.001) | -5.9 (p = 0.004) | +0.025 (p = 0.08) |
| FEMNIST | -0.051 (p < 0.001) | -3.2 (p = 0.001) | -- | -0.019 (p = 0.12) |
| FEMNIST + churn | -0.067 (p < 0.001) | -2.0 (p < 0.001) | -9.7 (p < 0.001) | -0.031 (p = 0.001) |

The per-decision credit works -- ARL-v2 learns to avoid exactly the
failures it penalises, choosing lighter modes and fewer risky invitations --
but avoiding failures does not translate into better class coverage: it
trains less per round, which costs accuracy on FEMNIST. ARL-v2 is therefore
an *efficiency and retention* mechanism, not a representation mechanism.
Those efficiency effects are large and consistent but were not the
pre-specified outcome; claiming them would need a confirmation on fresh
seeds. `arl2_shared` (same loss, no per-decision credit) sits between the
two on every measure, so the per-decision credit is what drives the change.

## Simulator fix, ARL-v3 and stronger baselines (frozen before testing)

Looking at *how* ARL-v2 got its efficiency (`arl2_*`, mean per round, 10
invitations) showed it had found a shortcut:

| CIFAR-10, weak clients | Light | Standard | Priority | failures | worst-10% acc |
|---|---|---|---|---|---|
| ARL-v2 (full) | 9.3 | 0.5 | 0.2 | 37% | 0.175 |
| plain failure count | 5.5 | 3.4 | 1.1 | 44% | 0.211 |

ARL-v2 put almost every client on Light (same pattern in all four settings),
because its loss -- success probability given up versus the safest mode --
is zero for Light, always the safest mode. Fewer failures came from asking
for about a third less training, and it invited the weak rare-data clients
no more often than before (1.76 vs 1.70 per round). Checking that shortcut
exposed two problems in the simulator and one in the comparison:

1. **Light's small upload was free.** Light (25% upload) and Standard (50%)
   were charged for a partial upload in time, energy and cost, but the server
   still received the client's full, uncompressed update -- compression was
   only applied for FedCG. Every policy that used Light, the RL policy above
   all, got the savings without the information loss.
   **Fix (`fl.compress_upload`):** the update a client sends is top-k
   sparsified to its mode's upload fraction, with error feedback (what is
   not sent is carried into the client's next upload), for every aggregator.
2. **Oort saw the true round time** (`t_expected`: this round's hidden
   compute jitter and channel draw). **Fix (`policy.oort_estimated_time`):**
   Oort uses the time estimated from observed speed and bandwidth, like
   every other policy.
3. **The baselines could not choose modes.** Random, Oort and
   Power-of-Choice always used Standard; only the RL policy, FedCS (always
   Light) and FedCER-like chose. Part of RL's advantage could be "it can
   choose modes" rather than "it chooses well".
   **Fix (`policy.mode_rule: expected_work`):** a fixed, sensible mode rule
   any baseline can use -- the mode with the most expected training,
   P(success) x local epochs, among modes with P(success) >= 0.5 (else the
   safest). `b_random_ew`, `b_oort_ew`, `b_poc_ew` are Random, Oort and
   Power-of-Choice with it; `rl_ew` is the RL policy's selection with the
   same rule (does *learning* modes beat a rule?); `rl_light` is the RL
   policy with every client on Light (is ARL-v2's efficiency just "use
   Light"?).

Both fixes are off by default; with them off every earlier condition gives
bit-identical results (checked on 7 conditions on the CPU), so all earlier
results stay valid as results *of the original simulator*. Every group from
here on runs with both fixes on (`sim2` in `configs/sweep_advanced.yaml`).

**ARL-v3** keeps ARL-v2's three good parts (only invited clients, value x
coverage-deficit weighting m_i = V_i (1 + sum_k c_ik u_k), per-decision
credit; V_i is now the Shapley share, see "Protocol fix" item 4) and
replaces the loss that caused the shortcut:

- work, not safety: W(m) = P(success | m) x local_epochs[m] for each mode
  safe enough to plan on (P >= 0.5, or the safest mode if none is), W = 0 for
  a gamble below that;
  **loss_i = m_i (W(best) - W(assigned)) / 4**, zero when the scheduler chose
  the expected-work mode. Light is no longer automatically right, and
  gambling on Priority is not rewarded either;
- credit for success: **gain_i = m_i Y_i**, so inviting a valuable,
  under-covered client and getting its update in is what pays. (A first
  draft weighted the gain by epochs too; on a 20-round synthetic smoke test
  it swung to Priority for most clients and failed 70% of invitations, so it
  was changed to the above before any run on real data.)

Per-client advantage (`arl3_full`) = -0.5 x norm(loss_i) + 0.5 x norm(gain_i);
`arl3_shared` puts both in the shared reward instead. Logged as
`arl_v3_loss` and `arl_v3_gain`.

**New published baselines** (`src/afl/policies/published.py`, each
implemented from the paper; departures forced by this setting are listed in
the class docstrings):

- **FAVOR** (Wang et al., INFOCOM 2020): Double-DQN selection on
  PCA-compressed model weights (100 dims per model, 2-layer MLP with 512
  hidden units, reward 64^(acc - target) - 1, target 99% MNIST, 55%
  CIFAR-10, 80% FEMNIST). The paper pre-trains its agent over ~200 FL runs;
  here it learns online in one run, like the proposed RL policy.
- **DELTA** (Wang et al., NeurIPS 2023), practical version: sampling
  probabilities proportional to sqrt(alpha zeta_i^2 + alpha sigma_i^2)
  (gradient diversity and local gradient variance, alpha = 0.5), updated for
  the clients that reported; K draws with replacement and the matching
  unbiased (Horvitz-Thompson) weights m_i q_i / (K p_i), not renormalised
  (see "Second review" below for why).
- **REFL** (Abdelmoniem et al., EuroSys 2023): picks the online clients
  *least* likely to be available next round (forecast right 90% of the time,
  as the paper assumes), clients sit out 5 rounds after submitting, fewer
  invitations when stragglers' late updates are due (APT), and
  staleness-aware aggregation of late updates (Eq. 5, beta = 0.35).

None of the three chooses modes, so they use Standard, as Random and Oort do.

### Protocol fix (after an external review, before any `s2_*` run)

An outside review of the code found problems in how results were measured.
Each was checked in the code and confirmed; all are fixed for the `s2_*`
groups, off by default, and every earlier condition still gives
bit-identical results with them off (re-checked on the same 7 conditions).

1. **The scheduler learned from the test set.** The reward (accuracy gain
   per round) was computed on test images, and the final score used the
   same images (on CIFAR-10, a 2,000-image subset of the 10,000 used for the
   final score). **Fix (`fl.val_frac: 0.3`):** 30% of the test pool becomes
   a server validation set; the reward and every learning baseline (FAVOR's
   reward too) see only it, and reported accuracy uses only the other 70%,
   which nothing ever learns from. Every accuracy comparison made before
   this fix needs confirming under it -- the `s2_*` groups are that
   confirmation.
2. **Same seed did not mean same starting model.** The RL policy's network
   was built from PyTorch's global generator before the FL model, so the RL
   policy and Random/Oort started from different weights; local training
   and RL sampling also shared that generator (the RL policy's own
   generator was created but never used). **Fix (`fl.clean_rng`):** the FL
   model is initialised from its own seed, identical for every method with
   that seed (a checksum, `init_checksum`, is saved with every run's
   config); the RL network, RL sampling and local training each have their
   own stream.
3. **The absence counter only counted failed invitations.** A client that
   was never invited for 50 rounds still showed 0 rounds since its last
   contribution. **Fix (`fl.absence_counter`):** it counts every round
   without a contribution.
4. **The value of a client can hide a joint loss.** Eq. (3.26) values a
   selected client by the coverage lost if *it alone* were removed, so two
   selected clients holding the same rare class are each worth 0 -- and if
   both fail, the sum of their values still says nothing was lost. This
   follows from the formula, and the original ARL, ARL-v2 and the first
   ARL-v3 all inherit it. **Fix:** ARL-v3 now weights each invited client by
   its **Shapley share** of the invited set's coverage (`metrics.coverage_shapley`:
   two holders of the same class get half each; the shares add up to the
   set's coverage; checked against brute force), and every run logs the
   **set-level** loss directly: `cov_lost` = Cov(invited) - Cov(delivered),
   and `cov_lost_avoid` = Cov(invited clients that could have succeeded in
   some mode this round, same random draws) - Cov(delivered).
5. **The policy could not see what ARL-v3 rewards.** It saw a static value
   and history, not the current class-coverage deficit or what it had
   already picked this round. **Fix (`policy.coverage_features`):** two
   more inputs, recomputed after every pick within a round -- the coverage
   client i would add to the clients picked so far, plain and weighted by
   each class's recent deficit -- so the policy can pick complementary
   clients. In `s2_*` every RL variant gets the same inputs (these, plus the
   corrected value input), so RL variants differ only in reward and mode rule.
6. **Baseline comparisons mixed two effects.** FAVOR, DELTA and REFL ran on
   Standard while the RL policy chose modes. **Fix:** each also runs with
   the shared expected-work mode rule (`b_favor_ew`, `b_delta_ew`,
   `b_refl_ew`), and a simple non-learning rule for the whole joint
   decision with the RL policy's information is added (`b_cov_ew`:
   expected-work modes + greedy expected coverage gain weighted by recent
   class deficits). FAVOR remains the single-run online adaptation
   described above, not the paper's multi-run pre-training.

#### Second review (also before any `s2_*` run)

A second outside review found eight more problems. All were checked in the
code and confirmed; all are fixed for `s2_*`, off by default, and every
earlier condition (16 checked, including all baselines whose code changed)
still gives bit-identical results with the fixes off.

1. **Per-client credit depended on processing order.** ARL-v2/v3 credit was
   standardised by a running normaliser updated after *each* client, so the
   same loss could be a penalty or a reward depending on where the client
   sat in the list. **Fix (`reward.batch_client_norm`):** the whole round is
   standardised with the statistics from before it, then they are updated.
2. **Offline clients were charged as if they had trained.** Every invitation
   cost the mode's full cost and battery, even for a client that was offline
   or could not start. **Fix (`system.charge_actual`):** cost and battery
   only for clients that actually ran.
3. **Multi-round credit silently dropped per-client ARL.** With
   `policy.horizon > 1` the per-client credit was never read, while the
   shared reward no longer contained it. No ARL-v2/v3 condition used this
   combination; it is now rejected with an error.
4. **The eligibility gate switched itself off.** With fewer than K clients
   passing it, the gate was dropped for everyone. **Fix
   (`policy.strict_gate`):** the clients that pass are always kept, the free
   slots are filled with the highest predicted success, and the number of
   fill-ins is logged (`gate_filled`). Used by the RL policy and the
   greedy-coverage rule, the two methods that use the gate.
5. **DELTA's weighting was not unbiased.** It drew without replacement but
   weighted by the draw probabilities and renormalised: two equal clients
   with updates 0 and 10 and draw weights 0.9 / 0.1 gave 9 instead of 5.
   **Fix:** draws with replacement and Horvitz-Thompson weights, which are
   unbiased when every sampled client reports (checked by simulation: 5.0).
6. **"Rounds to target" hid runs that never got there.** `report.py` now
   prints how many runs reached the target next to the rounds taken by
   those that did; `budget_s2.py` already did.
7. **A resumed sweep could reuse out-of-date results.** A run counted as
   done if its CSV existed. **Fix:** each run's saved config now carries a
   fingerprint of the simulator code (`code_hash`), and `sweep.py` re-runs
   any result whose config, code version or number of rounds does not match
   (`--ignore-code` to keep results across a code change deliberately).
8. **The confirmatory report could print "passes: yes" on partial data.**
   `prespec_s2.py` now prints no verdict for a family unless every run is
   present, finished, made with the protocol fix and by one code version.

**Availability was never really uncertain.** The "noisy" availability
hint is in [0.7, 1] for an online client and [0, 0.3] for an offline one,
so a threshold at 0.5 always tells them apart, and the success predictor
uses it. The RL policy therefore effectively knew who was online, while
Random, Oort, Power-of-Choice, FedCS and the rest invited offline clients and
counted those as failures. Real servers select among the devices that have
checked in, so that comparison was unfair to the baselines: **part of the
RL policy's earlier "30-45% fewer failed invitations" is likely just
avoiding offline clients.** **Fix (`system.checkin_pool`):** every policy
selects only from the clients online this round. The hint remains as it was
(knowing who is online is a fair assumption once everyone has it); what
stays uncertain is whether an online client finishes (deadline, battery,
link).

#### Third review (also before any `s2_*` run)

1. **Normalised credit could explode after a quiet period.** After ten
   zero losses, a loss of 0.2 was standardised to 200,000. **Fix
   (`reward.robust_norm`):** every reward normaliser uses a spread of at
   least 0.001, clips to +-3, and returns 0 until it has seen 10 values.
2. **The confirmatory report checked too little.** It trusted each run's own
   metadata, so one-round runs, missing code fingerprints, runs with the
   fixes off and NaN values could all produce "passes: yes". **Fix:** each
   run is checked against the frozen plan (`configs/sweep_advanced.yaml` via
   `sweep.run_status`): exact config, every round 1..T in order, a non-empty
   code fingerprint equal to the frozen one (`FROZEN_CODE_HASH` in
   `scripts/prespec_s2.py`), no source change during the run, and finite
   values of the tested measurement in every round. Anything else means no
   verdict.
3. **"Worst-client accuracy" was a label-mixture value**, so two clients with
   the same label mix always looked the same. **Fix:** on FEMNIST each
   client's accuracy is now measured on that writer's own held-out images
   (`acc_client_measured = 1` in the log). On the Dirichlet splits (CIFAR-10,
   MNIST) clients have no data of their own beyond their label mix -- their
   images are draws from the same pool -- so the label-mixture value is
   an estimate of a client's expected accuracy under the shared
   class-conditional distribution assumption and is kept; it is
   always also logged as `acc_labelmix_worst10`. Test A2 is read that way:
   on FEMNIST it is about the worst-off writers, on CIFAR-10 about the
   clients with the worst label mixes. With ~16 held-out images per writer
   the FEMNIST value is noisy, and the worst 10% are partly the unluckiest
   samples.
4. **"Equal spend" let expensive methods overshoot** (the first round at or
   above the budget, e.g. 18 against a budget of 10). **Fix:** the last
   round at or below the budget, with the share of the budget actually used
   shown.
5. **The code fingerprint was read at save time,** so editing files during a
   run could label an old-code result as new. **Fix:** the fingerprint is
   taken when the simulator is loaded (`provenance.LOADED`), and a run whose
   source files changed while it ran is flagged (`code_changed_during_run`)
   and never reused or tested. On Kaggle each session runs a fresh clone, so
   nobody edits the code under a running sweep.

#### Fourth review: direction, frozen inputs and consistent measurements

The corrected protocol supersedes the `c8a91ef` simulator used by the MNIST
development check below. The new simulator fingerprint and experiment inputs
are recorded in `configs/s2_frozen.json`; `scripts/prespec_s2.py` pins that
manifest's SHA-256 and the simulator fingerprint. Old CSVs are retained as
historical evidence, but cannot be used for this confirmatory protocol.

1. **A significant difference is not necessarily a win.** The existing
   one-/two-sided alternatives and Holm families are unchanged. Reports now
   distinguish **significantly better**, **significantly worse** (two-sided
   tests only), **inconclusive**, and **no verdict**. Higher accuracy is better;
   lower failure rate is better. A significantly losing method cannot pass.
2. **The freeze includes the experiment inputs.** The manifest pins the
   simulator, sweep YAML, referenced dataset configs, config-expansion and
   reporting code, seeds, comparisons and metric directions. Missing or
   mismatched pins, changed inputs, an unknown run fingerprint, or a different
   simulator checkout produce no verdict. The valid-data fixture uses the
   actual frozen hash; a consistent but incorrect hash is a negative test.
3. **Budget and target comparisons use one fixed test population.** Each new
   CSV logs `accuracy_fixed` on the same held-out examples in every round,
   including the last, and `eval_fixed_n` records its size. `accuracy` retains
   full-test-set final evaluation, as do final client metrics. The budget
   script uses `accuracy_fixed` both for equal-spend comparisons and for
   setting/reaching the Random-derived target. It refuses legacy CSVs missing
   that series. The general report uses the fixed series for new runs and
   labels the legacy fallback explicitly. No reported test score enters
   learning when the validation split is enabled.
4. **No invitations means no failed invitations.** An empty round has
   `fail_ratio = 0` and `empty_round = 1`. Mean round-level failure remains the
   specified endpoint in B/C2/D2; it must not be described as the fraction of
   all invitations that failed. The descriptive report additionally computes
   that fraction from total failures / total invitations within each seed,
   reports empty-round counts, and marks zero-invitation seeds as undefined.
   These counts concern on-time completion; late REFL deliveries remain
   separately logged.
5. **Acceptance rejection counts toward churn.** An update rejected by the
   norm/staleness acceptance rule records `rejected`, so it increments the
   failure streak instead of resetting it. A successful accepted update still
   resets the streak. This fix is not behind a switch, so it is the one change
   that alters an earlier condition: runs with client churn
   (`system.churn_after > 0`: `dev_churn_mnist`, `arl2_churn_cifar`,
   `arl2_churn_femnist`) would now lose slightly more clients if re-run (on a
   synthetic check, one more departure). Their saved results stay valid as
   results of the code that made them. Without churn, every earlier
   condition is still bit-identical (re-checked on 5 conditions).

The robust normaliser warms up over **10 observations, not 10 rounds**.
Per-client batches can reach that count earlier than round-level signals.

Validation of this revision: **127 tests pass on CPU**, including complete
20-seed fixtures where the reference loses on accuracy or failure rate,
incorrect/unset fingerprints, changed manifest inputs, fixed-set evaluation
through the final round, empty online pools and rejected-update churn. This
does not replace a new MNIST development check or the full confirmatory sweep.

When deliberately revising this protocol, first finish and review the code
and config changes, recompute the simulator fingerprint and all manifest
input digests, and update both pins together in one commit. Re-run the
regression suite before starting new confirmatory runs. Do not re-freeze a
protocol to admit existing results after inspecting their outcomes.

#### Fifth review (before any `s2_*` run)

1. **Per-client ARL credit was not a valid gradient of the round's total.**
   A client's loss and gain depend on which other clients were picked (its
   Shapley share, the coverage of the set), but its credit updated only its
   own pick. Example: with 3 clients and all 3 always picked, the total
   coverage cannot change, so the correct pick gradient is zero -- the old
   update gave a non-zero one. **Fix (`policy.credit_split`):** the picks get
   the round's *total* ARL term, standardised with the statistics from
   before the round; each client's own term goes to its *mode* only. That
   part is valid: given the picked set, client i's mode changes only its own
   loss and gain. The shared reward (accuracy, cost, latency, fairness)
   still goes to picks and modes as before. Clipping the standardised terms
   at +-3 (robust normaliser) makes the update a bounded, slightly biased
   version of that gradient; this is stated rather than hidden. Applied to
   both ARL-v2 (`arl2_full`) and ARL-v3 (`arl3_full`).
2. **FAVOR learned from clients it could not have picked.** Its Double-DQN
   target took the best next action over all clients, including offline
   ones. **Fix:** each stored transition keeps which clients were selectable
   in the next state, the target chooses only among them, and with none
   there is no bootstrap term.
3. **FedCER-like's time estimate left out the client's data volume**, so it
   chose lighter modes than needed (120 work units, unit speed: Priority
   takes 0.65 of the deadline, but it chose Light). **Fix
   (`policy.shared_time_estimate`):** the same workload-scaled estimate as
   the engine (`base.est_time`).
4. `scripts/run_experiments.py` (the older runner) now reuses a result only
   if `sweep.run_status` accepts it, like `sweep.py`.
5. With `policy.horizon > 1`, the last unfinished window is now learned
   from at the end of the run (no effect on anything logged; `s2_*` uses
   horizon 1).
6. `stats.py`'s curve average and `plot.py`'s accuracy curves use
   `accuracy_fixed` when the CSV has it; final results still use full-test
   `accuracy`.

The protocol was re-frozen with `scripts/freeze_s2.py`, which writes the
manifest and both pins in one step. **GPU smoke groups** (`smoke_s2_*`: every
`s2` condition, 5 rounds, seed 900, which no test uses) are run first on
Kaggle to check the code end to end on a GPU before the long runs.

#### Sixth review (before any `s2_*` run)

The fifth-review fixes were kept, with one change and stricter tests:

1. **One round objective for ARL credit.** The fifth-review credit split is
   valid but harder to justify. It is replaced in the `s2_*` runs by the
   simplest consistent rule (`policy.credit_total`): the policy learns from
   one round reward,
   R_t = shared reward + sum over invited clients of client credit_i,
   client credit_i = -0.5 z(loss_i) + 0.5 z(gain_i),
   applied to the log-probability of the *whole* joint action (all picks and
   modes), with a REINFORCE baseline equal to the mean of *previous* rounds'
   rewards, updated after the round (`policy.prior_baseline`; originally the
   current reward was folded in first). This is plain REINFORCE on R_t, so
   it accounts for how one pick changes the value of the others. A test
   enumerates every pick order and mode for 3 clients (2 picked) and checks
   that the expected update equals the exact gradient of E[R]; the
   constant-coverage case (3 of 3 picked) gives zero. The objective is
   non-stationary in one respect, stated rather than hidden: z standardises
   with running statistics and clips at +-3.
2. **FAVOR and availability.** Besides the next-state mask, a round with
   nobody online is now handled explicitly: no decision is made, and the
   last decision's transition skips forward to the next round with a choice,
   collecting the rewards in between discounted (r_1 + g r_2 + g^2 r_3 ...)
   and bootstrapping with g^k after k rounds. Test: an offline client with
   Q = 10^6 does not enter the target.
3. **One timing helper** (`base.est_time`) is used by the engine,
   FedCER-like and FedCG. Tests: workloads of 120 / 400 / 900 give Priority /
   Standard / Light, with estimates ordered by mode.
4. **Resume validation** in the older runner: truncated CSV, missing config,
   changed setting and changed code each trigger a re-run (tested).
5. **Last window:** a 23-round run with horizon 10 learns from 10 + 10 + 3
   rounds; a 3-round run from 3 (tested).
6. **Curves vs final scores:** curves, curve averages and rounds-to-target
   all use one helper (`report.accuracy_curve`): `accuracy_fixed` when
   logged; for older files the per-round accuracy without a final round that
   switched test sets (when `fl.eval_subset > 0`); refused when that cannot
   be checked (no saved config). Final headline scores keep full-test
   `accuracy`. Tested with a final full-test score deliberately different
   from the fixed-set curve.

**Checks of the current frozen code (commit `1349d2a`, fingerprint
`fd90310c1b7c34e9`), each run from a frozen snapshot of that commit:**

- *Real-data smoke, CPU* (`smoke_s2_*`, 50 runs, 5 rounds, seed 900): 50 of
  50 finished, every logged number finite (including `accuracy_fixed`), one
  fingerprint, one initial model per group, FEMNIST client accuracy measured.
- *MNIST bug check* (`dev_s2_mnist`, 69 runs, seeds 0-2, 100 rounds): 69 of
  69 finished, no errors, NaNs or mid-run source changes, one fingerprint,
  the same initial model for every method within a seed. With the single
  round objective, ARL-v3 mixes modes again (4.2 Light / 4.7 Standard / 1.1
  Priority per round). Nothing was changed after these checks.
- *GPU:* not checkable here; the `smoke_s2_*` groups are the first Kaggle
  step. The freeze is final once that run is clean.

**Checks of the code frozen before the sixth review (commit `434fe22`,
fingerprint `c4c4c85c373f99b1`; superseded), each run from a frozen snapshot:**

- *Real-data smoke, CPU* (`smoke_s2_*`: CIFAR-10 and FEMNIST, all 23 `s2`
  conditions plus the 4 churn ones, 5 rounds, seed 900): 50 of 50 runs
  finished, every logged number finite, one code fingerprint, one initial
  model per group, FEMNIST client accuracy measured on each writer's own
  images. Not yet run on a GPU: that is the first Kaggle step.
- *MNIST bug check* (`dev_s2_mnist`, seeds 0-2, 100 rounds): 69 of 69 runs
  finished, no errors or NaNs, one fingerprint, the same initial model for
  every method within a seed, no mid-run source change. One behaviour to
  note, not a bug and not tuned away: with the credit split, ARL-v3's modes
  went back to mostly Light (9.0 Light / 0.6 Standard / 0.4 Priority per
  round, against 6.3 / 1.2 / 2.4 before the split) -- the per-client gain
  for a delivered update outweighs the lost-work term. ARL-v3 was not
  changed after seeing this; the pre-specified tests decide whether it
  helps.

#### The `s2_*` runs and how they are checked (written before looking at outcomes)

All 1,540 `s2_*` runs were made on Kaggle (2x T4) in one session of 10.4 h,
0 failed (`sweep.log`), from commit `ae0d449` (code fingerprint
`fd90310c1b7c34e9`, the frozen one). Every run's saved config was compared
with the frozen plan before any result was looked at: 1,540 of 1,540 match
exactly in every experiment setting, all carry the frozen fingerprint, none
was flagged for a source change during the run. The one field that differs
is `out`, the folder the run was saved in (`/kaggle/working/adv/<group>` on
Kaggle, `results/adv/<group>` here). `scripts/prespec_s2.py` compares that
field too, so it reports "different config" for every run.

Decision, fixed before any outcome was read: the save folder is not part of
the experiment, so neither the data nor the frozen scripts are changed. The
confirmatory analysis uses the frozen script's own `--root` option with the
folder the runs were made in:
`/kaggle/working/adv` -> `results/adv` (a symbolic link), then
`python scripts/prespec_s2.py --root /kaggle/working/adv`. Every other
check stays in force.

#### Outcome of the pre-specified `s2_*` tests

Full output: `results/adv/s2_prespecified_tests.md` (every family complete,
no validation problem). Exact paired sign-flip, 20 seeds, Holm within family.

- **A / A2 -- ARL-v3 does not protect the worst-represented classes or
  clients.** 0 of 4 and 0 of 4 pass. Worst-10%-class accuracy vs the plain
  failure count: -0.000 (CIFAR-10), -0.004 (FEMNIST), -0.047 (CIFAR-10 +
  churn), -0.000 (FEMNIST + churn); worst-10%-client accuracy: -0.002,
  -0.025, -0.040, +0.011. All inconclusive; seven of eight point the wrong way.
- **B -- ARL-v2's efficiency is not more than "always Light".** 0 of 2 pass,
  and the direction is reversed: ARL-v2 fails *more* often than the RL
  policy with every client on Light (0.418 vs 0.324 on CIFAR-10, 0.546 vs
  0.308 on FEMNIST).
- **C -- proposed system (`c2_rl_fedprox`) vs the stronger baselines, final
  accuracy.** CIFAR-10: significantly better than DELTA (+0.063) and DELTA
  with the mode rule (+0.033); significantly worse than REFL with the mode
  rule (-0.048); the other six inconclusive. FEMNIST: significantly worse
  than six of nine (FAVOR, REFL, DELTA with the mode rule; Random, Oort,
  Power-of-Choice with the mode rule; by -0.17 to -0.20), the other three
  inconclusive and also lower.
- **C2 -- failure ratio.** CIFAR-10: the proposed system fails significantly
  less than eight of nine (0.440 vs 0.545-0.744); not different from Oort
  with the mode rule (0.449). FEMNIST: all nine inconclusive.
- **D -- learning vs a rule, same reward (`v2_rawfail`).** Accuracy: all six
  inconclusive (learned modes vs the expected-work rule, vs always Light,
  vs the greedy-coverage rule). **D2 -- failures:** learned modes fail
  significantly less than the expected-work rule (CIFAR-10) and the
  greedy-coverage rule (CIFAR-10), and significantly *more* than always
  Light on both datasets (0.398 vs 0.324; 0.507 vs 0.308).

**Descriptive (not pre-specified):**

- **RL policies that learn their own modes sometimes lock up.** In some runs
  every invitation fails for the rest of the run: after a few early
  failures the success predictor rates every client below 0.5, nobody
  passes the gate, all ten slots are fill-ins, the mode head has drifted to
  Priority, everything fails, and the model never trains again (e.g.
  `c2_rl_fedprox` FEMNIST seed 110: 1 success in rounds 21-200, final
  accuracy 0.066). Runs with 50+ consecutive rounds of zero successes:
  FEMNIST weak regime -- `c2_rl_fedprox` 4/20, `arl3_shared` 3/20,
  `arl3_full` 2/20, `arl2_full` 1/20, `v2_rawfail` 1/20; a few elsewhere
  (CIFAR-10 `arl2_full` 1/20; no-weak-client groups 1/10 for `c2`,
  `v2_rawfail`, `arl3_full`); **never** for the RL policy with a fixed mode
  rule (`rl_ew`, `rl_light`) or any baseline. This is most of the FEMNIST
  accuracy gap in family C (`c2` seed spread 0.24 vs ~0.02-0.06 for the
  baselines).
- **ARL cannot see this failure.** Inside a locked-up run ARL' is 0.000 and
  ARL-v3's loss ~0.006 while 100% of invitations fail: ARL counts only
  losses among clients the predictor expects to succeed, so a pessimistic
  predictor makes every loss look unavoidable. Standardised rewards add to
  it: a constant all-fail round becomes the new normal and gives no
  gradient. This is a concrete, reportable limitation of
  predictor-relative loss measures.
- **The mode choice is most of what matters.** Giving the selection-only
  baselines the expected-work mode rule raised their accuracy on every
  dataset (Random 0.510 -> 0.553 CIFAR-10, 0.715 -> 0.777 FEMNIST; Oort
  0.481 -> 0.564, 0.716 -> 0.785). The RL policy with every client on
  Light (`rl_light`) fails least of all RL variants (0.31-0.32), is
  cheapest (~7.0 per round), loses the fewest clients under churn (20-22
  vs 26-31) and is among the most accurate at equal spend.
- **Equal spend** (`results/adv/s2_*/budget.md`): at the same cumulative
  cost the proposed system is behind `rl_light`, REFL with the mode rule and
  MimiC on both datasets (CIFAR-10 -0.06 to -0.07; FEMNIST -0.16 to -0.17).
  Its earlier efficiency advantage does not survive equalising spend.
- **No weak clients** (`s2_main_*`, 10 seeds): the same order -- Oort and
  REFL with the mode rule, MimiC and `rl_light` lead; `c2_rl_fedprox` is
  last or near last on accuracy (0.477 CIFAR-10, 0.704 FEMNIST).

**What this supports for the thesis, after all fixes:** (1) ARL -- original,
v2 and v3 -- is not shown to improve accuracy, worst-class or worst-client
accuracy, or retention, and has an identified blind spot (a pessimistic
predictor hides all losses); (2) choosing the participation mode well is
what improves both accuracy and efficiency, and a simple rule (expected
work, or always Light for efficiency) does at least as well as the learned
RL mode choice, which can lock up; (3) the learned joint policy is not shown
to beat strong baselines given the same mode choice, and is worse on
FEMNIST. The original efficiency claim (fewer failures than Random/Oort)
holds on CIFAR-10 against baselines that always use Standard, but not
against mode-fair baselines on FEMNIST, nor at equal spend.

#### ARL-v3 as frozen for the `s2_*` runs

For each invited client i in round t (S_t = the invited set):

- m_i = phi_i(S_t) x (1 + sum_k c_ik u_k), where phi_i(S_t) is i's Shapley
  share of Cov(S_t) = sum_k max_{j in S_t} c_jk, c_ik its class shares, and
  u_k = 1 - q_k / max(q) the recent coverage deficit of class k (q = moving
  average, rate 0.1, of the classes covered by successful clients);
- W(m) = P(success | m) x local_epochs[m] for modes with P >= 0.5 (or the
  safest mode if none), else 0; best = the mode maximising W;
- loss_i = m_i (W(best) - W(assigned)) / 4, gain_i = m_i Y_i;
- `arl3_full`: the policy learns from R_t = shared reward + sum_i
  (-0.5 z(loss_i) + 0.5 z(gain_i)), z = the robust, order-free normaliser
  above, on the whole joint action, with a previous-rounds baseline (sixth
  review, item 1); `arl3_shared` instead puts -0.5 z(sum loss) + 0.5 z(sum
  gain) -- the round totals standardised once -- in the shared reward. The
  two are different objectives and are reported as such.
- The policy's inputs: the 12 original features with the corrected value
  input, plus the coverage client i adds to the clients already picked this
  round, plain and deficit-weighted, recomputed after each pick.

New per-run logs for "who benefits": `acc_client_worst10` (own held-out writer
accuracy on FEMNIST; on CIFAR/MNIST, estimated expected accuracy from
per-class held-out accuracy weighted by each client's label shares;
averaged over the worst 10% of clients),
`acc_client_weak`, `gap_max_all` / `gap_max_weak` (longest run of rounds
without a contribution), and predictor calibration by mode and by weak /
other clients (`ece_light`, ..., `ece_weak`, `ece_strong`).

> **Pre-specified tests (fixed before any `s2_*` run; fresh seeds 100-119,
> never used before; exact paired sign-flip; Holm within each family; a
> superiority requires Holm-adjusted p < 0.05 and a favourable difference;
> two-sided significant adverse differences establish inferiority; no verdict unless every run
> passes the validation in "Third review" item 2). Computed by
> `python scripts/prespec_s2.py`:**
>
> **A. Does ARL-v3 protect the worst-represented classes?** `arl3_full` vs
> `v2_rawfail`, final `acc_worst10`, one-sided (ARL-v3 better), in
> `s2_cifar`, `s2_femnist`, `s2_churn_cifar`, `s2_churn_femnist` (4 tests).
> **A2.** The same 4 pairs on final `acc_client_worst10` (FEMNIST: measured
> on the worst-off writers' own held-out images; CIFAR-10: the clients with
> the worst label mixes -- see "Third review" item 3).
>
> **B. Is ARL-v2's efficiency more than "always Light"?** `arl2_full` vs
> `rl_light`, mean `fail_ratio`, one-sided (ARL-v2 fewer), in `s2_cifar`
> and `s2_femnist` (2 tests).
>
> **C. Does the proposed system beat the stronger baselines on accuracy?**
> `c2_rl_fedprox` vs `b_favor`, `b_favor_ew`, `b_refl`, `b_refl_ew`,
> `b_delta`, `b_delta_ew`, `b_random_ew`, `b_oort_ew`, `b_poc_ew`, final
> `accuracy`, two-sided, in `s2_cifar` and `s2_femnist` (18 tests).
> **C2.** The same 18 pairs on mean `fail_ratio`.
>
> **D. Does learning the decision beat a simple rule, with the same
> reward?** `v2_rawfail` vs `rl_ew` (learned vs rule-based modes),
> `rl_light`, and `b_cov_ew` (no learning at all), final `accuracy`,
> two-sided, in `s2_cifar` and `s2_femnist` (6 tests). **D2.** The same
> 6 pairs on mean `fail_ratio`.
>
> An inconclusive test means "not shown", never "shown equal". A significantly
> worse result is reported as such. Everything else
> is descriptive, including `s2_main_*` (no weak clients), `arl3_shared`,
> the set-level coverage loss, gaps, calibration, and the **equal-spend
> analysis** (`python scripts/budget_s2.py --dir results/adv/<group>`):
> each method's fixed-set accuracy at its last checkpoint within the cheapest
> method's total spend for that seed, with actual budget use shown; and the
> cost to first reach 90% of Random's final fixed-set accuracy. Both comparisons
> use `accuracy_fixed`, including at the final round.

**Historical development check (`dev_s2_mnist`, seeds 0-2, CPU, fixes through
the third review; bug check only, predating the fourth-review changes):** run from a frozen snapshot of
commit `c8a91ef`; 69 of 69 runs finished, no errors or NaNs, one code
fingerprint (`cf531573f0858241`), every method started from the same model
within each seed (3 distinct initial models for 3 seeds), no mid-run source
change. Behaviour is as designed: with the check-in pool, Random's failure
ratio is 0.71 (it no longer invites offline clients), the expected-work
rule moves Random/Oort/Power-of-Choice/FAVOR/DELTA/REFL off all-Standard and
cuts their failures (e.g. Oort 0.74 -> 0.48), ARL-v3 mixes modes (6.3 Light /
1.2 Standard / 2.4 Priority per round), and the strict gate fills ~0.7-2
slots per round for the RL variants when fewer than 10 online clients pass
the gate (logged as `gate_filled`).

**Frozen.** The code the `s2_*` runs must be made with has fingerprint
`cf531573f0858241` (commit `c8a91ef`; `FROZEN_CODE_HASH` in
`scripts/prespec_s2.py`). Later commits that change only docs or scripts
keep it; any change to `src/afl` changes it, and `prespec_s2.py` then gives
no verdict on runs made with the new code.

**Still open, not addressed by these runs:** generalisation to fleets,
availability patterns and partitions the policy was not developed on; and
the thesis must say which fairness it means -- equal participation, equal
opportunity to contribute, or similar model quality across clients --
because the contribution-count CV and worst-class / worst-client accuracy
measure different things. The class histograms the scheduler uses
(Sec. 3.5) reveal each client's label proportions to the server; they are
not privacy-preserving by themselves.

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

- Supported before the fixes, to be re-confirmed by `s2_*` (3 datasets, 10
  seeds, robust to scale, skew and real device speeds): a learned joint
  selection-and-configuration policy wastes far fewer invitations than
  Random/Oort (about 0.45 vs 0.65-0.73; part of this gap came from the
  baselines inviting offline clients), costs ~20-30% less,
  protects weak and rare-data clients better, and clearly beats FedCER-like,
  FedCS and Power-of-Choice. On CIFAR-10 it is also more accurate than Random.
  On MNIST and FEMNIST its accuracy shows no significant difference from
  Random, Oort and imputation -- not shown to be equal.
- Not supported: that ARL's value weighting adds anything measurable beyond a
  plain failure count. It did not in the independent-failure pilot, in the
  persistent-failure regime, in the 30-seed confirmation, in FEMNIST, or in
  the weight sweep (including weight 0), and it produces slightly more
  failures. The success-based fairness term showed no effect either.
- Needs restating: "prevention beats imputation". Against a faithful
  correction method (MimiC) the RL policy is not shown to be more accurate
  and wins on efficiency.
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
- In every group before the simulator fix, Oort's system utility used each
  round's true (hidden) client times, which a real server would not know, so
  wins over Oort there understate the proposed policy's margin; and Light and
  Standard uploads were not really compressed, which flattered every policy
  that used them. Both are fixed in the `s2_*` groups.
