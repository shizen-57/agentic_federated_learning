# Does the claimed contribution actually hold up? (Tasks 1-5, run in serial)

This document reports the direct tests the user asked for after the novelty
discussion: isolate each of the thesis's two specific claims with a same-policy
ablation, add the two missing Table 3.5 baselines, re-check the literature one
more time, and report the results honestly -- including the ones that don't
flatter the thesis. All numbers below are mean +/- std across 5 seeds, N=30
clients, K=5, 100 rounds, MNIST, matching every other Chapter 5 result.

## Task 1: does the value-weighted counterfactual damage term (ARL) matter?

Compared `c2_rl_fedprox` (reward uses ARL') against `ablation_rawfail`
(identical config, reward's damage term replaced by the raw failed-selection
ratio D^fail_t instead of ARL').

| | ARL' reward (proposed) | raw failure-count reward |
|---|---|---|
| Final accuracy | 0.963 +/- 0.016 | 0.950 +/- 0.021 |
| CV fairness | 1.176 +/- 0.087 | 1.165 +/- 0.172 |
| Failed-participation ratio | 0.526 +/- 0.104 | **0.459 +/- 0.059** |
| Cost | 6.168 +/- 1.511 | **5.291 +/- 0.771** |
| Latency | 2.184 +/- 0.610 | **1.746 +/- 0.523** |
| Rounds to 90% accuracy | 17.4 +/- 9.2 | 18.0 +/- 12.1 |

**Honest verdict: this ablation does NOT clearly support the claim.** On three
independent metrics (failure ratio, cost, latency) the *simpler* raw
failure-count reward does *better*, not worse. ARL's only clear advantage is a
lower ARL' value itself (3.06 vs 3.59) -- which is close to circular, since
that policy is the one directly optimizing that exact quantity. CV and
accuracy are statistically indistinguishable given the overlapping standard
deviations. At N=30/K=5 with this reward-coefficient setting, the extra
complexity of value-weighting failures is not earning its keep over just
counting them. This is a genuine, reportable negative result, not a bug --
verified against the actual per-round curves (`figures/ablation_task1_task2.png`),
which show near-identical accuracy/CV trajectories and the raw-failure version
tracking noticeably lower failure/cost/latency for most of the run.

## Task 2: does fairness-over-accepted-contributions matter versus fairness-over-selection-count?

Compared `c2_rl_fedprox` (reward's fairness term uses CV over successful
contributions) against `ablation_cvsel` (identical config, fairness term uses
CV over raw selection counts instead).

| | CV-over-success (proposed) | CV-over-selection |
|---|---|---|
| Final accuracy | 0.963 +/- 0.016 | 0.954 +/- 0.031 |
| **CV over successful contributions** (the metric that matters) | 1.176 +/- 0.087 | 1.168 +/- 0.143 |
| Failed-participation ratio | 0.526 +/- 0.104 | 0.523 +/- 0.098 |
| Cost | 6.168 +/- 1.511 | 7.178 +/- 1.479 |
| Rounds to 90% accuracy | 17.4 +/- 9.2 | 21.6 +/- 14.9 |

**Honest verdict: also does not show a large effect.** The two variants land
on almost identical actual successful-contribution fairness (1.176 vs 1.168 --
well within noise). Optimizing for plain selection-count fairness in this
simulated environment nearly reproduces the same successful-contribution
fairness anyway. The proposed metric does show a real, if modest, edge in cost
(14% lower) and time-to-target (17% fewer rounds), but the headline fairness
number the claim is about does not differ meaningfully here.

**Likely reason for both null results**: in this simulator, failure causes
(availability, energy, deadline) are drawn independently per round rather than
being strongly and persistently correlated with specific clients, so a client
who is "selected often" and a client who "succeeds often" end up close to the
same set most of the time. The claims may show a bigger effect in a regime
with more persistent per-client unreliability (e.g., a fixed subset of clients
that are always low-power) -- that is a concrete, testable follow-up, not
covered here.

## Task 3: the two missing Table 3.5 baselines

**FedCER-like (greedy submodular selection + greedy per-client epoch config,
no learning, no fairness/success-awareness):**

| | FedCER-like | Proposed (c2_rl_fedprox) |
|---|---|---|
| Final accuracy | **0.720 +/- 0.251** | 0.963 +/- 0.016 |
| CV fairness | **3.493 +/- 1.182** | 1.176 +/- 0.087 |
| Rounds to 90% accuracy | 55.0 (only reached late/inconsistently) | 17.4 |

This is a large, clear gap -- and it is the result Table 3.5 predicts: the
thesis's own text says FedCER "includes no fairness or contribution-aware
mechanism." Reproducing it faithfully (diversity-only selection with zero
success-probability awareness) makes it pick clients that frequently fail,
tanking both accuracy and fairness. This *is* the "closest joint
selection-and-configuration competitor" losing badly, and it is losing for
exactly the reason the thesis argues it should. This result is evidence
*for* the thesis's positioning, not just implementation busywork.

**MIFA/MimiC-like (Oort selection + post-hoc substitution of a failed
client's last known-good update; prevention-vs-imputation test):**

| | Oort, no imputation | Oort + imputation (MIFA/MimiC-like) | Proposed RL |
|---|---|---|---|
| Final accuracy | 0.942 +/- 0.018 | 0.880 +/- 0.113 | **0.963 +/- 0.016** |
| CV fairness | 1.095 +/- 0.191 | 1.153 +/- 0.221 | 1.176 +/- 0.087 |
| Failed-participation ratio | 0.732 +/- 0.086 | 0.686 +/- 0.069 | **0.526 +/- 0.104** |
| Rounds to 90% accuracy | 15.4 +/- 12.5 | **10.2 +/- 5.5** | 17.4 +/- 9.2 |

Nuanced, honest result: imputation gives Oort a *faster early climb*
(10.2 vs 15.4 rounds to 90%) -- substituting a stale-but-real update clearly
beats leaving a hole in the aggregation early on. But it comes with **worse
and much more variable final accuracy** (0.880 +/- 0.113 vs 0.942 +/- 0.018
without imputation) -- stale substitutes eventually diverge from where the
global model has moved and start hurting more than they help. RL's
*prevention* strategy (avoid the failure in the first place via configuration)
gets both the best final accuracy and the lowest failure rate of the three,
without the late-training instability imputation introduces. This is a clean,
real result *for* the thesis's prevention-over-imputation argument (Sec.
3.11) -- on accuracy stability and failure rate specifically, not on fairness,
where the three are statistically close.

## Task 4: literature re-check, right before writing this up

Two more targeted searches, specifically on the two narrow claims:

- **Claim 1 (counterfactual/oracle-relative damage weighting)**: no exact
  match found. Closest hits are generic loss-weighted or contribution-based
  aggregation papers; none construct an explicit "what a perfectly-informed
  scheduler would have achieved" counterfactual. This claim still looks clear.

- **Claim 2 (fairness conditioned on genuine capability, not raw
  selection/outcome)**: found a close, very recent match --
  [Cumulative Utility Parity for Fair Federated Learning under Intermittent
  Client Participation](https://consensus.app/papers/details/a3d2c74ea53f577cb23d94681662a3a9/?utm_source=claude_desktop)
  (Behfar & Mortier, arXiv:2602.13651, 14 Feb 2026, Cambridge).

  **RESOLVED -- paper has now been read in full.** Correcting an earlier note
  in this document: it is dated February 2026, i.e. roughly seven months
  before this thesis is being finalised, so it is **prior art, not
  concurrent work**, and must be cited.

  What it actually does:
  - Utility is the benefit the client *receives* -- operationally "the change
    in per-client accuracy between consecutive rounds" -- not the contribution
    the client makes. Opposite direction to this thesis's X^succ ledger.
  - Its mechanism is *normalisation*: divide cumulative utility by the
    client's empirical availability rate (u_k / pi_k), plus inverse-
    availability sampling that provably equalises long-run selection
    frequency. Not a feasibility gate, and no data-rarity weighting.
  - Availability is **exogenous and stationary** by assumption (A_k(t) i.i.d.
    Bernoulli with fixed mean pi_k). The server samples among whoever is
    online; it cannot affect whether a client succeeds.
  - Its Lemma 3 / Theorem 3 handle missing clients by **stale-surrogate
    imputation** (substituting the last available update, with a staleness
    bound) -- i.e. exactly the MIFA/MimiC family this thesis argues against.
  - No RL, no per-client configuration assignment, no per-configuration
    success prediction. Baselines: q-FFL and PHP-FL; 100 clients, 50 rounds.

  **Consequences for the thesis:**
  1. The general principle "do not charge a client for unavoidable absence,
     and per-round fairness masks long-term under-representation" is now
     established prior art and can no longer be claimed as novel. PHP-FL
     (Wu et al., 2025), cited in their introduction as reweighting client
     contributions by estimated participation likelihood, is a second
     prior-art hit this thesis currently does not cite either. **Cite both.**
  2. The surviving, sharper distinction: prior availability-aware fairness
     treats availability as **exogenous** and corrects for it statistically
     after the fact. This thesis treats feasibility as partly **endogenous to
     the server's own configuration decision** -- a client that would have
     succeeded under Light but was assigned Priority was not "unavailable,"
     the scheduler broke it. That concept has no counterpart in their model.
     Claim 2 should be reframed around this, not around the generic
     avoidable/unavoidable split.
  3. Their Lemma 3 is the imputation approach this thesis opposes, and Task 3
     above provides the empirical contrast directly (imputation: faster early
     convergence, worse and far more variable final accuracy). Their own
     mechanism can be cited as a tested-and-beaten baseline.
  4. Claim 1 (value-weighted, oracle-relative counterfactual damage) is
     untouched by this paper -- it neither weights by data rarity nor
     constructs a counterfactual. It remains the cleaner of the two claims.

## Task 5: the honest overall verdict

Putting Tasks 1-4 together, plainly:

1. **The two isolated ablations (Tasks 1-2) did not demonstrate a strong,
   clean effect for either individual claim** in this experimental regime.
   Report this in Chapter 5/6 as-is -- it is a legitimate, useful negative
   result about *when* value-weighting and capability-conditioning help
   (not "never," but "not clearly in this exact independent-failure regime"),
   and it points to a concrete follow-up (persistent per-client
   unreliability) rather than undermining the thesis.
2. **The two missing baselines (Task 3) came out in the thesis's favor**,
   and for the reasons the thesis's own argument predicts: FedCER-like
   selection fails badly without a fairness/success mechanism, and
   imputation-based recovery (MIFA/MimiC-like) trades early speed for
   late-training instability that prevention avoids.
3. **The literature re-check (Task 4) surfaced one real, recent, close
   overlap** on claim 2 specifically, that needs to be read and addressed
   before submission -- this is exactly the kind of finding a committee
   member could otherwise raise first.

**Net assessment**: this is what an honestly-run validation looks like --
mixed, not a clean sweep. The strongest evidence for the thesis right now is
the baseline comparison (Task 3), not the isolated mechanism ablations (Tasks
1-2). That is worth stating plainly in the thesis itself: the contribution is
best supported as a *combination* (the joint system beats the closest
published competitors, clearly, for the reasons argued) rather than as two
cleanly-separable mechanisms each independently proven. That is a smaller, but
more defensible, claim than "both of our two ideas individually and
obviously help" -- and it will survive a committee's questions better than an
overclaimed version would.
