# Project review: gaps found (2026-10-29)

A read-through of the code, configs, docs and results, after the fourth
external review's milestone. No tests were run for this review. Each gap:
what is wrong, why it matters, and a suggested fix. Ordered by how much it
could change the thesis's conclusions.

## A. Fix before the next GPU runs (cheap now, costly later)

**A1. The main comparisons mix two changes: the scheduler AND the
aggregation method.**
- The tuned method is `ppo_full_mimic`: it combines updates with MimiC.
  Every primary comparator uses FedProx (`rl_light`, `arl3_full_pg`,
  `b_oort_ew`, `b_favor_ew`, `b_delta_ew`, `b_cov_ew`); REFL uses its own.
- So all 8 primary tests compare "new scheduler + MimiC" with "other
  scheduler + FedProx". In screen B (5 seeds), adding MimiC to the same PPO
  scheduler gave about +0.013 equal-spend accuracy and +0.034 worst-10%. A
  win could come from MimiC, not from the scheduler.
- The sensitivity screen (not run yet) has the same problem: tuned (MimiC)
  vs `rl_light` (FedProx).
- Fix: compare like with like. Use `rl_light_mimic` as the always-Light
  comparator in P1, P2 and the sensitivity screen, and give the best
  baseline and `arl3_full_pg` MimiC versions too (or run the tuned method
  with FedProx as well). Keep the FedProx comparisons as secondary.

**A2. The fairness endpoint (P2, `w10`) is measured at unequal spend.**
- `eq5` (P1) stops at a cumulative cost of 1,400. `w10` is the last 10
  rounds of the whole run, whatever was spent.
- In M7 stage 2 the tuned method spent a median 1,818 against always-Light's
  1,371, about 33% more. Part of a fairness gain could simply be bought
  with that extra spending.
- Fix: add an equal-spend fairness endpoint (worst-10% over the last rounds
  within the same 1,400 budget) and make it P2, or report both and say which
  one is primary before the freeze.

**A3. The cost constraint does not reach its target.**
- The soft constraint (Lagrangian) aims at 7.5 cost units per round
  (K x mean Light cost). The tuned method spends 9.1 per round (+21%);
  always-Light spends 6.9. The hard cap (3x = 22.5 per round) is far above
  that.
- So "safe, budget-constrained scheduler" is not quite true for c34. This
  also feeds A2.
- Fix: report it honestly; consider a larger dual step (`lagrange_lr`) or a
  tighter hard cap in screen C, decided before the run.

**A4. If the ARL screen says "ARL adds value", there is no defined route into
the final method.**
- `docs/RL2_ARL_SCREEN.md` says the final method "is re-examined with it",
  but screen C's rule (`decide`) can only choose `tuned_pt` or
  `tuned_scratch`, never an ARL version.
- Fix: write the route down now, before the ARL result is known. For
  example: "if arl3 or arl4 adds value, screen C's `tuned` method becomes
  the tuned settings plus that ARL version".

**A5. The confirmation report does not show what the docs promise.**
- `docs/RL2_CONFIRMATION.md` (written in the fourth-review work) says the
  report discloses actual spend, accuracy at common budget checkpoints and
  pre-training cost.
- `scripts/prespec_rl2.py` reports only the 8 tests and the secondary
  comparisons: no spend, no cost curves, no starvation, no pre-training cost.
- Fix: add these to `prespec_rl2.evaluate` (descriptive) before the freeze.

**A6. P2's second comparator is not the strongest fairness baseline.**
- P2 compares against always-Light and `arl3_full_pg` only. A baseline
  built to protect weak clients (for example the coverage-greedy
  `b_cov_ew`) is never a primary comparator on fairness. P1 uses "the best
  baseline", but P2 does not.
- Fix: choose P2's second comparator in screen C as the best baseline on
  `w10` (by the same rule as P1's), or add it.

## B. Validity gaps worth closing (or stating as limitations)

**B1. Unequal tuning effort.** SAFE-PPO had a 40-candidate random search
(M7). The baselines had none. Give the main baselines (at least Oort and the
best one) a small matched search on the same development seeds, or state it
plainly as a limitation.

**B2. Tuned on FEMNIST only.** M7 used `s2_femnist` alone; CIFAR-10 gets the
FEMNIST settings. Say so; consider a short CIFAR check of the top 5.

**B3. Pre-training is simulator-to-simulator.** Practice and test episodes
come from the same simulator family, so a pre-trained scheduler may learn
the simulator rather than something that transfers. The sensitivity screen
tests only the from-scratch method. Add a shifted-world check for the
pre-trained method (FedScale device trace, another weak share, churn).

**B4. One built-in scenario.** The fairness problem is built in: weak (slow)
devices hold the rarest classes. RL2 never tests the other settings that s2
had (random weak clients, no weak clients, churn), although the RL2 plan
(M9) promised churn and no-weak settings as secondary. Add them as secondary
settings in the freeze.

**B5. The measured device trace is unused.** `system.device_trace: fedscale`
(real phone speeds and bandwidths) exists, but no RL2 run uses it. It is the
cheapest way to show that the result does not depend on our invented device
tiers. Add it as one more setting in the sensitivity screen.

**B6. "Fair" is measured by one number.** The confirmation's only fairness
endpoint is worst-10% client accuracy. The thesis title promises fair
orchestration. The stress test found that about 14% of clients (18% after a
shock) were never invited in the last 60 rounds. Add descriptive secondary measures: the share of clients
that never contribute, the spread of contributions (already logged as a CV),
weak-client accuracy (`acc_client_weak`, logged) and the longest gaps
(`gap_max_weak`, logged).

**B7. `w10` is noisy on FEMNIST.** A client's own test set has about 13
examples (median: 10% held out, 70% of that for test). The mean of the 10
worst clients is therefore noisy; pairing by seed helps. State it, and report
the number of examples per client.

**B8. The seed count may come out too small.** The power analysis uses the
spread from screen C's 10 seeds, which is itself uncertain. Use a
conservative value (for example the larger of the M7 stage-2 and screen C
spreads, or an upper confidence limit).

**B9. CIFAR-10 has weaker data separation than FEMNIST.** FEMNIST's practice,
development and confirmation clients are different people. CIFAR-10's are
not: all runs draw clients from the same 50,000 training images and share
the validation pool; only the test images are split. State it.

**B10. Projection bias in PPO.** When the budget step changes or drops a
job, PPO still learns from the action it sampled, but the reward comes from
the action that ran. This is a known bias of action projection. It is small
if projections are rare: report how often they happen (`budget_dropped` is
logged).

**B11. The scheduler knows the simulator's time formula.** All policies get
device speed and bandwidth with only 15% noise, and the mode rule uses the
simulator's own time formula. A real server would know much less. This makes
feasibility look easier than in practice. State it (it is shared by every
method, so the comparison stays fair).

## C. Thesis and write-up gaps

- **C1. Chapter 3 no longer matches the method.** README says Chapter 3
  specifies the formulation (REINFORCE scheduler, ARL'). The method now
  under test is SAFE-PPO with predictor v2, a safety layer and, maybe, no
  ARL at all. Chapter 3 needs a new section, or the thesis must tell the
  story: the original method failed its frozen tests, the diagnosis, then
  RL2.
- **C2. The central claim may need reframing.** c34 uses no ARL term; its
  "contribution-aware" parts are only the successful-contribution fairness
  term (`lambda_fair`) and the validation protection reward. If the ARL
  screen finds no value, the title's claim rests on those, and the thesis
  should say exactly which part is contribution-aware.
- **C3. No thesis chapters yet** (memory.md, section 10), no RL2 figures
  (`scripts/plot_rl2.py` was planned in M10 and does not exist), and no
  `docs/RL2_RESULTS.md` yet.
- **C4. README is out of date:** it does not mention RL2, SAFE-PPO, the
  reserved data pools or the GPU workflow.
- **C5. Fairness-aware baselines from the literature are missing.** The
  published baselines are mostly efficiency-oriented (Oort, REFL, FAVOR,
  DELTA, FedCS, Power-of-Choice); the fairness-oriented ones are our own
  (`b_cov_ew`, the fairness-debt module).
  A published fairness-aware selection method (for example a long-term
  fairness scheduler with virtual queues) would make the fairness claim
  stronger, or the gap should be named in the related-work chapter.

## D. Small things

- **D1.** The `s2-frozen` tag exists only in this clone, not on GitHub. The
  validator falls back to the commit, which is in the pushed history, so
  nothing breaks; pushing the tag would make it visible to examiners.
- **D2.** The CPU-vs-GPU check is one seed and 30 rounds; it can show a gross
  difference, not a subtle one.
- **D3.** The predictor stress test is FEMNIST only; CIFAR-10 is not stressed.
- **D4.** Communication time is modelled, not measured: only the bytes
  were checked.

## Suggested order

1. Before the ARL and sensitivity screens: A1 (sensitivity reference
   `rl_light_mimic`), A4, B5 (add the FedScale trace setting).
2. Before the screen C re-run: A1 (matched aggregators), A2, A3, A6.
3. Before the freeze: A5, B4, B6, B8.
4. Thesis: C1-C5 and the stated limitations (B1-B3, B7, B9-B11).
