# Diagnostic findings: the FEMNIST collapse

Status: diagnosis done; **no algorithm has been changed**. Everything here
uses already-inspected seeds (110, 113) and is a failure reproduction or an
analysis-only intervention, not a test. The original s2 evaluation is
unchanged (`results/adv/PROVENANCE.md`, `scripts/validate_s2_original.sh`).

## 1. Observed symptoms

- Original s2 GPU runs (FEMNIST): ARL-v3 s113 4/2000 successes, accuracy
  0.103; c2 s110 5/2000, 0.066; c2 s113 8/2000, 0.141. Always-Light on the
  same seeds: 1,430+ successes, ~0.73.
- **Reproduced on CPU** with diagnostics on (`results/diagnostic_v1/full`):
  ARL-v3 s113 4/2000 (0.098), c2 s110 5/2000 (0.065), c2 s113 8/2000 (0.144).
  Not GPU noise. Escape is fragile, though: raw-fail RL s113 escaped on GPU
  (0.606) but not on CPU (0.173).
- In all 1,540 original runs, a fully fallback-filled start (all 10 slots
  from the gate's fallback in rounds 3-10) happens only to gated policies:
  FEMNIST learned-mode RL 12/20 seeds, `rl_ew` 16/20, `b_cov_ew` 18/20;
  never for any baseline (they do not use the gate). Only the policies that
  **learn their own modes** fail to recover.

## 2. Confirmed causal chain

Shown by the logs (`report.md` in each `results/diagnostic_v1` variant) and
by removing one link at a time (section 3):

1. **Cold start.** Rounds 1-2 invite 20 clients with an untrained policy;
   only 0-2 succeed.
2. **Predictor cliff at round 3.** `SuccessPredictor.predict` blends with a
   0.7 prior only while `seen < 20`. With 20 observations reached after two
   rounds, the median predicted success of every client drops from 0.54 to
   0.21-0.28 at once: 0 of ~94 candidates pass `elig_threshold` 0.5, so
   `strict_gate` fills all 10 slots with fallback clients.
3. **Inverted ranking.** The predictor is one logistic model on uncentred,
   non-negative features. Trained on almost only failures, every weight
   goes negative (log speed -0.43, energy -1.07, availability -0.99 at round
   30, s113), so *smaller* feature values mean *higher* predicted success.
   Synthetic check: after 30 failures it rates a slow, low-battery client at
   0.20 and a fast one at 0.10.
4. **Deterministic fallback picks the slowest devices.** `base.gate` fills
   with the highest-predicted non-eligible clients, i.e. the slowest: 85-97%
   of fallback invitations are tier-0 devices (observed speed 0.34-0.42 vs
   1.3-1.4 for Always-Light's picks). Most are **not** the "weak" clients
   (3-15%).
5. **Absorbing state.** All slots being fallback, the selection head gets
   no gradient; only the mode head learns. Learned modes give these slow
   devices Standard/Priority, which their battery cannot cover: 1,750-1,800
   of 2,000 invitations fail with reason `energy`. No successes, so the
   predictor keeps learning "everyone fails" and the ranking stays inverted.

Policies with a mode rule (`rl_ew`, `b_cov_ew`) enter the same state but give
fallback clients a safe mode, get successes, and recover.

## 3. Mechanism checks (analysis only)

`scripts/diagnose_mechanisms.py`: one link removed per run, in memory,
s2_femnist settings, 200 rounds (`results/diagnostic_v1/mechanisms/`).
These override the policy's actions, so they are not fixes.

| run | none | fallback in Light | random fallback | prior kept to 200 obs |
|---|---|---|---|---|
| ARL-v3 s110 | 388 / 0.731 | 999 / 0.696 | 1202 / 0.701 | 1217 / 0.778 |
| ARL-v3 s113 | **4 / 0.098** | 948 / 0.724 | 1217 / 0.684 | 1433 / 0.714 |
| c2 s110 | **5 / 0.065** | 933 / 0.661 | 1166 / 0.655 | 1367 / 0.689 |
| c2 s113 | **8 / 0.144** | 1091 / 0.702 | 1379 / 0.693 | 1317 / 0.749 |

(successes out of 2,000 / final accuracy.) Each single change removes the
lock-up. Random fallback recovers even though the learned modes stay heavy
(c2 s110: 1,968 Priority invitations), so the mode head is a contributing
link, not the root.

## 4. Hypotheses: status

| # | hypothesis | status |
|---|---|---|
| 1 | shared predictor depresses all predictions; threshold then repeats fallback | **confirmed** (steps 2-4) |
| 2 | eligible via one mode, run in a less feasible mode | **contributing**: fallback clients have no feasible-mode check; heavy modes on slow devices cause the `energy` failures |
| 3 | mode probabilities concentrate too early | **contributing, not necessary** (random fallback recovers with heavy modes) |
| 4 | reward scaling favours unproductive behaviour | **contributing**: in stuck rounds the advantage is positive in 51-94% of updates for c2/raw-fail; ARL-v3's credits make the total positive early on; not necessary (all interventions leave the reward unchanged) |
| 5 | energy/deadline constraints or an accounting bug | constraints **explain the outcomes** (energy failures); no accounting inconsistency found: sidecars reconcile with every CSV row |
| 6 | fallback removes the selection head's influence | **confirmed** (no pick gradient while all slots are fallback) |
| 7 | predictor features/labels differ from what the scheduler assumes | **confirmed**: uncentred features + failure-only labels invert the ranking |

## 5. What this changes in the thesis

- The FEMNIST losses of the learned-mode RL policies are mainly an
  **engineering failure of the eligibility gate and success predictor**
  (cold-start cliff, inverted ranking, deterministic fallback), amplified by
  learned modes. They are not evidence about ARL's idea itself.
- **ARL-v3's protection benefit remains unsupported**; nothing here tests it.
- The s2 results stand as the original evaluation; this explains them, it
  does not overturn them.

## 6. Candidate interventions (not implemented; one must be chosen)

Any one of these, behind its own flag, tested on development seeds 200-204
then fresh confirmation seeds (300+): (a) a longer or smoother cold-start
prior; (b) centred/standardised predictor features, or a per-mode intercept
only until enough successes; (c) a random or availability-weighted fallback;
(d) a feasible-mode mask for fallback clients. (b) targets the root cause
(inverted ranking); (a) and (c) remove the trigger and the absorbing choice.

## 7. Platform sensitivity (department RTX 3070, torch 2.5.1+cu121)

The same 8 diagnostic runs on the department GPU (`bash scripts/dept_gpu.sh
diag full`, 139 s): **none stayed stuck**. Every learned-mode run still hit
an all-fallback round 3, but escaped within a few rounds (longest zero-success
streak 0-1; 1,174-1,408 successes; accuracy 0.65-0.74). Kaggle's GPU and this
CPU kept ARL-v3 s113 and c2 s110/s113 locked for ~200 rounds.

So the lock-up is a **tipping point**, not a fixed outcome: the gate/predictor
chain puts nearly every learned-mode run into the same fragile state at
round 3, and tiny numerical differences between devices decide whether it
escapes. The cause in sections 2-3 stands; how often it bites depends on
the platform. Runs, checks and sidecars: `results/diagnostic_v1_rtx3070/`.

## 8. Unexecuted checks

- Other collapsing seeds (103, 112, 116) and CIFAR-10.
- Sidecars of the 200-round runs are not committed (28 MB); CPU runs are
  bit-reproducible: `python scripts/diagnose_collapse.py --variant full --jobs 4` (~9 min).

## How to run

    python scripts/diagnose_collapse.py --dry-run
    python scripts/diagnose_collapse.py --variant short --jobs 4    # 93 s on 4 CPU cores
    python scripts/diagnose_collapse.py --variant full --jobs 4     # 541 s
    python scripts/report_diagnostics.py results/diagnostic_v1/full --every 20
    python scripts/diagnose_mechanisms.py --rounds 200 --jobs 4     # analysis only

Kaggle cell (after the usual clone/data cells of `notebooks/run_kaggle.ipynb`):

    !python scripts/diagnose_collapse.py --variant full --jobs 4 --set fl.device=auto --out /kaggle/working/diag
    !python scripts/report_diagnostics.py /kaggle/working/diag/full > /kaggle/working/diag/report.md
