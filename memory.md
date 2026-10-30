# Project memory

Living notes for this thesis project: where things stand, what we know, and
how we work. Read this first in every session. Update it whenever progress is
made, a decision is taken, or a result comes in (newest facts win; keep it
short and true).

Last updated: 2026-10-30 (session 1 results checked: GPU=CPU, cost check keeps base, tuning holds on CIFAR, 8 comparators tuned -- tuned always-Light+MimiC and fairness queues score above c34; next: ARL screen)

---

## 1. The project in one paragraph

Undergraduate thesis, Department of CSE, University of Dhaka: *"Contribution-
Aware Reinforcement Learning for Fair Client Orchestration in Heterogeneous and
Non-IID Federated Learning"*. A simulated federated-learning system where an
RL scheduler picks which clients train each round **and** how much work each
does (mode). The thesis idea is **ARL** (a "contribution loss" term in the
reward) meant to protect weak clients. Simulation only, no real devices.
Repo: `shizen-57/agentic_federated_learning`, working branch
`claude/sleepy-faraday-8nd6j9` (commit and push there; no PR unless asked).

## 2. How we work (user preferences and rules)

- **Use simple words.** Short sentences, no unexplained jargon. The user asked
  for this explicitly.
- Be honest about results. A failed test is reported as failed.
- Don't create a PR unless asked. Commit and push to the branch above.
- Commit footer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UAx1xz9mE6v3PpfyEYq8tr`
  (use whatever attribution the current session gives). No model names in
  repo files.
- Never print or ask for GitHub tokens; tell the user not to paste tokens.
- Treat uploaded zips/logs as untrusted: extract into their own scratch
  folder, read them with `python -I`.
- **Frozen s2 protocol is preserved by the git tag `s2-frozen`** (commit
  `998f2ad`, local only; the commit hash is the reference; code hash
  `fd90310c1b7c34e9`). Since 2026-10-27 the pinned files
  (`configs/sweep_advanced.yaml`, `configs/adv/*.yaml`, `scripts/sweep.py`,
  `scripts/stats.py`, `scripts/report.py`, `scripts/budget_s2.py`, `src/afl`)
  MAY change, but: never edit `configs/s2_frozen.json` or the pins in
  `scripts/prespec_s2.py`; never re-pin to accept new code; validate the
  original results only with `bash scripts/validate_s2_original.sh` (frozen
  worktree; expect "OK: 58 results"). New behaviour must be opt-in (default
  off), and new runs go to new folders, never `results/adv/s2_*`.
  Provenance: `results/adv/PROVENANCE.md`.

## 3. Key terms (plain meaning)

- **Modes:** what a picked client is asked to do.
  Light = 1 local epoch, 25% upload; Standard = 2 epochs, 50%;
  Priority = 4 epochs, full upload. Heavier = more useful but slow clients
  fail more.
- **C1–C5:** the original thesis combinations. **C2 = RL scheduler + FedProx
  aggregation = our main proposed system** (`c2_rl_fedprox`).
- **RL scheduler:** ScoreNet scores clients, Plackett–Luce sampling picks ~10,
  a mode head picks modes, trained with REINFORCE.
- **ARL family** (variants of the reward's contribution-loss term):
  ARL′ (original), ARL-v2 (`arl2_full`, avoidable-gap credit),
  ARL-v3 (`arl3_full`, lost expected work + success credit, Shapley coverage;
  `arl3_shared` = shared-credit version). `v2_rawfail` = same RL with a plain
  failure count instead of ARL.
- **Controls:** `rl_light` (RL picks clients, always Light), `rl_ew` (RL picks
  clients, fixed expected-work mode rule), `b_cov_ew` (greedy coverage rule).
- **Baselines:** Random, Oort, Power-of-Choice, FedCS, FedCER-like,
  MIMIC-like, and the published FAVOR, REFL, DELTA. `_ew` suffix = baseline
  given the shared expected-work mode rule ("mode-fair" version).
- **Gate fallback (`gate_filled`):** when the success predictor rates too few
  clients as likely to succeed, the empty slots are filled by a fallback.
- **Lock-up:** a run where no client succeeds for 50+ rounds in a row.
- **Equal spend:** compare accuracy at the same total cost.

## 4. Where things are

- Code: `src/afl/` (engine.py, sim.py, policies/rl.py, policies/baselines.py,
  policies/published.py, predictor.py, ...).
- Tests: `python -m pytest tests -q` (core, advanced, review-fix tests).
- Experiment plan: `configs/sweep_advanced.yaml`; runner `scripts/sweep.py`.
- Frozen protocol: `configs/s2_frozen.json`, pins in `scripts/prespec_s2.py`
  (`FROZEN_CODE_HASH = fd90310c1b7c34e9`, manifest SHA
  `8d04e337...7640ffe`); `scripts/freeze_s2.py` regenerates pins.
- Pre-specified tests: `python scripts/prespec_s2.py` (needs the
  `/kaggle/working/adv -> results/adv` symlink trick and `--root
  /kaggle/working/adv`, because saved configs record the Kaggle output path).
  Output saved in `results/adv/s2_prespecified_tests.md`.
- Equal-spend analysis: `scripts/budget_s2.py` → `results/adv/<group>/budget.md`.
- Figures: `python scripts/plot_s2.py` → `figures/s2/fig1..fig8*.png`.
- Main write-up: `docs/ADVANCED_RESULTS.md` (headline list at the top;
  sections on s2 runs and their outcome).
- Kaggle notebooks: `notebooks/run_kaggle.ipynb` (smoke),
  `notebooks/run_kaggle_step2.ipynb` (main s2 runs).
- Results: `results/adv/s2_cifar`, `s2_femnist` (460 runs each, seeds
  100–119), `s2_churn_cifar`, `s2_churn_femnist` (80 each),
  `s2_main_cifar`, `s2_main_femnist` (230 each, seeds 100–109, no weak
  clients). Each run: `<cond>_s<seed>.csv` + `.config.json`.

## 5. History (short)

1. Built simulator, RL policy, baselines; MNIST/CIFAR-10/FEMNIST.
2. ARL′ failed its pre-specified tests (no effect on worst-class accuracy);
   ARL made more failures than a plain failure count.
3. Fair re-test of ARL failed. RL + MimiC looked good but its confirmation
   narrowly failed.
4. ARL-v2 redesign failed its pre-specified test (but was more economical).
5. Found simulator flaws (Light upload not really compressed; Oort saw true
   round times). Fixed. Built ARL-v3, mode-fair baselines, FAVOR/REFL/DELTA.
6. Several external reviews → fixed test-set leakage in the reward, init/RNG
   streams, absence counter, joint coverage loss, credit assignment, gate,
   charging, check-in pool, etc. Protocol re-frozen.
7. Ran 1,540 `s2_*` runs on Kaggle (4 parallel jobs, 10.4 h wall, ~41
   run-hours). All validated against the frozen protocol and analysed
   (commits `c8f8b0f`, `4286097`). Figures added (`ac6b5f3`).

## 6. Results of the s2 runs (the current "original evaluation")

- **ARL-v3: 0 of 8 pre-specified tests pass** (worst-10% accuracy and
  worst-10% client accuracy vs `v2_rawfail`, 4 settings). Protection benefit
  **not supported**.
- **ARL-v2 fails more than always-Light** (CIFAR 42% vs 32%, FEMNIST 55% vs
  31%).
- **C2 RL vs baselines:**
  - CIFAR-10: accuracy similar to most (0.521); beats DELTA and DELTA_ew;
    loses to REFL_ew. Fails significantly less than 8 of 9 baselines (44%).
  - FEMNIST: clearly worse (0.580 vs ~0.75–0.79); significantly worse than
    six mode-fair baselines.
- **Learned modes vs a rule (family D):** accuracy inconclusive.
- **Lock-up** (FEMNIST, 50+ rounds with zero successes): C2 4/20,
  ARL-v3 shared 3/20, ARL-v3 2/20, ARL-v2 1/20, v2_rawfail 1/20; never for
  `rl_light`, `rl_ew` or any baseline. ARL cannot see it (loss → ~0).
- **Mode rule helps every baseline:** +0.02 to +0.08 accuracy (e.g. Oort
  0.716 → 0.785 on FEMNIST). Choosing the mode matters more than choosing the
  client.
- **Always-Light (`rl_light`)**: cheapest (~7 cost/round vs C2 12.1 on CIFAR),
  fewest failures (~31–32%), fewest clients lost to churn (20–22 vs 26–31),
  near the top at equal spend (CIFAR 0.501, FEMNIST 0.717 vs C2 0.439/0.556).
  Not the most accurate overall (Oort_ew/REFL_ew higher, but ~1.8× cost).
  Always-Light vs C2 was **not** a pre-specified test → descriptive only.
- **Defensible thesis conclusion (agreed):** on CIFAR the RL scheduler failed
  less often than most baselines (an RL-family effect, not specific to
  ARL-v3; always-Light failed even less); ARL-v3's protection benefit is
  unsupported; FEMNIST exposed a reliability problem (lock-up).

## 7. Seed-113 diagnosis so far (FEMNIST)

- ARL-v3 (`arl3_full_s113`): 2,000 invitations, **4 successes**, final
  accuracy 0.103. Always-Light same seed: 1,430 successes, 0.736.
- Stuck on **Standard** mode (~99% of picks), **not Priority**. (An earlier
  explanation that stuck runs drift to Priority was wrong for this seed.)
- From round 3 on, **all 10 slots are gate fill-ins** (`gate_filled` = 10):
  the predictor rates everyone as likely to fail, so the RL's scores stop
  choosing clients.
- **Reward is not punishing the stuck state:** often positive (~+0.5 around
  rounds 6–13) while nothing succeeds. Main suspect.
- ARL-v3 loss shrinks to ~0.02: ARL reports "nothing lost" while nothing is
  learned.
- Same seed, other methods: C2 8 successes (0.141), arl3_shared 68 (0.394,
  mostly Priority), v2_rawfail 147 (0.606).

## 8. Agreed plan (current)

Do **not** run another full sweep or redesign ARL yet.

1. **Freeze** this experiment: keep code, configs, all seeds incl. failures.
   Add a git tag (e.g. `s2-frozen`) on the frozen code. Treat s2 results as
   the original evaluation.
2. **Reproduce** FEMNIST seed 113 (ARL-v3) locally on CPU; check it matches
   the Kaggle result (GPU runs are not bit-identical, so expect close, not
   exact).
3. **Add diagnostic logging (opt-in switch)** before changing any algorithm:
   selected client IDs, predicted success probabilities, actual failure
   reasons, battery, mode probabilities, gate fallback, reward components.
   Check that logging on gives the same run as logging off. Find **why** the
   policy kept choosing Standard despite repeated failures.
4. **Fix the demonstrated cause only**, test cheaply (short runs, development
   seeds). Separate predictor vs gate fallback vs mode learning vs their
   interaction; change one mechanism at a time.
5. **Freeze the revision**, write down fresh seeds and the test before
   running, then evaluate vs always-Light and strong baselines under equal
   spend. Do not reuse inspected seeds (100–119) alone.

Immediate coding task: tag + opt-in diagnostic logging + seed-113
reproduction. Steps 2–4 can run on CPU here; the dept GPU is used first for
step 2 (`dept_gpu.sh repro113`, frozen code) — status: waiting for the user
to run it and send back the results.

## 9. Compute notes

- Kaggle s2 run times: CIFAR ~2.0 min/run, FEMNIST ~1.1 min/run (4 parallel).
- Dept GPU: 8 GB VRAM is plenty (models are small, <1 GB per run). Speed is
  also limited by CPU cores; with 2 cores, double the times.
- Estimates (4 parallel): fixed C2 + ARL-v3 on CIFAR+FEMNIST, 20 seeds = 80
  runs ≈ 40 min; core 16-method comparison = 640 runs ≈ 4.3 h; full s2 redo ≈
  10–11 h. Runner skips finished runs, so work can be split across days.
- Free GPU options discussed: Kaggle (used), Colab, and others.
- **Dept GPU server** (access since 2026-10-27, via SSH; login details are
  kept by the user, never stored in the repo or this file). Server rules:
  install only into our own conda env (`afl`), never the base env; never
  change NVIDIA drivers (ask mosaddek@du.ac.bd if needed); files are deleted
  periodically, so copy results home after every session.
- Helper: `scripts/dept_gpu.sh` with steps `setup` (conda env, PyTorch for
  the installed driver, FEMNIST + CIFAR data), `check` (GPU, cores, timing),
  `repro113` (FEMNIST seed 113, all 23 methods, frozen code), `lockseeds`
  (seeds 103 110 112 116), `pack` (tar.gz of `runs/`). Run inside tmux.
  On this CPU container a FEMNIST run takes ~7 min alone.

## 10. Open questions / next decisions

- What exactly keeps the policy on Standard during lock-up (step 3 above).
- Whether the fix touches only RL code (then reuse old baseline runs) or
  shared code (then re-run the core comparison).
- Thesis chapters not written yet; charts 1, 5, 6 suggested for the main text.

## 11. Collapse diagnosis (2026-10-27) — see docs/DIAGNOSTIC_FINDINGS.md

- Done: resume-validation fix, window-baseline fix, float64 test, opt-in
  diagnostics (`diag.enabled`, proven not to change runs), runner
  `scripts/diagnose_collapse.py`, report `scripts/report_diagnostics.py`,
  analysis-only `scripts/diagnose_mechanisms.py`. Tests: 194 pass, 8 skipped
  (frozen-validator tests; they run in the frozen checkout).
- Collapse reproduced on CPU (ARL-v3 s113 4 successes, c2 s110 5, s113 8).
- Cause: cold-start prior switches off at 20 observations -> every client
  below 0.5 at round 3 -> all 10 slots gate fallback; predictor (uncentred
  features, failure-only labels) ranks slow/low-battery clients highest;
  deterministic fallback picks tier-0 devices; learned modes give them
  Standard/Priority -> energy failures -> no successes -> stuck.
- Removing any one link (fallback in Light, random fallback, longer prior)
  rescues all stuck runs at 200 rounds (acc 0.66-0.78 vs 0.07-0.14).
- User chose (b): `policy.predictor_norm=round` (standardise predictor
  client features per round), condition `arl3_full_pn`. Default off is
  row-identical to before. Pre-registered dev screen
  (docs/DEV_SCREEN_PREDICTOR_V1.md, FEMNIST seeds 200-204, 200 rounds):
  **FAILED** rule 3 -- median acc 0.701 vs unchanged ARL-v3 0.744; no
  starvation in any method on these seeds (so it could not show the fix);
  revision drives heavy modes and spends far more (equal-spend acc 0.676 vs
  0.744). Always-Light 0.781 / raw-fail 0.756 strongest. Per the rule: no
  tuning on these seeds, no confirmation study.
- PR: https://github.com/shizen-57/agentic_federated_learning/pull/1
- Dept GPU: user cloned the repo into a folder of a shared account (others
  use it; the path is not written here -- the repo is public). dept_gpu.sh refuses to run outside conda env
  `afl`. Planned there: `diag full` + dev screen on GPU as a cross-check.
- User chose (d): `policy.mode_mask=feasible` (draw only modes with
  p >= 0.5, log-prob of the masked draw), condition `arl3_full_fm`, on
  unchanged ARL-v3. Screen (docs/DEV_SCREEN_MODEMASK_V1.md, fresh seeds
  205-209): **FAILED** rule 2 (successes 943 vs 1,184) but acc +0.038
  (0.738 vs 0.700), spend much steadier; equal-spend level. It escapes both
  inspected lock-ups on CPU (s110 0.733, s113 0.759).
- Dept GPU (RTX 3070, 8 GB, 8 cores, ~0.6 min/FEMNIST run, 8 runs in 139 s):
  the 8 diagnostic runs did NOT stay stuck -> the lock-up is a tipping point
  sensitive to device numerics (Kaggle GPU and CPU lock, 3070 escapes).
- Seeds used now: 0-119, 200-209, 900. Next fresh dev seeds: 210+;
  confirmation: 300+.
- NEXT (user decides): thesis chapters (recommended), or a combined
  (b)+(d) screen on seeds 210-214 on the dept GPU (fast).

## 12. RL2 plan (2026-10-27): SAFE-PPO + ARL-v4

User wants every weakness fixed, no settling for simple. Decisions: pre-training
allowed (strict separation; report pre-trained and from-scratch); two primary
endpoints (P1 equal-spend accuracy, P2 worst-10% client accuracy; Holm);
CIFAR-10 + FEMNIST. Design: docs/RL2_DESIGN.md (components C1-C9 + ARL-v4).
Seeds: configs/seed_registry.yaml, guard scripts/seeds.py (dev 210-299,
practice 1000-1999, confirmation 300-399 needs frozen manifest).
Milestones: M0 spec+seeds (done) -> M1 predictor v2 + gate v2 -> M2 PPO ->
M3 set encoder + memory -> M4 safety -> M5/M5b reward v2 + ARL-v4 ->
M6 pre-training -> M7 tuning -> M8 ablation -> M9 freeze -> M10 confirmation
(GPU, after user go-ahead). Each milestone: tests, pre-registered dev screen.

### RL2 progress (2026-10-27)
- Built and pushed: M0 (spec + seeds), M1 (predictor v2 + Thompson gate,
  `arl3_full_pg`), M2 (SafePPOPolicy, `ppo_v3`), M3 (attention set encoder +
  pointer picks + memory, `ppo_att`), M4 (Lagrangian cost constraint, trust
  curriculum, mask margin, `ppo_safe`), M5/M5b (ARL-v4 + validation-only
  protection reward, `ppo_full`; `ppo_full_mimic` = + MimiC, user asked),
  M6 (practice mode, scripts/pretrain.py, checkpoints with provenance).
  Every switch defaults off; defaults row-identical to earlier runs.
  Suite 316 tests pass on torch 2.14 and 2.5.1 (8 frozen-validator tests skip; they run in the
  frozen checkout).
- LESSON: never edit src/afl while a screen runs here -- runs get flagged
  "source changed while it ran" (happened once to screen A; restarted).
  Run screens from a frozen git worktree
  (scratchpad/wt_screen, data symlinked) with --out into the main repo.
- Screen A (M1, seeds 210-214 + lock-check 110/113, CPU, per-seed split):
  **PASSED** (docs/RL2_M1_SCREEN.md). Median final / equal-spend / worst-10%:
  arl3_full_pg 0.768 / 0.769 / 0.529, 0 starved; arl3_full 0.748 / 0.738 /
  0.483, 1 starved; rl_light 0.754 / 0.754 / 0.460; v2_rawfail 0.744 / 0.744 /
  0.511. Lock-check s110 0.754, s113 0.766 (both escape).
- Second external review (of e105c56), all 7 findings fixed and tested
  (docs/RL2_DESIGN.md "Revision 2"): (1) persistent data split
  (`data.split: persistent`, fixed val/test pools, FEMNIST practice vs
  evaluation writers, fingerprint; checkpoint refused on another split);
  (2) pretrain command runs; (3) one shared start network; (4) mask x
  curriculum rule (feasible AND unlocked, else Light only if optimistic, else
  abstain); (5) ARL-v4 joint reward = set-level coverage f(D), not summed
  difference credits; (6) empty rounds kept in the PPO trajectory; (7)
  report_dev_screen.py --spec validates every expected run, else
  "INCOMPLETE -- no decision" (exit 2). Also: counterfactual on the joint
  budget, hard per-round cap (`hard_budget_mult` 2.0), predictor evidence
  ageing (`pred_decay` 0.98), pre-training cost recorded.
- Third external review (of 4c69be3), all 4 findings confirmed and fixed
  (docs/RL2_DESIGN.md "Revision 3"): (1) budget projection only moves a
  client to a mode SAFE-PPO allows, else drops it; curriculum calibrates on
  the executed job; PPO keeps the sampled action; (2) report/runner accept
  only `fl.device` overrides, spec/source/plan hashes checked; (3) starvation
  = 50+ rounds with no successful contribution, invited or not (new log
  column `n_online`; offline rounds skipped) -- screen A unchanged by this;
  (4) ARL-v4 gap = 0 when no feasible affordable alternative exists.
- Screen B attempt 1 on the dept GPU (commit ec4866f) was VOID: 65/100 runs
  (all masked conditions) crashed -- server torch is 2.5.1 (newest cu121
  build), which cannot make a tensor from numpy bools; local torch 2.14 can.
  Results not inspected. Fixed (plain bools); regression test; GPU script
  now runs scripts/smoke_screen.py (2 rounds per condition, seed 900) first.
  LESSON: test against the server's torch too -- a torch 2.5.1 venv lives in
  the scratchpad (venv251); re-create with `pip install --index-url
  https://download.pytorch.org/whl/cpu torch==2.5.1` if gone.
- Screen B attempt 2 (commit 61eff94): 100/100 runs finished, 0 failed,
  25 min on the RTX 3070. Report said INCOMPLETE ("different config" for all):
  saved "out" relative vs rebuilt absolute. Fixed in scripts/ only
  (report ignores "out"); simulator hash unchanged, so the runs stand and
  only the report is re-run. LESSON: unit tests used absolute paths on both
  sides; tests/test_screen_e2e.py now runs runner + report for real.
- **Screen B OUTCOME** (report re-run with the fix; evidence complete,
  100/100): ppo_full PASS, ppo_full_mimic PASS -> **ppo_full_mimic carried
  forward**. Paired vs ppo_full (eq-spend / final / worst10): rl_light
  +0.016/-0.004/+0.028 (always-Light still slightly ahead of ppo_full);
  ppo_full_mimic +0.013/+0.021/+0.034 (closes most of the gap). Removing
  predictor v2 + Thompson gate: -0.587/-0.459, 3/5 starve (essential).
  8 of 12 components are drop candidates (attention, evidence ageing, cost
  constraint, hard cap, curriculum, protection, ARL-v4->v3, ARL-v4->rawfail):
  ARL-v4 shows no benefit yet. Decide in M7 on seeds 220-239.
  Runs saved in results/rl2_screen_b/full (re-validated here: identical
  decision). Exploratory (results/rl2_screen_b/EXPLORATORY.md): mimic vs
  rl_light: equal spend +0.002 (tie), last-10 mean acc +0.029, worst-10%
  +0.052, 4/5 seeds each, p 0.125/0.188; RL spends ~30% more. Seed 219
  last-round-only dip (0.79 -> 0.49) -> from M7 on, "final" = mean of last
  10 rounds (decided after screen B, before M7; say so). rl_light_pg is
  worse than rl_light (0.720 vs 0.767).
  Small results pack for upload: tar MANIFEST/plan/report/sweep.log + CSVs
  + config.json (5 MB) instead of the 110 MB full pack.
- **M7 OUTCOME** (docs/RL2_M7_TUNING.md; results/rl2_tune; re-checked here,
  same winner): **c34** selected (stage 2 score 0.6378 vs base c00 0.6311,
  rl_light 0.5866; top 5 within 0.002 = near coin toss). c34: attention on,
  curriculum OFF, cost target 1.0 (beta 0), cap 3.0, protection 1.0, ageing
  0.98, mask z 0.25, gate z 1.5, PPO lr 0.005/4 epochs/batch 8/clip 0.3/
  ent 0.003/gamma 0.9, **damage = raw_fail (NO ARL)**. ARL-v4 option had the
  lowest stage-1 mean. Exploratory on stage-2 seeds: c34 vs rl_light eq5
  +0.036 (13/15), w10 +0.067 (14/15), acc10 +0.056 (15/15); screen B seeds
  were lucky for rl_light (gap there +0.005/+0.032). configs/rl2_tuned.json =
  SELECTED.json. Screen C: tuned == tuned_noarl (alias) -> 2 pre-trained
  methods (tuned, tuned_arl4) x 2 datasets = 4 checkpoints, 240 runs.
  Re-check fix: tune --report-only compares the recorded ranking (Python 3.11
  vs 3.13 float sums differ in the 16th digit) instead of rewriting it.
- Screen C pre-training attempt 1: CUDA out of memory at FEMNIST iteration
  11/20 with 8 episodes at once (~1 GB each on the 8 GB RTX 3070, mostly
  per-process overhead + eval batches, not data). Nothing saved. Fixed in
  scripts only (simulator hash unchanged a442febc89a1a7d3): procs femnist 4 /
  cifar 3 (same result), pretrain.py resumable (.partial after each
  iteration; same weights), MAXPROCS=2 cap if needed,
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, screen-c / confirm
  default JOBS=3 (CIFAR ~1.8 GB per run). LESSON: budget ~1 GB GPU per
  FEMNIST process and ~1.8 GB per CIFAR process on the 8 GB card.
- **M7-M10 built** (all rules written before any run):
  - shared library scripts/rl2lib.py (plan groups with an `override` applied
    after the condition -- new in sweep.build_jobs; manifests; evidence
    check; 2-round smoke on seed 900; score; sign-flip/Holm; seed count);
  - M7 scripts/tune_rl2.py + configs/rl2_tune.yaml + docs/RL2_M7_TUNING.md:
    40 candidates of ppo_full_mimic (8 drop candidates as switches + PPO and
    reward settings), seeds 220-224, top 5 (+c00) on 225-239, winner ->
    runs/rl2_tune/SELECTED.json -> commit as configs/rl2_tuned.json;
  - M8 scripts/screen_c_rl2.py + configs/rl2_screen_c.yaml +
    docs/RL2_SCREEN_C.md: methods tuned / tuned_arl4 / tuned_noarl (aliases
    by effective config), pretrain per dataset (pretrain.py --overrides),
    pt vs scratch, refs incl. rl_light_mimic (new condition), seeds 240-249,
    both datasets -> DECISION.json;
  - M9 scripts/freeze_rl2.py (power: delta 0.01 P1 / 0.02 P2, 8 tests, n in
    [20,100], seeds 300+; pins into scripts/prespec_rl2.py; refuses dirty /
    other code / second freeze); M10 scripts/confirm_rl2.py (needs --go =
    user's go-ahead; checks pins, code, inputs, checkpoints, seeds) and
    scripts/prespec_rl2.py (8 one-sided sign-flip tests, Holm) --
    docs/RL2_CONFIRMATION.md;
  - endpoints from M7 on: eq5 (mean fixed-set acc, last 5 rounds within cost
    1,400) and w10 (mean worst-10%, last 10 rounds); score = mean of both;
  - GPU steps: tune, screen-c-pretrain, screen-c, confirm [go], packsmall <name>.
  - Tested end to end here at tiny size (tune, pretrain, screen C, freeze,
    confirm checks); tests/test_rl2_pipeline.py.
- Screen B **version 2** (docs/RL2_SCREEN_B.md, configs/rl2_screen_b.yaml,
  seeds 215-219, persistent split, 100 runs): ppo_full, each component
  removed one at a time (12 leave-one-out), ppo_full_mimic, ppo_v3, and
  references incl. rl_light_pg (repaired simple scheduler) and
  ppo_full_rawfail / ppo_full_arl3 (the stack without ARL-v4). Rule is in the
  spec. Version 1 was never run.
- Seed plan: 210-214 screen A (used), 215-219 screen B, 220-239 tuning (M7,
  and the comparators' matched tuning), 240-249 screen C (M8), 250-269 ARL
  screen, 270-279 sensitivity, 280-284 cost check, 285-289 CIFAR check,
  290-292 predictor stress, 900-902 device checks, confirmation 300+ (user
  go-ahead).
- **Fourth external review (2026-10-28)**: user agreed "make the simulator
  trustworthy and find which parts of the method help; postpone framework
  migration (Flower etc.)". Milestone = verified repairs + genuinely reserved
  confirmation data + controlled ARL-vs-plain comparison; no architecture
  change unless a check shows a specific need. Done:
  - (1)+(7) tests/test_validation_matrix.py: integrated select -> budget
    change/drop -> learn case with spies (oracle calls excluded); nobody
    online; everyone abstains then recovers; all updates fail / rejected;
    cap too tight; sudden bandwidth+battery drop and recovery; cost/time/
    energy equations; sensitivity scales. Map of every matrix row to its
    test: docs/VALIDATION.md. LESSON: the ARL* oracle calls env.outcome for
    every client every round -- any spy on outcome must skip calls made
    inside env.oracle_eligible (the recovery test was passing vacuously).
  - (2) reserved pools: `data.eval_pool: dev|confirm` (default dev).
    FEMNIST writers practice 1,798 / dev 899 / confirm 900; CIFAR-10 test
    examples validation 3,006 / dev 3,488 / confirm 3,506; pinned in
    configs/rl2_data_roles.json (scripts/data_roles.py); dev runners refuse
    confirm (rl2lib.check_pool); confirmation pre-trains afresh in M10 under
    the frozen code. HISTORY (disclose in thesis): old-code screens drew from
    the whole eval pool -- 715 of 799 usable FEMNIST confirm writers were
    used (screen B 234, M7 552, screen C 385); old screen C read all CIFAR
    confirm examples. So M7 = development evidence; **screen C must be
    re-run on the new code** (same spec and seeds); the freeze refuses a
    DECISION.json without `eval_pool: dev` + matching `data_roles_sha256`.
  - (3) ARL screen: configs/rl2_arl_screen.yaml + docs/RL2_ARL_SCREEN.md,
    tuned c34 with only the damage term changed (plain / arl3 / arl4), both
    datasets, seeds 250-269, 120 runs, two-sided sign-flip + Holm over 8
    tests. Matched tuning budget from M7 (expected best of 9): v3 0.6404,
    v4 0.6441, plain 0.6486. Runner scripts/screen_rl2.py (generic).
  - (4) docs/SIMULATOR_MODEL.md (equations, units, sources, not calibrated);
    timing check (results/simulator_checks/timing.md): real time ~ n x E,
    R^2 0.997 FEMNIST / 0.999 CIFAR; sensitivity screen
    configs/rl2_sensitivity.yaml + docs/RL2_SENSITIVITY.md (FEMNIST, seeds
    270-279, deadline/availability/battery/bandwidth low+high, 180 runs);
    new knobs system.energy_scale / bandwidth_scale.
  - (5) scripts/predictor_stress.py (seeds 290-292, results/predictor_stress,
    docs/VALIDATION.md section 5): calibrated overall (0.615 predicted vs
    0.597 observed, ECE 0.038); too optimistic for Priority (0.72 vs 0.64),
    weak clients (0.48 vs 0.44), cold start (0.53 vs 0.47); after a shock
    0.38 vs 0.04 for the few jobs that still ran, abstentions 16 -> 54.
    Recovery: 91% of hit clients re-invited, median 9 rounds, then success
    0.655. Avoided for good (ruled out >= 90% of last 60 online rounds): 35
    stable / 47 shock of 300 clients; ground truth: they could succeed in a
    median 21% of rounds, only 1 in >= half -> mostly right (threshold 0.5).
    DECISION: no method change; a budgeted Light probe is the candidate if
    more inclusion is wanted (new version, needs its own dev screen).
  - (6) scripts/cost_curves.py: accuracy at budgets 500-2000 with actual
    spend; spends within ~2% at each checkpoint; rl_light ahead at 500,
    tuned ahead from ~800; rl_light only spends ~1,371 in total.
  - New log columns n_abstain, n_light_override (simulator hash changed).
- **Gap review (2026-10-29, no tests run): docs/GAP_REVIEW.md.** Most
  important: (A1) the tuned method uses MimiC but every primary comparator
  (and the sensitivity screen's rl_light) uses FedProx -> scheduler effect
  confounded with aggregation; use rl_light_mimic / matched aggregators;
  (A2) P2 `w10` is measured at unequal spend (tuned ~1,818 vs rl_light
  ~1,371); (A3) the soft cost constraint overshoots its 7.5/round target
  (tuned 9.1, rl_light 6.9); (A4) no defined route for an ARL win into
  screen C's final method; (A5) prespec_rl2 does not yet report spend /
  cost curves / starvation / pre-training cost as RL2_CONFIRMATION promises;
  (A6) P2 lacks a fairness-oriented baseline. Also: baselines untuned (B1),
  M7 FEMNIST-only (B2), pre-training sim-to-sim (B3), only the weak=rare
  scenario (B4), FedScale trace unused in RL2 (B5), fairness = one number
  (B6), w10 noisy (~13 test examples per FEMNIST client, B7); thesis:
  Chapter 3 out of date, no chapters/figures yet, README stale (C1-C5).
  User said "fix all of them first" -> all fixed (2026-10-29), status table at
  the end of docs/GAP_REVIEW.md. Also found and fixed A7: the `_ew`
  baselines used the ORIGINAL predictor v1; their tuning may now pick v2.
  What changed:
  - comparators: `*_mimic` versions (conditions.py); new baseline
    `b_fairq_ew` (policy `fairq`, Lyapunov fairness queues after RBCS-F,
    Huang et al. TPDS 2021); Oort eps/alpha and FAVOR lr are config knobs
    (defaults unchanged);
  - matched tuning of 8 comparators: configs/rl2_tune_<name>.yaml, same
    budget/seeds/score as M7 (seeds 220-239 reused), small spaces searched
    exhaustively (tune_rl2.sample_configs); winners -> commit as
    configs/rl2_tuned_<name>.json (I do it after checking the pack);
  - endpoints: P2 = `w10b` (worst-10% over the last 10 rounds WITHIN the
    1,400 budget); rl2lib.score_eq = (eq5 + w10b)/2 for screen C and the
    screens; M7's rl2lib.score unchanged (its re-check still matches);
  - screen C v2: tuned refs (`tuned_references`, `control:
    rl_light_mimic_tuned`), P1/P2 second comparator = best tuned comparator
    on eq5 / w10b; tuned file = configs/rl2_tuned_final.json;
  - chain (A3/A4): cost check (seeds 280-284) -> `python
    scripts/finalize_tuned.py cost --spec configs/rl2_cost_check.yaml --out
    <folder>` -> commit configs/rl2_tuned_cost.json -> ARL screen ->
    `finalize_tuned.py arl ...` -> commit configs/rl2_tuned_final.json ->
    sensitivity + screen C. Runners stop if a file in the chain is missing;
  - freeze: secondary worlds (fedscale_devices, weak_random, no_weak,
    churn; final vs control; first 20 confirmation seeds), power with the
    80% upper confidence limit of screen C's sd (sd 0.02 from 10 seeds ->
    75 seeds); prespec reports spend, cost curves, starvation, n_never,
    n_gap50, Jain, weak-client acc, budget_lighter/dropped, pre-training
    cost, worlds;
  - new log columns: cost_ref, budget_lighter, n_never, n_gap50 (code hash
    changed -> every new run is on the new code);
  - CIFAR check of M7's top 5 (seeds 285-289); sensitivity v2 (13 settings,
    260 runs, reference = tuned rl_light_mimic); device check 3 seeds x 60
    rounds; predictor stress --dataset cifar; scripts/plot_rl2.py;
    docs: RL2_METHOD.md (thesis method draft, which parts are
    contribution-aware), RL2_BASELINE_TUNING.md, RL2_COST_AND_CIFAR_CHECKS.md,
    README RL2 section.
  - Verified 2026-10-29: 366 tests pass on torch 2.14 and 2.5.1 (8 skip);
    validate_s2_original.sh OK (58/58); new CPU device reference
    results/devicecheck/cpu.json (code a1be0f1e34704d16, seeds 900-902, 60
    rounds).
  - Server dry run (2026-10-29, Python 3.11.17 + torch 2.5.1, like the
    server; scratchpad venv311): every group of every new step ran 2 rounds
    (tuning specs, cost/CIFAR checks, ARL, sensitivity, screen C + worlds,
    device check). It caught one bug: predictor_stress's report crashed
    when a group had no jobs (None with a number format) -- fixed, with a
    regression test; `--rounds N` added for quick checks of the exact CLI.
  - D1: tag push refused (this session may push only the working branch).
    The user can push it from their own computer: `git fetch origin && git
    tag s2-frozen 998f2adb1c6cb244da311b3837319276d480d8f3 && git push origin
    s2-frozen` (never from the shared server).
- **Session 1 results (2026-10-30, RTX 3070, commit 2812614; all evidence
  re-checked here, same verdicts as the server; results/ committed):**
  - device check: GPU vs CPU no material difference (largest: tuned FEMNIST
    cost +12%, limit 15%); results/devicecheck/{gpu.json,compare.md}.
  - CIFAR stress: like FEMNIST (ECE 0.046; recovery 89%, median 8 rounds;
    avoided-for-good clients would succeed in a median 2% of rounds) BUT
    Priority badly over-predicted (0.69 vs 0.38; 2% of jobs).
  - cost check: rule -> base kept (median second-half spend 1.007 x target).
    Hidden by the median: seeds 282/283 at 1.35x/1.95x; cap_target held
    every seed (0.87-0.95) and scored +0.005; fast_dual made spend worse.
    Applied as written (no post-hoc switch); thesis must not claim the soft
    constraint holds spend. configs/rl2_tuned_cost.json = c34 unchanged.
  - CIFAR check: rule -> tuning holds (c34 rank 1/6, +0.007 vs c00) BUT on
    w10b c34 is the weakest of 6 on CIFAR (0.345 vs 0.378) -- caution for P2.
  - comparator tuning (stage 2, seeds 225-239, M7 score): fairq 0.6532,
    rl_light_mimic 0.6519 (eq5 0.784, spend 1,468), refl 0.6491, oort
    0.6430, favor 0.6334, arl3 0.6324, cov 0.6157, delta 0.6099 -- vs c34
    0.6378 in M7 (old code/pool, not a paired comparison). Suggests c34's
    lead over always-Light came mostly from MimiC + tuning effort. Screen C
    will decide on fresh seeds; a negative result is a valid thesis finding.
    Winners in configs/rl2_tuned_<name>.json (5 of 8 picked predictor v2).
- Next on the dept GPU, after `git pull` (stop or finish the old-code
  screen C first; it is exploratory; move `runs/rl2_screen_c` and
  `checkpoints` aside before the new screen C):
  1. DONE: session 1 (no dependencies): `devicecheck`, `screen2
     configs/rl2_cost_check.yaml`, `screen2 configs/rl2_cifar_check.yaml`,
     `stress cifar`, then `tunebase all` (~8-9 h, overnight; resumable);
     packsmall each and send.
  2. DONE (2026-10-30).
  3. session 2 (NEXT): `screen2 configs/rl2_arl_screen.yaml`; send; I run
     finalize_tuned arl and commit.
  4. session 3: `screen2 configs/rl2_sensitivity.yaml`,
     `screen-c-pretrain`, `screen-c`; send; then freeze; confirm only with
     the user's go-ahead.
