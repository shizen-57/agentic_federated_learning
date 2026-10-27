# Project memory

Living notes for this thesis project: where things stand, what we know, and
how we work. Read this first in every session. Update it whenever progress is
made, a decision is taken, or a result comes in (newest facts win; keep it
short and true).

Last updated: 2026-10-27 (diagnosis work started)

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
