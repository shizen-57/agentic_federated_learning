# Handover: where the project stands (2026-10-30)

For the next assistant (GPT) taking over or reviewing. Read this first, then
`memory.md` (the living project notes), then the documents named below.
Everything here is on branch `claude/sleepy-faraday-8nd6j9` of
`shizen-57/agentic_federated_learning` (open PR:
https://github.com/shizen-57/agentic_federated_learning/pull/1), last commit
`fd68c8d`. Simulator code hash: `a1be0f1e34704d16`.

## 1. The project in a few lines

Undergraduate thesis (CSE, University of Dhaka): *Contribution-Aware
Reinforcement Learning for Fair Client Orchestration in Heterogeneous and
Non-IID Federated Learning*. Everything is a **simulation**: one machine plays
a server and 100 phones. Each round a scheduler picks up to 10 phones and a
job size (mode) for each; the thesis method is an RL scheduler with a
"contribution-aware" reward (ARL) meant to protect weak clients.

**Current verdict, in one sentence:** the original method failed its frozen
tests; the redesigned method (SAFE-PPO, "RL2") looked better than the simple
always-Light scheduler, but once every rival got the same aggregation and
the same tuning effort, the tuned always-Light and a fairness-queue baseline
score *higher* than our method on development data. The fair head-to-head
(screen C) has not run yet.

## 2. Working rules (please keep them)

**With the user**
- Use simple words, short sentences, and explain jargon. The user asked
  for this explicitly.
- Be honest: a failed test is reported as failed. A negative result is an
  acceptable thesis outcome.
- Never ask for or print passwords or GitHub tokens. If the user pastes
  them, tell them not to. Server login details stay with the user and are
  **never** written into the repo: it is **public**. No server address,
  username, port, server folder path or token in any committed file.

**Protocol** (`memory.md` section 2)
- **The frozen s2 evaluation** is preserved by commit `998f2ad`
  (tag `s2-frozen`; the tag exists only in one clone and still needs pushing).
  Never edit `configs/s2_frozen.json` or the pins in `scripts/prespec_s2.py`,
  and never re-pin. Validate the originals only with
  `bash scripts/validate_s2_original.sh`; expect "OK: 58 results".
- **Every step has a rule written before it runs**, in its spec and doc.
  Apply the rule as written; never change it after seeing results. If a
  rule looks weak afterwards, say so in the doc; don't switch decisions.
- **Data roles** (fixed by `data.split_seed`, pinned in
  `configs/rl2_data_roles.json`):
  - FEMNIST writers: practice 1,798 / development 899 / confirmation 900;
  - CIFAR-10 test images: validation 3,006 / development 3,488 /
    confirmation 3,506.
  - Development runs may never read the confirmation pool; the runners
    refuse (`rl2lib.check_pool`).
- **Seeds** (`configs/seed_registry.yaml`, guarded by `scripts/seeds.py`):
  development 210-299, practice 1000-1999, confirmation 300+ (only from a
  frozen manifest).
- **Evidence first.** Every runner checks every run before applying a rule:
  - the expected config;
  - all rounds present, with finite values;
  - one code version, made from a clean checkout and equal to the code
    running the check;
  - recorded input hashes.

  Otherwise it prints "INCOMPLETE -- no decision".
- **The GPU server runs only from a clean `git pull`.** Never edit files
  there and never `git pull` while a run is going. Nothing is pushed from the
  server.
- **Results folders are gitignored.** Commit evidence with `git add -f`.
- **`memory.md`** must be updated (with its "Last updated" line) after any
  progress, decision or result (`CLAUDE.md`).

## 3. Short history

1. Built the simulator, the RL scheduler (REINFORCE, ARL') and many baselines.
2. **s2: the frozen original evaluation** (1,540 runs, seeds 100-119).
   ARL-v3 passed 0 of 8 pre-specified tests. FEMNIST exposed "lock-ups"
   (50+ rounds with no successful client). Always-Light was cheap and near
   the top. See `docs/ADVANCED_RESULTS.md`.
3. Diagnosis (`docs/DIAGNOSTIC_FINDINGS.md`). The success predictor inverted
   its ranking after failures, and a deterministic fallback got stuck.
4. **RL2: SAFE-PPO** (`docs/RL2_DESIGN.md`, `docs/RL2_METHOD.md`):
   - repaired predictor v2 (Beta posterior, uncertainty, ageing);
   - Thompson-sampled fallback;
   - safety layer (mode mask, hard cap, soft cost constraint);
   - attention encoder, PPO, optional pre-training.
5. Development screens:
   - A and B passed;
   - **M7 tuning** picked candidate **c34**, which uses **no ARL term** (the
     plain failure count);
   - first comparison: c34 was ahead of always-Light by +0.036 in
     equal-spend accuracy.
6. Four external reviews (by GPT), each fixed. The fourth added: reserved
   confirmation data, a validation matrix, the ARL screen and the
   sensitivity screen. Docs: `docs/VALIDATION.md`, `docs/SIMULATOR_MODEL.md`.
7. **Project gap review** (`docs/GAP_REVIEW.md`, 2026-10-29). Its two biggest
   findings:
   - **MimiC vs FedProx.** c34 combines updates with MimiC, but every
     comparator used FedProx;
   - **unequal tuning.** c34 had 40-candidate tuning, the comparators had
     none.

   Also: fairness was measured at unequal spend, and the baselines used the
   old faulty predictor. All were fixed (status table at the end of that
   doc).
8. **Session 1 on the GPU** (2026-10-30): results below.

## 4. The setup in brief

Details: `docs/SIMULATOR_MODEL.md`.

**The federated learning part**
- 100 clients, 10 per round, 200 rounds.
- Small CNNs (about 220k parameters for FEMNIST, about 190k for CIFAR-10).
- Local training: SGD, learning rate 0.05, batch 32, FedProx term 0.01.
- Combining updates: FedProx, or MimiC (what the method and now every
  comparator use, except REFL, which keeps its own published aggregation).

**Job sizes (modes)**

| mode | local epochs | share of the update uploaded |
|---|---|---|
| Light | 1 | 25% |
| Standard | 2 | 50% |
| Priority | 4 | 100% |

**Data**
- FEMNIST: one writer = one client, 150-400 images each.
- CIFAR-10: Dirichlet(0.3) split, 50-600 images per client.

**Clients**
- Three device tiers (35% slow, 40% medium, 25% fast), each with its own
  speed, bandwidth, battery and link reliability.
- A daily online pattern; a client stays in its state from the last round
  with probability 0.8.
- A job fails if the client is offline, short of battery, loses the link,
  misses the deadline, or is in the 2% randomly rejected.

**The built-in fairness problem:** 30% of the clients are weak (slow,
unreliable), and they hold the **rarest classes**.

**What is assumed, not measured:** the energy and cost units, the tier
values, the deadline and the 2% rejection rate. A FedScale trace of real
phones is available as an option (`system.device_trace: fedscale`).

**Endpoints, all at equal spend** (cumulative cost <= 1,400):
- `eq5`: mean accuracy over the last 5 rounds within the budget;
- `w10b`: mean worst-10% client accuracy over the last 10 rounds within the
  budget;
- `score_eq` = (eq5 + w10b) / 2.

M7 and the comparator tuning ranked by `score` = (eq5 + w10) / 2, where `w10`
is the last 10 rounds whatever was spent; it is kept for parity.

## 5. Results so far (all development evidence)

**Session 1** (RTX 3070, commit `2812614`; every report re-run here gave the
server's verdict):

- **Device check:** GPU vs CPU, no material difference
  (`results/devicecheck/compare.md`).
- **CIFAR-10 stress test** (`results/predictor_stress_cifar`):
  - the predictor is calibrated overall (ECE 0.046);
  - 89% of the clients hit by the shock are invited again after recovery;
  - clients avoided for good would have succeeded in only about 2% of their
    rounds;
  - **but Priority jobs are badly over-predicted** (0.69 predicted vs 0.38
    observed; 2% of jobs).
- **Cost check** (`results/rl2_cost_check`): the rule kept the base (median
  spend 1.007 x target).
  - Hidden by the median: 2 of 5 seeds overspend (1.35x and 1.95x).
  - Only the hard cap at the target held every seed.
  - So the soft spending constraint is **not** reliable.
- **CIFAR-10 check of the tuning** (`results/rl2_cifar_check`): the tuning
  holds by its rule (c34 ranks first of six), **but** c34 is the weakest of
  the six on `w10b`.
- **Matched comparator tuning** (`results/rl2_tune_*`,
  `docs/RL2_BASELINE_TUNING.md`). Stage 2, seeds 225-239, M7 score:

| comparator (tuned) | score |
|---|---|
| fairness queues (`b_fairq_ew_mimic`, after RBCS-F) | 0.6532 |
| always-Light + MimiC (`rl_light_mimic`, the control) | 0.6519, at the lowest spend |
| REFL | 0.6491 |
| Oort | 0.6430 |
| **c34, our method (M7, old code and pool)** | **0.6378** |
| FAVOR | 0.6334 |
| original RL, repaired | 0.6324 |
| greedy coverage | 0.6157 |
| DELTA | 0.6099 |

Reading: c34's earlier lead over always-Light probably came mostly from MimiC
and extra tuning. Not yet a paired comparison: c34's number is from the old
code and data pool, and every winner is a best-of-six on these seeds.

## 6. Where we are, and what comes next

The order is enforced by the runners: a missing file stops them.

| step | status | how |
|---|---|---|
| cost check, then `finalize_tuned.py cost` | **done** | `configs/rl2_tuned_cost.json` = c34 unchanged |
| CIFAR-10 check, CIFAR stress, device check | **done** | see section 5 |
| matched tuning of 8 comparators | **done** | `configs/rl2_tuned_<name>.json` |
| **ARL screen** (does ARL-v3/v4 add value? 120 runs) | **next (session 2)** | `bash scripts/dept_gpu.sh screen2 configs/rl2_arl_screen.yaml` |
| `finalize_tuned.py arl` | after it | writes `configs/rl2_tuned_final.json`; commit it |
| sensitivity screen (260 runs) | after that | `screen2 configs/rl2_sensitivity.yaml` |
| screen C: pre-train, then run (240 runs) | after that | `screen-c-pretrain`, then `screen-c` |
| freeze (M9) | after screen C | `python scripts/freeze_rl2.py --decision <screen C folder>/DECISION.json`; commit the frozen file and the pins together |
| confirmation (M10) | **only with the user's explicit go-ahead** | `bash scripts/dept_gpu.sh confirm go` |

**Session 2 on the server** (the user has the commands): `git pull`; move the
old-code `runs/rl2_screen_c` and `checkpoints` aside; run the ARL screen;
then `bash scripts/dept_gpu.sh packsmall rl2_arl_screen`. The user copies the
pack home with scp and uploads it.

**Then, here:**
1. Check the upload (section 7).
2. Run `python scripts/screen_rl2.py report --spec configs/rl2_arl_screen.yaml
   --out results/rl2_arl_screen`.
3. Run `python scripts/finalize_tuned.py arl --spec configs/rl2_arl_screen.yaml
   --out results/rl2_arl_screen`.
4. Commit `configs/rl2_tuned_final.json` and the results.
5. Give the user the session 3 commands.

**A decision for the user after screen C.** If our method does not beat the
tuned always-Light, the options are:
- (a) freeze and confirm anyway: an honest test, likely "not supported";
- (b) stop and write up the negative result;
- (c) more method work, on fresh development seeds, before any freeze.

Do not choose for the user.

## 7. How uploaded results are checked (please keep doing it this way)

1. Treat archives as untrusted. List them first (`tar -tzvf`) and look for
   links, absolute paths or `..`.
2. Extract each into a new, empty folder with `--no-same-owner
   --no-same-permissions`. Read with `python -I`.
3. Check every `MANIFEST.json`: the code hash must equal this checkout's, and
   `git_dirty` must be false.
4. Privacy scan before committing: no usernames, server addresses, home or
   server paths.
5. Copy into `results/<name>`. Re-run the runner's own report (it re-checks
   every run), and compare with the server's verdict.
6. Commit the evidence with `git add -f`, and record the outcome in the
   step's doc and in `memory.md`.

## 8. Known weaknesses and open items

**The method**
- The predictor over-predicts Priority jobs (strongly on CIFAR-10).
- The soft cost constraint does not hold spending on some seeds.
- About 14% of clients (18% after a shock) are never invited late in a run;
  the stress tests show they would mostly fail anyway.
- On CIFAR-10, c34 is weak on the equal-spend fairness measure.

**Data and setup**
- Earlier development screens (B, M7) drew clients from the whole
  evaluation pool, before the confirmation pool existed. Disclosed in
  `docs/RL2_CONFIRMATION.md` ("History"). Every decision that feeds the
  freeze is re-made on the development pool.
- CIFAR-10 separates only its test images; FEMNIST separates whole writers
  (`docs/SIMULATOR_MODEL.md`).
- The scheduler knows the simulator's own time formula, with 15% noise on
  speed and bandwidth (stated as a limitation).

**The thesis**
- Chapter 3 still describes the original method. `docs/RL2_METHOD.md` is a
  draft of the new method chapter, including exactly which parts are
  "contribution-aware".
- No RL2 results chapter yet. `scripts/plot_rl2.py` makes the figures.

**Housekeeping**
- The `s2-frozen` tag is not on GitHub (this session could push only the
  branch). The user can push it from their own computer:
  `git tag s2-frozen 998f2adb1c6cb244da311b3837319276d480d8f3 && git push
  origin s2-frozen`.
- The old git history of `memory.md` (27-29 October) contains the name of
  the user's server folder (not a login detail). It is removed from the
  current version; rewriting history was left to the user to decide.

## 9. Where things are

| what | where |
|---|---|
| project notes (read first) | `memory.md` |
| simulator, engine, policies | `src/afl/` (`engine.py`, `sim.py`, `policies/safe_rl.py`, `policies/published.py`, `predictor.py`, `conditions.py`) |
| RL2 runners | `scripts/rl2lib.py`, `tune_rl2.py`, `screen_rl2.py`, `screen_c_rl2.py`, `finalize_tuned.py`, `freeze_rl2.py`, `prespec_rl2.py`, `confirm_rl2.py` |
| GPU helper (every server step) | `scripts/dept_gpu.sh` (run it with no argument for the list) |
| specs (rules inside) | `configs/rl2_*.yaml` |
| tuned settings | `configs/rl2_tuned.json` (c34), `rl2_tuned_cost.json`, `rl2_tuned_<comparator>.json` |
| docs per step | `docs/RL2_*.md`, `docs/GAP_REVIEW.md`, `docs/VALIDATION.md`, `docs/SIMULATOR_MODEL.md` |
| results (evidence) | `results/` (s2 in `results/adv/`, RL2 in `results/rl2_*`) |

## 10. How to verify

- **Tests:** `python -m pytest tests -q`. The full suite passed on torch 2.14
  and 2.5.1 (the server's version): 366 passed, 8 skipped (the skipped ones
  run only in the frozen checkout). One regression test was added since.
- **s2 originals:** `bash scripts/validate_s2_original.sh` should print
  "OK: 58 results".
- **A short pre-flight on the server's versions** (Python 3.11 + torch 2.5.1)
  before giving the user new GPU steps. It caught a crash once.
- **Dry runs:** each runner's `plan` command shows what would run;
  `rl2lib.smoke` runs 2 rounds per group.
