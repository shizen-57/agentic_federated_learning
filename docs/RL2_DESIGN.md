# RL2 design: SAFE-PPO scheduler with ARL-v4

Written before any RL2 code or run; revised after the second external review
(what changed and why: "Revision 2" and "Revision 3" at the end). Plan summarised in `memory.md`
section 12. Every component is a switch (default off), so the original
behaviour and the frozen s2 evaluation stay reproducible.

## Why (evidence)

From `docs/DIAGNOSTIC_FINDINGS.md` and the two development screens: the
predictor inverts its ranking after failures and drops its prior abruptly; the
deterministic gate fallback creates an absorbing stuck state; learned job sizes
are unsafe and overspend; REINFORCE with ~200 updates from zero is too weak;
the policy has no memory and scores clients one by one; ARL-v1..v3 measured the
wrong quantity and are blind to the stuck state.

## Components

| id | component | switch | default (pre-tuning) | tunable on dev seeds |
|---|---|---|---|---|
| C1 | Predictor v2: standardised features + per-client Beta correction shrunk to the shared model (weight n/(n+k)), smooth cold start, posterior variance; a client's own evidence ages by `pred_decay` per round, so drift is tracked and uncertainty grows back | `policy.predictor: v2`, `policy.pred_decay` | k = 5, decay 0.98 | k, prior strength, decay |
| C2 | Gate v2: eligibility on an optimistic bound p + z*sd; fallback by Thompson sampling from the posterior | `policy.gate_mode: thompson` | z = 1.0 | z |
| C3 | PPO actor-critic: GAE, clipped objective, value clipping, advantage normalisation, entropy schedule | `policy.name: safe_ppo` | clip 0.2, gamma 0.9, lambda 0.95, 4 epochs, batch = 8 rounds | lr, clip, epochs, batch, entropy |
| C4 | Set encoder: 2-layer self-attention over candidates (d=64, 4 heads), pointer selection with exact sequential log-prob, mode head conditioned on chosen set + budget state | `policy.encoder: attention` | d 64 | d, layers |
| C5 | Memory: per-client per-mode success EMAs, last-5 outcomes, trend, staleness; GRU over a round-summary vector as global context | `policy.memory: true` | EMA 0.2, GRU 32 | EMA rate |
| C6 | Safety: mode mask with margin (p - z*sd >= threshold), SOFT Lagrangian cost constraint (dual ascent toward a mean spend target), HARD per-round cap (action projection), trust curriculum (heavier modes unlock when calibration error < e and n > m); composition rule below | `policy.mode_mask: feasible`, `mode_mask_z`, `cost_target_mult`, `hard_budget_mult`, `curriculum` | target 1.25 x (K x mean Light cost), cap 2.0 x, eta 0.05, e 0.1, m 200 | target and cap multipliers, eta |
| C7 | Reward v2: validation accuracy gain + protection term (worst-class validation accuracy gain, successful-contribution fairness) | `reward.protect_weight` | 0.5 | weights |
| C8 | ARL-v4 (below) | `reward.damage_metric: arl_v4` | lambda 0.5 | lambda |
| C9 | Pre-training on practice seeds 1000-1999, validation-only reward, no test evaluation, domain randomisation; then online fine-tuning | `policy.init_checkpoint` | 2 x 160 episodes per dataset | episodes, randomisation ranges |

## ARL-v4

For each invited client i in round t, with calibrated success probabilities
p_i(m) and uncertainty sd_i(m) from C1, and the round's cost constraint from C6:

- **Contribution value** v_i: Shapley share of class coverage (as in ARL-v3),
  times (1 + the client's share in under-covered classes), times a quality
  factor from the alignment of its past delivered updates with the validation
  gradient (validation data only; 1 until it has delivered).
- **Feasible counterfactual**: the best mode m* among modes affordable within
  the round's JOINT remaining budget -- the spend target minus the other
  invited clients' planned cost -- and with p - z*sd >= threshold (never
  "heaviest"; never a per-client allowance that the set as a whole could not
  afford). If no mode is both, there was no alternative and L_i = 0;
  overspending is charged by the constraint and the cap, a stuck client by
  the deficit term.
- **Loss**: L_i = v_i * max(0, E[contribution | m*] - E[contribution | chosen]),
  using confident probabilities (p - z*sd), so uncertain predictions are not punished.
- **Deficit term**: D_t = sum over classes of the coverage deficit EMA, times
  how long it has lasted. A round with no successful contributions raises D_t,
  so the stuck state is visible and costly.
- **Set-level gain** G_t = f(D_t) = sum over classes k of w_k * max over
  clients i in D_t of c_ik: the deficit-weighted (w_k = 1 + deficit_k) class
  coverage of the set D_t of clients that DELIVERED. Adding a successful client
  never lowers it. This is the coverage term of the joint objective.
- **Joint objective** (used in all RL2 conditions, `policy.credit_total: true`):
  R_t = shared reward - lambda * mu * z(D_t) + lambda_cov * z(G_t)
  - lambda * sum_i z(L_i). Per-client difference credits f(D) - f(D \ i) are
  NOT summed into it (their sum is not the coverage of the set); they are used
  only as individual advantages when `credit_total` is off.
- ARL-v4_t = sum_i L_i + mu * D_t. ARL* (the simulator oracle) stays
  evaluation-only.

## Aggregation

The scheduler choices above are independent of how updates are combined.
RL selection + MimiC aggregation was the best pair in the early sweep, but its
pre-registered confirmation narrowly failed (docs/ADVANCED_RESULTS.md item 9),
so it is carried as a candidate, not assumed: condition `ppo_full_mimic`
(= `ppo_full` with `fl.aggregator: mimic`) is in the ablation (M8) and enters
the confirmation only if it helps there on fresh dev seeds.

## Protocol

Seeds: `configs/seed_registry.yaml` (guarded by `scripts/seeds.py`). Each
milestone ends with a development screen whose rule is committed before it
runs, in its spec, and is applied by `scripts/report_dev_screen.py --spec`
only to complete, valid evidence. Screen A 210-214 (done), screen B 215-219,
tuning 220-239, ablation and screen C 240-259, confirmation on 300+ from a
frozen manifest, after the user's go-ahead. Primary endpoints
(both, Holm across): P1 accuracy at a pre-fixed equal-spend budget vs
always-Light and vs the best baseline; P2 final worst-10% client accuracy vs
always-Light and ARL-v3. Results reported for every seed, pass or fail.

## Revision 2 (after the second external review)

Each finding and what changed. Nothing below had been run on GPU before the fix.

1. **Data roles (critical).** With `data.split: per_seed` every seed draws its
   own validation/test split, so practice episodes on other seeds could train
   on another seed's test examples. Now `data.split: persistent` fixes the
   roles once (`split_seed`): one validation pool, one test pool, and for
   FEMNIST disjoint practice and evaluation writer pools. The split has a
   fingerprint; a pre-trained checkpoint is refused on any other split.
   Every RL2 screen and confirmation run uses the persistent split, for every
   condition (spec `set: {data.split: persistent}`). s2 keeps `per_seed`
   (within a run it was always clean).
2. **Pre-training path.** `scripts/pretrain.py` applies the condition to the
   group's settings itself, so the documented command runs.
3. **Shared start.** One starting network is built once and sent to every
   worker (checked by checksum), so averaging never mixes independent
   initialisations. `--mode average|sequential` is recorded.
4. **Mask composition.** Allowed modes = feasible (p - z*sd >= threshold)
   AND unlocked by the curriculum. If that is empty: Light, only if Light's
   optimistic bound (p + z*sd) clears the threshold (counted as
   `light_override`); otherwise the client abstains (not invited this round,
   counted). The curriculum never silently overrides feasibility.
5. **ARL-v4 objective.** The joint reward uses the set-level gain f(D), not a
   sum of difference credits (above).
6. **Empty rounds.** A round with no action stays in the PPO trajectory:
   its reward, time step and value are kept for GAE; only the actor loss and
   entropy are masked to rounds that acted.
7. **Screen evidence.** The report rebuilds every expected run from the spec
   and checks it (config, all rounds, finite values, no source change, one
   code version, clean checkout, same code as the report). Anything missing
   or invalid prints "INCOMPLETE -- no decision".

Research points from the same review:

- **Soft vs hard budget.** The Lagrangian keeps the MEAN spend near its target
  but allows single expensive rounds; `policy.hard_budget_mult` adds a hard
  per-round cap. Both are reported; "safe" means the hard cap, not the
  Lagrangian alone.
- **Stale uncertainty.** Predictor evidence ages (`pred_decay`), so a client
  whose behaviour changed is not trusted on old outcomes.
- **Unbundled ablations.** Screen B removes ONE component at a time from the
  full stack (docs/RL2_SCREEN_B.md), instead of adding several at once.
- **Required comparisons.** The same PPO stack without ARL-v4
  (`ppo_full_rawfail`: plain failure count; `ppo_full_arl3`: ARL-v3), and a
  repaired simple scheduler (`rl_light_pg`: always Light with the same
  predictor v2, Thompson gate and evidence ageing as the full stack).
- **Pre-training cost.** Checkpoints record the practice episodes, rounds,
  simulated cost and time. Results report deployment cost (the evaluated run)
  and total cost (practice + deployment) side by side; a pre-trained
  scheduler is never compared on deployment cost alone without saying so.

## Revision 3 (after the third external review)

1. **Budget projection vs the safety mask.** The engine's budget step (hard
   cap, `system.cost_max`, `latency_max`) used to move a client to the next
   lighter mode whatever the mask said, so a masked job (e.g. Light at
   p = 0.1) could run. SAFE-PPO now publishes its allowed modes each round;
   the projection only moves a client to a lighter ALLOWED mode and otherwise
   drops it for the round. Policies without a mask are unchanged.
   The curriculum's calibration now uses the prediction for the job that
   actually ran, not the one first sampled; PPO still learns from the
   sampled action (an action projection).
2. **Report overrides.** The report used to apply any `--set` recorded in the
   manifest when rebuilding the expected runs, so `data.split=per_seed`
   could silently replace the spec's persistent split. Now the runner and the
   report accept only operational overrides (`fl.device`), none is applied,
   and the spec, source plan and generated plan must match their recorded
   hashes.
3. **Starvation includes abstention.** A scheduler that stops inviting used to
   pass "no starvation". Starvation is now the longest run of rounds with no
   successful contribution, invited or not; rounds with nobody online
   (new log column `n_online`) are skipped. Re-checked on screen A: no empty
   rounds there, the same two runs starve under either definition, so its
   decision is unchanged.
4. **ARL-v4 with no alternative.** With no feasible, affordable mode the
   opportunity loss is now 0 (it used to reopen every mode, and when nothing
   was confidently feasible it compared against an unsafe one).
