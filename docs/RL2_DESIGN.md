# RL2 design: SAFE-PPO scheduler with ARL-v4

Written before any RL2 code or run. Plan: `/root/.claude/plans` copy summarised
in `memory.md` section 12. Every component is a switch (default off), so the
original behaviour and the frozen s2 evaluation stay reproducible.

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
| C1 | Predictor v2: standardised features + per-client Beta correction shrunk to the shared model (weight n/(n+k)), smooth cold start, posterior variance | `policy.predictor: v2` | k = 5 | k, prior strength |
| C2 | Gate v2: eligibility on an optimistic bound p + z*sd; fallback by Thompson sampling from the posterior | `policy.gate_mode: thompson` | z = 1.0 | z |
| C3 | PPO actor-critic: GAE, clipped objective, value clipping, advantage normalisation, entropy schedule | `policy.name: safe_ppo` | clip 0.2, gamma 0.9, lambda 0.95, 4 epochs, batch = 8 rounds | lr, clip, epochs, batch, entropy |
| C4 | Set encoder: 2-layer self-attention over candidates (d=64, 4 heads), pointer selection with exact sequential log-prob, mode head conditioned on chosen set + budget state | `policy.encoder: attention` | d 64 | d, layers |
| C5 | Memory: per-client per-mode success EMAs, last-5 outcomes, trend, staleness; GRU over a round-summary vector as global context | `policy.memory: true` | EMA 0.2, GRU 32 | EMA rate |
| C6 | Safety: mode mask with margin (p - z*sd >= threshold), Lagrangian cost constraint (dual ascent toward a spend target), trust curriculum (heavier modes unlock when calibration error < e and n > m) | `policy.safety: full` | target = rl_light's mean cost per round x 1.25, eta 0.05, e 0.1, m 200 | target multiplier, eta |
| C7 | Reward v2: validation accuracy gain + protection term (worst-class validation accuracy gain, successful-contribution fairness) | `reward.version: 2` | weights 1.0 / 0.5 / 0.3 | weights |
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
  the remaining budget and with p - z*sd >= threshold (never "heaviest").
- **Loss**: L_i = v_i * max(0, E[contribution | m*] - E[contribution | chosen]),
  using confident probabilities (p - z*sd), so uncertain predictions are not punished.
- **Deficit term**: D_t = sum over classes of the coverage deficit EMA, times
  how long it has lasted. A round with no successful contributions raises D_t,
  so the stuck state is visible and costly.
- **Per-client credit**: difference reward R(S) - R(S \ i) on the coverage
  model, for PPO's per-client advantage.
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
runs. Tuning only on dev seeds 215-239, ablation on 240-259, confirmation on
300+ from a frozen manifest, after the user's go-ahead. Primary endpoints
(both, Holm across): P1 accuracy at a pre-fixed equal-spend budget vs
always-Light and vs the best baseline; P2 final worst-10% client accuracy vs
always-Light and ARL-v3. Results reported for every seed, pass or fail.
