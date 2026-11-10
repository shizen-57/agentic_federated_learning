# The RL2 method (SAFE-PPO): a draft for the thesis method chapter

Gap review C1 and C2. Chapter 3 of the thesis describes the original method:
a REINFORCE scheduler (ScoreNet + Plackett-Luce picks) with the ARL' reward.
That method failed its frozen tests (s2, `docs/ADVANCED_RESULTS.md`), and
the diagnosis (`docs/DIAGNOSTIC_FINDINGS.md`) found why. The method now
under test is SAFE-PPO. This document describes it as implemented, in the
order a method chapter needs, so Chapter 3 can be extended (or a new
chapter written) without reading the code. The design rationale and its
revisions are in `docs/RL2_DESIGN.md`.

## 1. The decision each round

In round t, the clients that check in form the candidate set A_t (online
this round). The scheduler chooses a set S_t of at most K = 10 clients from
A_t, and a mode m_i for each chosen client:

| mode | local epochs E | upload fraction f |
|---|---|---|
| Light | 1 | 0.25 |
| Standard | 2 | 0.5 |
| Priority | 4 | 1.0 |

A job succeeds if the client stays online, has the battery, keeps the link
and finishes within the deadline (`docs/SIMULATOR_MODEL.md`). The goal:
high accuracy at a given spend, and protection of the weakest clients,
without starving anyone.

## 2. What the scheduler sees

Per client (`policies/rl.py FEATURES`, plus memory):
- device estimates: speed and bandwidth (15% noise), battery level;
- data: its size and its marginal value for class coverage;
- its predicted success in each mode (below), with uncertainty;
- its record: success rate, rounds since its last success, how often it
  was picked, its last local loss;
- memory: success averages per mode, the last outcomes, a short-minus-long
  trend, time since last seen, how often seen;
- coverage gain: how much each client would add to the classes of the
  clients picked so far in this round.

Per round: the round's progress, mean best success probability, eligible
share, cost multiplier, recent cost against the target, and a GRU summary of
the last few rounds.

## 3. Success predictor v2

A shared logistic model on per-round standardised features gives p0, mixed
with a prior of 0.7 by weight seen / (seen + 20), so there is no cold-start
cliff. Each (client, mode) then has a Beta posterior centred on p0:

    a = w s + k p0,   b = w f + k (1 - p0),   p = a / (a + b),
    sd = sqrt(a b / ((a + b)^2 (a + b + 1))),

with s and f the client's own successes and failures in that mode, k = 5,
and w = 0.98^(rounds since the last update), so old evidence fades and the
uncertainty grows again (evidence ageing). Calibration is checked against
the mode that actually ran (`docs/VALIDATION.md`, section 5).

## 4. The safety layer

1. **Feasible-mode mask.** Mode m is allowed for client i if
   p_i(m) - z_mask sd_i(m) >= 0.5 (z_mask = 0.25 in the tuned method).
2. **If no mode is allowed**: Light, if its optimistic bound
   p_i(Light) + z_gate sd_i(Light) >= 0.5 (a "Light override", z_gate = 1.5);
   otherwise the client abstains this round. Both are logged every round
   (`n_light_override`, `n_abstain`).
3. **Gate.** Clients that pass the eligibility gate are candidates; if fewer
   than K pass, the free slots are filled by Thompson sampling from the
   posteriors, never by the same deterministic set (the absorbing state
   that caused the original lock-ups).
4. **Hard cap.** The chosen jobs' cost per round may not exceed
   h x (K x mean Light cost) (h = 3.0 tuned). Over the cap, the budget step
   moves the last-picked client to a lighter *allowed* mode, or drops it.
5. **Soft constraint.** A Lagrange multiplier lambda on cost: PPO learns
   from r_t - lambda_t c_t / c*, and lambda_{t+1} = max(0, lambda_t +
   eta (c_t / c* - 1)), with c* = 1.0 x K x mean Light cost and eta = 0.05.
   In M7 the tuned method still spent about 25% above c*; the cost check
   (`docs/RL2_COST_AND_CIFAR_CHECKS.md`) decides whether a tighter variant
   replaces it.

## 5. The policy network and learning

- **Encoder** (attention): two self-attention layers (64 dimensions, 4
  heads) over all candidates plus a global token, so each client is judged
  next to the others.
- **Selection**: pointer-style picks, one at a time, each conditioned on the
  clients already picked, over the allowed clients; the exact log-probability
  of the whole ordered pick is kept.
- **Modes**: a mode head per picked client, restricted to its allowed modes.
- **Critic**: a value of the state only.
- **PPO**: clipped objective (clip 0.3), 4 epochs per batch of 8 rounds,
  learning rate 0.005, discount 0.9, GAE lambda 0.95, advantage
  normalisation, entropy bonus decaying from 0.003. Rounds with no action
  stay in the trajectory (their reward and value count; only the actor
  terms are masked).
- **Projection bias** (gap review B10). When the budget step changes or
  drops a job, PPO still learns from the sampled action, while the reward
  comes from the executed one: a known bias of action projection. The
  report shows how often it happens (`budget_lighter`, `budget_dropped`).

## 6. The reward, and what is contribution-aware (C2)

    r_t = alpha z(d_acc) - beta z(cost) - delta z(latency) - lambda_arl D_t
          - lambda_fair CV_t + w_prot z(d_worst),

with z(.) a running standardisation (clipped to +-3) and:
- d_acc: the change in validation accuracy (never the test set);
- D_t: the damage term. In the tuned method c34 it is the **plain failure
  ratio** (`raw_fail`). ARL-v3 or ARL-v4 can replace it;
- CV_t: the coefficient of variation of each client's **successful
  contributions** so far (lambda_fair = 0.3);
- d_worst: the change in the worst-10%-class validation accuracy (the
  protection term, w_prot = 1.0);
- tuned weights: alpha 1, beta 0 (cost is handled by the constraint),
  delta 0.1, lambda_arl 0.5.

**Which parts are contribution-aware:**
1. the successful-contribution fairness term CV_t, which counts what each
   client actually delivered, not how often it was asked;
2. the protection term d_worst, which rewards progress on the classes the
   model knows worst (the rare data that weak devices hold);
3. the coverage-gain inputs and the marginal-value input, which tell the
   policy what each client would add;
4. ARL-v4 (or v3), *if* the ARL screen finds that it adds value
   (`docs/RL2_ARL_SCREEN.md`). It is the thesis's original idea, but the
   tuned method does not use it unless that screen says so.

The thesis should say precisely which of these the final method contains.
If the ARL screen finds no value, the claim is "contribution-aware through
1-3", and the ARL result is reported as a negative finding.

## 7. Pre-training (optional)

Practice episodes on practice seeds (1000-1999) and practice data only
(FEMNIST practice writers), with the reward from the validation pool and no
test evaluation, the device world randomised (mean availability, its daily
swing, the deadline). 20 iterations x 8 parallel episodes, starting from one
shared network; the parameters are averaged after each iteration. Then the
run starts from this checkpoint and keeps learning online. Screen C decides
whether pre-training is used. Its cost is reported separately.

## 8. From Chapter 3 to SAFE-PPO

| Chapter 3 (original) | SAFE-PPO | why (evidence) |
|---|---|---|
| ScoreNet, clients scored one by one | attention over all clients, pointer picks | clients were judged without each other |
| REINFORCE, one update per round | PPO with a critic and GAE | noisy learning signal, ~200 updates |
| success predictor v1 | predictor v2 (Beta posterior, uncertainty, ageing) | v1 inverted its ranking after failures |
| deterministic gate fallback | Thompson-sampled fallback | absorbing "stuck" state (lock-ups) |
| any mode may be drawn | feasible-mode mask, Light override, abstention | unsafe heavy jobs |
| fixed cost weight | hard cap + Lagrangian constraint | overspending |
| ARL' damage term | failure count, or ARL-v3/v4 if the ARL screen supports it | ARL' failed its pre-specified tests |
| no memory | per-client memory and a GRU | no use of history |
| from zero | optional pre-training on practice episodes | too little training |
| FedProx aggregation | MimiC aggregation | the best pairing in the early sweep; every comparator now uses it too |

## 9. Limitations to state

Simulation only; energy and cost in simulated units; the scheduler knows
the simulator's time formula; CIFAR-10's data separation is weaker than
FEMNIST's; per-client test sets are small; the tuning was done on FEMNIST
only; earlier development screens saw data that is now reserved for the
confirmation. Each is explained, with what was done about it, in
`docs/GAP_REVIEW.md`, `docs/SIMULATOR_MODEL.md` and
`docs/RL2_CONFIRMATION.md`.
