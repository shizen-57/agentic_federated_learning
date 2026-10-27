# RL2 screen B: the SAFE-PPO stack, layer by layer (from scratch)

Written and committed before any run. Development gate; exploratory.

**Conditions** (`configs/rl2_screen_b.yaml`; FEMNIST, s2 settings, 200 rounds,
dev seeds 215-219, no pre-training): references `arl3_full`, `v2_rawfail`,
`rl_light`; stack `ppo_v3` (PPO), `ppo_att` (+ attention + memory),
`ppo_safe` (+ predictor v2, Thompson gate, safety layer), `ppo_full`
(+ ARL-v4 + protection reward, cost via constraint), `ppo_full_mimic`
(+ MimiC aggregation). Run on the department RTX 3070
(`bash scripts/dept_gpu.sh screen configs/rl2_screen_b.yaml`) or CPU; the
device is recorded in MANIFEST.json.

**Measures:** as in docs/DEV_SCREEN_PREDICTOR_V1.md, plus the median final
worst-10% client accuracy (`acc_client_worst10`, the P2 measure), reported.

**Rule** for each stack condition c:
1. no seed starves (50+ rounds with invitations and no success);
2. median accuracy at equal spend (B = 1,400) >= median of `rl_light` - 0.02;
3. median final accuracy >= median of `arl3_full` - 0.02.

**Carried forward:** among conditions that pass, the one with the highest
median equal-spend accuracy goes to pre-training (M6) and tuning (M7). If
`ppo_full_mimic` is that one, MimiC aggregation is kept. If none passes,
`ppo_full` still goes to pre-training: pre-training is part of the design and
from-scratch learning over 200 rounds is the known weak point (W7); screen C
then judges the pre-trained version on new dev seeds. Results are reported
for every seed either way; nothing is tuned on seeds 215-219.

Report: `python scripts/report_dev_screen.py <out>/full/s2_femnist
--screen-b ppo_v3,ppo_att,ppo_safe,ppo_full,ppo_full_mimic`.
