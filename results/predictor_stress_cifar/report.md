# Predictor and safety-layer stress test

Tuned method, real CIFAR-10, seeds 290, 291, 292, 200 rounds. Shock world: half the devices lose 98% bandwidth and 95% battery at round 80, restored at 140. Calibration of the executed job's predicted success.

## stable

| group | jobs | mean predicted | observed | ECE | Brier |
|---|---|---|---|---|---|
| all | 6000 | 0.610 | 0.588 | 0.046 | 0.227 |
| mode light | 4915 | 0.586 | 0.575 | 0.047 | 0.228 |
| mode standard | 947 | 0.721 | 0.682 | 0.039 | 0.206 |
| mode priority | 138 | 0.688 | 0.377 | 0.311 | 0.333 |
| weak clients | 1911 | 0.535 | 0.523 | 0.082 | 0.258 |
| strong clients | 4089 | 0.644 | 0.618 | 0.036 | 0.212 |
| first 3 invitations | 893 | 0.542 | 0.480 | 0.077 | 0.238 |
| later invitations | 5107 | 0.621 | 0.606 | 0.044 | 0.225 |
| before (all) | 2370 | 0.591 | 0.558 | 0.054 | 0.232 |
| shock (all) | 1800 | 0.617 | 0.592 | 0.049 | 0.229 |
| after (all) | 1830 | 0.626 | 0.621 | 0.038 | 0.217 |

Exceptions per round (mean over seeds and rounds):

- before: abstain 13.02, Light override 36.03, gate fill 0.00, invited 10.00
- shock: abstain 24.76, Light override 24.98, gate fill 0.00, invited 10.00
- after: abstain 25.25, Light override 24.38, gate fill 0.00, invited 10.00
- clients not invited at all in the last 60 rounds: [10, 7, 18] of 100 per seed
- of those 37 clients (all seeds): 3% weak; median 39 of 60 rounds online; ruled out by the safety layer in a median 96% of their online rounds; 31 ruled out in >= 90% (avoided for good)
- were they right to be avoided? Of the 31 avoided for good, some mode would have succeeded in a median 2% of their online rounds (simulator ground truth); 0 could have succeeded in at least half of them (avoided wrongly)
- last 60 rounds, every online client-round the safety layer ruled out (3190): a job was possible in 13% (Light alone in 13%); for allowed client-rounds 62%

## shock

| group | jobs | mean predicted | observed | ECE | Brier |
|---|---|---|---|---|---|
| all | 6000 | 0.598 | 0.570 | 0.047 | 0.224 |
| mode light | 5112 | 0.576 | 0.559 | 0.038 | 0.224 |
| mode standard | 767 | 0.736 | 0.673 | 0.063 | 0.211 |
| mode priority | 121 | 0.679 | 0.364 | 0.315 | 0.317 |
| weak clients | 1786 | 0.513 | 0.494 | 0.082 | 0.255 |
| strong clients | 4214 | 0.635 | 0.602 | 0.037 | 0.211 |
| first 3 invitations | 890 | 0.545 | 0.482 | 0.070 | 0.238 |
| later invitations | 5110 | 0.608 | 0.585 | 0.044 | 0.222 |
| before (hit) | 1166 | 0.603 | 0.582 | 0.060 | 0.230 |
| before (untouched) | 1204 | 0.596 | 0.557 | 0.054 | 0.234 |
| shock (hit) | 282 | 0.402 | 0.035 | 0.367 | 0.179 |
| shock (untouched) | 1518 | 0.622 | 0.596 | 0.059 | 0.221 |
| after (hit) | 938 | 0.582 | 0.633 | 0.068 | 0.224 |
| after (untouched) | 892 | 0.633 | 0.629 | 0.058 | 0.222 |

Exceptions per round (mean over seeds and rounds):

- before: abstain 12.99, Light override 36.40, gate fill 0.00, invited 10.00
- shock: abstain 54.02, Light override 21.64, gate fill 0.00, invited 10.00
- after: abstain 28.02, Light override 26.73, gate fill 0.00, invited 10.00
- clients not invited at all in the last 60 rounds: [20, 11, 7] of 100 per seed
- of those 38 clients (all seeds): 8% weak; median 40.0 of 60 rounds online; ruled out by the safety layer in a median 97% of their online rounds; 23 ruled out in >= 90% (avoided for good)
- were they right to be avoided? Of the 23 avoided for good, some mode would have succeeded in a median 2% of their online rounds (simulator ground truth); 2 could have succeeded in at least half of them (avoided wrongly)
- last 60 rounds, every online client-round the safety layer ruled out (3458): a job was possible in 17% (Light alone in 17%); for allowed client-rounds 62%

Recovery of the hit clients after round 140: 89% invited again; median wait 8 rounds; success rate 0.582 before the shock vs 0.625 from round 160 on.
