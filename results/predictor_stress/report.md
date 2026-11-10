# Predictor and safety-layer stress test

Tuned method, real FEMNIST, seeds 290, 291, 292, 200 rounds. Shock world: half the devices lose 98% bandwidth and 95% battery at round 80, restored at 140. Calibration of the executed job's predicted success.

## stable

| group | jobs | mean predicted | observed | ECE | Brier |
|---|---|---|---|---|---|
| all | 6000 | 0.615 | 0.597 | 0.038 | 0.223 |
| mode light | 4880 | 0.591 | 0.583 | 0.034 | 0.223 |
| mode standard | 541 | 0.722 | 0.680 | 0.050 | 0.215 |
| mode priority | 579 | 0.719 | 0.639 | 0.080 | 0.228 |
| weak clients | 1615 | 0.475 | 0.436 | 0.069 | 0.250 |
| strong clients | 4385 | 0.667 | 0.657 | 0.051 | 0.212 |
| first 3 invitations | 877 | 0.529 | 0.465 | 0.064 | 0.240 |
| later invitations | 5123 | 0.630 | 0.620 | 0.039 | 0.220 |
| before (all) | 2370 | 0.576 | 0.549 | 0.037 | 0.231 |
| shock (all) | 1800 | 0.640 | 0.624 | 0.033 | 0.213 |
| after (all) | 1830 | 0.641 | 0.634 | 0.051 | 0.221 |

Exceptions per round (mean over seeds and rounds):

- before: abstain 16.33, Light override 34.03, gate fill 0.00, invited 10.00
- shock: abstain 25.05, Light override 23.78, gate fill 0.00, invited 10.00
- after: abstain 26.39, Light override 22.60, gate fill 0.00, invited 10.00
- clients not invited at all in the last 60 rounds: [6, 24, 12] of 100 per seed
- of those 43 clients (all seeds): 14% weak; median 39 of 60 rounds online; ruled out by the safety layer in a median 97% of their online rounds; 35 ruled out in >= 90% (avoided for good)
- were they right to be avoided? Of the 35 avoided for good, some mode would have succeeded in a median 21% of their online rounds (simulator ground truth); 1 could have succeeded in at least half of them (avoided wrongly)
- last 60 rounds, every online client-round the safety layer ruled out (3510): a job was possible in 23% (Light alone in 23%); for allowed client-rounds 63%

## shock

| group | jobs | mean predicted | observed | ECE | Brier |
|---|---|---|---|---|---|
| all | 6000 | 0.587 | 0.562 | 0.037 | 0.224 |
| mode light | 5042 | 0.564 | 0.550 | 0.032 | 0.222 |
| mode standard | 471 | 0.708 | 0.658 | 0.050 | 0.216 |
| mode priority | 487 | 0.704 | 0.595 | 0.108 | 0.248 |
| weak clients | 1633 | 0.458 | 0.415 | 0.060 | 0.247 |
| strong clients | 4367 | 0.635 | 0.618 | 0.056 | 0.215 |
| first 3 invitations | 883 | 0.528 | 0.462 | 0.066 | 0.239 |
| later invitations | 5117 | 0.597 | 0.580 | 0.037 | 0.221 |
| before (hit) | 1250 | 0.584 | 0.556 | 0.046 | 0.237 |
| before (untouched) | 1120 | 0.568 | 0.540 | 0.039 | 0.225 |
| shock (hit) | 259 | 0.378 | 0.042 | 0.336 | 0.168 |
| shock (untouched) | 1541 | 0.600 | 0.582 | 0.053 | 0.227 |
| after (hit) | 1008 | 0.609 | 0.656 | 0.087 | 0.222 |
| after (untouched) | 822 | 0.631 | 0.614 | 0.033 | 0.215 |

Exceptions per round (mean over seeds and rounds):

- before: abstain 16.33, Light override 34.03, gate fill 0.00, invited 10.00
- shock: abstain 53.81, Light override 22.17, gate fill 0.00, invited 10.00
- after: abstain 29.92, Light override 24.73, gate fill 0.00, invited 10.00
- clients not invited at all in the last 60 rounds: [18, 20, 15] of 100 per seed
- of those 53 clients (all seeds): 15% weak; median 40 of 60 rounds online; ruled out by the safety layer in a median 98% of their online rounds; 47 ruled out in >= 90% (avoided for good)
- were they right to be avoided? Of the 47 avoided for good, some mode would have succeeded in a median 20% of their online rounds (simulator ground truth); 1 could have succeeded in at least half of them (avoided wrongly)
- last 60 rounds, every online client-round the safety layer ruled out (3788): a job was possible in 24% (Light alone in 24%); for allowed client-rounds 64%

Recovery of the hit clients after round 140: 91% invited again; median wait 9.0 rounds; success rate 0.556 before the shock vs 0.655 from round 160 on.
