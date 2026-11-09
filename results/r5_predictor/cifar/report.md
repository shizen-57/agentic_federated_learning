# Predictor check: cifar, seeds [403, 404, 405], 200 rounds

Predicted / observed success of the invited jobs after the first quarter of the run. Lower Brier and calibration error are better.

**Random clients, random modes (same invitations for every version)**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.149 | 0.046 | 0.51 / 0.51 | 0.31 / 0.31 | 0.16 / 0.15 | 0.23 / 0.24 | 0.11 / 0.05 |
| v2 | 0.141 | 0.048 | 0.51 / 0.51 | 0.31 / 0.31 | 0.16 / 0.15 | 0.23 / 0.24 | 0.09 / 0.05 |
| v3 | 0.141 | 0.053 | 0.52 / 0.51 | 0.31 / 0.31 | 0.16 / 0.15 | 0.24 / 0.24 | 0.08 / 0.05 |

**Random clients, the version's own expected-work mode**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.220 | 0.045 | 0.42 / 0.43 | 0.57 / 0.54 | 0.64 / 0.56 | 0.45 / 0.46 | 0.24 / 0.16 |
| v2 | 0.209 | 0.046 | 0.40 / 0.41 | 0.61 / 0.58 | 0.69 / 0.62 | 0.46 / 0.46 | 0.17 / 0.16 |
| v3 | 0.207 | 0.051 | 0.40 / 0.41 | 0.65 / 0.62 | 0.68 / 0.60 | 0.48 / 0.46 | 0.17 / 0.15 |

**The expected-work rule applied to every online client** (judged by each round's true outcome; from the invitations of the `rule` case)

| predictor | Light / Standard / Priority share | jobs that would succeed | epochs delivered per unit cost |
|---|---|---|---|
| v1 | 75% / 11% / 14% | 46% | 0.70 |
| v2 | 68% / 18% / 14% | 47% | 0.74 |
| v3 | 64% / 22% / 15% | 48% | 0.78 |

**Ways of choosing the job, applied to every online client** (predictor v2; judged by each round's true outcome; the oracle skips clients that cannot finish any job)

| rule | Light / Standard / Priority share | jobs that succeed | epochs delivered per unit cost | successful jobs per unit cost |
|---|---|---|---|---|
| always Light | 100% / 0% / 0% | 51% | 0.68 | 0.68 |
| expected work (the baselines' rule) | 68% / 18% / 14% | 47% | 0.74 | 0.42 |
| expected work per unit cost | 98% / 2% / 0% | 51% | 0.67 | 0.66 |
| oracle: the heaviest job that finishes | 37% / 32% / 31% | 100% | 1.58 | 0.71 |
