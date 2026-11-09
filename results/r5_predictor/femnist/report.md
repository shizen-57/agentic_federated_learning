# Predictor check: femnist, seeds [403, 404, 405], 200 rounds

Predicted / observed success of the invited jobs after the first quarter of the run. Lower Brier and calibration error are better.

**Random clients, random modes (same invitations for every version)**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.131 | 0.049 | 0.49 / 0.50 | 0.31 / 0.31 | 0.15 / 0.13 | 0.20 / 0.14 | 0.10 / 0.06 |
| v2 | 0.123 | 0.046 | 0.49 / 0.50 | 0.31 / 0.31 | 0.15 / 0.13 | 0.18 / 0.14 | 0.08 / 0.06 |
| v3 | 0.124 | 0.043 | 0.50 / 0.50 | 0.31 / 0.31 | 0.14 / 0.13 | 0.15 / 0.14 | 0.07 / 0.06 |

**Random clients, the version's own expected-work mode**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.217 | 0.031 | 0.38 / 0.39 | 0.57 / 0.54 | 0.68 / 0.62 | 0.41 / 0.37 | 0.23 / 0.18 |
| v2 | 0.209 | 0.036 | 0.39 / 0.39 | 0.58 / 0.54 | 0.72 / 0.69 | 0.40 / 0.37 | 0.18 / 0.18 |
| v3 | 0.208 | 0.046 | 0.39 / 0.38 | 0.63 / 0.59 | 0.70 / 0.65 | 0.39 / 0.38 | 0.18 / 0.17 |

**The expected-work rule applied to every online client** (judged by each round's true outcome; from the invitations of the `rule` case)

| predictor | Light / Standard / Priority share | jobs that would succeed | epochs delivered per unit cost |
|---|---|---|---|
| v1 | 69% / 15% / 16% | 44% | 0.70 |
| v2 | 71% / 14% / 15% | 46% | 0.74 |
| v3 | 68% / 18% / 15% | 46% | 0.74 |

**Ways of choosing the job, applied to every online client** (predictor v2; judged by each round's true outcome; the oracle skips clients that cannot finish any job)

| rule | Light / Standard / Priority share | jobs that succeed | epochs delivered per unit cost | successful jobs per unit cost |
|---|---|---|---|---|
| always Light | 100% / 0% / 0% | 50% | 0.67 | 0.67 |
| expected work (the baselines' rule) | 71% / 14% / 15% | 46% | 0.74 | 0.40 |
| expected work per unit cost | 98% / 2% / 0% | 50% | 0.66 | 0.66 |
| oracle: the heaviest job that finishes | 40% / 33% / 26% | 100% | 1.47 | 0.69 |
