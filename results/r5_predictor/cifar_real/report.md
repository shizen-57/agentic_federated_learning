# Predictor check: cifar, seeds [403, 404, 405], 200 rounds, with ['system.drop_model=exposure', 'system.drop_ref_time=1.0', 'system.job_overhead=1.0', 'system.job_overhead_time=0.05', 'system.energy_by_work=true', 'system.cap_latency=true']

Predicted / observed success of the invited jobs after the first quarter of the run. Lower Brier and calibration error are better.

**Random clients, random modes (same invitations for every version)**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.124 | 0.032 | 0.57 / 0.58 | 0.34 / 0.34 | 0.17 / 0.15 | 0.26 / 0.27 | 0.11 / 0.05 |
| v2 | 0.112 | 0.034 | 0.56 / 0.58 | 0.34 / 0.34 | 0.17 / 0.15 | 0.25 / 0.27 | 0.08 / 0.05 |
| v3 | 0.104 | 0.033 | 0.58 / 0.58 | 0.34 / 0.34 | 0.16 / 0.15 | 0.27 / 0.27 | 0.08 / 0.05 |

**Random clients, the version's own expected-work mode**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.202 | 0.037 | 0.40 / 0.41 | 0.60 / 0.58 | 0.66 / 0.59 | 0.48 / 0.49 | 0.21 / 0.13 |
| v2 | 0.187 | 0.035 | 0.40 / 0.41 | 0.62 / 0.59 | 0.72 / 0.67 | 0.49 / 0.49 | 0.14 / 0.12 |
| v3 | 0.177 | 0.031 | 0.42 / 0.42 | 0.66 / 0.64 | 0.71 / 0.68 | 0.50 / 0.50 | 0.15 / 0.13 |

**The expected-work rule applied to every online client** (judged by each round's true outcome; from the invitations of the `rule` case)

| predictor | Light / Standard / Priority share | jobs that would succeed | epochs delivered per unit cost |
|---|---|---|---|
| v1 | 64% / 19% / 16% | 47% | 0.40 |
| v2 | 63% / 21% / 16% | 49% | 0.43 |
| v3 | 62% / 21% / 17% | 51% | 0.45 |

**Ways of choosing the job, applied to every online client** (predictor v2; judged by each round's true outcome; the oracle skips clients that cannot finish any job)

| rule | Light / Standard / Priority share | jobs that succeed | epochs delivered per unit cost | successful jobs per unit cost |
|---|---|---|---|---|
| always Light | 100% / 0% / 0% | 57% | 0.33 | 0.33 |
| expected work (the baselines' rule) | 63% / 21% / 16% | 49% | 0.43 | 0.23 |
| expected work per unit cost | 59% / 27% / 13% | 52% | 0.43 | 0.24 |
| oracle: the heaviest job that finishes | 40% / 32% / 28% | 100% | 0.92 | 0.43 |
