# Predictor check: femnist, seeds [403, 404, 405], 200 rounds, with ['system.drop_model=exposure', 'system.drop_ref_time=1.0', 'system.job_overhead=1.0', 'system.job_overhead_time=0.05', 'system.energy_by_work=true', 'system.cap_latency=true']

Predicted / observed success of the invited jobs after the first quarter of the run. Lower Brier and calibration error are better.

**Random clients, random modes (same invitations for every version)**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.109 | 0.044 | 0.52 / 0.53 | 0.32 / 0.32 | 0.15 / 0.13 | 0.20 / 0.13 | 0.09 / 0.03 |
| v2 | 0.102 | 0.039 | 0.52 / 0.53 | 0.32 / 0.32 | 0.16 / 0.13 | 0.17 / 0.13 | 0.06 / 0.03 |
| v3 | 0.099 | 0.035 | 0.53 / 0.53 | 0.32 / 0.32 | 0.14 / 0.13 | 0.13 / 0.13 | 0.06 / 0.03 |

**Random clients, the version's own expected-work mode**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.202 | 0.029 | 0.37 / 0.37 | 0.59 / 0.58 | 0.69 / 0.64 | 0.41 / 0.36 | 0.20 / 0.12 |
| v2 | 0.187 | 0.032 | 0.38 / 0.38 | 0.62 / 0.60 | 0.72 / 0.69 | 0.40 / 0.37 | 0.13 / 0.12 |
| v3 | 0.184 | 0.034 | 0.38 / 0.37 | 0.66 / 0.66 | 0.71 / 0.68 | 0.36 / 0.37 | 0.14 / 0.12 |

**The expected-work rule applied to every online client** (judged by each round's true outcome; from the invitations of the `rule` case)

| predictor | Light / Standard / Priority share | jobs that would succeed | epochs delivered per unit cost |
|---|---|---|---|
| v1 | 68% / 16% / 17% | 44% | 0.39 |
| v2 | 69% / 16% / 15% | 46% | 0.40 |
| v3 | 67% / 19% / 14% | 47% | 0.41 |

**Ways of choosing the job, applied to every online client** (predictor v2; judged by each round's true outcome; the oracle skips clients that cannot finish any job)

| rule | Light / Standard / Priority share | jobs that succeed | epochs delivered per unit cost | successful jobs per unit cost |
|---|---|---|---|---|
| always Light | 100% / 0% / 0% | 54% | 0.31 | 0.31 |
| expected work (the baselines' rule) | 69% / 16% / 15% | 46% | 0.40 | 0.21 |
| expected work per unit cost | 68% / 21% / 11% | 49% | 0.40 | 0.23 |
| oracle: the heaviest job that finishes | 41% / 34% / 24% | 100% | 0.86 | 0.42 |
