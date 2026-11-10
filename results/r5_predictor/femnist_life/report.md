# Predictor check: femnist, seeds [403, 404, 405], 200 rounds, with ['system.drop_model=exposure', 'system.drop_ref_time=1.0', 'system.job_overhead=1.0', 'system.job_overhead_time=0.05', 'system.energy_by_work=true', 'system.cap_latency=true', 'system.state_model=diurnal']

Predicted / observed success of the invited jobs after the first quarter of the run. Lower Brier and calibration error are better.

**Random clients, random modes (same invitations for every version)**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.101 | 0.045 | 0.47 / 0.48 | 0.27 / 0.27 | 0.14 / 0.11 | 0.19 / 0.12 | 0.09 / 0.06 |
| v2 | 0.099 | 0.027 | 0.47 / 0.48 | 0.27 / 0.27 | 0.14 / 0.11 | 0.17 / 0.12 | 0.06 / 0.06 |
| v3 | 0.092 | 0.029 | 0.49 / 0.48 | 0.27 / 0.27 | 0.12 / 0.11 | 0.13 / 0.12 | 0.07 / 0.06 |

**Random clients, the version's own expected-work mode**

| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs | weak clients pred / obs | slow devices pred / obs |
|---|---|---|---|---|---|---|---|
| v1 | 0.185 | 0.047 | 0.33 / 0.34 | 0.62 / 0.60 | 0.73 / 0.64 | 0.37 / 0.27 | 0.20 / 0.16 |
| v2 | 0.179 | 0.036 | 0.33 / 0.33 | 0.61 / 0.59 | 0.75 / 0.67 | 0.36 / 0.29 | 0.16 / 0.17 |
| v3 | 0.168 | 0.023 | 0.34 / 0.33 | 0.68 / 0.66 | 0.73 / 0.70 | 0.33 / 0.30 | 0.18 / 0.16 |

**The expected-work rule applied to every online client** (judged by each round's true outcome; from the invitations of the `rule` case)

| predictor | Light / Standard / Priority share | jobs that would succeed | epochs delivered per unit cost |
|---|---|---|---|
| v1 | 71% / 16% / 13% | 42% | 0.36 |
| v2 | 70% / 18% / 12% | 42% | 0.37 |
| v3 | 69% / 20% / 11% | 44% | 0.38 |

**Ways of choosing the job, applied to every online client** (predictor v2; judged by each round's true outcome; the oracle skips clients that cannot finish any job)

| rule | Light / Standard / Priority share | jobs that succeed | epochs delivered per unit cost | successful jobs per unit cost |
|---|---|---|---|---|
| always Light | 100% / 0% / 0% | 49% | 0.28 | 0.28 |
| expected work (the baselines' rule) | 70% / 18% / 12% | 42% | 0.37 | 0.20 |
| expected work per unit cost | 65% / 24% / 10% | 41% | 0.35 | 0.20 |
| oracle: the heaviest job that finishes | 45% / 33% / 22% | 100% | 0.85 | 0.42 |
