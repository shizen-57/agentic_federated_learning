# Pre-specified s2 tests (scripts/prespec_s2.py --root /kaggle/working/adv; see docs/ADVANCED_RESULTS.md)

## Family A (greater, final acc_worst10)

| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| s2_cifar | arl3_full | v2_rawfail | 0.1537 | 0.1540 | -0.0002 | 20 | 0.5035 | 1.0000 | inconclusive |
| s2_femnist | arl3_full | v2_rawfail | 0.0502 | 0.0540 | -0.0038 | 20 | 0.6227 | 1.0000 | inconclusive |
| s2_churn_cifar | arl3_full | v2_rawfail | 0.1417 | 0.1889 | -0.0472 | 20 | 0.9776 | 1.0000 | inconclusive |
| s2_churn_femnist | arl3_full | v2_rawfail | 0.0694 | 0.0696 | -0.0002 | 20 | 0.5072 | 1.0000 | inconclusive |

## Family A2 (greater, final acc_client_worst10)

| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| s2_cifar | arl3_full | v2_rawfail | 0.3290 | 0.3310 | -0.0020 | 20 | 0.5412 | 1.0000 | inconclusive |
| s2_femnist | arl3_full | v2_rawfail | 0.4137 | 0.4386 | -0.0249 | 20 | 0.7761 | 1.0000 | inconclusive |
| s2_churn_cifar | arl3_full | v2_rawfail | 0.3122 | 0.3522 | -0.0399 | 20 | 0.9851 | 1.0000 | inconclusive |
| s2_churn_femnist | arl3_full | v2_rawfail | 0.4635 | 0.4528 | +0.0107 | 20 | 0.2364 | 0.9456 | inconclusive |

## Family B (less, mean fail_ratio)

| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| s2_cifar | arl2_full | rl_light | 0.4179 | 0.3244 | +0.0935 | 20 | 0.9948 | 1.0000 | inconclusive |
| s2_femnist | arl2_full | rl_light | 0.5455 | 0.3083 | +0.2372 | 20 | 1.0000 | 1.0000 | inconclusive |

## Family C (two-sided, final accuracy)

| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| s2_cifar | c2_rl_fedprox | b_favor | 0.5209 | 0.5140 | +0.0069 | 20 | 0.5834 | 0.8111 | inconclusive |
| s2_cifar | c2_rl_fedprox | b_favor_ew | 0.5209 | 0.5356 | -0.0147 | 20 | 0.2182 | 0.8111 | inconclusive |
| s2_cifar | c2_rl_fedprox | b_refl | 0.5209 | 0.5369 | -0.0160 | 20 | 0.2028 | 0.8111 | inconclusive |
| s2_cifar | c2_rl_fedprox | b_refl_ew | 0.5209 | 0.5684 | -0.0475 | 20 | 0.0003 | 0.0042 | significantly worse |
| s2_cifar | c2_rl_fedprox | b_delta | 0.5209 | 0.4584 | +0.0625 | 20 | 0.0000 | 0.0003 | significantly better |
| s2_cifar | c2_rl_fedprox | b_delta_ew | 0.5209 | 0.4875 | +0.0334 | 20 | 0.0022 | 0.0238 | significantly better |
| s2_cifar | c2_rl_fedprox | b_random_ew | 0.5209 | 0.5528 | -0.0319 | 20 | 0.0147 | 0.1180 | inconclusive |
| s2_cifar | c2_rl_fedprox | b_oort_ew | 0.5209 | 0.5638 | -0.0429 | 20 | 0.0273 | 0.1638 | inconclusive |
| s2_cifar | c2_rl_fedprox | b_poc_ew | 0.5209 | 0.5059 | +0.0150 | 20 | 0.3015 | 0.8111 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_favor | 0.5804 | 0.7219 | -0.1414 | 20 | 0.0212 | 0.1482 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_favor_ew | 0.5804 | 0.7480 | -0.1676 | 20 | 0.0037 | 0.0369 | significantly worse |
| s2_femnist | c2_rl_fedprox | b_refl | 0.5804 | 0.7498 | -0.1694 | 20 | 0.0067 | 0.0602 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_refl_ew | 0.5804 | 0.7690 | -0.1885 | 20 | 0.0010 | 0.0116 | significantly worse |
| s2_femnist | c2_rl_fedprox | b_delta | 0.5804 | 0.6858 | -0.1054 | 20 | 0.0666 | 0.3331 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_delta_ew | 0.5804 | 0.7582 | -0.1778 | 20 | 0.0002 | 0.0026 | significantly worse |
| s2_femnist | c2_rl_fedprox | b_random_ew | 0.5804 | 0.7767 | -0.1963 | 20 | 0.0000 | 0.0001 | significantly worse |
| s2_femnist | c2_rl_fedprox | b_oort_ew | 0.5804 | 0.7850 | -0.2046 | 20 | 0.0000 | 0.0001 | significantly worse |
| s2_femnist | c2_rl_fedprox | b_poc_ew | 0.5804 | 0.7648 | -0.1843 | 20 | 0.0000 | 0.0004 | significantly worse |

## Family C2 (two-sided, mean fail_ratio)

| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| s2_cifar | c2_rl_fedprox | b_favor | 0.4402 | 0.6752 | -0.2350 | 20 | 0.0000 | 0.0002 | significantly better |
| s2_cifar | c2_rl_fedprox | b_favor_ew | 0.4402 | 0.5622 | -0.1221 | 20 | 0.0023 | 0.0299 | significantly better |
| s2_cifar | c2_rl_fedprox | b_refl | 0.4402 | 0.7442 | -0.3040 | 20 | 0.0000 | 0.0001 | significantly better |
| s2_cifar | c2_rl_fedprox | b_refl_ew | 0.4402 | 0.5665 | -0.1263 | 20 | 0.0008 | 0.0115 | significantly better |
| s2_cifar | c2_rl_fedprox | b_delta | 0.4402 | 0.6924 | -0.2522 | 20 | 0.0000 | 0.0002 | significantly better |
| s2_cifar | c2_rl_fedprox | b_delta_ew | 0.4402 | 0.5449 | -0.1048 | 20 | 0.0040 | 0.0435 | significantly better |
| s2_cifar | c2_rl_fedprox | b_random_ew | 0.4402 | 0.5517 | -0.1115 | 20 | 0.0026 | 0.0313 | significantly better |
| s2_cifar | c2_rl_fedprox | b_oort_ew | 0.4402 | 0.4492 | -0.0090 | 20 | 0.7939 | 1.0000 | inconclusive |
| s2_cifar | c2_rl_fedprox | b_poc_ew | 0.4402 | 0.5979 | -0.1577 | 20 | 0.0003 | 0.0039 | significantly better |
| s2_femnist | c2_rl_fedprox | b_favor | 0.6428 | 0.6935 | -0.0506 | 20 | 0.3943 | 1.0000 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_favor_ew | 0.6428 | 0.5706 | +0.0723 | 20 | 0.2428 | 1.0000 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_refl | 0.6428 | 0.7644 | -0.1215 | 20 | 0.0638 | 0.5743 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_refl_ew | 0.6428 | 0.5707 | +0.0721 | 20 | 0.2386 | 1.0000 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_delta | 0.6428 | 0.7078 | -0.0650 | 20 | 0.2881 | 1.0000 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_delta_ew | 0.6428 | 0.5560 | +0.0868 | 20 | 0.1651 | 1.0000 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_random_ew | 0.6428 | 0.5598 | +0.0830 | 20 | 0.1833 | 1.0000 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_oort_ew | 0.6428 | 0.4838 | +0.1590 | 20 | 0.0173 | 0.1730 | inconclusive |
| s2_femnist | c2_rl_fedprox | b_poc_ew | 0.6428 | 0.6186 | +0.0243 | 20 | 0.6806 | 1.0000 | inconclusive |

## Family D (two-sided, final accuracy)

| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| s2_cifar | v2_rawfail | rl_ew | 0.5178 | 0.5195 | -0.0017 | 20 | 0.9412 | 1.0000 | inconclusive |
| s2_cifar | v2_rawfail | rl_light | 0.5178 | 0.5244 | -0.0066 | 20 | 0.5420 | 1.0000 | inconclusive |
| s2_cifar | v2_rawfail | b_cov_ew | 0.5178 | 0.4921 | +0.0257 | 20 | 0.0348 | 0.2088 | inconclusive |
| s2_femnist | v2_rawfail | rl_ew | 0.7117 | 0.7392 | -0.0275 | 20 | 0.1639 | 0.4917 | inconclusive |
| s2_femnist | v2_rawfail | rl_light | 0.7117 | 0.7468 | -0.0351 | 20 | 0.0557 | 0.2785 | inconclusive |
| s2_femnist | v2_rawfail | b_cov_ew | 0.7117 | 0.7409 | -0.0291 | 20 | 0.1178 | 0.4713 | inconclusive |

## Family D2 (two-sided, mean fail_ratio)

| group | ref | vs | ref mean | other mean | diff | n | p | p (Holm) | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| s2_cifar | v2_rawfail | rl_ew | 0.3977 | 0.5170 | -0.1193 | 20 | 0.0038 | 0.0151 | significantly better |
| s2_cifar | v2_rawfail | rl_light | 0.3977 | 0.3244 | +0.0733 | 20 | 0.0153 | 0.0460 | significantly worse |
| s2_cifar | v2_rawfail | b_cov_ew | 0.3977 | 0.5528 | -0.1552 | 20 | 0.0003 | 0.0014 | significantly better |
| s2_femnist | v2_rawfail | rl_ew | 0.5071 | 0.5550 | -0.0479 | 20 | 0.4102 | 0.8203 | inconclusive |
| s2_femnist | v2_rawfail | rl_light | 0.5071 | 0.3083 | +0.1988 | 20 | 0.0001 | 0.0008 | significantly worse |
| s2_femnist | v2_rawfail | b_cov_ew | 0.5071 | 0.5499 | -0.0429 | 20 | 0.4622 | 0.8203 | inconclusive |
