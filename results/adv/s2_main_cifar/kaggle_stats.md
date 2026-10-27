# /kaggle/working/adv/s2_main_cifar/: c2_rl_fedprox vs others (paired by seed; diff = ref - other; Holm-corrected within each metric; two-sided)


## final_accuracy (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 0.4773 | 0.5098 | -0.0325 | [-0.1250, +0.0319] | 0.6680 | 1.0000 | no significant difference |
| arl3_full | 10 | 0.4773 | 0.5292 | -0.0519 | [-0.1325, +0.0070] | 0.2422 | 1.0000 | no significant difference |
| arl3_shared | 10 | 0.4773 | 0.4872 | -0.0099 | [-0.0801, +0.0413] | 0.9414 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 0.4773 | 0.4738 | +0.0035 | [-0.0882, +0.0810] | 0.9434 | 1.0000 | no significant difference |
| b_delta | 10 | 0.4773 | 0.4861 | -0.0088 | [-0.0969, +0.0623] | 0.8730 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 0.4773 | 0.5223 | -0.0450 | [-0.1355, +0.0295] | 0.3828 | 1.0000 | no significant difference |
| b_favor | 10 | 0.4773 | 0.5481 | -0.0708 | [-0.1629, +0.0072] | 0.1641 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 0.4773 | 0.5195 | -0.0423 | [-0.1456, +0.0439] | 0.5078 | 1.0000 | no significant difference |
| b_fedcer | 10 | 0.4773 | 0.4589 | +0.0184 | [-0.0881, +0.1103] | 0.7461 | 1.0000 | no significant difference |
| b_fedcs | 10 | 0.4773 | 0.5071 | -0.0299 | [-0.1189, +0.0448] | 0.5586 | 1.0000 | no significant difference |
| b_mimic | 10 | 0.4773 | 0.5769 | -0.0997 | [-0.1925, -0.0315] | 0.0020 | 0.0430 | ref worse |
| b_oort | 10 | 0.4773 | 0.5237 | -0.0465 | [-0.1347, +0.0369] | 0.3359 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 0.4773 | 0.5792 | -0.1019 | [-0.1991, -0.0197] | 0.0508 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 0.4773 | 0.5375 | -0.0603 | [-0.1690, +0.0323] | 0.3301 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 0.4773 | 0.4381 | +0.0391 | [-0.0643, +0.1266] | 0.4531 | 1.0000 | no significant difference |
| b_random | 10 | 0.4773 | 0.5178 | -0.0405 | [-0.1423, +0.0471] | 0.4863 | 1.0000 | no significant difference |
| b_random_ew | 10 | 0.4773 | 0.5607 | -0.0834 | [-0.1797, +0.0019] | 0.1113 | 1.0000 | no significant difference |
| b_refl | 10 | 0.4773 | 0.5166 | -0.0394 | [-0.1594, +0.0649] | 0.5977 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 0.4773 | 0.5605 | -0.0833 | [-0.1789, -0.0031] | 0.1055 | 1.0000 | no significant difference |
| rl_ew | 10 | 0.4773 | 0.5117 | -0.0345 | [-0.1165, +0.0506] | 0.4609 | 1.0000 | no significant difference |
| rl_light | 10 | 0.4773 | 0.5405 | -0.0633 | [-0.1716, +0.0281] | 0.2988 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 0.4773 | 0.4655 | +0.0118 | [-0.0700, +0.0851] | 0.7969 | 1.0000 | no significant difference |

## final_acc_worst10 (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 0.1369 | 0.1494 | -0.0125 | [-0.0627, +0.0430] | 0.6855 | 1.0000 | no significant difference |
| arl3_full | 10 | 0.1369 | 0.1987 | -0.0618 | [-0.1353, +0.0183] | 0.1699 | 1.0000 | no significant difference |
| arl3_shared | 10 | 0.1369 | 0.1946 | -0.0577 | [-0.1059, -0.0058] | 0.0625 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 0.1369 | 0.1804 | -0.0435 | [-0.1089, +0.0168] | 0.2559 | 1.0000 | no significant difference |
| b_delta | 10 | 0.1369 | 0.1503 | -0.0134 | [-0.0919, +0.0721] | 0.7559 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 0.1369 | 0.2053 | -0.0685 | [-0.1479, +0.0031] | 0.1309 | 1.0000 | no significant difference |
| b_favor | 10 | 0.1369 | 0.2125 | -0.0756 | [-0.1835, +0.0332] | 0.2305 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 0.1369 | 0.1887 | -0.0518 | [-0.1372, +0.0389] | 0.2910 | 1.0000 | no significant difference |
| b_fedcer | 10 | 0.1369 | 0.1248 | +0.0121 | [-0.0543, +0.0893] | 0.7676 | 1.0000 | no significant difference |
| b_fedcs | 10 | 0.1369 | 0.1811 | -0.0442 | [-0.1409, +0.0560] | 0.4082 | 1.0000 | no significant difference |
| b_mimic | 10 | 0.1369 | 0.3209 | -0.1840 | [-0.2543, -0.1073] | 0.0039 | 0.0859 | no significant difference |
| b_oort | 10 | 0.1369 | 0.1538 | -0.0169 | [-0.1186, +0.0826] | 0.7793 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 0.1369 | 0.2900 | -0.1531 | [-0.2283, -0.0780] | 0.0078 | 0.1641 | no significant difference |
| b_poc_ew | 10 | 0.1369 | 0.1677 | -0.0308 | [-0.1118, +0.0510] | 0.5117 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 0.1369 | 0.0962 | +0.0407 | [-0.0632, +0.1435] | 0.4980 | 1.0000 | no significant difference |
| b_random | 10 | 0.1369 | 0.1706 | -0.0337 | [-0.1244, +0.0661] | 0.5215 | 1.0000 | no significant difference |
| b_random_ew | 10 | 0.1369 | 0.2172 | -0.0803 | [-0.1793, +0.0237] | 0.1777 | 1.0000 | no significant difference |
| b_refl | 10 | 0.1369 | 0.1451 | -0.0082 | [-0.0813, +0.0697] | 0.8574 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 0.1369 | 0.2036 | -0.0667 | [-0.1399, +0.0198] | 0.1582 | 1.0000 | no significant difference |
| rl_ew | 10 | 0.1369 | 0.1514 | -0.0146 | [-0.0856, +0.0670] | 0.7520 | 1.0000 | no significant difference |
| rl_light | 10 | 0.1369 | 0.2150 | -0.0781 | [-0.1767, +0.0219] | 0.1641 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 0.1369 | 0.1612 | -0.0243 | [-0.1173, +0.0647] | 0.6289 | 1.0000 | no significant difference |

## mean_accuracy_fixed (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 0.3448 | 0.3769 | -0.0322 | [-0.0883, +0.0082] | 0.2891 | 1.0000 | no significant difference |
| arl3_full | 10 | 0.3448 | 0.3963 | -0.0516 | [-0.1137, -0.0077] | 0.0547 | 1.0000 | no significant difference |
| arl3_shared | 10 | 0.3448 | 0.3520 | -0.0072 | [-0.0411, +0.0171] | 0.7949 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 0.3448 | 0.3334 | +0.0114 | [-0.0897, +0.1063] | 0.8379 | 1.0000 | no significant difference |
| b_delta | 10 | 0.3448 | 0.3793 | -0.0345 | [-0.1288, +0.0508] | 0.5000 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 0.3448 | 0.3858 | -0.0411 | [-0.1333, +0.0422] | 0.3984 | 1.0000 | no significant difference |
| b_favor | 10 | 0.3448 | 0.4227 | -0.0779 | [-0.1709, +0.0047] | 0.1445 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 0.3448 | 0.4206 | -0.0758 | [-0.1641, +0.0031] | 0.1289 | 1.0000 | no significant difference |
| b_fedcer | 10 | 0.3448 | 0.3786 | -0.0338 | [-0.1285, +0.0532] | 0.5059 | 1.0000 | no significant difference |
| b_fedcs | 10 | 0.3448 | 0.3897 | -0.0450 | [-0.1384, +0.0432] | 0.3848 | 1.0000 | no significant difference |
| b_mimic | 10 | 0.3448 | 0.4648 | -0.1200 | [-0.2089, -0.0391] | 0.0020 | 0.0430 | ref worse |
| b_oort | 10 | 0.3448 | 0.4614 | -0.1166 | [-0.2099, -0.0335] | 0.0312 | 0.6250 | no significant difference |
| b_oort_ew | 10 | 0.3448 | 0.4889 | -0.1441 | [-0.2357, -0.0619] | 0.0020 | 0.0430 | ref worse |
| b_poc_ew | 10 | 0.3448 | 0.4128 | -0.0680 | [-0.1619, +0.0168] | 0.1836 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 0.3448 | 0.3703 | -0.0255 | [-0.1187, +0.0586] | 0.6094 | 1.0000 | no significant difference |
| b_random | 10 | 0.3448 | 0.4207 | -0.0760 | [-0.1675, +0.0047] | 0.1621 | 1.0000 | no significant difference |
| b_random_ew | 10 | 0.3448 | 0.4268 | -0.0821 | [-0.1717, -0.0019] | 0.1270 | 1.0000 | no significant difference |
| b_refl | 10 | 0.3448 | 0.4072 | -0.0624 | [-0.1553, +0.0204] | 0.2207 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 0.3448 | 0.4359 | -0.0912 | [-0.1829, -0.0092] | 0.1172 | 1.0000 | no significant difference |
| rl_ew | 10 | 0.3448 | 0.3424 | +0.0023 | [-0.0768, +0.0831] | 0.9570 | 1.0000 | no significant difference |
| rl_light | 10 | 0.3448 | 0.4155 | -0.0707 | [-0.1599, +0.0090] | 0.1699 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 0.3448 | 0.3406 | +0.0042 | [-0.0271, +0.0334] | 0.5742 | 1.0000 | no significant difference |

## final_cv (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 1.2968 | 1.1820 | +0.1148 | [-0.2706, +0.6068] | 0.7441 | 1.0000 | no significant difference |
| arl3_full | 10 | 1.2968 | 1.1044 | +0.1924 | [-0.2416, +0.7558] | 0.6113 | 1.0000 | no significant difference |
| arl3_shared | 10 | 1.2968 | 1.3140 | -0.0171 | [-0.4215, +0.5305] | 0.9668 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 1.2968 | 1.8960 | -0.5992 | [-0.9548, -0.0923] | 0.0449 | 0.7188 | no significant difference |
| b_delta | 10 | 1.2968 | 1.0618 | +0.2350 | [-0.0971, +0.6984] | 0.3828 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 1.2968 | 0.6814 | +0.6154 | [+0.2860, +1.0643] | 0.0020 | 0.0430 | ref worse |
| b_favor | 10 | 1.2968 | 1.2675 | +0.0293 | [-0.3048, +0.4864] | 0.9551 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 1.2968 | 0.9542 | +0.3426 | [+0.0016, +0.7741] | 0.1250 | 1.0000 | no significant difference |
| b_fedcer | 10 | 1.2968 | 2.5649 | -1.2681 | [-1.6302, -0.7672] | 0.0039 | 0.0703 | no significant difference |
| b_fedcs | 10 | 1.2968 | 2.1612 | -0.8644 | [-1.1993, -0.4560] | 0.0078 | 0.1328 | no significant difference |
| b_mimic | 10 | 1.2968 | 1.0457 | +0.2511 | [-0.0643, +0.6959] | 0.2832 | 1.0000 | no significant difference |
| b_oort | 10 | 1.2968 | 1.3766 | -0.0798 | [-0.4023, +0.3641] | 0.7773 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 1.2968 | 1.1726 | +0.1242 | [-0.1864, +0.5449] | 0.7109 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 1.2968 | 0.9209 | +0.3759 | [+0.0484, +0.8234] | 0.0547 | 0.8203 | no significant difference |
| b_powerofchoice | 10 | 1.2968 | 1.4045 | -0.1077 | [-0.4625, +0.3745] | 0.6973 | 1.0000 | no significant difference |
| b_random | 10 | 1.2968 | 1.0454 | +0.2514 | [-0.0640, +0.6962] | 0.2832 | 1.0000 | no significant difference |
| b_random_ew | 10 | 1.2968 | 0.6431 | +0.6537 | [+0.3306, +1.0879] | 0.0020 | 0.0430 | ref worse |
| b_refl | 10 | 1.2968 | 0.8270 | +0.4698 | [+0.1520, +0.9144] | 0.0020 | 0.0430 | ref worse |
| b_refl_ew | 10 | 1.2968 | 0.3944 | +0.9024 | [+0.5674, +1.3406] | 0.0020 | 0.0430 | ref worse |
| rl_ew | 10 | 1.2968 | 1.3002 | -0.0034 | [-0.3532, +0.3225] | 1.0000 | 1.0000 | no significant difference |
| rl_light | 10 | 1.2968 | 1.0335 | +0.2633 | [-0.0327, +0.6291] | 0.1777 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 1.2968 | 1.5349 | -0.2381 | [-0.8263, +0.3959] | 0.4629 | 1.0000 | no significant difference |

## mean_fail_ratio (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 0.5148 | 0.4334 | +0.0814 | [-0.0031, +0.2002] | 0.1953 | 1.0000 | no significant difference |
| arl3_full | 10 | 0.5148 | 0.4355 | +0.0793 | [-0.0150, +0.2069] | 0.2324 | 1.0000 | no significant difference |
| arl3_shared | 10 | 0.5148 | 0.5261 | -0.0112 | [-0.0616, +0.0580] | 0.7910 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 0.5148 | 0.5525 | -0.0377 | [-0.2223, +0.1606] | 0.7031 | 1.0000 | no significant difference |
| b_delta | 10 | 0.5148 | 0.6517 | -0.1369 | [-0.3015, +0.0470] | 0.1836 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 0.5148 | 0.5467 | -0.0319 | [-0.1961, +0.1494] | 0.7441 | 1.0000 | no significant difference |
| b_favor | 10 | 0.5148 | 0.6199 | -0.1050 | [-0.2636, +0.0703] | 0.2676 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 0.5148 | 0.5528 | -0.0380 | [-0.2055, +0.1455] | 0.6895 | 1.0000 | no significant difference |
| b_fedcer | 10 | 0.5148 | 0.5458 | -0.0310 | [-0.2034, +0.1584] | 0.7480 | 1.0000 | no significant difference |
| b_fedcs | 10 | 0.5148 | 0.2300 | +0.2848 | [+0.0901, +0.4792] | 0.0293 | 0.5859 | no significant difference |
| b_mimic | 10 | 0.5148 | 0.6489 | -0.1340 | [-0.2954, +0.0455] | 0.1777 | 1.0000 | no significant difference |
| b_oort | 10 | 0.5148 | 0.6125 | -0.0976 | [-0.2603, +0.0825] | 0.3008 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 0.5148 | 0.4210 | +0.0938 | [-0.0630, +0.2695] | 0.3164 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 0.5148 | 0.6259 | -0.1111 | [-0.2753, +0.0709] | 0.2500 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 0.5148 | 0.7871 | -0.2723 | [-0.4384, -0.0887] | 0.0234 | 0.4922 | no significant difference |
| b_random | 10 | 0.5148 | 0.6483 | -0.1335 | [-0.2947, +0.0459] | 0.1777 | 1.0000 | no significant difference |
| b_random_ew | 10 | 0.5148 | 0.5476 | -0.0328 | [-0.1960, +0.1470] | 0.7207 | 1.0000 | no significant difference |
| b_refl | 10 | 0.5148 | 0.7272 | -0.2124 | [-0.3750, -0.0317] | 0.0586 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 0.5148 | 0.5778 | -0.0630 | [-0.2301, +0.1213] | 0.5078 | 1.0000 | no significant difference |
| rl_ew | 10 | 0.5148 | 0.5657 | -0.0509 | [-0.2138, +0.1050] | 0.5605 | 1.0000 | no significant difference |
| rl_light | 10 | 0.5148 | 0.2741 | +0.2407 | [+0.0753, +0.4234] | 0.0078 | 0.1719 | no significant difference |
| v2_rawfail | 10 | 0.5148 | 0.5317 | -0.0168 | [-0.0874, +0.0480] | 0.6016 | 1.0000 | no significant difference |

## mean_arl_oracle (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 5.8990 | 5.9535 | -0.0545 | [-0.4672, +0.3028] | 0.8242 | 1.0000 | no significant difference |
| arl3_full | 10 | 5.8990 | 5.8210 | +0.0780 | [-0.2521, +0.5070] | 0.8008 | 1.0000 | no significant difference |
| arl3_shared | 10 | 5.8990 | 6.1094 | -0.2104 | [-0.5102, +0.1783] | 0.2852 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 5.8990 | 3.4917 | +2.4073 | [+1.3324, +3.3620] | 0.0059 | 0.1113 | no significant difference |
| b_delta | 10 | 5.8990 | 6.5776 | -0.6786 | [-1.0065, -0.3305] | 0.0156 | 0.2656 | no significant difference |
| b_delta_ew | 10 | 5.8990 | 6.3689 | -0.4699 | [-0.7833, -0.1283] | 0.0293 | 0.4688 | no significant difference |
| b_favor | 10 | 5.8990 | 6.1292 | -0.2302 | [-0.5829, +0.1596] | 0.2754 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 5.8990 | 5.9117 | -0.0127 | [-0.3939, +0.3710] | 0.9473 | 1.0000 | no significant difference |
| b_fedcer | 10 | 5.8990 | 0.9038 | +4.9952 | [+4.4129, +5.5938] | 0.0020 | 0.0430 | ref worse |
| b_fedcs | 10 | 5.8990 | 5.4328 | +0.4662 | [-0.0538, +1.0260] | 0.1426 | 1.0000 | no significant difference |
| b_mimic | 10 | 5.8990 | 6.1560 | -0.2570 | [-0.5882, +0.0973] | 0.2031 | 1.0000 | no significant difference |
| b_oort | 10 | 5.8990 | 6.3034 | -0.4044 | [-0.7839, +0.0023] | 0.0918 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 5.8990 | 5.9998 | -0.1008 | [-0.4768, +0.3035] | 0.6562 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 5.8990 | 6.3426 | -0.4436 | [-0.8002, -0.0643] | 0.0566 | 0.7930 | no significant difference |
| b_powerofchoice | 10 | 5.8990 | 6.7122 | -0.8132 | [-1.1845, -0.4027] | 0.0098 | 0.1758 | no significant difference |
| b_random | 10 | 5.8990 | 6.1553 | -0.2563 | [-0.5875, +0.0977] | 0.2031 | 1.0000 | no significant difference |
| b_random_ew | 10 | 5.8990 | 5.9407 | -0.0417 | [-0.3907, +0.3314] | 0.8301 | 1.0000 | no significant difference |
| b_refl | 10 | 5.8990 | 7.2024 | -1.3034 | [-1.6789, -0.9472] | 0.0020 | 0.0430 | ref better |
| b_refl_ew | 10 | 5.8990 | 7.3849 | -1.4859 | [-1.8350, -1.1095] | 0.0020 | 0.0430 | ref better |
| rl_ew | 10 | 5.8990 | 6.1276 | -0.2286 | [-0.6978, +0.2094] | 0.3730 | 1.0000 | no significant difference |
| rl_light | 10 | 5.8990 | 5.6823 | +0.2167 | [-0.2125, +0.6469] | 0.3633 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 5.8990 | 6.1144 | -0.2154 | [-0.3907, -0.0471] | 0.0391 | 0.5859 | no significant difference |

## mean_cost (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 7.5512 | 8.6597 | -1.1085 | [-2.6698, +0.2999] | 0.2207 | 1.0000 | no significant difference |
| arl3_full | 10 | 7.5512 | 12.0637 | -4.5124 | [-6.8318, -2.1332] | 0.0117 | 0.1992 | no significant difference |
| arl3_shared | 10 | 7.5512 | 8.7852 | -1.2340 | [-2.8233, +0.3578] | 0.1777 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 7.5512 | 10.6029 | -3.0516 | [-6.7348, +0.2778] | 0.1387 | 1.0000 | no significant difference |
| b_delta | 10 | 7.5512 | 9.9522 | -2.4010 | [-4.9469, +0.0944] | 0.1172 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 7.5512 | 11.7465 | -4.1953 | [-6.6735, -1.8987] | 0.0078 | 0.1406 | no significant difference |
| b_favor | 10 | 7.5512 | 10.5341 | -2.9829 | [-5.5448, -0.5059] | 0.0605 | 0.8477 | no significant difference |
| b_favor_ew | 10 | 7.5512 | 11.7088 | -4.1575 | [-6.4944, -1.9181] | 0.0059 | 0.1172 | no significant difference |
| b_fedcer | 10 | 7.5512 | 12.1360 | -4.5847 | [-7.5285, -1.7568] | 0.0176 | 0.2812 | no significant difference |
| b_fedcs | 10 | 7.5512 | 6.9962 | +0.5551 | [-1.8980, +2.8636] | 0.6836 | 1.0000 | no significant difference |
| b_mimic | 10 | 7.5512 | 10.3739 | -2.8227 | [-5.3611, -0.3639] | 0.0684 | 0.8887 | no significant difference |
| b_oort | 10 | 7.5512 | 16.4650 | -8.9138 | [-11.3699, -6.5783] | 0.0020 | 0.0430 | ref better |
| b_oort_ew | 10 | 7.5512 | 14.3843 | -6.8331 | [-9.3107, -4.4913] | 0.0020 | 0.0430 | ref better |
| b_poc_ew | 10 | 7.5512 | 10.2801 | -2.7288 | [-5.2505, -0.3049] | 0.0801 | 0.8887 | no significant difference |
| b_powerofchoice | 10 | 7.5512 | 8.0296 | -0.4783 | [-3.0735, +2.1115] | 0.7383 | 1.0000 | no significant difference |
| b_random | 10 | 7.5512 | 10.3739 | -2.8227 | [-5.3611, -0.3639] | 0.0684 | 0.8887 | no significant difference |
| b_random_ew | 10 | 7.5512 | 12.0275 | -4.4763 | [-6.8879, -2.2430] | 0.0059 | 0.1172 | no significant difference |
| b_refl | 10 | 7.5512 | 7.3331 | +0.2181 | [-2.3354, +2.6870] | 0.8730 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 7.5512 | 9.3523 | -1.8011 | [-4.2130, +0.4652] | 0.1738 | 1.0000 | no significant difference |
| rl_ew | 10 | 7.5512 | 11.2687 | -3.7174 | [-6.2720, -0.8984] | 0.0391 | 0.5859 | no significant difference |
| rl_light | 10 | 7.5512 | 7.2835 | +0.2678 | [-2.2436, +2.6343] | 0.8477 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 7.5512 | 7.3512 | +0.2000 | [-1.5585, +2.0045] | 0.8379 | 1.0000 | no significant difference |

## mean_latency (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 3.2757 | 2.0750 | +1.2008 | [+0.0333, +2.8244] | 0.1680 | 1.0000 | no significant difference |
| arl3_full | 10 | 3.2757 | 2.1086 | +1.1671 | [-0.0722, +2.9139] | 0.1895 | 1.0000 | no significant difference |
| arl3_shared | 10 | 3.2757 | 3.0276 | +0.2481 | [-0.6068, +1.4813] | 0.8711 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 3.2757 | 1.9667 | +1.3090 | [-0.2285, +3.1753] | 0.2129 | 1.0000 | no significant difference |
| b_delta | 10 | 3.2757 | 4.2801 | -1.0044 | [-2.5537, +0.8861] | 0.2930 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 3.2757 | 2.3834 | +0.8923 | [-0.6301, +2.7686] | 0.4160 | 1.0000 | no significant difference |
| b_favor | 10 | 3.2757 | 4.3189 | -1.0432 | [-2.5336, +0.8290] | 0.2754 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 3.2757 | 2.4221 | +0.8536 | [-0.6714, +2.7336] | 0.4297 | 1.0000 | no significant difference |
| b_fedcer | 10 | 3.2757 | 1.9899 | +1.2858 | [-0.3145, +3.2203] | 0.2266 | 1.0000 | no significant difference |
| b_fedcs | 10 | 3.2757 | 0.7186 | +2.5571 | [+1.0317, +4.3681] | 0.0059 | 0.1230 | no significant difference |
| b_mimic | 10 | 3.2757 | 4.4235 | -1.1478 | [-2.7095, +0.7862] | 0.2617 | 1.0000 | no significant difference |
| b_oort | 10 | 3.2757 | 2.8925 | +0.3832 | [-1.1665, +2.2929] | 0.7363 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 3.2757 | 1.7037 | +1.5720 | [+0.0561, +3.4379] | 0.1250 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 3.2757 | 2.6776 | +0.5981 | [-0.9239, +2.4669] | 0.5938 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 3.2757 | 5.0458 | -1.7700 | [-3.3095, +0.1383] | 0.0996 | 1.0000 | no significant difference |
| b_random | 10 | 3.2757 | 4.4235 | -1.1478 | [-2.7095, +0.7862] | 0.2617 | 1.0000 | no significant difference |
| b_random_ew | 10 | 3.2757 | 2.4262 | +0.8496 | [-0.6727, +2.7342] | 0.4375 | 1.0000 | no significant difference |
| b_refl | 10 | 3.2757 | 4.8335 | -1.5577 | [-3.1088, +0.4643] | 0.1445 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 3.2757 | 2.5336 | +0.7421 | [-0.7897, +2.6767] | 0.5352 | 1.0000 | no significant difference |
| rl_ew | 10 | 3.2757 | 2.1123 | +1.1634 | [-0.2827, +2.8619] | 0.2129 | 1.0000 | no significant difference |
| rl_light | 10 | 3.2757 | 0.9968 | +2.2789 | [+0.7330, +4.1216] | 0.0020 | 0.0430 | ref worse |
| v2_rawfail | 10 | 3.2757 | 3.1653 | +0.1105 | [-0.6555, +1.0925] | 0.7930 | 1.0000 | no significant difference |

wrote /kaggle/working/adv/s2_main_cifar/stats_c2_rl_fedprox.csv
