# /kaggle/working/adv/s2_cifar/: c2_rl_fedprox vs others (paired by seed; diff = ref - other; Holm-corrected within each metric; two-sided)


## final_accuracy (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 0.5209 | 0.5115 | +0.0094 | [-0.0073, +0.0283] | 0.3484 | 1.0000 | no significant difference |
| arl3_full | 20 | 0.5209 | 0.5219 | -0.0010 | [-0.0253, +0.0222] | 0.9348 | 1.0000 | no significant difference |
| arl3_shared | 20 | 0.5209 | 0.5284 | -0.0075 | [-0.0271, +0.0121] | 0.4765 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 0.5209 | 0.4921 | +0.0288 | [+0.0048, +0.0527] | 0.0334 | 0.4340 | no significant difference |
| b_delta | 20 | 0.5209 | 0.4584 | +0.0625 | [+0.0443, +0.0786] | 0.0000 | 0.0004 | ref better |
| b_delta_ew | 20 | 0.5209 | 0.4875 | +0.0334 | [+0.0157, +0.0513] | 0.0022 | 0.0367 | ref better |
| b_favor | 20 | 0.5209 | 0.5140 | +0.0069 | [-0.0159, +0.0308] | 0.5834 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 0.5209 | 0.5356 | -0.0147 | [-0.0375, +0.0063] | 0.2182 | 1.0000 | no significant difference |
| b_fedcer | 20 | 0.5209 | 0.4359 | +0.0851 | [+0.0582, +0.1117] | 0.0000 | 0.0004 | ref better |
| b_fedcs | 20 | 0.5209 | 0.4992 | +0.0217 | [+0.0020, +0.0401] | 0.0445 | 0.5343 | no significant difference |
| b_mimic | 20 | 0.5209 | 0.5689 | -0.0480 | [-0.0723, -0.0228] | 0.0015 | 0.0264 | ref worse |
| b_oort | 20 | 0.5209 | 0.4814 | +0.0395 | [+0.0084, +0.0722] | 0.0299 | 0.4179 | no significant difference |
| b_oort_ew | 20 | 0.5209 | 0.5638 | -0.0429 | [-0.0737, -0.0061] | 0.0273 | 0.4096 | no significant difference |
| b_poc_ew | 20 | 0.5209 | 0.5059 | +0.0150 | [-0.0129, +0.0420] | 0.3015 | 1.0000 | no significant difference |
| b_powerofchoice | 20 | 0.5209 | 0.4222 | +0.0987 | [+0.0664, +0.1289] | 0.0000 | 0.0007 | ref better |
| b_random | 20 | 0.5209 | 0.5101 | +0.0108 | [-0.0203, +0.0435] | 0.5255 | 1.0000 | no significant difference |
| b_random_ew | 20 | 0.5209 | 0.5528 | -0.0319 | [-0.0547, -0.0097] | 0.0147 | 0.2360 | no significant difference |
| b_refl | 20 | 0.5209 | 0.5369 | -0.0160 | [-0.0395, +0.0066] | 0.2028 | 1.0000 | no significant difference |
| b_refl_ew | 20 | 0.5209 | 0.5684 | -0.0475 | [-0.0706, -0.0255] | 0.0003 | 0.0061 | ref worse |
| rl_ew | 20 | 0.5209 | 0.5195 | +0.0014 | [-0.0285, +0.0388] | 0.9500 | 1.0000 | no significant difference |
| rl_light | 20 | 0.5209 | 0.5244 | -0.0035 | [-0.0275, +0.0188] | 0.7753 | 1.0000 | no significant difference |
| v2_rawfail | 20 | 0.5209 | 0.5178 | +0.0031 | [-0.0121, +0.0182] | 0.7026 | 1.0000 | no significant difference |

## final_acc_worst10 (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 0.1716 | 0.1427 | +0.0289 | [-0.0271, +0.0808] | 0.3180 | 1.0000 | no significant difference |
| arl3_full | 20 | 0.1716 | 0.1537 | +0.0179 | [-0.0444, +0.0793] | 0.5842 | 1.0000 | no significant difference |
| arl3_shared | 20 | 0.1716 | 0.1922 | -0.0206 | [-0.0701, +0.0311] | 0.4470 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 0.1716 | 0.1476 | +0.0240 | [-0.0216, +0.0715] | 0.3331 | 1.0000 | no significant difference |
| b_delta | 20 | 0.1716 | 0.0985 | +0.0731 | [+0.0270, +0.1199] | 0.0075 | 0.1506 | no significant difference |
| b_delta_ew | 20 | 0.1716 | 0.1354 | +0.0362 | [-0.0149, +0.0848] | 0.1729 | 1.0000 | no significant difference |
| b_favor | 20 | 0.1716 | 0.1558 | +0.0158 | [-0.0382, +0.0635] | 0.5635 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 0.1716 | 0.1893 | -0.0177 | [-0.0710, +0.0345] | 0.5289 | 1.0000 | no significant difference |
| b_fedcer | 20 | 0.1716 | 0.0962 | +0.0755 | [+0.0156, +0.1322] | 0.0248 | 0.4711 | no significant difference |
| b_fedcs | 20 | 0.1716 | 0.1808 | -0.0092 | [-0.0591, +0.0422] | 0.7301 | 1.0000 | no significant difference |
| b_mimic | 20 | 0.1716 | 0.2967 | -0.1251 | [-0.1921, -0.0526] | 0.0037 | 0.0769 | no significant difference |
| b_oort | 20 | 0.1716 | 0.1243 | +0.0473 | [-0.0014, +0.0985] | 0.0884 | 1.0000 | no significant difference |
| b_oort_ew | 20 | 0.1716 | 0.2366 | -0.0650 | [-0.1446, +0.0165] | 0.1412 | 1.0000 | no significant difference |
| b_poc_ew | 20 | 0.1716 | 0.1525 | +0.0191 | [-0.0440, +0.0781] | 0.5608 | 1.0000 | no significant difference |
| b_powerofchoice | 20 | 0.1716 | 0.0573 | +0.1143 | [+0.0640, +0.1643] | 0.0004 | 0.0089 | ref better |
| b_random | 20 | 0.1716 | 0.1483 | +0.0233 | [-0.0488, +0.0978] | 0.5482 | 1.0000 | no significant difference |
| b_random_ew | 20 | 0.1716 | 0.2278 | -0.0562 | [-0.1250, +0.0150] | 0.1404 | 1.0000 | no significant difference |
| b_refl | 20 | 0.1716 | 0.1400 | +0.0317 | [-0.0235, +0.0811] | 0.2707 | 1.0000 | no significant difference |
| b_refl_ew | 20 | 0.1716 | 0.2186 | -0.0470 | [-0.1159, +0.0220] | 0.2070 | 1.0000 | no significant difference |
| rl_ew | 20 | 0.1716 | 0.1586 | +0.0130 | [-0.0354, +0.0600] | 0.6045 | 1.0000 | no significant difference |
| rl_light | 20 | 0.1716 | 0.1441 | +0.0275 | [-0.0209, +0.0791] | 0.3005 | 1.0000 | no significant difference |
| v2_rawfail | 20 | 0.1716 | 0.1540 | +0.0176 | [-0.0245, +0.0602] | 0.4362 | 1.0000 | no significant difference |

## mean_accuracy_fixed (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 0.3790 | 0.3743 | +0.0047 | [-0.0087, +0.0251] | 0.8764 | 1.0000 | no significant difference |
| arl3_full | 20 | 0.3790 | 0.3866 | -0.0076 | [-0.0202, +0.0039] | 0.2513 | 1.0000 | no significant difference |
| arl3_shared | 20 | 0.3790 | 0.3876 | -0.0086 | [-0.0320, +0.0132] | 0.4925 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 0.3790 | 0.3138 | +0.0652 | [+0.0252, +0.1049] | 0.0061 | 0.1101 | no significant difference |
| b_delta | 20 | 0.3790 | 0.3446 | +0.0344 | [+0.0027, +0.0614] | 0.0383 | 0.6129 | no significant difference |
| b_delta_ew | 20 | 0.3790 | 0.3684 | +0.0106 | [-0.0199, +0.0367] | 0.4945 | 1.0000 | no significant difference |
| b_favor | 20 | 0.3790 | 0.3955 | -0.0165 | [-0.0453, +0.0083] | 0.2674 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 0.3790 | 0.4064 | -0.0274 | [-0.0570, -0.0019] | 0.0683 | 0.9106 | no significant difference |
| b_fedcer | 20 | 0.3790 | 0.3561 | +0.0229 | [-0.0084, +0.0517] | 0.1636 | 1.0000 | no significant difference |
| b_fedcs | 20 | 0.3790 | 0.3758 | +0.0032 | [-0.0345, +0.0371] | 0.8716 | 1.0000 | no significant difference |
| b_mimic | 20 | 0.3790 | 0.4505 | -0.0715 | [-0.1027, -0.0454] | 0.0000 | 0.0000 | ref worse |
| b_oort | 20 | 0.3790 | 0.4247 | -0.0457 | [-0.0753, -0.0196] | 0.0024 | 0.0472 | ref worse |
| b_oort_ew | 20 | 0.3790 | 0.4734 | -0.0944 | [-0.1248, -0.0680] | 0.0000 | 0.0000 | ref worse |
| b_poc_ew | 20 | 0.3790 | 0.4033 | -0.0243 | [-0.0538, +0.0019] | 0.1154 | 1.0000 | no significant difference |
| b_powerofchoice | 20 | 0.3790 | 0.3500 | +0.0290 | [-0.0007, +0.0549] | 0.0623 | 0.9106 | no significant difference |
| b_random | 20 | 0.3790 | 0.3983 | -0.0193 | [-0.0495, +0.0062] | 0.2115 | 1.0000 | no significant difference |
| b_random_ew | 20 | 0.3790 | 0.4125 | -0.0335 | [-0.0640, -0.0076] | 0.0244 | 0.4150 | no significant difference |
| b_refl | 20 | 0.3790 | 0.4070 | -0.0280 | [-0.0569, -0.0028] | 0.0607 | 0.9106 | no significant difference |
| b_refl_ew | 20 | 0.3790 | 0.4199 | -0.0409 | [-0.0710, -0.0152] | 0.0058 | 0.1093 | no significant difference |
| rl_ew | 20 | 0.3790 | 0.3609 | +0.0181 | [-0.0181, +0.0549] | 0.3514 | 1.0000 | no significant difference |
| rl_light | 20 | 0.3790 | 0.4004 | -0.0214 | [-0.0515, +0.0048] | 0.1667 | 1.0000 | no significant difference |
| v2_rawfail | 20 | 0.3790 | 0.3851 | -0.0061 | [-0.0212, +0.0077] | 0.4541 | 1.0000 | no significant difference |

## final_cv (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 1.5575 | 1.3001 | +0.2574 | [-0.0684, +0.5938] | 0.1531 | 1.0000 | no significant difference |
| arl3_full | 20 | 1.5575 | 1.0824 | +0.4751 | [+0.2466, +0.7181] | 0.0008 | 0.0110 | ref worse |
| arl3_shared | 20 | 1.5575 | 1.3974 | +0.1602 | [-0.0690, +0.3864] | 0.1945 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 1.5575 | 1.7380 | -0.1804 | [-0.4113, +0.0633] | 0.1645 | 1.0000 | no significant difference |
| b_delta | 20 | 1.5575 | 1.1348 | +0.4227 | [+0.1713, +0.6987] | 0.0056 | 0.0616 | no significant difference |
| b_delta_ew | 20 | 1.5575 | 0.6380 | +0.9195 | [+0.6579, +1.2067] | 0.0000 | 0.0000 | ref worse |
| b_favor | 20 | 1.5575 | 1.3494 | +0.2081 | [-0.0388, +0.4853] | 0.1502 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 1.5575 | 0.8673 | +0.6903 | [+0.4232, +0.9852] | 0.0000 | 0.0005 | ref worse |
| b_fedcer | 20 | 1.5575 | 2.4051 | -0.8476 | [-1.1286, -0.5384] | 0.0001 | 0.0020 | ref better |
| b_fedcs | 20 | 1.5575 | 2.3294 | -0.7719 | [-1.0272, -0.4937] | 0.0001 | 0.0019 | ref better |
| b_mimic | 20 | 1.5575 | 1.1286 | +0.4290 | [+0.1824, +0.7017] | 0.0039 | 0.0507 | no significant difference |
| b_oort | 20 | 1.5575 | 1.4994 | +0.0582 | [-0.1922, +0.3362] | 0.6807 | 1.0000 | no significant difference |
| b_oort_ew | 20 | 1.5575 | 1.1737 | +0.3839 | [+0.1190, +0.6762] | 0.0161 | 0.1448 | no significant difference |
| b_poc_ew | 20 | 1.5575 | 0.9288 | +0.6287 | [+0.3598, +0.9244] | 0.0001 | 0.0021 | ref worse |
| b_powerofchoice | 20 | 1.5575 | 1.3579 | +0.1996 | [-0.0542, +0.4763] | 0.1726 | 1.0000 | no significant difference |
| b_random | 20 | 1.5575 | 1.1286 | +0.4290 | [+0.1824, +0.7017] | 0.0039 | 0.0507 | no significant difference |
| b_random_ew | 20 | 1.5575 | 0.6183 | +0.9392 | [+0.6826, +1.2240] | 0.0000 | 0.0000 | ref worse |
| b_refl | 20 | 1.5575 | 0.6667 | +0.8909 | [+0.6391, +1.1704] | 0.0000 | 0.0000 | ref worse |
| b_refl_ew | 20 | 1.5575 | 0.3500 | +1.2075 | [+0.9418, +1.5022] | 0.0000 | 0.0000 | ref worse |
| rl_ew | 20 | 1.5575 | 1.3178 | +0.2398 | [-0.0257, +0.5417] | 0.1292 | 1.0000 | no significant difference |
| rl_light | 20 | 1.5575 | 1.0902 | +0.4674 | [+0.1709, +0.7774] | 0.0086 | 0.0862 | no significant difference |
| v2_rawfail | 20 | 1.5575 | 1.4180 | +0.1395 | [-0.1089, +0.4150] | 0.3375 | 1.0000 | no significant difference |

## mean_fail_ratio (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 0.4402 | 0.4179 | +0.0223 | [-0.0348, +0.0648] | 0.4719 | 1.0000 | no significant difference |
| arl3_full | 20 | 0.4402 | 0.4222 | +0.0180 | [-0.0149, +0.0570] | 0.3849 | 1.0000 | no significant difference |
| arl3_shared | 20 | 0.4402 | 0.4180 | +0.0222 | [-0.0276, +0.0776] | 0.4560 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 0.4402 | 0.5528 | -0.1126 | [-0.1911, -0.0321] | 0.0143 | 0.1001 | no significant difference |
| b_delta | 20 | 0.4402 | 0.6924 | -0.2522 | [-0.3067, -0.1888] | 0.0000 | 0.0002 | ref better |
| b_delta_ew | 20 | 0.4402 | 0.5449 | -0.1048 | [-0.1598, -0.0415] | 0.0040 | 0.0317 | ref better |
| b_favor | 20 | 0.4402 | 0.6752 | -0.2350 | [-0.2888, -0.1719] | 0.0000 | 0.0002 | ref better |
| b_favor_ew | 20 | 0.4402 | 0.5622 | -0.1221 | [-0.1816, -0.0550] | 0.0023 | 0.0230 | ref better |
| b_fedcer | 20 | 0.4402 | 0.5738 | -0.1337 | [-0.1960, -0.0681] | 0.0009 | 0.0114 | ref better |
| b_fedcs | 20 | 0.4402 | 0.2400 | +0.2002 | [+0.1203, +0.2856] | 0.0002 | 0.0025 | ref worse |
| b_mimic | 20 | 0.4402 | 0.6909 | -0.2507 | [-0.3062, -0.1861] | 0.0000 | 0.0002 | ref better |
| b_oort | 20 | 0.4402 | 0.7069 | -0.2667 | [-0.3258, -0.1979] | 0.0000 | 0.0002 | ref better |
| b_oort_ew | 20 | 0.4402 | 0.4492 | -0.0090 | [-0.0663, +0.0570] | 0.7939 | 1.0000 | no significant difference |
| b_poc_ew | 20 | 0.4402 | 0.5979 | -0.1577 | [-0.2130, -0.0926] | 0.0003 | 0.0036 | ref better |
| b_powerofchoice | 20 | 0.4402 | 0.8088 | -0.3686 | [-0.4230, -0.3070] | 0.0000 | 0.0001 | ref better |
| b_random | 20 | 0.4402 | 0.6909 | -0.2507 | [-0.3062, -0.1861] | 0.0000 | 0.0002 | ref better |
| b_random_ew | 20 | 0.4402 | 0.5517 | -0.1115 | [-0.1669, -0.0482] | 0.0026 | 0.0235 | ref better |
| b_refl | 20 | 0.4402 | 0.7442 | -0.3040 | [-0.3555, -0.2438] | 0.0000 | 0.0001 | ref better |
| b_refl_ew | 20 | 0.4402 | 0.5665 | -0.1263 | [-0.1788, -0.0656] | 0.0008 | 0.0106 | ref better |
| rl_ew | 20 | 0.4402 | 0.5170 | -0.0768 | [-0.1506, -0.0010] | 0.0633 | 0.3422 | no significant difference |
| rl_light | 20 | 0.4402 | 0.3244 | +0.1158 | [+0.0530, +0.1886] | 0.0011 | 0.0122 | ref worse |
| v2_rawfail | 20 | 0.4402 | 0.3977 | +0.0425 | [+0.0026, +0.0850] | 0.0570 | 0.3422 | no significant difference |

## mean_arl_oracle (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 5.6477 | 5.6777 | -0.0300 | [-0.2009, +0.1060] | 0.7574 | 1.0000 | no significant difference |
| arl3_full | 20 | 5.6477 | 5.5411 | +0.1066 | [-0.0297, +0.2670] | 0.2027 | 1.0000 | no significant difference |
| arl3_shared | 20 | 5.6477 | 5.7103 | -0.0626 | [-0.1885, +0.0674] | 0.3616 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 5.6477 | 3.6839 | +1.9638 | [+1.3816, +2.5043] | 0.0000 | 0.0005 | ref worse |
| b_delta | 20 | 5.6477 | 6.2695 | -0.6218 | [-0.7797, -0.4507] | 0.0000 | 0.0002 | ref better |
| b_delta_ew | 20 | 5.6477 | 5.9529 | -0.3052 | [-0.4571, -0.1469] | 0.0022 | 0.0332 | ref better |
| b_favor | 20 | 5.6477 | 5.8091 | -0.1614 | [-0.3066, -0.0097] | 0.0548 | 0.4471 | no significant difference |
| b_favor_ew | 20 | 5.6477 | 5.5339 | +0.1138 | [-0.0528, +0.2946] | 0.2282 | 1.0000 | no significant difference |
| b_fedcer | 20 | 5.6477 | 0.8894 | +4.7583 | [+4.5211, +5.0128] | 0.0000 | 0.0000 | ref worse |
| b_fedcs | 20 | 5.6477 | 5.5765 | +0.0712 | [-0.2152, +0.3783] | 0.6519 | 1.0000 | no significant difference |
| b_mimic | 20 | 5.6477 | 5.8079 | -0.1602 | [-0.2864, -0.0303] | 0.0290 | 0.3475 | no significant difference |
| b_oort | 20 | 5.6477 | 6.1467 | -0.4990 | [-0.6616, -0.3384] | 0.0000 | 0.0004 | ref better |
| b_oort_ew | 20 | 5.6477 | 5.6538 | -0.0061 | [-0.1509, +0.1403] | 0.9366 | 1.0000 | no significant difference |
| b_poc_ew | 20 | 5.6477 | 5.9101 | -0.2624 | [-0.4349, -0.0838] | 0.0109 | 0.1417 | no significant difference |
| b_powerofchoice | 20 | 5.6477 | 6.3035 | -0.6558 | [-0.7995, -0.5082] | 0.0000 | 0.0001 | ref better |
| b_random | 20 | 5.6477 | 5.8079 | -0.1602 | [-0.2864, -0.0303] | 0.0290 | 0.3475 | no significant difference |
| b_random_ew | 20 | 5.6477 | 5.5060 | +0.1417 | [+0.0161, +0.2679] | 0.0447 | 0.4471 | no significant difference |
| b_refl | 20 | 5.6477 | 7.3445 | -1.6968 | [-1.8775, -1.5106] | 0.0000 | 0.0000 | ref better |
| b_refl_ew | 20 | 5.6477 | 6.8604 | -1.2127 | [-1.3587, -1.0538] | 0.0000 | 0.0000 | ref better |
| rl_ew | 20 | 5.6477 | 5.8961 | -0.2484 | [-0.4252, -0.0804] | 0.0096 | 0.1338 | no significant difference |
| rl_light | 20 | 5.6477 | 5.4508 | +0.1969 | [+0.0073, +0.3688] | 0.0493 | 0.4471 | no significant difference |
| v2_rawfail | 20 | 5.6477 | 5.6468 | +0.0009 | [-0.1910, +0.1890] | 0.9929 | 1.0000 | no significant difference |

## mean_cost (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 12.0858 | 8.7485 | +3.3373 | [+1.0921, +5.8454] | 0.0121 | 0.2054 | no significant difference |
| arl3_full | 20 | 12.0858 | 9.4342 | +2.6516 | [+0.4785, +4.8533] | 0.0323 | 0.5166 | no significant difference |
| arl3_shared | 20 | 12.0858 | 10.3193 | +1.7665 | [-0.1052, +3.7459] | 0.0971 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 12.0858 | 10.2608 | +1.8250 | [-0.1918, +4.1566] | 0.1274 | 1.0000 | no significant difference |
| b_delta | 20 | 12.0858 | 10.9997 | +1.0861 | [-1.0230, +3.4639] | 0.3713 | 1.0000 | no significant difference |
| b_delta_ew | 20 | 12.0858 | 10.5002 | +1.5856 | [-0.5611, +3.9943] | 0.2029 | 1.0000 | no significant difference |
| b_favor | 20 | 12.0858 | 11.6527 | +0.4331 | [-1.6582, +2.7689] | 0.7151 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 12.0858 | 10.5926 | +1.4932 | [-0.6747, +3.8846] | 0.2296 | 1.0000 | no significant difference |
| b_fedcer | 20 | 12.0858 | 10.1043 | +1.9815 | [-0.3067, +4.5314] | 0.1377 | 1.0000 | no significant difference |
| b_fedcs | 20 | 12.0858 | 7.0007 | +5.0851 | [+3.0975, +7.3202] | 0.0000 | 0.0006 | ref worse |
| b_mimic | 20 | 12.0858 | 11.5428 | +0.5430 | [-1.5902, +2.9359] | 0.6561 | 1.0000 | no significant difference |
| b_oort | 20 | 12.0858 | 16.5323 | -4.4465 | [-6.4846, -2.1475] | 0.0017 | 0.0335 | ref better |
| b_oort_ew | 20 | 12.0858 | 12.6927 | -0.6069 | [-2.7172, +1.7544] | 0.6089 | 1.0000 | no significant difference |
| b_poc_ew | 20 | 12.0858 | 9.5906 | +2.4952 | [+0.3859, +4.8646] | 0.0459 | 0.6884 | no significant difference |
| b_powerofchoice | 20 | 12.0858 | 10.6509 | +1.4349 | [-0.6945, +3.8522] | 0.2454 | 1.0000 | no significant difference |
| b_random | 20 | 12.0858 | 11.5428 | +0.5430 | [-1.5902, +2.9359] | 0.6561 | 1.0000 | no significant difference |
| b_random_ew | 20 | 12.0858 | 10.9091 | +1.1767 | [-1.0302, +3.6593] | 0.3514 | 1.0000 | no significant difference |
| b_refl | 20 | 12.0858 | 8.8551 | +3.2307 | [+1.1456, +5.5721] | 0.0083 | 0.1489 | no significant difference |
| b_refl_ew | 20 | 12.0858 | 8.5964 | +3.4894 | [+1.4122, +5.8406] | 0.0045 | 0.0857 | no significant difference |
| rl_ew | 20 | 12.0858 | 11.8006 | +0.2852 | [-2.1862, +3.1372] | 0.8419 | 1.0000 | no significant difference |
| rl_light | 20 | 12.0858 | 7.1109 | +4.9749 | [+2.9469, +7.2476] | 0.0001 | 0.0020 | ref worse |
| v2_rawfail | 20 | 12.0858 | 10.3128 | +1.7730 | [-0.2240, +3.6862] | 0.1014 | 1.0000 | no significant difference |

## mean_latency (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 2.4793 | 2.3335 | +0.1458 | [-0.5767, +0.6408] | 0.8052 | 1.0000 | no significant difference |
| arl3_full | 20 | 2.4793 | 2.0633 | +0.4160 | [-0.0124, +0.8579] | 0.0853 | 1.0000 | no significant difference |
| arl3_shared | 20 | 2.4793 | 2.3591 | +0.1202 | [-0.4432, +0.6786] | 0.6645 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 2.4793 | 1.9761 | +0.5031 | [-0.0725, +1.1806] | 0.1443 | 1.0000 | no significant difference |
| b_delta | 20 | 2.4793 | 4.0248 | -1.5455 | [-2.1232, -0.8785] | 0.0004 | 0.0060 | ref better |
| b_delta_ew | 20 | 2.4793 | 2.2179 | +0.2614 | [-0.2957, +0.9301] | 0.4615 | 1.0000 | no significant difference |
| b_favor | 20 | 2.4793 | 3.9678 | -1.4885 | [-2.0612, -0.8111] | 0.0005 | 0.0078 | ref better |
| b_favor_ew | 20 | 2.4793 | 2.3523 | +0.1270 | [-0.4749, +0.8275] | 0.7428 | 1.0000 | no significant difference |
| b_fedcer | 20 | 2.4793 | 1.9823 | +0.4970 | [-0.1063, +1.2142] | 0.1723 | 1.0000 | no significant difference |
| b_fedcs | 20 | 2.4793 | 0.7380 | +1.7412 | [+1.1255, +2.4483] | 0.0000 | 0.0002 | ref worse |
| b_mimic | 20 | 2.4793 | 4.1150 | -1.6357 | [-2.2048, -0.9749] | 0.0002 | 0.0039 | ref better |
| b_oort | 20 | 2.4793 | 2.8830 | -0.4038 | [-0.9793, +0.2730] | 0.2378 | 1.0000 | no significant difference |
| b_oort_ew | 20 | 2.4793 | 1.6316 | +0.8477 | [+0.2991, +1.5017] | 0.0067 | 0.0943 | no significant difference |
| b_poc_ew | 20 | 2.4793 | 2.4565 | +0.0228 | [-0.5489, +0.7138] | 0.9533 | 1.0000 | no significant difference |
| b_powerofchoice | 20 | 2.4793 | 4.5586 | -2.0793 | [-2.6872, -1.3608] | 0.0001 | 0.0015 | ref better |
| b_random | 20 | 2.4793 | 4.1150 | -1.6357 | [-2.2048, -0.9749] | 0.0002 | 0.0039 | ref better |
| b_random_ew | 20 | 2.4793 | 2.2639 | +0.2153 | [-0.3378, +0.8747] | 0.5458 | 1.0000 | no significant difference |
| b_refl | 20 | 2.4793 | 4.2744 | -1.7951 | [-2.3510, -1.1160] | 0.0001 | 0.0022 | ref better |
| b_refl_ew | 20 | 2.4793 | 2.2835 | +0.1958 | [-0.3514, +0.8510] | 0.5848 | 1.0000 | no significant difference |
| rl_ew | 20 | 2.4793 | 1.9221 | +0.5572 | [-0.0174, +1.2380] | 0.1019 | 1.0000 | no significant difference |
| rl_light | 20 | 2.4793 | 1.0506 | +1.4286 | [+0.8762, +2.1011] | 0.0000 | 0.0000 | ref worse |
| v2_rawfail | 20 | 2.4793 | 2.1336 | +0.3457 | [-0.0709, +0.8105] | 0.1530 | 1.0000 | no significant difference |

wrote /kaggle/working/adv/s2_cifar/stats_c2_rl_fedprox.csv
