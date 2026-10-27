# /kaggle/working/adv/s2_femnist/: c2_rl_fedprox vs others (paired by seed; diff = ref - other; Holm-corrected within each metric; two-sided)


## final_accuracy (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 0.5804 | 0.6855 | -0.1051 | [-0.1915, -0.0322] | 0.0069 | 0.0931 | no significant difference |
| arl3_full | 20 | 0.5804 | 0.6816 | -0.1012 | [-0.1865, -0.0294] | 0.0103 | 0.1129 | no significant difference |
| arl3_shared | 20 | 0.5804 | 0.6448 | -0.0643 | [-0.1369, -0.0043] | 0.0824 | 0.1862 | no significant difference |
| b_cov_ew | 20 | 0.5804 | 0.7409 | -0.1604 | [-0.2729, -0.0614] | 0.0067 | 0.0931 | no significant difference |
| b_delta | 20 | 0.5804 | 0.6858 | -0.1054 | [-0.2155, -0.0069] | 0.0666 | 0.1862 | no significant difference |
| b_delta_ew | 20 | 0.5804 | 0.7582 | -0.1778 | [-0.2875, -0.0822] | 0.0002 | 0.0036 | ref worse |
| b_favor | 20 | 0.5804 | 0.7219 | -0.1414 | [-0.2501, -0.0461] | 0.0212 | 0.1482 | no significant difference |
| b_favor_ew | 20 | 0.5804 | 0.7480 | -0.1676 | [-0.2815, -0.0686] | 0.0037 | 0.0590 | no significant difference |
| b_fedcer | 20 | 0.5804 | 0.7301 | -0.1496 | [-0.2619, -0.0518] | 0.0156 | 0.1401 | no significant difference |
| b_fedcs | 20 | 0.5804 | 0.6878 | -0.1074 | [-0.2135, -0.0151] | 0.0465 | 0.1862 | no significant difference |
| b_mimic | 20 | 0.5804 | 0.7302 | -0.1498 | [-0.2613, -0.0493] | 0.0156 | 0.1401 | no significant difference |
| b_oort | 20 | 0.5804 | 0.7157 | -0.1353 | [-0.2461, -0.0379] | 0.0280 | 0.1552 | no significant difference |
| b_oort_ew | 20 | 0.5804 | 0.7850 | -0.2046 | [-0.3167, -0.1057] | 0.0000 | 0.0001 | ref worse |
| b_poc_ew | 20 | 0.5804 | 0.7648 | -0.1843 | [-0.2948, -0.0906] | 0.0000 | 0.0005 | ref worse |
| b_powerofchoice | 20 | 0.5804 | 0.6849 | -0.1044 | [-0.2109, -0.0105] | 0.0583 | 0.1862 | no significant difference |
| b_random | 20 | 0.5804 | 0.7145 | -0.1341 | [-0.2459, -0.0341] | 0.0259 | 0.1552 | no significant difference |
| b_random_ew | 20 | 0.5804 | 0.7767 | -0.1963 | [-0.3072, -0.0983] | 0.0000 | 0.0001 | ref worse |
| b_refl | 20 | 0.5804 | 0.7498 | -0.1694 | [-0.2824, -0.0664] | 0.0067 | 0.0931 | no significant difference |
| b_refl_ew | 20 | 0.5804 | 0.7690 | -0.1885 | [-0.3041, -0.0871] | 0.0010 | 0.0165 | ref worse |
| rl_ew | 20 | 0.5804 | 0.7392 | -0.1588 | [-0.2729, -0.0574] | 0.0113 | 0.1129 | no significant difference |
| rl_light | 20 | 0.5804 | 0.7468 | -0.1664 | [-0.2725, -0.0728] | 0.0008 | 0.0148 | ref worse |
| v2_rawfail | 20 | 0.5804 | 0.7117 | -0.1313 | [-0.2293, -0.0472] | 0.0061 | 0.0920 | no significant difference |

## final_acc_worst10 (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 0.0543 | 0.0625 | -0.0082 | [-0.0298, +0.0138] | 0.4980 | 1.0000 | no significant difference |
| arl3_full | 20 | 0.0543 | 0.0502 | +0.0041 | [-0.0141, +0.0231] | 0.6900 | 1.0000 | no significant difference |
| arl3_shared | 20 | 0.0543 | 0.0453 | +0.0091 | [-0.0080, +0.0263] | 0.3370 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 0.0543 | 0.0659 | -0.0116 | [-0.0396, +0.0169] | 0.4461 | 1.0000 | no significant difference |
| b_delta | 20 | 0.0543 | 0.0212 | +0.0331 | [+0.0076, +0.0610] | 0.0257 | 0.5139 | no significant difference |
| b_delta_ew | 20 | 0.0543 | 0.0657 | -0.0114 | [-0.0414, +0.0170] | 0.4630 | 1.0000 | no significant difference |
| b_favor | 20 | 0.0543 | 0.0623 | -0.0079 | [-0.0399, +0.0263] | 0.6512 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 0.0543 | 0.0714 | -0.0171 | [-0.0490, +0.0141] | 0.3152 | 1.0000 | no significant difference |
| b_fedcer | 20 | 0.0543 | 0.0378 | +0.0165 | [-0.0057, +0.0398] | 0.1813 | 1.0000 | no significant difference |
| b_fedcs | 20 | 0.0543 | 0.0647 | -0.0104 | [-0.0346, +0.0129] | 0.4158 | 1.0000 | no significant difference |
| b_mimic | 20 | 0.0543 | 0.0957 | -0.0414 | [-0.0731, -0.0106] | 0.0203 | 0.4471 | no significant difference |
| b_oort | 20 | 0.0543 | 0.0472 | +0.0071 | [-0.0248, +0.0407] | 0.6824 | 1.0000 | no significant difference |
| b_oort_ew | 20 | 0.0543 | 0.0768 | -0.0225 | [-0.0528, +0.0075] | 0.1783 | 1.0000 | no significant difference |
| b_poc_ew | 20 | 0.0543 | 0.0543 | +0.0000 | [-0.0278, +0.0276] | 0.9985 | 1.0000 | no significant difference |
| b_powerofchoice | 20 | 0.0543 | 0.0230 | +0.0313 | [+0.0050, +0.0584] | 0.0366 | 0.6952 | no significant difference |
| b_random | 20 | 0.0543 | 0.0819 | -0.0276 | [-0.0574, +0.0010] | 0.0884 | 1.0000 | no significant difference |
| b_random_ew | 20 | 0.0543 | 0.0930 | -0.0387 | [-0.0659, -0.0087] | 0.0213 | 0.4482 | no significant difference |
| b_refl | 20 | 0.0543 | 0.0723 | -0.0179 | [-0.0490, +0.0117] | 0.2708 | 1.0000 | no significant difference |
| b_refl_ew | 20 | 0.0543 | 0.0703 | -0.0160 | [-0.0414, +0.0084] | 0.2373 | 1.0000 | no significant difference |
| rl_ew | 20 | 0.0543 | 0.0854 | -0.0311 | [-0.0588, -0.0056] | 0.0386 | 0.6952 | no significant difference |
| rl_light | 20 | 0.0543 | 0.0818 | -0.0274 | [-0.0508, -0.0037] | 0.0424 | 0.7200 | no significant difference |
| v2_rawfail | 20 | 0.0543 | 0.0540 | +0.0003 | [-0.0183, +0.0220] | 0.9823 | 1.0000 | no significant difference |

## mean_accuracy_fixed (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 0.4607 | 0.5143 | -0.0536 | [-0.1022, -0.0133] | 0.0172 | 0.1024 | no significant difference |
| arl3_full | 20 | 0.4607 | 0.5281 | -0.0674 | [-0.1247, -0.0194] | 0.0128 | 0.1024 | no significant difference |
| arl3_shared | 20 | 0.4607 | 0.4948 | -0.0340 | [-0.0854, +0.0141] | 0.2142 | 0.2142 | no significant difference |
| b_cov_ew | 20 | 0.4607 | 0.5638 | -0.1031 | [-0.2103, -0.0066] | 0.0686 | 0.2057 | no significant difference |
| b_delta | 20 | 0.4607 | 0.5453 | -0.0846 | [-0.1753, -0.0014] | 0.0795 | 0.2057 | no significant difference |
| b_delta_ew | 20 | 0.4607 | 0.6112 | -0.1505 | [-0.2439, -0.0646] | 0.0040 | 0.0399 | ref worse |
| b_favor | 20 | 0.4607 | 0.6173 | -0.1566 | [-0.2498, -0.0705] | 0.0027 | 0.0297 | ref worse |
| b_favor_ew | 20 | 0.4607 | 0.6572 | -0.1965 | [-0.2913, -0.1096] | 0.0001 | 0.0014 | ref worse |
| b_fedcer | 20 | 0.4607 | 0.6365 | -0.1758 | [-0.2715, -0.0860] | 0.0013 | 0.0152 | ref worse |
| b_fedcs | 20 | 0.4607 | 0.5887 | -0.1280 | [-0.2237, -0.0386] | 0.0156 | 0.1024 | no significant difference |
| b_mimic | 20 | 0.4607 | 0.6311 | -0.1704 | [-0.2636, -0.0848] | 0.0009 | 0.0131 | ref worse |
| b_oort | 20 | 0.4607 | 0.6464 | -0.1856 | [-0.2812, -0.0964] | 0.0005 | 0.0072 | ref worse |
| b_oort_ew | 20 | 0.4607 | 0.6995 | -0.2388 | [-0.3336, -0.1517] | 0.0000 | 0.0000 | ref worse |
| b_poc_ew | 20 | 0.4607 | 0.6662 | -0.2054 | [-0.2997, -0.1196] | 0.0000 | 0.0002 | ref worse |
| b_powerofchoice | 20 | 0.4607 | 0.5925 | -0.1318 | [-0.2294, -0.0418] | 0.0145 | 0.1024 | no significant difference |
| b_random | 20 | 0.4607 | 0.6309 | -0.1702 | [-0.2636, -0.0845] | 0.0009 | 0.0131 | ref worse |
| b_random_ew | 20 | 0.4607 | 0.6724 | -0.2117 | [-0.3079, -0.1232] | 0.0000 | 0.0003 | ref worse |
| b_refl | 20 | 0.4607 | 0.6702 | -0.2094 | [-0.3040, -0.1229] | 0.0000 | 0.0002 | ref worse |
| b_refl_ew | 20 | 0.4607 | 0.6792 | -0.2185 | [-0.3149, -0.1306] | 0.0000 | 0.0001 | ref worse |
| rl_ew | 20 | 0.4607 | 0.5721 | -0.1114 | [-0.2173, -0.0141] | 0.0498 | 0.1991 | no significant difference |
| rl_light | 20 | 0.4607 | 0.6398 | -0.1791 | [-0.2697, -0.0953] | 0.0004 | 0.0066 | ref worse |
| v2_rawfail | 20 | 0.4607 | 0.5412 | -0.0805 | [-0.1366, -0.0291] | 0.0086 | 0.0777 | no significant difference |

## final_cv (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 1.8069 | 1.3022 | +0.5047 | [+0.2153, +0.8367] | 0.0018 | 0.0236 | ref worse |
| arl3_full | 20 | 1.8069 | 1.2603 | +0.5466 | [+0.1203, +1.0109] | 0.0238 | 0.1191 | no significant difference |
| arl3_shared | 20 | 1.8069 | 1.5105 | +0.2964 | [+0.1468, +0.4586] | 0.0005 | 0.0081 | ref worse |
| b_cov_ew | 20 | 1.8069 | 1.7234 | +0.0835 | [-0.2815, +0.5085] | 0.7068 | 0.7068 | no significant difference |
| b_delta | 20 | 1.8069 | 1.2282 | +0.5787 | [+0.2176, +1.0061] | 0.0064 | 0.0578 | no significant difference |
| b_delta_ew | 20 | 1.8069 | 0.6562 | +1.1507 | [+0.7858, +1.5732] | 0.0000 | 0.0000 | ref worse |
| b_favor | 20 | 1.8069 | 1.3767 | +0.4302 | [+0.0678, +0.8574] | 0.0487 | 0.1462 | no significant difference |
| b_favor_ew | 20 | 1.8069 | 1.0344 | +0.7725 | [+0.3715, +1.2287] | 0.0013 | 0.0185 | ref worse |
| b_fedcer | 20 | 1.8069 | 2.5410 | -0.7341 | [-1.1101, -0.3020] | 0.0038 | 0.0454 | ref better |
| b_fedcs | 20 | 1.8069 | 2.3155 | -0.5086 | [-0.8736, -0.0872] | 0.0258 | 0.1191 | no significant difference |
| b_mimic | 20 | 1.8069 | 1.2156 | +0.5913 | [+0.2363, +1.0084] | 0.0041 | 0.0454 | ref worse |
| b_oort | 20 | 1.8069 | 1.2860 | +0.5209 | [+0.1504, +0.9569] | 0.0171 | 0.1058 | no significant difference |
| b_oort_ew | 20 | 1.8069 | 0.9747 | +0.8322 | [+0.4666, +1.2625] | 0.0001 | 0.0014 | ref worse |
| b_poc_ew | 20 | 1.8069 | 0.7826 | +1.0243 | [+0.6488, +1.4621] | 0.0000 | 0.0002 | ref worse |
| b_powerofchoice | 20 | 1.8069 | 1.3634 | +0.4435 | [+0.0583, +0.8873] | 0.0549 | 0.1462 | no significant difference |
| b_random | 20 | 1.8069 | 1.2152 | +0.5917 | [+0.2364, +1.0087] | 0.0041 | 0.0454 | ref worse |
| b_random_ew | 20 | 1.8069 | 0.6180 | +1.1889 | [+0.8280, +1.6043] | 0.0000 | 0.0000 | ref worse |
| b_refl | 20 | 1.8069 | 0.6730 | +1.1339 | [+0.7733, +1.5543] | 0.0000 | 0.0000 | ref worse |
| b_refl_ew | 20 | 1.8069 | 0.3155 | +1.4914 | [+1.1200, +1.9195] | 0.0000 | 0.0000 | ref worse |
| rl_ew | 20 | 1.8069 | 1.0245 | +0.7824 | [+0.4148, +1.2142] | 0.0002 | 0.0033 | ref worse |
| rl_light | 20 | 1.8069 | 1.2588 | +0.5481 | [+0.1803, +0.9603] | 0.0132 | 0.1058 | no significant difference |
| v2_rawfail | 20 | 1.8069 | 1.3714 | +0.4355 | [+0.1271, +0.7921] | 0.0145 | 0.1058 | no significant difference |

## mean_fail_ratio (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 0.6428 | 0.5455 | +0.0974 | [+0.0512, +0.1508] | 0.0004 | 0.0069 | ref worse |
| arl3_full | 20 | 0.6428 | 0.5657 | +0.0772 | [+0.0254, +0.1357] | 0.0111 | 0.1772 | no significant difference |
| arl3_shared | 20 | 0.6428 | 0.5633 | +0.0795 | [+0.0288, +0.1382] | 0.0050 | 0.0843 | no significant difference |
| b_cov_ew | 20 | 0.6428 | 0.5499 | +0.0929 | [-0.0338, +0.2211] | 0.1766 | 1.0000 | no significant difference |
| b_delta | 20 | 0.6428 | 0.7078 | -0.0650 | [-0.1795, +0.0499] | 0.2881 | 1.0000 | no significant difference |
| b_delta_ew | 20 | 0.6428 | 0.5560 | +0.0868 | [-0.0284, +0.2023] | 0.1651 | 1.0000 | no significant difference |
| b_favor | 20 | 0.6428 | 0.6935 | -0.0506 | [-0.1634, +0.0616] | 0.3943 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 0.6428 | 0.5706 | +0.0723 | [-0.0429, +0.1879] | 0.2428 | 1.0000 | no significant difference |
| b_fedcer | 20 | 0.6428 | 0.5829 | +0.0599 | [-0.0561, +0.1746] | 0.3306 | 1.0000 | no significant difference |
| b_fedcs | 20 | 0.6428 | 0.1892 | +0.4536 | [+0.3251, +0.5811] | 0.0000 | 0.0001 | ref worse |
| b_mimic | 20 | 0.6428 | 0.6977 | -0.0549 | [-0.1704, +0.0611] | 0.3720 | 1.0000 | no significant difference |
| b_oort | 20 | 0.6428 | 0.7676 | -0.1248 | [-0.2391, -0.0099] | 0.0533 | 0.7464 | no significant difference |
| b_oort_ew | 20 | 0.6428 | 0.4838 | +0.1590 | [+0.0454, +0.2746] | 0.0173 | 0.2595 | no significant difference |
| b_poc_ew | 20 | 0.6428 | 0.6186 | +0.0243 | [-0.0889, +0.1374] | 0.6806 | 1.0000 | no significant difference |
| b_powerofchoice | 20 | 0.6428 | 0.8679 | -0.2250 | [-0.3382, -0.1120] | 0.0018 | 0.0322 | ref better |
| b_random | 20 | 0.6428 | 0.6970 | -0.0542 | [-0.1699, +0.0624] | 0.3789 | 1.0000 | no significant difference |
| b_random_ew | 20 | 0.6428 | 0.5598 | +0.0830 | [-0.0318, +0.1989] | 0.1833 | 1.0000 | no significant difference |
| b_refl | 20 | 0.6428 | 0.7644 | -0.1215 | [-0.2380, -0.0043] | 0.0638 | 0.8296 | no significant difference |
| b_refl_ew | 20 | 0.6428 | 0.5707 | +0.0721 | [-0.0418, +0.1864] | 0.2386 | 1.0000 | no significant difference |
| rl_ew | 20 | 0.6428 | 0.5550 | +0.0879 | [-0.0415, +0.2191] | 0.2185 | 1.0000 | no significant difference |
| rl_light | 20 | 0.6428 | 0.3083 | +0.3346 | [+0.2209, +0.4512] | 0.0000 | 0.0001 | ref worse |
| v2_rawfail | 20 | 0.6428 | 0.5071 | +0.1358 | [+0.0833, +0.1947] | 0.0000 | 0.0000 | ref worse |

## mean_arl_oracle (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 1.6618 | 1.6133 | +0.0485 | [-0.1197, +0.2155] | 0.5795 | 1.0000 | no significant difference |
| arl3_full | 20 | 1.6618 | 1.5715 | +0.0903 | [-0.0707, +0.2528] | 0.3034 | 1.0000 | no significant difference |
| arl3_shared | 20 | 1.6618 | 1.4960 | +0.1658 | [+0.0308, +0.2972] | 0.0238 | 0.1905 | no significant difference |
| b_cov_ew | 20 | 1.6618 | 1.1531 | +0.5087 | [+0.3103, +0.7024] | 0.0002 | 0.0016 | ref worse |
| b_delta | 20 | 1.6618 | 1.1071 | +0.5547 | [+0.3887, +0.7278] | 0.0000 | 0.0000 | ref worse |
| b_delta_ew | 20 | 1.6618 | 1.0348 | +0.6269 | [+0.4637, +0.7956] | 0.0000 | 0.0000 | ref worse |
| b_favor | 20 | 1.6618 | 1.0265 | +0.6353 | [+0.4832, +0.7959] | 0.0000 | 0.0000 | ref worse |
| b_favor_ew | 20 | 1.6618 | 0.9433 | +0.7185 | [+0.5659, +0.8860] | 0.0000 | 0.0000 | ref worse |
| b_fedcer | 20 | 1.6618 | 0.2701 | +1.3917 | [+1.2415, +1.5475] | 0.0000 | 0.0000 | ref worse |
| b_fedcs | 20 | 1.6618 | 1.7668 | -0.1050 | [-0.3128, +0.0937] | 0.3457 | 1.0000 | no significant difference |
| b_mimic | 20 | 1.6618 | 1.0417 | +0.6201 | [+0.4569, +0.7884] | 0.0000 | 0.0000 | ref worse |
| b_oort | 20 | 1.6618 | 1.4661 | +0.1957 | [-0.0024, +0.3995] | 0.0834 | 0.5839 | no significant difference |
| b_oort_ew | 20 | 1.6618 | 0.9622 | +0.6996 | [+0.5112, +0.8897] | 0.0000 | 0.0000 | ref worse |
| b_poc_ew | 20 | 1.6618 | 0.9007 | +0.7610 | [+0.6003, +0.9256] | 0.0000 | 0.0000 | ref worse |
| b_powerofchoice | 20 | 1.6618 | 1.0398 | +0.6220 | [+0.4567, +0.7901] | 0.0000 | 0.0000 | ref worse |
| b_random | 20 | 1.6618 | 1.0416 | +0.6202 | [+0.4569, +0.7885] | 0.0000 | 0.0000 | ref worse |
| b_random_ew | 20 | 1.6618 | 0.9669 | +0.6949 | [+0.5308, +0.8646] | 0.0000 | 0.0000 | ref worse |
| b_refl | 20 | 1.6618 | 1.3148 | +0.3470 | [+0.1784, +0.5197] | 0.0012 | 0.0106 | ref worse |
| b_refl_ew | 20 | 1.6618 | 1.2394 | +0.4224 | [+0.2567, +0.5970] | 0.0001 | 0.0009 | ref worse |
| rl_ew | 20 | 1.6618 | 1.6252 | +0.0365 | [-0.1899, +0.2434] | 0.7524 | 1.0000 | no significant difference |
| rl_light | 20 | 1.6618 | 1.5225 | +0.1392 | [-0.1131, +0.3768] | 0.2942 | 1.0000 | no significant difference |
| v2_rawfail | 20 | 1.6618 | 1.5839 | +0.0779 | [-0.0905, +0.2445] | 0.3854 | 1.0000 | no significant difference |

## mean_cost (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 8.8405 | 6.6290 | +2.2116 | [-0.1637, +4.9937] | 0.1229 | 1.0000 | no significant difference |
| arl3_full | 20 | 8.8405 | 7.8812 | +0.9593 | [-1.5943, +3.8680] | 0.5172 | 1.0000 | no significant difference |
| arl3_shared | 20 | 8.8405 | 7.0449 | +1.7957 | [+0.2086, +3.6529] | 0.0532 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 8.8405 | 11.6920 | -2.8514 | [-5.6596, +0.1095] | 0.0756 | 1.0000 | no significant difference |
| b_delta | 20 | 8.8405 | 11.1848 | -2.3443 | [-5.1464, +0.8639] | 0.1480 | 1.0000 | no significant difference |
| b_delta_ew | 20 | 8.8405 | 10.8068 | -1.9663 | [-4.7572, +1.2154] | 0.2177 | 1.0000 | no significant difference |
| b_favor | 20 | 8.8405 | 11.9501 | -3.1096 | [-5.8677, +0.0564] | 0.0581 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 8.8405 | 10.9322 | -2.0917 | [-4.7966, +1.0351] | 0.1840 | 1.0000 | no significant difference |
| b_fedcer | 20 | 8.8405 | 9.9727 | -1.1321 | [-3.9897, +2.0058] | 0.4750 | 1.0000 | no significant difference |
| b_fedcs | 20 | 8.8405 | 6.5501 | +2.2905 | [-0.5161, +5.4790] | 0.1603 | 1.0000 | no significant difference |
| b_mimic | 20 | 8.8405 | 11.8021 | -2.9615 | [-5.7611, +0.2646] | 0.0735 | 1.0000 | no significant difference |
| b_oort | 20 | 8.8405 | 16.1227 | -7.2822 | [-10.1556, -4.0395] | 0.0005 | 0.0115 | ref better |
| b_oort_ew | 20 | 8.8405 | 11.7142 | -2.8737 | [-5.6635, +0.3094] | 0.0792 | 1.0000 | no significant difference |
| b_poc_ew | 20 | 8.8405 | 9.6061 | -0.7656 | [-3.5353, +2.3985] | 0.6246 | 1.0000 | no significant difference |
| b_powerofchoice | 20 | 8.8405 | 11.6365 | -2.7960 | [-5.6787, +0.4634] | 0.0960 | 1.0000 | no significant difference |
| b_random | 20 | 8.8405 | 11.8021 | -2.9615 | [-5.7611, +0.2646] | 0.0735 | 1.0000 | no significant difference |
| b_random_ew | 20 | 8.8405 | 11.2689 | -2.4284 | [-5.2564, +0.7942] | 0.1370 | 1.0000 | no significant difference |
| b_refl | 20 | 8.8405 | 9.3737 | -0.5332 | [-3.3366, +2.6378] | 0.7368 | 1.0000 | no significant difference |
| b_refl_ew | 20 | 8.8405 | 8.7053 | +0.1353 | [-2.6837, +3.3517] | 0.9336 | 1.0000 | no significant difference |
| rl_ew | 20 | 8.8405 | 11.2536 | -2.4130 | [-5.4234, +0.7750] | 0.1540 | 1.0000 | no significant difference |
| rl_light | 20 | 8.8405 | 6.9703 | +1.8702 | [-0.8946, +4.9989] | 0.2436 | 1.0000 | no significant difference |
| v2_rawfail | 20 | 8.8405 | 7.1372 | +1.7033 | [-0.3279, +4.0110] | 0.1527 | 1.0000 | no significant difference |

## mean_latency (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 20 | 3.5730 | 2.5457 | +1.0273 | [+0.5215, +1.6172] | 0.0002 | 0.0028 | ref worse |
| arl3_full | 20 | 3.5730 | 2.6467 | +0.9263 | [+0.3189, +1.6336] | 0.0075 | 0.0673 | no significant difference |
| arl3_shared | 20 | 3.5730 | 3.0607 | +0.5123 | [-0.2017, +1.2957] | 0.2154 | 1.0000 | no significant difference |
| b_cov_ew | 20 | 3.5730 | 1.7528 | +1.8202 | [+0.8226, +2.9171] | 0.0020 | 0.0311 | ref worse |
| b_delta | 20 | 3.5730 | 3.2943 | +0.2787 | [-0.6878, +1.3323] | 0.6054 | 1.0000 | no significant difference |
| b_delta_ew | 20 | 3.5730 | 1.8634 | +1.7096 | [+0.7419, +2.7775] | 0.0031 | 0.0405 | ref worse |
| b_favor | 20 | 3.5730 | 3.2805 | +0.2925 | [-0.6531, +1.3365] | 0.5821 | 1.0000 | no significant difference |
| b_favor_ew | 20 | 3.5730 | 1.9156 | +1.6574 | [+0.6887, +2.7290] | 0.0041 | 0.0455 | ref worse |
| b_fedcer | 20 | 3.5730 | 1.6045 | +1.9685 | [+0.9999, +3.0440] | 0.0003 | 0.0054 | ref worse |
| b_fedcs | 20 | 3.5730 | 0.4750 | +3.0980 | [+2.0932, +4.2021] | 0.0000 | 0.0000 | ref worse |
| b_mimic | 20 | 3.5730 | 3.3200 | +0.2531 | [-0.7093, +1.3028] | 0.6386 | 1.0000 | no significant difference |
| b_oort | 20 | 3.5730 | 2.8143 | +0.7587 | [-0.1888, +1.8024] | 0.1647 | 1.0000 | no significant difference |
| b_oort_ew | 20 | 3.5730 | 1.5649 | +2.0081 | [+1.0487, +3.0695] | 0.0001 | 0.0024 | ref worse |
| b_poc_ew | 20 | 3.5730 | 1.9386 | +1.6344 | [+0.6665, +2.6975] | 0.0044 | 0.0455 | ref worse |
| b_powerofchoice | 20 | 3.5730 | 3.6133 | -0.0403 | [-1.0006, +1.0106] | 0.9396 | 1.0000 | no significant difference |
| b_random | 20 | 3.5730 | 3.3200 | +0.2531 | [-0.7093, +1.3028] | 0.6386 | 1.0000 | no significant difference |
| b_random_ew | 20 | 3.5730 | 1.8843 | +1.6887 | [+0.7296, +2.7480] | 0.0031 | 0.0405 | ref worse |
| b_refl | 20 | 3.5730 | 3.3233 | +0.2497 | [-0.7289, +1.3176] | 0.6472 | 1.0000 | no significant difference |
| b_refl_ew | 20 | 3.5730 | 1.8136 | +1.7594 | [+0.7945, +2.8277] | 0.0019 | 0.0311 | ref worse |
| rl_ew | 20 | 3.5730 | 1.7986 | +1.7744 | [+0.7881, +2.8559] | 0.0026 | 0.0368 | ref worse |
| rl_light | 20 | 3.5730 | 0.9667 | +2.6063 | [+1.6314, +3.6858] | 0.0000 | 0.0000 | ref worse |
| v2_rawfail | 20 | 3.5730 | 2.4099 | +1.1631 | [+0.4796, +1.9653] | 0.0001 | 0.0028 | ref worse |

wrote /kaggle/working/adv/s2_femnist/stats_c2_rl_fedprox.csv
