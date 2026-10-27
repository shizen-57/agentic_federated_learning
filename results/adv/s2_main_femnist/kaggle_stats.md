# /kaggle/working/adv/s2_main_femnist/: c2_rl_fedprox vs others (paired by seed; diff = ref - other; Holm-corrected within each metric; two-sided)


## final_accuracy (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 0.7042 | 0.7356 | -0.0313 | [-0.0760, +0.0079] | 0.2266 | 1.0000 | no significant difference |
| arl3_full | 10 | 0.7042 | 0.7070 | -0.0027 | [-0.0499, +0.0445] | 0.9180 | 1.0000 | no significant difference |
| arl3_shared | 10 | 0.7042 | 0.6906 | +0.0136 | [-0.0909, +0.1225] | 0.6738 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 0.7042 | 0.7282 | -0.0239 | [-0.1118, +0.0453] | 0.6660 | 1.0000 | no significant difference |
| b_delta | 10 | 0.7042 | 0.7448 | -0.0406 | [-0.1328, +0.0280] | 0.4727 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 0.7042 | 0.7609 | -0.0567 | [-0.1478, +0.0120] | 0.2949 | 1.0000 | no significant difference |
| b_favor | 10 | 0.7042 | 0.7436 | -0.0393 | [-0.1331, +0.0321] | 0.4863 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 0.7042 | 0.7399 | -0.0357 | [-0.1311, +0.0389] | 0.5449 | 1.0000 | no significant difference |
| b_fedcer | 10 | 0.7042 | 0.7208 | -0.0166 | [-0.1146, +0.0589] | 0.8047 | 1.0000 | no significant difference |
| b_fedcs | 10 | 0.7042 | 0.7198 | -0.0155 | [-0.1103, +0.0554] | 0.8320 | 1.0000 | no significant difference |
| b_mimic | 10 | 0.7042 | 0.7616 | -0.0573 | [-0.1553, +0.0151] | 0.2930 | 1.0000 | no significant difference |
| b_oort | 10 | 0.7042 | 0.7555 | -0.0512 | [-0.1434, +0.0206] | 0.3086 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 0.7042 | 0.7697 | -0.0655 | [-0.1723, +0.0265] | 0.2695 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 0.7042 | 0.7461 | -0.0418 | [-0.1421, +0.0371] | 0.4766 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 0.7042 | 0.7041 | +0.0002 | [-0.0962, +0.0900] | 0.9980 | 1.0000 | no significant difference |
| b_random | 10 | 0.7042 | 0.7513 | -0.0471 | [-0.1404, +0.0214] | 0.3867 | 1.0000 | no significant difference |
| b_random_ew | 10 | 0.7042 | 0.7519 | -0.0476 | [-0.1412, +0.0263] | 0.4180 | 1.0000 | no significant difference |
| b_refl | 10 | 0.7042 | 0.7515 | -0.0473 | [-0.1460, +0.0407] | 0.3848 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 0.7042 | 0.7804 | -0.0761 | [-0.1693, -0.0027] | 0.1113 | 1.0000 | no significant difference |
| rl_ew | 10 | 0.7042 | 0.7085 | -0.0043 | [-0.0877, +0.0735] | 0.9141 | 1.0000 | no significant difference |
| rl_light | 10 | 0.7042 | 0.7717 | -0.0674 | [-0.1538, -0.0016] | 0.1152 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 0.7042 | 0.7231 | -0.0188 | [-0.0634, +0.0176] | 0.4785 | 1.0000 | no significant difference |

## final_acc_worst10 (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 0.0608 | 0.0730 | -0.0122 | [-0.0350, +0.0072] | 0.3516 | 1.0000 | no significant difference |
| arl3_full | 10 | 0.0608 | 0.0571 | +0.0037 | [-0.0201, +0.0303] | 0.8438 | 1.0000 | no significant difference |
| arl3_shared | 10 | 0.0608 | 0.0579 | +0.0029 | [-0.0195, +0.0291] | 0.8594 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 0.0608 | 0.0292 | +0.0316 | [-0.0050, +0.0717] | 0.1562 | 1.0000 | no significant difference |
| b_delta | 10 | 0.0608 | 0.0651 | -0.0043 | [-0.0367, +0.0280] | 0.7773 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 0.0608 | 0.0901 | -0.0293 | [-0.0723, +0.0090] | 0.2266 | 1.0000 | no significant difference |
| b_favor | 10 | 0.0608 | 0.0627 | -0.0019 | [-0.0563, +0.0525] | 0.9453 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 0.0608 | 0.0690 | -0.0082 | [-0.0590, +0.0374] | 0.7637 | 1.0000 | no significant difference |
| b_fedcer | 10 | 0.0608 | 0.0637 | -0.0029 | [-0.0511, +0.0432] | 0.9141 | 1.0000 | no significant difference |
| b_fedcs | 10 | 0.0608 | 0.0812 | -0.0204 | [-0.0748, +0.0274] | 0.5078 | 1.0000 | no significant difference |
| b_mimic | 10 | 0.0608 | 0.0846 | -0.0238 | [-0.0768, +0.0255] | 0.4062 | 1.0000 | no significant difference |
| b_oort | 10 | 0.0608 | 0.0821 | -0.0213 | [-0.0618, +0.0212] | 0.3555 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 0.0608 | 0.0854 | -0.0246 | [-0.0670, +0.0161] | 0.3027 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 0.0608 | 0.0444 | +0.0164 | [-0.0335, +0.0643] | 0.5469 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 0.0608 | 0.0414 | +0.0194 | [-0.0271, +0.0649] | 0.4844 | 1.0000 | no significant difference |
| b_random | 10 | 0.0608 | 0.0972 | -0.0364 | [-0.0848, +0.0026] | 0.1387 | 1.0000 | no significant difference |
| b_random_ew | 10 | 0.0608 | 0.1210 | -0.0602 | [-0.1144, -0.0119] | 0.0469 | 1.0000 | no significant difference |
| b_refl | 10 | 0.0608 | 0.0718 | -0.0110 | [-0.0448, +0.0240] | 0.5586 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 0.0608 | 0.0779 | -0.0171 | [-0.0663, +0.0317] | 0.5371 | 1.0000 | no significant difference |
| rl_ew | 10 | 0.0608 | 0.0506 | +0.0103 | [-0.0205, +0.0442] | 0.5703 | 1.0000 | no significant difference |
| rl_light | 10 | 0.0608 | 0.0832 | -0.0224 | [-0.0681, +0.0161] | 0.3945 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 0.0608 | 0.0728 | -0.0120 | [-0.0404, +0.0175] | 0.4375 | 1.0000 | no significant difference |

## mean_accuracy_fixed (higher is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 0.5174 | 0.5724 | -0.0549 | [-0.1240, -0.0037] | 0.1074 | 1.0000 | no significant difference |
| arl3_full | 10 | 0.5174 | 0.5532 | -0.0358 | [-0.0713, -0.0038] | 0.1016 | 1.0000 | no significant difference |
| arl3_shared | 10 | 0.5174 | 0.5557 | -0.0382 | [-0.0849, +0.0003] | 0.2363 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 0.5174 | 0.5110 | +0.0064 | [-0.1198, +0.1291] | 0.9238 | 1.0000 | no significant difference |
| b_delta | 10 | 0.5174 | 0.6290 | -0.1116 | [-0.2447, +0.0128] | 0.1719 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 0.5174 | 0.6332 | -0.1157 | [-0.2504, +0.0099] | 0.1660 | 1.0000 | no significant difference |
| b_favor | 10 | 0.5174 | 0.6660 | -0.1485 | [-0.2854, -0.0216] | 0.1191 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 0.5174 | 0.6735 | -0.1560 | [-0.2930, -0.0284] | 0.1074 | 1.0000 | no significant difference |
| b_fedcer | 10 | 0.5174 | 0.6423 | -0.1248 | [-0.2693, +0.0110] | 0.1641 | 1.0000 | no significant difference |
| b_fedcs | 10 | 0.5174 | 0.6067 | -0.0893 | [-0.2326, +0.0424] | 0.2734 | 1.0000 | no significant difference |
| b_mimic | 10 | 0.5174 | 0.6666 | -0.1491 | [-0.2876, -0.0194] | 0.1172 | 1.0000 | no significant difference |
| b_oort | 10 | 0.5174 | 0.6997 | -0.1823 | [-0.3194, -0.0549] | 0.0137 | 0.2871 | no significant difference |
| b_oort_ew | 10 | 0.5174 | 0.7090 | -0.1916 | [-0.3276, -0.0666] | 0.0039 | 0.0859 | no significant difference |
| b_poc_ew | 10 | 0.5174 | 0.6628 | -0.1454 | [-0.2858, -0.0142] | 0.1172 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 0.5174 | 0.6475 | -0.1300 | [-0.2692, -0.0001] | 0.1367 | 1.0000 | no significant difference |
| b_random | 10 | 0.5174 | 0.6789 | -0.1614 | [-0.2979, -0.0352] | 0.1035 | 1.0000 | no significant difference |
| b_random_ew | 10 | 0.5174 | 0.6735 | -0.1560 | [-0.2899, -0.0294] | 0.0918 | 1.0000 | no significant difference |
| b_refl | 10 | 0.5174 | 0.6768 | -0.1593 | [-0.2958, -0.0318] | 0.0957 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 0.5174 | 0.6930 | -0.1756 | [-0.3136, -0.0488] | 0.0352 | 0.7031 | no significant difference |
| rl_ew | 10 | 0.5174 | 0.4769 | +0.0405 | [-0.1133, +0.1851] | 0.6152 | 1.0000 | no significant difference |
| rl_light | 10 | 0.5174 | 0.6653 | -0.1479 | [-0.2857, -0.0227] | 0.1211 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 0.5174 | 0.5705 | -0.0531 | [-0.1163, -0.0077] | 0.0957 | 1.0000 | no significant difference |

## final_cv (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 1.2358 | 1.1019 | +0.1339 | [-0.0332, +0.3105] | 0.1875 | 1.0000 | no significant difference |
| arl3_full | 10 | 1.2358 | 1.1198 | +0.1160 | [-0.0014, +0.2467] | 0.1172 | 1.0000 | no significant difference |
| arl3_shared | 10 | 1.2358 | 1.3367 | -0.1009 | [-0.4616, +0.2141] | 0.6367 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 1.2358 | 1.7861 | -0.5503 | [-0.8891, -0.1310] | 0.0469 | 0.7031 | no significant difference |
| b_delta | 10 | 1.2358 | 1.0172 | +0.2186 | [-0.0272, +0.5664] | 0.2051 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 1.2358 | 0.7120 | +0.5238 | [+0.2631, +0.8956] | 0.0020 | 0.0430 | ref worse |
| b_favor | 10 | 1.2358 | 1.2615 | -0.0257 | [-0.2659, +0.3299] | 0.9355 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 1.2358 | 1.0370 | +0.1988 | [-0.0615, +0.5506] | 0.2949 | 1.0000 | no significant difference |
| b_fedcer | 10 | 1.2358 | 2.8065 | -1.5707 | [-1.8745, -1.1413] | 0.0039 | 0.0742 | no significant difference |
| b_fedcs | 10 | 1.2358 | 2.2130 | -0.9772 | [-1.2700, -0.5742] | 0.0039 | 0.0742 | no significant difference |
| b_mimic | 10 | 1.2358 | 1.0100 | +0.2258 | [-0.0216, +0.5734] | 0.1816 | 1.0000 | no significant difference |
| b_oort | 10 | 1.2358 | 1.1419 | +0.0939 | [-0.1587, +0.4657] | 0.8594 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 1.2358 | 1.0653 | +0.1705 | [-0.1015, +0.5525] | 0.4902 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 1.2358 | 0.8334 | +0.4024 | [+0.1171, +0.8030] | 0.0098 | 0.1562 | no significant difference |
| b_powerofchoice | 10 | 1.2358 | 1.1491 | +0.0867 | [-0.1779, +0.4522] | 0.7930 | 1.0000 | no significant difference |
| b_random | 10 | 1.2358 | 1.0099 | +0.2259 | [-0.0211, +0.5734] | 0.1797 | 1.0000 | no significant difference |
| b_random_ew | 10 | 1.2358 | 0.6730 | +0.5628 | [+0.3082, +0.9222] | 0.0020 | 0.0430 | ref worse |
| b_refl | 10 | 1.2358 | 0.8248 | +0.4110 | [+0.1577, +0.7766] | 0.0039 | 0.0742 | no significant difference |
| b_refl_ew | 10 | 1.2358 | 0.3872 | +0.8486 | [+0.5727, +1.2323] | 0.0020 | 0.0430 | ref worse |
| rl_ew | 10 | 1.2358 | 1.2889 | -0.0531 | [-0.2498, +0.1279] | 0.6152 | 1.0000 | no significant difference |
| rl_light | 10 | 1.2358 | 1.0884 | +0.1474 | [-0.1946, +0.5469] | 0.5469 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 1.2358 | 1.2870 | -0.0512 | [-0.2968, +0.1607] | 0.6816 | 1.0000 | no significant difference |

## mean_fail_ratio (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 0.5519 | 0.4596 | +0.0924 | [+0.0145, +0.1871] | 0.0586 | 1.0000 | no significant difference |
| arl3_full | 10 | 0.5519 | 0.5161 | +0.0357 | [-0.0179, +0.1053] | 0.4590 | 1.0000 | no significant difference |
| arl3_shared | 10 | 0.5519 | 0.5201 | +0.0318 | [-0.0173, +0.1003] | 0.4414 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 0.5519 | 0.6474 | -0.0955 | [-0.2782, +0.0787] | 0.3379 | 1.0000 | no significant difference |
| b_delta | 10 | 0.5519 | 0.6400 | -0.0881 | [-0.2580, +0.0896] | 0.3535 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 0.5519 | 0.5604 | -0.0085 | [-0.1861, +0.1764] | 0.9277 | 1.0000 | no significant difference |
| b_favor | 10 | 0.5519 | 0.6202 | -0.0683 | [-0.2338, +0.1045] | 0.4648 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 0.5519 | 0.5760 | -0.0241 | [-0.2030, +0.1616] | 0.7988 | 1.0000 | no significant difference |
| b_fedcer | 10 | 0.5519 | 0.5735 | -0.0215 | [-0.2048, +0.1701] | 0.8281 | 1.0000 | no significant difference |
| b_fedcs | 10 | 0.5519 | 0.2222 | +0.3296 | [+0.1410, +0.5225] | 0.0078 | 0.1562 | no significant difference |
| b_mimic | 10 | 0.5519 | 0.6321 | -0.0802 | [-0.2547, +0.1020] | 0.4082 | 1.0000 | no significant difference |
| b_oort | 10 | 0.5519 | 0.5938 | -0.0419 | [-0.2115, +0.1353] | 0.6562 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 0.5519 | 0.4360 | +0.1159 | [-0.0590, +0.2972] | 0.2480 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 0.5519 | 0.6641 | -0.1122 | [-0.2794, +0.0620] | 0.2500 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 0.5519 | 0.7972 | -0.2453 | [-0.4122, -0.0717] | 0.0352 | 0.6680 | no significant difference |
| b_random | 10 | 0.5519 | 0.6319 | -0.0800 | [-0.2545, +0.1022] | 0.4082 | 1.0000 | no significant difference |
| b_random_ew | 10 | 0.5519 | 0.5561 | -0.0042 | [-0.1805, +0.1814] | 0.9688 | 1.0000 | no significant difference |
| b_refl | 10 | 0.5519 | 0.7169 | -0.1650 | [-0.3412, +0.0179] | 0.1270 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 0.5519 | 0.5904 | -0.0385 | [-0.2201, +0.1500] | 0.7090 | 1.0000 | no significant difference |
| rl_ew | 10 | 0.5519 | 0.6974 | -0.1455 | [-0.3554, +0.0837] | 0.2559 | 1.0000 | no significant difference |
| rl_light | 10 | 0.5519 | 0.2603 | +0.2916 | [+0.1135, +0.4742] | 0.0020 | 0.0430 | ref worse |
| v2_rawfail | 10 | 0.5519 | 0.4597 | +0.0922 | [+0.0296, +0.1728] | 0.0039 | 0.0820 | no significant difference |

## mean_arl_oracle (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 1.1102 | 1.1676 | -0.0575 | [-0.1281, +0.0055] | 0.1738 | 1.0000 | no significant difference |
| arl3_full | 10 | 1.1102 | 1.2022 | -0.0920 | [-0.2330, +0.0235] | 0.2559 | 1.0000 | no significant difference |
| arl3_shared | 10 | 1.1102 | 1.2514 | -0.1412 | [-0.3036, -0.0101] | 0.0898 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 1.1102 | 1.0917 | +0.0184 | [-0.3410, +0.3586] | 0.9219 | 1.0000 | no significant difference |
| b_delta | 10 | 1.1102 | 1.1961 | -0.0859 | [-0.2185, +0.0650] | 0.2871 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 1.1102 | 1.1604 | -0.0502 | [-0.1874, +0.1082] | 0.5391 | 1.0000 | no significant difference |
| b_favor | 10 | 1.1102 | 1.1029 | +0.0073 | [-0.1462, +0.1699] | 0.9277 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 1.1102 | 1.0935 | +0.0167 | [-0.0974, +0.1377] | 0.7988 | 1.0000 | no significant difference |
| b_fedcer | 10 | 1.1102 | 0.3082 | +0.8020 | [+0.6433, +0.9785] | 0.0020 | 0.0430 | ref worse |
| b_fedcs | 10 | 1.1102 | 0.9721 | +0.1381 | [-0.0330, +0.3137] | 0.1758 | 1.0000 | no significant difference |
| b_mimic | 10 | 1.1102 | 1.0920 | +0.0181 | [-0.1091, +0.1672] | 0.8164 | 1.0000 | no significant difference |
| b_oort | 10 | 1.1102 | 1.1689 | -0.0587 | [-0.1802, +0.0746] | 0.4082 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 1.1102 | 1.1070 | +0.0032 | [-0.1376, +0.1613] | 0.9727 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 1.1102 | 1.0259 | +0.0843 | [-0.0431, +0.2337] | 0.3125 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 1.1102 | 1.0822 | +0.0279 | [-0.1198, +0.1941] | 0.7500 | 1.0000 | no significant difference |
| b_random | 10 | 1.1102 | 1.0920 | +0.0182 | [-0.1091, +0.1672] | 0.8164 | 1.0000 | no significant difference |
| b_random_ew | 10 | 1.1102 | 1.0628 | +0.0474 | [-0.0804, +0.1981] | 0.5957 | 1.0000 | no significant difference |
| b_refl | 10 | 1.1102 | 1.3382 | -0.2281 | [-0.3452, -0.0829] | 0.0234 | 0.4688 | no significant difference |
| b_refl_ew | 10 | 1.1102 | 1.4611 | -0.3509 | [-0.4827, -0.2022] | 0.0059 | 0.1230 | no significant difference |
| rl_ew | 10 | 1.1102 | 1.3494 | -0.2392 | [-0.4605, -0.0475] | 0.0391 | 0.7422 | no significant difference |
| rl_light | 10 | 1.1102 | 1.1792 | -0.0690 | [-0.2330, +0.1027] | 0.4785 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 1.1102 | 1.1539 | -0.0437 | [-0.1036, +0.0228] | 0.2363 | 1.0000 | no significant difference |

## mean_cost (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 7.9929 | 8.0937 | -0.1007 | [-1.8236, +1.9963] | 0.9355 | 1.0000 | no significant difference |
| arl3_full | 10 | 7.9929 | 8.5657 | -0.5727 | [-2.2772, +1.2833] | 0.5566 | 1.0000 | no significant difference |
| arl3_shared | 10 | 7.9929 | 8.3870 | -0.3941 | [-1.4429, +0.5532] | 0.4688 | 1.0000 | no significant difference |
| b_cov_ew | 10 | 7.9929 | 9.5037 | -1.5108 | [-4.6773, +1.9274] | 0.4102 | 1.0000 | no significant difference |
| b_delta | 10 | 7.9929 | 9.8925 | -1.8996 | [-4.7029, +0.9820] | 0.2461 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 7.9929 | 11.9702 | -3.9772 | [-6.6625, -1.1781] | 0.0332 | 0.5977 | no significant difference |
| b_favor | 10 | 7.9929 | 10.4967 | -2.5038 | [-5.2540, +0.3309] | 0.1348 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 7.9929 | 11.8153 | -3.8224 | [-6.4167, -1.1625] | 0.0332 | 0.5977 | no significant difference |
| b_fedcer | 10 | 7.9929 | 11.6696 | -3.6767 | [-6.1277, -1.1812] | 0.0312 | 0.5938 | no significant difference |
| b_fedcs | 10 | 7.9929 | 7.1425 | +0.8505 | [-1.8138, +3.5257] | 0.5664 | 1.0000 | no significant difference |
| b_mimic | 10 | 7.9929 | 10.2936 | -2.3007 | [-5.1275, +0.6049] | 0.1699 | 1.0000 | no significant difference |
| b_oort | 10 | 7.9929 | 14.8909 | -6.8980 | [-9.8616, -3.8332] | 0.0078 | 0.1719 | no significant difference |
| b_oort_ew | 10 | 7.9929 | 13.7849 | -5.7920 | [-8.4990, -3.0387] | 0.0078 | 0.1719 | no significant difference |
| b_poc_ew | 10 | 7.9929 | 10.0550 | -2.0621 | [-4.7774, +0.7725] | 0.2051 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 7.9929 | 7.2311 | +0.7618 | [-1.9972, +3.6894] | 0.6250 | 1.0000 | no significant difference |
| b_random | 10 | 7.9929 | 10.2936 | -2.3007 | [-5.1275, +0.6049] | 0.1699 | 1.0000 | no significant difference |
| b_random_ew | 10 | 7.9929 | 12.1438 | -4.1508 | [-6.7981, -1.4720] | 0.0273 | 0.5469 | no significant difference |
| b_refl | 10 | 7.9929 | 7.3234 | +0.6695 | [-2.2043, +3.6134] | 0.6719 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 7.9929 | 9.4474 | -1.4545 | [-4.1628, +1.2974] | 0.3496 | 1.0000 | no significant difference |
| rl_ew | 10 | 7.9929 | 9.6283 | -1.6354 | [-6.2117, +2.5545] | 0.5215 | 1.0000 | no significant difference |
| rl_light | 10 | 7.9929 | 7.2739 | +0.7190 | [-2.0086, +3.4840] | 0.6387 | 1.0000 | no significant difference |
| v2_rawfail | 10 | 7.9929 | 7.2303 | +0.7626 | [-0.5855, +2.1799] | 0.3398 | 1.0000 | no significant difference |

## mean_latency (lower is better)

| vs | n | ref | other | diff | 95% CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| arl2_full | 10 | 3.6790 | 2.1245 | +1.5545 | [+0.3902, +2.8738] | 0.0410 | 0.7383 | no significant difference |
| arl3_full | 10 | 3.6790 | 2.6672 | +1.0117 | [+0.1884, +1.8984] | 0.0840 | 1.0000 | no significant difference |
| arl3_shared | 10 | 3.6790 | 2.6222 | +1.0568 | [+0.2097, +2.2633] | 0.0176 | 0.3340 | no significant difference |
| b_cov_ew | 10 | 3.6790 | 2.2546 | +1.4243 | [-0.2938, +3.3642] | 0.2227 | 1.0000 | no significant difference |
| b_delta | 10 | 3.6790 | 4.0696 | -0.3906 | [-2.1504, +1.5553] | 0.6992 | 1.0000 | no significant difference |
| b_delta_ew | 10 | 3.6790 | 2.3297 | +1.3492 | [-0.4375, +3.3401] | 0.2500 | 1.0000 | no significant difference |
| b_favor | 10 | 3.6790 | 4.0624 | -0.3835 | [-2.0995, +1.5011] | 0.7031 | 1.0000 | no significant difference |
| b_favor_ew | 10 | 3.6790 | 2.4908 | +1.1881 | [-0.6303, +3.2131] | 0.3086 | 1.0000 | no significant difference |
| b_fedcer | 10 | 3.6790 | 2.0578 | +1.6212 | [-0.1639, +3.6378] | 0.1738 | 1.0000 | no significant difference |
| b_fedcs | 10 | 3.6790 | 0.6692 | +3.0097 | [+1.2095, +5.0565] | 0.0039 | 0.0781 | no significant difference |
| b_mimic | 10 | 3.6790 | 4.1739 | -0.4950 | [-2.2836, +1.4798] | 0.6367 | 1.0000 | no significant difference |
| b_oort | 10 | 3.6790 | 2.8937 | +0.7852 | [-0.9433, +2.7147] | 0.4648 | 1.0000 | no significant difference |
| b_oort_ew | 10 | 3.6790 | 1.7230 | +1.9560 | [+0.1920, +3.9401] | 0.1230 | 1.0000 | no significant difference |
| b_poc_ew | 10 | 3.6790 | 2.6654 | +1.0135 | [-0.7318, +2.9913] | 0.3730 | 1.0000 | no significant difference |
| b_powerofchoice | 10 | 3.6790 | 4.8523 | -1.1733 | [-2.9046, +0.8019] | 0.2656 | 1.0000 | no significant difference |
| b_random | 10 | 3.6790 | 4.1739 | -0.4950 | [-2.2836, +1.4798] | 0.6367 | 1.0000 | no significant difference |
| b_random_ew | 10 | 3.6790 | 2.3189 | +1.3601 | [-0.4301, +3.3596] | 0.2480 | 1.0000 | no significant difference |
| b_refl | 10 | 3.6790 | 4.5053 | -0.8264 | [-2.6115, +1.1547] | 0.4336 | 1.0000 | no significant difference |
| b_refl_ew | 10 | 3.6790 | 2.3874 | +1.2916 | [-0.5021, +3.3019] | 0.2754 | 1.0000 | no significant difference |
| rl_ew | 10 | 3.6790 | 2.3940 | +1.2850 | [-0.5003, +3.2172] | 0.2500 | 1.0000 | no significant difference |
| rl_light | 10 | 3.6790 | 0.8877 | +2.7913 | [+0.9796, +4.8043] | 0.0020 | 0.0430 | ref worse |
| v2_rawfail | 10 | 3.6790 | 2.1575 | +1.5215 | [+0.5010, +2.6604] | 0.0020 | 0.0430 | ref worse |

wrote /kaggle/working/adv/s2_main_femnist/stats_c2_rl_fedprox.csv
