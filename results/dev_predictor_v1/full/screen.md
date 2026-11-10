# Development screen v1 (exploratory; seeds 200-204)

Starved = 50+ rounds in a row with invitations and no success. Equal spend: fixed-set accuracy at the last round with cumulative cost <= 1400.

| condition | seed | successes/invites | fallback share | longest zero streak | starved | modes % L/S/P | final acc | total spend | acc at equal spend (round) |
|---|---|---|---|---|---|---|---|---|---|
| arl3_full | 200 | 1303/2000 | 0.01 | 0 | no | 81/14/4 | 0.785 | 1775 | 0.786 (r157) |
| arl3_full | 201 | 1325/2000 | 0.01 | 0 | no | 4/91/5 | 0.676 | 2668 | 0.664 (r100) |
| arl3_full | 202 | 1276/2000 | 0.01 | 0 | no | 80/13/7 | 0.777 | 1847 | 0.758 (r136) |
| arl3_full | 203 | 859/2000 | 0.39 | 7 | no | 81/9/10 | 0.744 | 1147 | 0.744 (r200) |
| arl3_full | 204 | 1374/2000 | 0.01 | 0 | no | 14/26/60 | 0.682 | 3915 | 0.659 (r88) |
| arl3_full_pn | 200 | 1340/2000 | 0.00 | 0 | no | 1/98/1 | 0.738 | 2626 | 0.715 (r104) |
| arl3_full_pn | 201 | 1363/2000 | 0.02 | 0 | no | 2/38/60 | 0.584 | 4069 | 0.641 (r74) |
| arl3_full_pn | 202 | 1389/2000 | 0.04 | 0 | no | 3/51/46 | 0.719 | 3826 | 0.676 (r58) |
| arl3_full_pn | 203 | 1251/2000 | 0.01 | 0 | no | 20/7/72 | 0.696 | 4095 | 0.714 (r87) |
| arl3_full_pn | 204 | 1460/2000 | 0.00 | 0 | no | 1/65/33 | 0.701 | 3451 | 0.666 (r65) |
| rl_light | 200 | 1313/2000 | 0.01 | 0 | no | 100/0/0 | 0.782 | 1448 | 0.749 (r192) |
| rl_light | 201 | 1353/2000 | 0.00 | 0 | no | 100/0/0 | 0.737 | 1411 | 0.744 (r198) |
| rl_light | 202 | 1351/2000 | 0.01 | 0 | no | 100/0/0 | 0.787 | 1377 | 0.787 (r200) |
| rl_light | 203 | 1379/2000 | 0.00 | 0 | no | 100/0/0 | 0.781 | 1398 | 0.781 (r200) |
| rl_light | 204 | 1602/2000 | 0.00 | 0 | no | 100/0/0 | 0.679 | 1273 | 0.679 (r200) |
| v2_rawfail | 200 | 1285/2000 | 0.01 | 0 | no | 71/26/3 | 0.747 | 1877 | 0.762 (r147) |
| v2_rawfail | 201 | 1362/2000 | 0.01 | 0 | no | 86/8/6 | 0.756 | 1678 | 0.695 (r159) |
| v2_rawfail | 202 | 1337/2000 | 0.01 | 0 | no | 85/3/12 | 0.783 | 1891 | 0.761 (r128) |
| v2_rawfail | 203 | 1303/2000 | 0.03 | 0 | no | 80/6/14 | 0.760 | 1914 | 0.769 (r133) |
| v2_rawfail | 204 | 1421/2000 | 0.01 | 1 | no | 81/4/15 | 0.741 | 1984 | 0.724 (r112) |

| condition | median successes | median final acc | median acc at equal spend | starved seeds |
|---|---|---|---|---|
| arl3_full | 1303 | 0.744 | 0.744 | 0/5 |
| arl3_full_pn | 1363 | 0.701 | 0.676 | 0/5 |
| rl_light | 1353 | 0.781 | 0.749 | 0/5 |
| v2_rawfail | 1337 | 0.756 | 0.761 | 0/5 |

Decision rule: (1) no revision seed starves: pass; (2) median successes >= unchanged ARL-v3: pass; (3) median final accuracy within 0.02 of unchanged ARL-v3: FAIL.
**Screen FAILED** (passing means worth a confirmation study on fresh seeds, not better).
