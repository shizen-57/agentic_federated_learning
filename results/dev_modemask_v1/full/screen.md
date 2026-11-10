# Development screen v1 (exploratory; seeds 200-204)

Starved = 50+ rounds in a row with invitations and no success. Equal spend: fixed-set accuracy at the last round with cumulative cost <= 1400.

| condition | seed | successes/invites | fallback share | longest zero streak | starved | modes % L/S/P | final acc | total spend | acc at equal spend (round) |
|---|---|---|---|---|---|---|---|---|---|
| arl3_full | 205 | 1184/2000 | 0.05 | 2 | no | 2/95/3 | 0.700 | 2558 | 0.697 (r110) |
| arl3_full | 206 | 1285/2000 | 0.02 | 1 | no | 1/26/73 | 0.690 | 4477 | 0.646 (r81) |
| arl3_full | 207 | 1357/2000 | 0.01 | 0 | no | 93/2/4 | 0.759 | 1617 | 0.757 (r169) |
| arl3_full | 208 | 752/2000 | 0.49 | 6 | no | 83/12/5 | 0.737 | 1175 | 0.737 (r200) |
| arl3_full | 209 | 772/2000 | 0.40 | 23 | no | 27/39/34 | 0.662 | 2340 | 0.646 (r164) |
| arl3_full_fm | 205 | 839/2000 | 0.33 | 4 | no | 77/16/7 | 0.797 | 1687 | 0.754 (r174) |
| arl3_full_fm | 206 | 1214/2000 | 0.02 | 1 | no | 66/30/3 | 0.787 | 1970 | 0.751 (r138) |
| arl3_full_fm | 207 | 1268/2000 | 0.00 | 0 | no | 34/65/1 | 0.643 | 2254 | 0.700 (r122) |
| arl3_full_fm | 208 | 943/2000 | 0.30 | 4 | no | 65/10/26 | 0.738 | 2243 | 0.667 (r144) |
| arl3_full_fm | 209 | 857/2000 | 0.37 | 5 | no | 64/29/7 | 0.683 | 1727 | 0.681 (r178) |
| rl_light | 205 | 1322/2000 | 0.01 | 0 | no | 100/0/0 | 0.739 | 1411 | 0.751 (r198) |
| rl_light | 206 | 1249/2000 | 0.00 | 0 | no | 100/0/0 | 0.723 | 1456 | 0.737 (r191) |
| rl_light | 207 | 1340/2000 | 0.00 | 0 | no | 100/0/0 | 0.707 | 1426 | 0.752 (r196) |
| rl_light | 208 | 1472/2000 | 0.00 | 0 | no | 100/0/0 | 0.707 | 1369 | 0.707 (r200) |
| rl_light | 209 | 1542/2000 | 0.00 | 0 | no | 100/0/0 | 0.655 | 1315 | 0.655 (r200) |
| v2_rawfail | 205 | 1191/2000 | 0.05 | 2 | no | 59/17/24 | 0.778 | 2385 | 0.690 (r79) |
| v2_rawfail | 206 | 1333/2000 | 0.02 | 1 | no | 52/38/10 | 0.694 | 2198 | 0.738 (r128) |
| v2_rawfail | 207 | 1503/2000 | 0.01 | 0 | no | 88/9/3 | 0.715 | 1556 | 0.708 (r175) |
| v2_rawfail | 208 | 841/2000 | 0.45 | 5 | no | 72/12/16 | 0.716 | 1084 | 0.716 (r200) |
| v2_rawfail | 209 | 854/2000 | 0.41 | 40 | no | 1/98/2 | 0.668 | 1718 | 0.646 (r175) |

| condition | median successes | median final acc | median acc at equal spend | starved seeds |
|---|---|---|---|---|
| arl3_full | 1184 | 0.700 | 0.697 | 0/5 |
| arl3_full_fm | 943 | 0.738 | 0.700 | 0/5 |
| rl_light | 1340 | 0.707 | 0.737 | 0/5 |
| v2_rawfail | 1191 | 0.715 | 0.708 | 0/5 |

Decision rule: (1) no revision seed starves: pass; (2) median successes >= unchanged ARL-v3: FAIL; (3) median final accuracy within 0.02 of unchanged ARL-v3: pass.
**Screen FAILED** (passing means worth a confirmation study on fresh seeds, not better).
