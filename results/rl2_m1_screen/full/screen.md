# Development screen v1 (exploratory; seeds 200-204)

Starved = 50+ rounds in a row with invitations and no success. Equal spend: fixed-set accuracy at the last round with cumulative cost <= 1400.

| condition | seed | successes/invites | fallback share | longest zero streak | starved | modes % L/S/P | final acc | total spend | acc at equal spend (round) |
|---|---|---|---|---|---|---|---|---|---|
| arl3_full | 210 | 585/2000 | 0.68 | 11 | no | 77/22/1 | 0.741 | 970 | 0.741 (r200) |
| arl3_full | 211 | 197/2000 | 0.86 | 98 | YES | 2/97/1 | 0.596 | 660 | 0.596 (r200) |
| arl3_full | 212 | 1348/2000 | 0.01 | 1 | no | 6/88/6 | 0.767 | 2765 | 0.738 (r97) |
| arl3_full | 213 | 803/2000 | 0.45 | 5 | no | 75/13/12 | 0.748 | 1196 | 0.748 (r200) |
| arl3_full | 214 | 1157/2000 | 0.02 | 1 | no | 10/76/14 | 0.764 | 2965 | 0.712 (r82) |
| arl3_full_pg | 210 | 1084/2000 | 0.00 | 0 | no | 77/13/10 | 0.735 | 2065 | 0.765 (r114) |
| arl3_full_pg | 211 | 1175/2000 | 0.00 | 1 | no | 89/9/2 | 0.775 | 1664 | 0.769 (r162) |
| arl3_full_pg | 212 | 1211/2000 | 0.00 | 0 | no | 84/13/4 | 0.792 | 1817 | 0.795 (r144) |
| arl3_full_pg | 213 | 1196/2000 | 0.00 | 1 | no | 54/44/2 | 0.768 | 2043 | 0.718 (r113) |
| arl3_full_pg | 214 | 957/2000 | 0.00 | 1 | no | 53/31/16 | 0.739 | 2610 | 0.772 (r139) |
| rl_light | 210 | 1547/2000 | 0.01 | 0 | no | 100/0/0 | 0.639 | 1261 | 0.639 (r200) |
| rl_light | 211 | 1513/2000 | 0.00 | 0 | no | 100/0/0 | 0.708 | 1370 | 0.708 (r200) |
| rl_light | 212 | 1349/2000 | 0.01 | 0 | no | 100/0/0 | 0.790 | 1460 | 0.786 (r191) |
| rl_light | 213 | 1351/2000 | 0.01 | 0 | no | 100/0/0 | 0.754 | 1357 | 0.754 (r200) |
| rl_light | 214 | 1498/2000 | 0.00 | 0 | no | 100/0/0 | 0.765 | 1393 | 0.765 (r200) |
| v2_rawfail | 210 | 523/2000 | 0.63 | 69 | YES | 0/98/2 | 0.649 | 1240 | 0.649 (r200) |
| v2_rawfail | 211 | 402/2000 | 0.77 | 11 | no | 75/21/4 | 0.723 | 785 | 0.723 (r200) |
| v2_rawfail | 212 | 1396/2000 | 0.01 | 0 | no | 93/6/2 | 0.761 | 1540 | 0.793 (r179) |
| v2_rawfail | 213 | 469/2000 | 0.74 | 9 | no | 84/11/5 | 0.744 | 925 | 0.744 (r200) |
| v2_rawfail | 214 | 1389/2000 | 0.02 | 0 | no | 93/4/4 | 0.778 | 1618 | 0.756 (r168) |

| condition | median successes | median final acc | median acc at equal spend | median worst-10% client acc | starved seeds |
|---|---|---|---|---|---|
| arl3_full | 803 | 0.748 | 0.738 | 0.483 | 1/5 |
| arl3_full_pg | 1175 | 0.768 | 0.769 | 0.529 | 0/5 |
| rl_light | 1498 | 0.754 | 0.754 | 0.460 | 0/5 |
| v2_rawfail | 523 | 0.744 | 0.744 | 0.511 | 1/5 |

Decision rule: (1) no revision seed starves: pass; (2) median successes >= unchanged ARL-v3: pass; (3) median final accuracy within 0.02 of unchanged ARL-v3: pass.
**Screen PASSED** (passing means worth a confirmation study on fresh seeds, not better).
