# Development screen v1 (exploratory; seeds 200-204)

Starved = 50+ rounds in a row with invitations and no success. Equal spend: fixed-set accuracy at the last round with cumulative cost <= 1400.

| condition | seed | successes/invites | fallback share | longest zero streak | starved | modes % L/S/P | final acc | total spend | acc at equal spend (round) |
|---|---|---|---|---|---|---|---|---|---|
| arl3_full_fm | 110 | 904/2000 | 0.32 | 7 | no | 71/11/17 | 0.733 | 1907 | 0.755 (r164) |
| arl3_full_fm | 113 | 689/2000 | 0.48 | 7 | no | 74/14/11 | 0.759 | 1671 | 0.734 (r173) |

| condition | median successes | median final acc | median acc at equal spend | starved seeds |
|---|---|---|---|---|
| arl3_full_fm | 796 | 0.746 | 0.744 | 0/2 |
