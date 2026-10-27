# Development screen v1 (exploratory; seeds 200-204)

Starved = 50+ rounds in a row with invitations and no success. Equal spend: fixed-set accuracy at the last round with cumulative cost <= 1400.

| condition | seed | successes/invites | fallback share | longest zero streak | starved | modes % L/S/P | final acc | total spend | acc at equal spend (round) |
|---|---|---|---|---|---|---|---|---|---|
| arl3_full_pg | 110 | 1224/2000 | 0.00 | 0 | no | 91/2/7 | 0.754 | 1754 | 0.721 (r149) |
| arl3_full_pg | 113 | 1249/2000 | 0.00 | 1 | no | 88/6/6 | 0.766 | 1773 | 0.747 (r146) |

| condition | median successes | median final acc | median acc at equal spend | median worst-10% client acc | starved seeds |
|---|---|---|---|---|---|
| arl3_full_pg | 1236 | 0.760 | 0.734 | 0.525 | 0/2 |
