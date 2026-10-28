# Screen B (FEMNIST, seeds 215-219): accuracy vs cumulative cost

Median over seeds of fixed-set accuracy at the last round with cumulative cost <= the budget (simulated cost units), the median actual spend at that round, and the seeds that reached it. eq5 (development endpoint): mean accuracy of the last 5 rounds within 1,400, with the mean actual spend of those rounds.

| series | seeds | B=500: acc (spend) [n] | B=800: acc (spend) [n] | B=1100: acc (spend) [n] | B=1400: acc (spend) [n] | B=1700: acc (spend) [n] | B=2000: acc (spend) [n] | eq5 (spend) | total spend |
|---|---|---|---|---|---|---|---|---|---|
| rl_light | 5 | 0.678 (497) [5] | 0.752 (796) [5] | 0.714 (1098) [5] | 0.783 (1395) [5] | 0.767 (1413) [5] | 0.767 (1413) [5] | 0.774 (1381) | 1413 |
| arl3_full_pg | 5 | 0.617 (496) [5] | 0.738 (799) [5] | 0.737 (1095) [5] | 0.746 (1397) [5] | 0.770 (1694) [5] | 0.772 (1879) [5] | 0.763 (1379) | 1879 |
| ppo_full | 5 | 0.610 (495) [5] | 0.719 (794) [5] | 0.776 (1093) [5] | 0.770 (1395) [5] | 0.786 (1697) [5] | 0.763 (1902) [5] | 0.759 (1375) | 1902 |
| ppo_full_mimic | 5 | 0.669 (495) [5] | 0.738 (798) [5] | 0.766 (1093) [5] | 0.780 (1396) [5] | 0.785 (1693) [5] | 0.792 (1836) [5] | 0.778 (1376) | 1836 |
