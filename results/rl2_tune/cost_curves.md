# M7 stage 2 (FEMNIST, seeds 225-239): accuracy vs cumulative cost

Median over seeds of fixed-set accuracy at the last round with cumulative cost <= the budget (simulated cost units), the median actual spend at that round, and the seeds that reached it. eq5 (development endpoint): mean accuracy of the last 5 rounds within 1,400, with the mean actual spend of those rounds.

| series | seeds | B=500: acc (spend) [n] | B=800: acc (spend) [n] | B=1100: acc (spend) [n] | B=1400: acc (spend) [n] | B=1700: acc (spend) [n] | B=2000: acc (spend) [n] | eq5 (spend) | total spend |
|---|---|---|---|---|---|---|---|---|---|
| rl_light | 15 | 0.684 (496) [15] | 0.708 (795) [15] | 0.726 (1099) [15] | 0.729 (1371) [15] | 0.737 (1371) [15] | 0.737 (1371) [15] | 0.728 (1357) | 1371 |
| untuned c00 | 15 | 0.675 (494) [15] | 0.718 (796) [15] | 0.747 (1094) [15] | 0.766 (1394) [15] | 0.769 (1696) [15] | 0.773 (1893) [15] | 0.765 (1374) | 1893 |
| tuned c34 | 15 | 0.666 (496) [15] | 0.730 (797) [15] | 0.756 (1096) [15] | 0.772 (1397) [15] | 0.775 (1693) [15] | 0.783 (1818) [15] | 0.763 (1380) | 1818 |
