A: cpu torch 2.14.1+cpu code a1be0f1e34704d16
B: NVIDIA GeForce RTX 3070 torch 2.5.1+cu121 code a1be0f1e34704d16

| run (mean over seeds) | seeds | acc10 A / B | successes A / B | cost A / B | heavy share A / B | material |
|---|---|---|---|---|---|---|
| cifar_light | 3/3 | 0.331 / 0.327 | 400 / 367 | 428 / 442 | 0.00 / 0.00 | no |
| cifar_tuned | 3/3 | 0.400 / 0.395 | 327 / 320 | 598 / 603 | 0.24 / 0.19 | no |
| femnist_light | 3/3 | 0.648 / 0.646 | 393 / 371 | 428 / 435 | 0.00 / 0.00 | no |
| femnist_tuned | 3/3 | 0.683 / 0.683 | 335 / 338 | 534 / 599 | 0.14 / 0.24 | no |

0 group(s) with a material difference (limits: accuracy 0.03, successes and cost 15%)
