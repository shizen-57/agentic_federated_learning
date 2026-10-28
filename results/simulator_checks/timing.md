# Timing and cost scaling check

Machine: x86_64, 1 CPU thread, median of 3 repeats. Real seconds are machine-specific; the simulator only claims proportionality.

## s2_femnist

| examples n | epochs E | n x E | measured seconds | fit a*n*E + b |
|---|---|---|---|---|
| 100 | 1 | 100 | 0.0399 | 0.0359 |
| 100 | 2 | 200 | 0.0746 | 0.0764 |
| 100 | 4 | 400 | 0.1484 | 0.1573 |
| 200 | 1 | 200 | 0.0839 | 0.0764 |
| 200 | 2 | 400 | 0.1529 | 0.1573 |
| 200 | 4 | 800 | 0.3070 | 0.3191 |
| 400 | 1 | 400 | 0.1668 | 0.1573 |
| 400 | 2 | 800 | 0.3250 | 0.3191 |
| 400 | 4 | 1600 | 0.5965 | 0.6427 |
| 800 | 1 | 800 | 0.3202 | 0.3191 |
| 800 | 2 | 1600 | 0.6847 | 0.6427 |
| 800 | 4 | 3200 | 1.2934 | 1.2900 |

Fit: time = 4.045e-04 x (n x E) - 0.0045 s; R^2 = 0.9969; largest relative error 10.0%. The simulator's t_train is proportional to n x E (no constant term).

Model parameters: 222,078; bytes uploaded per mode (float32 x upload fraction): light 222,078 (0.25), standard 444,156 (0.50), priority 888,312 (1.00). The simulator's t_comm and cost are proportional to the same fraction.

## s2_cifar

| examples n | epochs E | n x E | measured seconds | fit a*n*E + b |
|---|---|---|---|---|
| 100 | 1 | 100 | 0.1377 | 0.1050 |
| 100 | 2 | 200 | 0.2063 | 0.2034 |
| 100 | 4 | 400 | 0.4043 | 0.4003 |
| 200 | 1 | 200 | 0.1971 | 0.2034 |
| 200 | 2 | 400 | 0.4183 | 0.4003 |
| 200 | 4 | 800 | 0.7925 | 0.7941 |
| 400 | 1 | 400 | 0.3930 | 0.4003 |
| 400 | 2 | 800 | 0.8115 | 0.7941 |
| 400 | 4 | 1600 | 1.5815 | 1.5816 |
| 800 | 1 | 800 | 0.7335 | 0.7941 |
| 800 | 2 | 1600 | 1.5663 | 1.5816 |
| 800 | 4 | 3200 | 3.1731 | 3.1568 |

Fit: time = 9.844e-04 x (n x E) + 0.0065 s; R^2 = 0.9993; largest relative error 23.8%. The simulator's t_train is proportional to n x E (no constant term).

Model parameters: 188,810; bytes uploaded per mode (float32 x upload fraction): light 188,810 (0.25), standard 377,620 (0.50), priority 755,240 (1.00). The simulator's t_comm and cost are proportional to the same fraction.

## Cost model

C = E x n / 1000 + f (simulated units): linear in examples, in epochs and in the uploaded fraction, by construction (tests/test_validation_matrix.py checks it on the code). These are relative units, not joules or currency.
