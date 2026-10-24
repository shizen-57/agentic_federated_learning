# Device traces

`fedscale_device_sample.npz` is a random sample (20,000 of 500,000 entries,
`numpy.random.default_rng(0)`) of FedScale's `client_device_capacity` trace:
per-device computation latency (`computation_ms`, ms per sample, measured on
real smartphones via AI Benchmark) and network throughput
(`communication_kbps`, from MobiPerf measurements).

- Source: https://github.com/SymbioticLab/FedScale,
  `benchmark/dataset/data/device_info/client_device_capacity`
- Paper: Lai et al., "FedScale: Benchmarking Model and System Performance of
  Federated Learning at Scale", ICML 2022.
- License: Apache License 2.0, see `LICENSE-FEDSCALE`. Modifications: random
  subsample, converted from a pickled dict to float32 numpy arrays.

Used when `system.device_trace: fedscale` (see `src/afl/sim.py`).
