# Diagnostic runs on the department GPU (RTX 3070)

The 8 runs of `configs/diagnostic_v1.yaml` (variant `full`), run by the
user on the department server with `bash scripts/dept_gpu.sh diag full`.

- Device: NVIDIA GeForce RTX 3070 (8 GB), torch 2.5.1+cu121, Python 3.11;
  139 s wall with 4 parallel jobs.
- Code: commit d391e32, simulator hash 64487e6f86fe1001 (the predictor and
  mode-mask switches exist but are off; off reproduces earlier runs row for
  row on CPU).
- Checked on receipt: every config equals its planned job, every CSV passes
  the resume validation (200 rounds, finite, exact schema), no source change
  during a run. Archive SHA-256
  191aa6d0fdf37e7f395b0badaa104949b3aa63be1b76390a051f17769841c561.
- `sidecars.tar.gz` holds the diagnostic JSON-lines files (GPU runs are not
  bit-reproducible, so they are kept). The server's own `diag.log` is left
  out because it contains the server account's home path.

Result: none of the 8 runs stayed stuck (see `report.md`), unlike the same
runs on Kaggle's GPU and on CPU; see docs/DIAGNOSTIC_FINDINGS.md section 7.
