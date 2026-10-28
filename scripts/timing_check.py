#!/usr/bin/env python3
"""Does the simulator's time and cost model scale like real work? (fourth
external review, finding 4; docs/SIMULATOR_MODEL.md)

    python scripts/timing_check.py --out results/simulator_checks/timing

The simulator assumes local training time is proportional to examples x
epochs (t_train = E * n / 1200 / speed * jitter), upload time to the uploaded
fraction of the model (t_comm = 0.25 * f / bandwidth * channel), and cost
C = E * n / 1000 + f. This measures real local training on this machine
(the FEMNIST and CIFAR-10 models, n = 100..800 examples, E = 1, 2, 4
epochs), fits time = a * n * E + b, and reports the fit and the largest
relative error; it also computes the real bytes each mode would upload.
Real seconds are machine-specific; only the shape (proportionality) is the
simulator's claim.
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

SIZES, EPOCHS, REPEATS = (100, 200, 400, 800), (1, 2, 4), 3


def measure(group: str) -> tuple[list[tuple[int, int, float]], dict]:
    import numpy as np
    import torch
    import rl2lib
    from afl.config import Cfg
    from afl.local import local_train
    from afl.models import build_model, get_flat
    from sweep import build_jobs
    torch.set_num_threads(1)
    src = rl2lib.source_group("configs/sweep_advanced.yaml", group)
    e = rl2lib.group_entry(src, {}, 1, ["rl_light"], [900])
    cfg = Cfg.from_dict(build_jobs({"seeds": [900], "sets": {}, "groups": {"g": e}}, None, None, "/x", [])[0]["raw"])
    from afl.data import build
    d = build(cfg)
    model = build_model(cfg, d["dim"], d["num_classes"])
    flat = get_flat(model)
    gen = torch.Generator().manual_seed(0)
    out = []
    for n in SIZES:
        x = torch.from_numpy(d["x_train"][:n])
        y = torch.from_numpy(d["y_train"][:n])
        for ep in EPOCHS:
            ts = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                local_train(cfg, flat, x, y, ep, d["dim"], d["num_classes"], gen)
                ts.append(time.perf_counter() - t0)
            out.append((n, ep, st.median(ts)))
    n_params = int(flat.numel())
    bytes_ = {m: int(n_params * 4 * f) for m, f in cfg.system.upload_frac.items()}
    return out, {"params": n_params, "upload_bytes": bytes_, "upload_frac": dict(cfg.system.upload_frac)}


def fit(points):
    import numpy as np
    w = np.array([n * e for n, e, _ in points], float)
    t = np.array([s for _, _, s in points])
    A = np.vstack([w, np.ones_like(w)]).T
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = a * w + b
    r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    rel = np.abs(pred - t) / t
    return a, b, float(r2), float(rel.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import platform
    lines = ["# Timing and cost scaling check", "",
             f"Machine: {platform.processor() or platform.machine()}, 1 CPU thread, median of {REPEATS} repeats. "
             "Real seconds are machine-specific; the simulator only claims proportionality.", ""]
    for group in ("s2_femnist", "s2_cifar"):
        pts, info = measure(group)
        a, b, r2, rel = fit(pts)
        lines += [f"## {group}", "", "| examples n | epochs E | n x E | measured seconds | fit a*n*E + b |",
                  "|---|---|---|---|---|"]
        lines += [f"| {n} | {e} | {n * e} | {s:.4f} | {a * n * e + b:.4f} |" for n, e, s in pts]
        lines += ["", f"Fit: time = {a:.3e} x (n x E) {'-' if b < 0 else '+'} {abs(b):.4f} s; R^2 = {r2:.4f}; largest relative error "
                      f"{rel:.1%}. The simulator's t_train is proportional to n x E (no constant term).",
                  "", f"Model parameters: {info['params']:,}; bytes uploaded per mode (float32 x upload fraction): "
                      + ", ".join(f"{m} {v:,} ({info['upload_frac'][m]:.2f})" for m, v in info["upload_bytes"].items())
                      + ". The simulator's t_comm and cost are proportional to the same fraction.", ""]
    lines += ["## Cost model", "",
              "C = E x n / 1000 + f (simulated units): linear in examples, in epochs and in the uploaded "
              "fraction, by construction (tests/test_validation_matrix.py checks it on the code). "
              "These are relative units, not joules or currency."]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
