"""Opt-in diagnostic sidecars (diag.enabled; docs/DIAGNOSTIC_FINDINGS.md).

Three JSON-lines files next to a run's CSV, streamed and flushed every round:

    <tag>.diag_round.jsonl    one record per round: pool, gate, fallback,
                              invitations and outcomes, reward parts, predictor
    <tag>.diag_invite.jsonl   one record per invitation: who, how picked, which
                              mode was proposed and run, predictions for every
                              mode, what the server saw, what happened
    <tag>.diag_update.jsonl   one record per policy update (RL policies): the
                              baseline, advantage and gradient norms

Each file starts with a header line {"schema": "afl-diag", "version": ...}.
Files are written as .part and renamed when the run is saved, so a run killed
mid-way never leaves a complete-looking sidecar behind.

Recording must never change the run. Everything here only reads values the
run has already computed: no random draws, no model forwards, no predictor
updates, no environment calls that mutate state. Values marked "analysis
only" are simulator truths a real server cannot see; they are logged for
analysis and never fed to any policy.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

SCHEMA = "afl-diag"
VERSION = 1
KINDS = ("round", "invite", "update")


def clean(x):
    """JSON-safe copy: numpy/torch scalars and arrays to Python, non-finite
    floats to None (strict JSON has no NaN)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if hasattr(x, "detach"):                      # torch tensor
        x = x.detach().cpu()
        return clean(x.item() if x.numel() == 1 else x.tolist())
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def norm_state(nz, z: float | None = None) -> dict:
    """What a reward Normalizer looked like after its call this round."""
    out = {"n": int(nz.n), "mean": float(nz.mean),
           "warmup": bool(nz.n < max(nz.warmup, 2))}
    if nz.n > 1:
        out["std"] = float((nz.m2 / (nz.n - 1)) ** 0.5)
    if z is not None and nz.clip is not None:
        out["clipped"] = bool(abs(z) >= nz.clip)
    return out


class DiagWriter:
    def __init__(self, out: str, tag: str):
        os.makedirs(out, exist_ok=True)
        self.final = {k: os.path.join(out, f"{tag}.diag_{k}.jsonl") for k in KINDS}
        self.fh = {}
        for k, path in self.final.items():
            fh = open(path + ".part", "w")
            fh.write(json.dumps({"schema": SCHEMA, "version": VERSION, "kind": k,
                                 "tag": tag}) + "\n")
            self.fh[k] = fh
        self.closed = False

    def write(self, kind: str, rec: dict) -> None:
        self.fh[kind].write(json.dumps(clean(rec), allow_nan=False) + "\n")

    def flush(self) -> None:
        for fh in self.fh.values():
            fh.flush()

    def close(self) -> None:
        """Finish the files: rename .part to the final names."""
        if self.closed:
            return
        for k, fh in self.fh.items():
            fh.close()
            os.replace(self.final[k] + ".part", self.final[k])
        self.closed = True


def read(path: str) -> tuple[dict, list[dict]]:
    """(header, records) of one sidecar; refuses a different schema/version."""
    with open(path) as fh:
        header = json.loads(fh.readline())
        if header.get("schema") != SCHEMA or header.get("version") != VERSION:
            raise ValueError(f"{path}: not an {SCHEMA} v{VERSION} sidecar")
        return header, [json.loads(line) for line in fh if line.strip()]
