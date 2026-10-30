#!/usr/bin/env python3
"""The fixed data roles of the persistent split, written down and checked
(docs/RL2_CONFIRMATION.md, "Reserved data").

    python scripts/data_roles.py            # check against configs/rl2_data_roles.json
    python scripts/data_roles.py --write    # (re)write it -- only before any confirmation

FEMNIST: practice / development / confirmation writers. Pooled test sets
(CIFAR-10): validation / development-test / confirmation-test example ids.
Records each id list's size and SHA-256 and checks that the lists are
disjoint; the freeze pins this file, so the confirmation's held-out data is
fixed by id, not just by a seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

PATH = os.path.join(ROOT, "configs", "rl2_data_roles.json")


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()


def roles() -> dict:
    from afl.config import DataCfg
    from afl.data import femnist_role_writers, pooled_confirm_roles, pooled_roles
    d, val_frac = DataCfg(), 0.3          # the s2 settings: fl.val_frac 0.3
    out = {"split_seed": d.split_seed, "practice_writer_share": d.practice_writer_share,
           "confirm_share": d.confirm_share, "val_frac": val_frac}
    wpath = os.path.join(ROOT, "data", "femnist_writer.npy")
    if os.path.exists(wpath):
        n = int(np.load(wpath).max()) + 1
        prac, dev, conf = femnist_role_writers(n, d.split_seed, d.practice_writer_share, d.confirm_share)
        out["femnist_writers"] = {k: {"n": len(v), "sha256": _h(v)}
                                  for k, v in (("practice", prac), ("dev", dev), ("confirm", conf))}
        out["femnist_disjoint"] = not (set(prac) & set(dev) or set(prac) & set(conf) or set(dev) & set(conf))
    n_test = 10000                        # CIFAR-10's test set
    v = pooled_roles(n_test, d.split_seed, val_frac)
    c = pooled_confirm_roles(n_test, d.split_seed, d.confirm_share)
    ids = {"validation": np.where(v)[0], "dev_test": np.where(~v & ~c)[0], "confirm_test": np.where(~v & c)[0]}
    out["cifar10_examples"] = {k: {"n": len(x), "sha256": _h(x)} for k, x in ids.items()}
    out["cifar10_disjoint"] = sum(len(x) for x in ids.values()) == n_test == len(set().union(*map(set, ids.values())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    now = roles()
    if not (now.get("femnist_disjoint", True) and now["cifar10_disjoint"]):
        raise SystemExit("the data roles overlap")
    if args.write:
        with open(PATH, "w") as fh:
            json.dump(now, fh, indent=1, sort_keys=True)
        print(f"wrote {PATH}")
        return 0
    with open(PATH) as fh:
        saved = json.load(fh)
    diff = [k for k in saved if saved[k] != now.get(k)]
    print("data roles match the recorded ones" if not diff else f"DIFFERENT: {diff}")
    return 1 if diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
