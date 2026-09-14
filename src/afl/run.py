"""CLI entry point.

    python -m afl.run --config configs/quick.yaml
    python -m afl.run --config configs/mnist.yaml --set policy.name=oort --tag oort_s0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import Cfg
from .engine import Experiment


def _load(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _apply_override(d: dict, expr: str) -> None:
    key, _, raw = expr.partition("=")
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        val = raw
    node = d
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = val


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[],
                    help="dotted override, e.g. --set fl.rounds=10")
    ap.add_argument("--tag")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    raw = _load(args.config)
    for expr in args.set:
        _apply_override(raw, expr)
    if args.tag:
        raw["tag"] = args.tag
    if args.seed is not None:
        raw["seed"] = args.seed
        raw.setdefault("data", {}).setdefault("seed", args.seed)
    if args.out:
        raw["out"] = args.out

    cfg = Cfg.from_dict(raw)
    exp = Experiment(cfg)
    exp.run(verbose=not args.quiet)
    path = exp.save()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
