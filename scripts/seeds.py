"""Seed bookkeeping (configs/seed_registry.yaml).

check(seeds, purpose) raises if a seed is used for the wrong purpose:
confirmation seeds need a frozen manifest, practice seeds are never evaluated,
and dev runs may not reuse a confirmation or practice seed.
"""
from __future__ import annotations

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "configs", "seed_registry.yaml")


def registry(path: str = PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _in(seed: int, rng) -> bool:
    return rng[0] <= seed <= rng[1]


def kind(seed: int, reg: dict | None = None) -> str:
    reg = reg or registry()
    for name in ("confirmation", "practice", "dev"):
        if _in(seed, reg[name]):
            return name
    if any(_in(seed, r) for r in reg["used"]):
        return "used"
    return "unregistered"


def check(seeds, purpose: str, frozen_manifest: str | None = None, reg: dict | None = None) -> None:
    """purpose: 'diagnostic' (inspected seeds allowed), 'dev', 'practice', 'confirmation'."""
    reg = reg or registry()
    for s in seeds:
        k = kind(int(s), reg)
        if k == "confirmation" and (purpose != "confirmation" or not frozen_manifest):
            raise ValueError(f"seed {s} is reserved for confirmation; it needs a frozen manifest")
        if k == "practice" and purpose != "practice":
            raise ValueError(f"seed {s} is a practice seed; practice episodes are never evaluated")
        if purpose == "practice" and k != "practice":
            raise ValueError(f"practice episodes must use practice seeds, not {s} ({k})")
        if purpose == "dev" and k != "dev":
            raise ValueError(f"dev runs must use dev seeds, not {s} ({k})")
        if purpose == "confirmation" and k != "confirmation":
            raise ValueError(f"confirmation must use fresh confirmation seeds, not {s} ({k})")
