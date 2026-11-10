"""Seed registry guard (scripts/seeds.py)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import seeds  # noqa: E402


def test_kinds():
    assert seeds.kind(113) == "used" and seeds.kind(205) == "used"
    assert seeds.kind(210) == "dev" and seeds.kind(300) == "confirmation"
    assert seeds.kind(1500) == "practice"


@pytest.mark.parametrize("seed,purpose,ok", [
    (210, "dev", True), (113, "dev", False), (300, "dev", False), (1000, "dev", False),
    (113, "diagnostic", True), (300, "diagnostic", False), (1200, "diagnostic", False),
    (1200, "practice", True), (210, "practice", False),
])
def test_check(seed, purpose, ok):
    if ok:
        seeds.check([seed], purpose)
    else:
        with pytest.raises(ValueError):
            seeds.check([seed], purpose)


def test_confirmation_needs_a_frozen_manifest():
    with pytest.raises(ValueError):
        seeds.check([300], "confirmation")
    seeds.check([300], "confirmation", frozen_manifest="configs/rl2_frozen.json")
    with pytest.raises(ValueError):
        seeds.check([210], "confirmation", frozen_manifest="x")


def test_runner_refuses_confirmation_seeds(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("version: 1\nsource_plan: configs/sweep_advanced.yaml\nsource_group: s2_femnist\n"
                    "seeds: [300]\nconditions: [rl_light]\nvariants: {full: 200}\nout: x\n")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "diagnose_collapse.py"),
                        "--spec", str(spec), "--variant", "full", "--dry-run"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "reserved for confirmation" in r.stderr
