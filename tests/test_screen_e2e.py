"""End to end, as on the server: the real runner (diagnose_collapse.py ->
sweep.py) with a RELATIVE output folder, then the report, from the repo and
from elsewhere on a moved copy. Screen B's first report failed exactly here
(every run "different config") although all unit tests passed."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "femnist_x.npy")),
                                reason="needs the FEMNIST data")

SPEC = """version: 1
title: end-to-end check
purpose: diagnostic
source_plan: configs/sweep_advanced.yaml
source_group: s2_femnist
seeds: [900]
set: {data.split: persistent}
conditions: [arl3_full, ppo_full]
reference: arl3_full
new_conditions: [ppo_full]
variants: {full: 2}
out: unused
"""


def _py(args, cwd):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=cwd)


def _valid(r) -> bool:
    """Complete evidence -- or, while this checkout has uncommitted changes
    (as during development), that is the ONLY problem the report finds."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if not dirty:
        return r.returncode == 0 and "Evidence complete: all 2 runs" in r.stdout
    problems = [ln for ln in r.stdout.split("INCOMPLETE")[-1].splitlines() if ln.startswith("- ")]
    return r.returncode == 2 and problems == [
        "- the runs were made from a checkout with uncommitted changes (or unknown)"]


def test_runner_then_report_with_a_relative_output_folder(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(SPEC)
    rel = os.path.relpath(tmp_path / "out", ROOT)
    r = _py([os.path.join(ROOT, "scripts", "diagnose_collapse.py"), "--spec", str(spec),
             "--variant", "full", "--jobs", "2", "--set", "fl.device=cpu", "--out", rel], ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    report = os.path.join(ROOT, "scripts", "report_dev_screen.py")
    r = _py([report, os.path.join(rel, "full", "s2_femnist"), "--spec", str(spec)], ROOT)
    assert _valid(r), r.stdout[-3000:]
    # the results folder copied elsewhere (as when it is sent home) still validates
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "out", moved)
    r = _py([report, str(moved / "full" / "s2_femnist"), "--spec", str(spec)], str(tmp_path))
    assert _valid(r), r.stdout[-3000:]
