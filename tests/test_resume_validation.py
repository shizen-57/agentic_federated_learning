"""Resume validation (scripts/sweep.py::run_status): a damaged earlier result
must be re-run, never reused and never crash the sweep."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PLAN = ("seeds: [0]\nsets: {}\ngroups:\n  g: {config: configs/quick.yaml, "
        "set: {fl.rounds: 3}, conditions: [b_random]}\n")


def _sweep(plan, out, *extra):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "sweep.py"), "--plan", str(plan),
           "--out", str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def good_run(tmp_path_factory):
    """One real, valid 3-round run made by the current code."""
    d = tmp_path_factory.mktemp("good")
    plan = d / "plan.yaml"
    plan.write_text(PLAN)
    r = _sweep(plan, d / "out")
    assert r.returncode == 0, r.stdout + r.stderr
    return plan, d / "out"


@pytest.fixture
def run(good_run, tmp_path):
    """A fresh copy of the valid run, plus its job, for one test to damage."""
    from afl.provenance import code_hash
    from afl.run import _load
    from sweep import build_jobs
    plan, src = good_run
    out = tmp_path / "out"
    _copy(src, out)
    raw = build_jobs(_load(str(plan)), None, None, str(out), [])[0]["raw"]
    base = out / "g" / "b_random_s0"
    return raw, base.with_suffix(".csv"), base.with_suffix(".config.json"), code_hash()


def _copy(src, out):
    """Copy a finished run folder; the saved config records its output path,
    so point it at the copy (as a relocation would need)."""
    shutil.copytree(src, out)
    for meta_p in out.glob("*/*.config.json"):
        meta = json.loads(meta_p.read_text())
        meta["out"] = str(meta_p.parent)
        meta_p.write_text(json.dumps(meta))


def _edit_csv(path, fn):
    lines = path.read_text().splitlines()
    path.write_text("\n".join(fn(lines)) + "\n")


def _set_cell(lines, row, col, value):
    header = lines[0].split(",")
    cells = lines[row].split(",")
    cells[header.index(col)] = value
    lines[row] = ",".join(cells)
    return lines


def test_valid_result_is_reused(run):
    from sweep import run_status
    raw, *_, code = run
    assert run_status(raw, code) == "ok"
    assert run_status(raw, None) == "ok"


@pytest.mark.parametrize("damage,reason", [
    (lambda l: _set_cell(l, 2, "accuracy", "nan"), "non-finite value"),
    (lambda l: _set_cell(l, 3, "reward", "inf"), "non-finite value"),
    (lambda l: _set_cell(l, 1, "cost", "-Infinity"), "non-finite value"),
    (lambda l: _set_cell(l, 2, "cost", "abc"), "non-numeric value"),
    (lambda l: _set_cell(l, 2, "round", "2.5"), "fractional round"),
    (lambda l: _set_cell(l, 3, "round", "2"), "incomplete"),            # duplicate round
    (lambda l: [l[0], l[2], l[1], l[3]], "incomplete"),                  # out of order
    (lambda l: l[:3], "incomplete"),                                     # missing last round
    (lambda l: l[:3] + [l[3][: len(l[3]) // 2]], "truncated row"),       # cut mid-row
    (lambda l: [], "empty csv"),
])
def test_damaged_csv_is_rerun(run, damage, reason):
    from sweep import run_status
    raw, csv_p, _, code = run
    _edit_csv(csv_p, damage)
    assert run_status(raw, code) == reason


def test_column_rules(run):
    from sweep import run_status
    raw, csv_p, _, code = run
    keep = csv_p.read_text()
    # A column the code never writes is not accepted.
    _edit_csv(csv_p, lambda l: [l[0] + ",extra"] + [r + ",0" for r in l[1:]])
    assert run_status(raw, code) == "unexpected columns"
    # A column dropped: with the same code the schema must match exactly ...
    csv_p.write_text(keep)
    def drop(col):
        return lambda l: [",".join(c for k, c in enumerate(r.split(","))
                                   if k != l[0].split(",").index(col)) for r in l]
    _edit_csv(csv_p, drop("gate_filled"))
    assert run_status(raw, code) == "different columns"
    # ... while an older code version's documented schema is fine under --ignore-code,
    assert run_status(raw, None) == "ok"
    # but never without the core columns.
    _edit_csv(csv_p, drop("accuracy"))
    assert run_status(raw, None) == "missing columns"
    csv_p.write_text(keep)
    _edit_csv(csv_p, lambda l: [l[0] + ",round"] + [r + ",1" for r in l[1:]])
    assert run_status(raw, None) == "duplicate columns"


@pytest.mark.parametrize("text,reason", [
    ("{not json", "unreadable config"),
    ("", "unreadable config"),
    ("[1, 2, 3]", "malformed config"),
    ('{"fl": 5}', "malformed config"),
])
def test_damaged_config_is_rerun_not_a_crash(run, text, reason):
    from sweep import run_status
    raw, _, meta_p, code = run
    meta_p.write_text(text)
    assert run_status(raw, code) == reason


def test_missing_config_field_is_a_different_config(run):
    from sweep import run_status
    raw, _, meta_p, code = run
    meta = json.loads(meta_p.read_text())
    del meta["reward"]
    meta_p.write_text(json.dumps(meta))
    assert run_status(raw, code) == "different config"


def test_sweep_survives_corrupt_metadata_and_quarantines_it(good_run, tmp_path):
    plan, src = good_run
    out = tmp_path / "out"
    _copy(src, out)
    meta_p = out / "g" / "b_random_s0.config.json"
    meta_p.write_text("{corrupt")
    r = _sweep(plan, out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "will be re-run: unreadable config" in r.stdout
    assert "moved 1 earlier result(s) to quarantine/" in r.stdout
    kept = list((out / "g" / "quarantine").glob("*/b_random_s0.config.json"))
    assert len(kept) == 1 and kept[0].read_text() == "{corrupt"
    assert "1 already done" in _sweep(plan, out).stdout


def test_sweep_never_writes_into_an_original_evaluation(good_run, tmp_path):
    plan, _ = good_run
    out = tmp_path / "adv"
    out.mkdir()
    (out / "PROVENANCE.md").write_text("original evaluation")   # as in results/adv
    r = _sweep(plan, out)
    assert r.returncode == 1 and "refusing to write into an original evaluation" in r.stdout
    assert not (out / "g").exists()
