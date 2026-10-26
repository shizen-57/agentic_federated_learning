"""Regressions for misleading verdicts, evaluation changes and failure accounting."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from test_advanced import ROOT, Experiment, _fake_s2_runs, _tiny
import prespec_s2
from report import invitation_failure_rate, rounds_to_target


@pytest.mark.parametrize("col,diff,expected", [
    ("accuracy", -0.8, "significantly worse"),
    ("accuracy", 0.8, "significantly better"),
    ("fail_ratio", -0.8, "significantly better"),
    ("fail_ratio", 0.8, "significantly worse"),
    ("accuracy", 0.0, "inconclusive"),
])
def test_two_sided_verdict_has_the_correct_direction(col, diff, expected):
    assert prespec_s2.conclusion(col, diff, 0.001, "two-sided") == expected
    assert prespec_s2.conclusion(col, diff, 0.1, "two-sided") == "inconclusive"
    assert prespec_s2.conclusion(col, diff, 0.001, "two-sided", False) == "no verdict"


def test_one_sided_superiority_does_not_claim_inferiority():
    assert prespec_s2.conclusion("accuracy", -0.8, 0.001, "greater") == "inconclusive"


@pytest.mark.parametrize("family", ["D", "D2"])
def test_complete_but_losing_method_is_reported_as_worse(tmp_path, monkeypatch, capsys, family):
    jobs = prespec_s2.load_plan(str(tmp_path))
    tests = prespec_s2.FAMILIES[family]
    # Reference accuracy is 10% vs 90%, or failure rate is 90% vs 10%.
    values = (0.1, 0.9) if family == "D" else (0.9, 0.1)
    _fake_s2_runs(tmp_path, jobs, tests,
                  lambda c: values[0] if c == "v2_rawfail" else values[1])
    monkeypatch.setattr(sys, "argv", ["prespec_s2.py", "--root", str(tmp_path)])
    prespec_s2.main()
    section = capsys.readouterr().out.split(f"## Family {family} ")[1].split("## Family")[0]
    assert "INCOMPLETE" not in section
    assert section.count("| significantly worse |") == len(tests)
    assert "significantly better" not in section


def test_unset_frozen_hash_refuses_every_verdict(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prespec_s2, "FROZEN_CODE_HASH", None)
    monkeypatch.setattr(sys, "argv", ["prespec_s2.py", "--root", str(tmp_path)])
    assert prespec_s2.main() == 1
    assert "no verdict" in capsys.readouterr().out
    runs, problems, _ = prespec_s2.runs({}, "s2_cifar", "arl3_full", "acc_worst10")
    assert not runs and problems


def test_modified_manifest_refuses_verdict(tmp_path, monkeypatch, capsys):
    path = tmp_path / "manifest.json"
    path.write_bytes(Path(prespec_s2.MANIFEST_PATH).read_bytes() + b"\n")
    monkeypatch.setattr(prespec_s2, "MANIFEST_PATH", str(path))
    monkeypatch.setattr(sys, "argv", ["prespec_s2.py", "--root", str(tmp_path)])
    assert prespec_s2.main() == 1
    assert "manifest fingerprint mismatch" in capsys.readouterr().out


@pytest.mark.parametrize("changed", ["configs/sweep_advanced.yaml", "configs/adv/cifar10.yaml",
                                    "scripts/sweep.py"])
def test_live_plan_inputs_cannot_replace_frozen_inputs(tmp_path, monkeypatch, changed):
    for name in prespec_s2.MANIFEST_INPUTS:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(ROOT) / name, dest)
    with (tmp_path / changed).open("a") as fh:
        fh.write("\n# altered after freezing\n")
    monkeypatch.setattr(prespec_s2, "ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="differs from the frozen manifest"):
        prespec_s2.load_plan(str(tmp_path / "runs"))


@pytest.mark.parametrize("val_frac,subset", [(0.3, 50), (0.0, 50), (0.3, 0)])
def test_fixed_accuracy_uses_same_examples_even_on_final_round(val_frac, subset, monkeypatch):
    exp = Experiment(_tiny({"b_random": None, "fl.val_frac": val_frac,
                            "fl.eval_subset": subset}, rounds=2))
    original = exp._evaluate
    recorded = []
    def evaluate(full=True, idx=None):
        result = original(full=full, idx=idx)
        if idx is None and not full:
            used = exp._last_idx
            recorded.append((used.cpu().tolist() if used is not None else
                             list(range(len(exp.y_test))), result[0]))
        return result
    monkeypatch.setattr(exp, "_evaluate", evaluate)
    rows = exp.run(verbose=False)
    assert recorded[0][0] == recorded[1][0]
    assert [r["accuracy_fixed"] for r in rows] == [r[1] for r in recorded]
    assert all(r["eval_fixed_n"] == len(recorded[0][0]) for r in rows)
    assert rows[-1]["accuracy"] == pytest.approx(original(full=True)[0])
    # Client metrics must still see the full final held-out population.
    assert len(exp._last_ok) == (len(exp.test_idx) if exp.test_idx is not None else len(exp.y_test))


def _budget_run(root, name, costs, fixed, headline):
    with (root / f"{name}_s0.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["round", "cost", "accuracy_fixed", "accuracy"])
        wr.writeheader()
        for t, (cost, a, h) in enumerate(zip(costs, fixed, headline), 1):
            wr.writerow(dict(round=t, cost=cost, accuracy_fixed=a, accuracy=h))


def test_budget_and_target_use_fixed_series_not_full_final_score(tmp_path):
    _budget_run(tmp_path, "b_random", [5, 5], [0.3, 0.4], [0.3, 0.99])
    _budget_run(tmp_path, "c2_rl_fedprox", [9, 9], [0.5, 0.6], [0.8, 0.99])
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "budget_s2.py"),
                           "--dir", str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "target accuracy 0.360 (accuracy_fixed)" in proc.stdout
    line = next(l for l in proc.stdout.splitlines() if l.startswith("| c2_rl_fedprox "))
    assert "| 0.5000 |" in line and "90%" in line and line.endswith("| 100% | 9 |")
    assert math.isnan(rounds_to_target([
        {"round": 1, "accuracy_fixed": 0.2, "accuracy": 0.2},
        {"round": 2, "accuracy_fixed": 0.3, "accuracy": 0.99}], 0.8))


def test_budget_refuses_legacy_series_and_empty_directory(tmp_path):
    command = [sys.executable, os.path.join(ROOT, "scripts", "budget_s2.py"), "--dir", str(tmp_path)]
    assert subprocess.run(command, capture_output=True).returncode == 1
    (tmp_path / "b_random_s0.csv").write_text("round,cost,accuracy\n1,1,0.5\n")
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 1 and "accuracy_fixed" in proc.stdout


def test_empty_online_pool_has_no_failed_invitations(monkeypatch):
    exp = Experiment(_tiny({"b_random": None, "system.checkin_pool": True}, rounds=1))
    original = exp.env.new_round
    def offline(t):
        original(t)
        exp.env._draw["available"][:] = False
    monkeypatch.setattr(exp.env, "new_round", offline)
    row = exp.run_round(1)
    assert row["n_selected"] == row["n_success"] == 0
    assert row["fail_ratio"] == 0 and row["empty_round"] == 1
    assert np.isfinite(row["reward"])


def test_failure_summary_weights_invitations_and_marks_empty_runs():
    rows = [{"n_selected": 1, "n_success": 0}, {"n_selected": 9, "n_success": 9},
            {"n_selected": 0, "n_success": 0}]
    assert invitation_failure_rate(rows) == pytest.approx(0.1)
    assert math.isnan(invitation_failure_rate(rows[-1:]))


def test_rejected_updates_count_toward_churn_and_success_resets_streak(monkeypatch):
    import afl.engine as engine
    exp = Experiment(_tiny({"b_random": None, "system.churn_after": 2,
                            "system.churn_prob": 1.0}, rounds=4))
    monkeypatch.setattr(exp.policy, "act", lambda state, t: ([0], {0: "light"}))
    monkeypatch.setattr(exp.env, "outcome", lambda *a, **kw:
                        {"success": 1, "reason": "ok", "latency": 1.0})
    # Reject using the real norm acceptance rule, then permit one successful
    # contribution before two consecutive rejected attempts cause departure.
    monkeypatch.setattr(engine, "local_train", lambda *a, **kw:
                        (exp.global_flat.new_ones(exp.global_flat.shape), 1))
    for t, limit, expected_streak in [(1, 0, 1), (2, 1e9, 0), (3, 0, 1), (4, 0, 2)]:
        exp.cfg.system.norm_clip = limit
        row = exp.run_round(t)
        assert row["n_success"] == int(limit > 0)
        assert exp.env.fail_streak[0] == expected_streak
        assert bool(exp.env.departed[0]) == (t == 4)
