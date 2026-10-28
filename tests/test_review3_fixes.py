"""Third external review: the budget projection respects SAFE-PPO's allowed
modes and the curriculum scores the job that ran (finding 1); the screen
report accepts no experimental override (2) and counts abstention as
starvation (3); ARL-v4 invents no alternative when none is affordable (4)."""
from __future__ import annotations

import csv
import os
import subprocess
import sys

from test_review2_fixes import _engine
from test_safe_ppo import _ppo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

MODES = ("light", "standard", "priority")


def _round(exp, t=1):
    exp.env.new_round(t)
    return exp._state(t)


def test_budget_projection_never_executes_a_masked_mode(tmp_path):
    exp = _engine(tmp_path)
    state = _round(exp)
    sel = list(range(5))
    assign = {i: "standard" for i in sel}
    cap = 0.5 * sum(exp.env.cost(i, "standard") for i in sel)
    # the reviewer's case: Standard allowed (p 0.9), Light masked (p 0.1)
    only_std = {i: frozenset({"standard"}) for i in sel}
    keep, out, dropped = exp._enforce_budget(sel, assign, state, {}, cap, only_std)
    assert all(out[i] == "standard" for i in keep)           # never moved to Light
    assert dropped == len(sel) - len(keep) > 0              # dropped instead
    assert sum(exp.env.cost(i, out[i]) for i in keep) <= cap + 1e-9
    # with Light allowed (or no mask: the original behaviour) it downgrades
    for allowed in ({i: frozenset(MODES) for i in sel}, None):
        keep2, out2, _ = exp._enforce_budget(sel, assign, state, {}, cap, allowed)
        assert "light" in out2.values() and len(keep2) >= len(keep)
    assert (exp._enforce_budget(sel, assign, state, {}, cap, {i: frozenset(MODES) for i in sel})
            == exp._enforce_budget(sel, assign, state, {}, cap, None))


def test_executed_modes_are_allowed_and_ppo_keeps_the_sampled_action(tmp_path):
    exp = _engine(tmp_path, mode_mask="feasible", mode_mask_z=0.5, hard_budget_mult=0.6,
                  cost_target_mult=0.0)
    pol = exp.policy
    real_act, real_obs, real_budget = pol.act, pol.observe_outcome, exp._enforce_budget
    seen = {"downgraded": 0, "rounds": 0}

    def act(state, t):
        sel, assign = real_act(state, t)
        seen["sampled"] = dict(assign)
        return sel, assign

    def budget(sel, assign, state, fracs, cap=None, allowed=None):
        assert allowed is pol.allowed_modes
        keep, out, dropped = real_budget(sel, assign, state, fracs, cap, allowed)
        assert all(out[i] in allowed[i] for i in keep)
        seen["downgraded"] += sum(out[i] != assign[i] for i in keep) + dropped
        return keep, out, dropped

    def observe(state, action, reward, info):
        real_obs(state, action, reward, info)
        entry = pol.buf[-1] if pol.buf else None
        if entry is not None and entry["act"]:
            chosen = entry["picks"] + entry["inp"]["fill"]
            assert {i: pol.modes[m] for i, m in zip(chosen, entry["modes"])} == seen["sampled"]
            seen["rounds"] += 1

    pol.act, pol.observe_outcome, exp._enforce_budget = act, observe, budget
    exp.run(verbose=False)
    assert seen["rounds"] > 0 and seen["downgraded"] > 0      # the cap did bind


def test_curriculum_calibrates_the_job_that_ran():
    pol = _ppo(2, 2, **{"policy.curriculum": True})
    pol._p_round = {0: {"light": 0.1, "standard": 0.9, "priority": 0.5},
                    1: {"light": 0.8, "standard": 0.6, "priority": 0.2}}
    # sampled Standard for client 0, but the budget made it Light
    pol._curriculum(([0, 1], {0: "light", 1: "light"}), {"success": {0: 0, 1: 1}})
    assert pol.cal == [(0.1, 0), (0.8, 1)]


def test_arl4_gap_is_zero_without_a_feasible_affordable_alternative():
    from afl import metrics as M
    ep = {"light": 1, "standard": 2, "priority": 4}
    p = {"light": 0.9, "standard": 0.9, "priority": 0.9}
    assert M.arl4_expected_gap(p, "light", ep, 0.5, set()) == 0.0        # the reviewer's case
    low = {"light": 0.3, "standard": 0.4, "priority": 0.2}                # nothing confidently feasible
    assert M.arl4_expected_gap(low, "light", ep, 0.5) == 0.0
    assert M.arl4_expected_gap(low, "light", ep, 0.5, {"light", "standard"}) == 0.0
    assert M.arl4_expected_gap(p, "light", ep, 0.5, {"light", "standard"}) > 0   # real alternative


def _csv(path, rows):
    from afl.engine import LOG_FIELDS
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LOG_FIELDS)
        for t, (sel, succ, online) in enumerate(rows, 1):
            row = dict.fromkeys(LOG_FIELDS, 0.0)
            row.update(round=t, n_selected=sel, n_success=succ, n_online=online, cost=1.0,
                       empty_round=int(not sel), accuracy=0.5, accuracy_fixed=0.5)
            w.writerow([row[k] for k in LOG_FIELDS])
    return path


def test_complete_abstention_is_starvation(tmp_path):
    import report_dev_screen as rep
    m = rep.measures(_csv(tmp_path / "a.csv", [(0, 0, 50)] * 200))      # the reviewer's fixture
    assert m["starved"] and m["longest"] == 200 and m["longest_empty"] == 200 and m["empty"] == 200
    mixed = [(10, 0, 50)] * 30 + [(0, 0, 50)] * 30 + [(10, 3, 50)] * 140
    m = rep.measures(_csv(tmp_path / "b.csv", mixed))
    assert m["starved"] and m["longest"] == 60
    assert (m["longest_fail"], m["longest_empty"]) == (30, 30)          # each part alone < 50
    # nobody online is not the scheduler's fault: skipped, neither extends nor resets
    m = rep.measures(_csv(tmp_path / "c.csv", [(0, 0, 0)] * 200))
    assert not m["starved"] and m["offline"] == 200
    gap = [(10, 0, 50)] * 30 + [(0, 0, 0)] * 5 + [(10, 0, 50)] * 30 + [(10, 3, 50)] * 135
    assert rep.measures(_csv(tmp_path / "d.csv", gap))["longest"] == 60


def test_runner_refuses_experimental_overrides():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "diagnose_collapse.py"),
                        "--spec", os.path.join(ROOT, "configs", "rl2_screen_b.yaml"), "--variant",
                        "full", "--dry-run", "--set", "data.split=per_seed"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "spec's `set`" in r.stderr
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "diagnose_collapse.py"),
                        "--spec", os.path.join(ROOT, "configs", "rl2_screen_b.yaml"), "--variant",
                        "full", "--dry-run", "--set", "fl.device=auto"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
