"""RL2 M4: the safety layer -- Lagrangian cost constraint, trust curriculum,
and the uncertainty margin on the feasible-mode mask."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from test_safe_ppo import _ppo
from test_sixth_review import _state


def test_dual_ascent_raises_the_price_when_overspending_and_drops_it_to_zero():
    pol = _ppo(4, 2, **{"policy.cost_target_mult": 1.0, "policy.lagrange_lr": 0.1})
    pol.cost_target = 10.0
    r = pol.shaped_reward(1.0, 15.0)                  # lam was 0: no penalty yet
    assert r == pytest.approx(1.0) and pol.lam == pytest.approx(0.05)
    r = pol.shaped_reward(1.0, 15.0)
    assert r == pytest.approx(1.0 - 0.05 * 1.5) and pol.lam == pytest.approx(0.10)
    for _ in range(10):
        pol.shaped_reward(1.0, 5.0)                   # under target
    assert pol.lam == 0.0
    off = _ppo(4, 2)
    assert off.shaped_reward(1.0, 99.0) == 1.0        # constraint off


def test_spend_target_comes_from_the_light_cost_scale():
    pol = _ppo(4, 2, **{"policy.cost_target_mult": 1.25})
    st = _state(4)
    st["cost_ref"] = 8.0
    pol._inputs(st, 1)
    assert pol.cost_target == pytest.approx(10.0)


def test_calibration_error():
    from afl.policies.safe_rl import SafePPOPolicy
    ce = SafePPOPolicy.calibration_error
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    good = [(float(a), int(rng.random() < a)) for a in p]
    bad = [(float(a), 0) for a in p]
    assert ce(good) < 0.03 and ce(bad) > 0.4


def test_curriculum_unlocks_on_calibration_and_locks_again():
    pol = _ppo(4, 2, **{"policy.curriculum": True, "policy.curr_min": 20, "policy.curr_err": 0.1})
    assert pol.level == 0
    st = _state(4)
    inp = pol._inputs(st, 1)
    assert not inp["pm"][:, 1:].any() and inp["pm"][:, 0].all()     # Light only
    pol._pred = {0: 0.5, 1: 0.5}
    for k in range(10):                                  # 20 well-calibrated outcomes
        pol._curriculum(([0, 1], {}), {"success": {0: k % 2, 1: (k + 1) % 2}})
    assert pol.level == 1
    inp = pol._inputs(st, 2)
    assert inp["pm"][:, :2].all() and not inp["pm"][:, 2].any()
    pol._pred = {0: 0.9, 1: 0.9}
    for _ in range(5):                                   # predicted 0.9, all fail
        pol._curriculum(([0, 1], {}), {"success": {}})
    assert pol.level == 0


def test_mask_margin_uses_the_posterior_sd():
    pol = _ppo(3, 2, **{"policy.mode_mask": "feasible", "policy.mode_mask_z": 1.0})
    st = _state(3)
    st["p_succ"][0] = {"light": 0.9, "standard": 0.6, "priority": 0.55}
    st["p_sd"] = {i: {"light": 0.05, "standard": 0.2, "priority": 0.01} for i in range(3)}
    row = pol._mode_row(st, 0, torch.zeros(3))
    assert (row > -1).tolist() == [True, False, True]  # standard: 0.6 - 0.2 < 0.5


def test_safety_layer_runs_end_to_end_and_reports(tmp_path):
    from afl.config import Cfg
    from afl.diagnostics import read
    from afl.engine import Experiment
    raw = {"data": {"dataset": "synthetic", "num_clients": 20}, "out": str(tmp_path),
           "fl": {"rounds": 12, "clients_per_round": 5, "model": "mlp", "clean_rng": True},
           "policy": {"name": "safe_ppo", "predictor": "v2", "gate_mode": "thompson",
                      "strict_gate": True, "mode_mask": "feasible", "mode_mask_z": 0.5,
                      "cost_target_mult": 1.0, "curriculum": True, "curr_min": 10,
                      "encoder": "attention", "memory": True},
           "diag": {"enabled": True}}
    exp = Experiment(Cfg.from_dict(raw))
    rows = exp.run(verbose=False)
    exp.save()
    _, upd = read(str(tmp_path / "run.diag_update.jsonl"))
    assert all("lam" in u and "level" in u for u in upd)
    assert exp.policy.cost_target > 0 and np.isfinite(rows[-1]["accuracy"])
