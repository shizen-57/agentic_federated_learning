"""RL2 M5/M5b: ARL-v4 and the validation-only protection reward."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from afl import metrics as M  # noqa: E402

EP = {"light": 1, "standard": 2, "priority": 4}


def test_stuck_state_is_visible_and_grows():
    C = 5
    cov_ema, peak, dur = np.full(C, 0.4), np.full(C, 0.4), np.zeros(C)
    seen = []
    for _ in range(30):                               # nothing delivered
        cov_ema = 0.9 * cov_ema
        peak, dur = M.deficit_update(cov_ema, peak, dur, np.zeros(C))
        seen.append(M.deficit_term(cov_ema, peak, dur))
    assert all(b > a for a, b in zip(seen, seen[1:])) and seen[-1] > 2.0
    # normal service keeps it low
    cov_ema, peak, dur = np.full(C, 0.4), np.full(C, 0.4), np.zeros(C)
    peak, dur = M.deficit_update(cov_ema, peak, dur, np.full(C, 0.4))
    assert M.deficit_term(cov_ema, peak, dur) == pytest.approx(0.0)


def test_both_failed_holders_of_a_rare_class_are_charged():
    sk = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])    # clients 0 and 1 hold the rare class
    phi = M.coverage_shapley(sk, [0, 1, 2])
    p = {"light": 0.9, "standard": 0.8, "priority": 0.1}
    for i in (0, 1):                                 # each sent on Priority (infeasible)
        assert phi[i] > 0
        assert phi[i] * M.arl4_expected_gap(p, "priority", EP, 0.5) > 0


def test_unaffordable_heavier_mode_is_never_rewarded_and_optimum_costs_nothing():
    p = {"light": 0.95, "standard": 0.9, "priority": 0.85}
    afford = {"light", "standard"}
    best = M.arl4_expected_gap(p, "standard", EP, 0.5, afford)
    assert best == 0.0                                 # the constrained optimum
    assert M.arl4_expected_gap(p, "priority", EP, 0.5, afford) == 0.0    # not rewarded either
    assert M.arl4_expected_gap(p, "light", EP, 0.5, afford) > 0          # gave work up
    # without a budget the optimum is priority
    assert M.arl4_expected_gap(p, "priority", EP, 0.5) == 0.0
    assert M.arl4_expected_gap(p, "standard", EP, 0.5) > 0


def test_unsafe_modes_are_not_the_counterfactual():
    p = {"light": 0.9, "standard": 0.45, "priority": 0.2}   # only light is confidently feasible
    assert M.arl4_expected_gap(p, "light", EP, 0.5) == 0.0
    assert M.arl4_expected_gap(p, "priority", EP, 0.5) > 0


def test_difference_credit():
    sk = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    w = np.array([1.0, 2.0])
    both = M.difference_credit(sk, [0, 1, 2], w)
    assert both[0] == both[1] == 0.0 and both[2] == pytest.approx(1.0)
    one = M.difference_credit(sk, [0, 2], w)
    assert one[0] == pytest.approx(2.0)


def _raw(tmp_path, **pol):
    return {"data": {"dataset": "synthetic", "num_clients": 16}, "out": str(tmp_path),
            "fl": {"rounds": 6, "clients_per_round": 4, "model": "mlp", "clean_rng": True,
                   "val_frac": 0.3},
            "policy": {"name": "safe_ppo", "predictor": "v2", "per_client_credit": True,
                       "credit_total": True, **pol},
            "reward": {"damage_metric": "arl_v4", "batch_client_norm": True, "robust_norm": True,
                       "protect_weight": 0.5}}


def test_reward_reads_validation_data_only(tmp_path):
    from afl.config import Cfg
    from afl.engine import Experiment
    exp = Experiment(Cfg.from_dict(_raw(tmp_path)))
    used = []
    orig_eval, orig_grad = exp._evaluate, exp._val_grad

    def ev(full=True, idx=None):
        used.append("val" if idx is exp.val_idx else "reported")
        return orig_eval(full=full, idx=idx)
    exp._evaluate = ev
    exp.run(verbose=False)
    vg = set(exp._val_grad_idx.tolist())
    assert vg <= set(exp.val_idx.tolist())
    assert not vg & set(exp.test_idx.tolist())
    assert used.count("val") == 6                      # one validation pass per round feeds the reward
    rows = exp.rows
    assert all(np.isfinite(r["arl_v4_loss"]) and r["arl_v4_deficit"] >= 0 for r in rows)


def test_needs_a_validation_set(tmp_path):
    from afl.config import Cfg
    from afl.engine import Experiment
    raw = _raw(tmp_path)
    raw["fl"]["val_frac"] = 0.0
    with pytest.raises(ValueError):
        Experiment(Cfg.from_dict(raw))
