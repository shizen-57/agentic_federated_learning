"""Tests for the parts that would silently produce a wrong thesis result.

The priority here is not coverage. It is the handful of invariants that, if they
broke, would still produce plausible-looking CSVs and plots: the common-random-
numbers property that ARL* depends on, the submodularity of the coverage
function, the sign convention of CV, and the fact that the three aggregation
rules actually differ.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from afl.aggregate import aggregate           # noqa: E402
from afl.config import Cfg                    # noqa: E402
from afl.data import class_sketch, dirichlet_partition  # noqa: E402
from afl.engine import Experiment             # noqa: E402
from afl import metrics as M                  # noqa: E402
from afl.predictor import SuccessPredictor    # noqa: E402
from afl.sim import ClientEnv                 # noqa: E402


def _cfg(**over):
    base = {"data": {"num_clients": 12}, "fl": {"rounds": 4, "clients_per_round": 4}}
    base.update(over)
    return Cfg.from_dict(base)


# --------------------------------------------------------------- partition

def test_dirichlet_partition_is_disjoint_and_non_iid():
    y = np.tile(np.arange(10), 400)
    parts = dirichlet_partition(y, 12, 0.2, seed=0, min_samples=40, max_samples=200)
    flat = np.concatenate(parts)
    assert len(flat) == len(set(flat.tolist())), "a sample was given to two clients"
    sk = class_sketch(y, parts, 10)
    # With alpha=0.2 clients should be far from the uniform 0.1-per-class mix.
    assert sk.max(axis=1).mean() > 0.4


# -------------------------------------------------------------- coverage

def test_coverage_is_submodular():
    sketch = np.eye(5)
    first = M.representation_value(sketch, [], 0)
    again = M.representation_value(sketch, [0], 0)
    assert first > 0 and again == 0, "a redundant client must add no value"
    rare = M.representation_value(sketch, [0, 1], 4)
    assert rare > 0, "a client holding an uncovered class must add value"


def test_cv_sign_convention():
    even = M.cv_fairness(np.array([4.0, 4.0, 4.0]))
    skewed = M.cv_fairness(np.array([12.0, 0.0, 0.0]))
    assert even < skewed, "CV must increase as contributions concentrate"
    assert M.cv_fairness(np.zeros(3)) == 0.0


def test_jain_bounds():
    assert M.jain(np.array([3.0, 3.0, 3.0])) == 1.0
    assert M.jain(np.array([9.0, 0.0, 0.0])) < 0.4


# ------------------------------------------------ common random numbers

def test_outcome_is_deterministic_within_a_round():
    cfg = _cfg()
    env = ClientEnv(cfg, np.full(12, 200.0), seed=0)
    env.new_round(5)
    a = [env.outcome(i, "standard") for i in range(12)]
    b = [env.outcome(i, "standard") for i in range(12)]
    assert [x["success"] for x in a] == [x["success"] for x in b]


def test_counterfactual_uses_the_same_draws():
    """The whole point of CRN: changing the mode must be the only difference.

    An unavailable client stays unavailable under every mode, and a client that
    fails only on the deadline under Priority must do strictly better under
    Light, because the jitter and channel draws are shared.
    """
    cfg = _cfg()
    env = ClientEnv(cfg, np.full(12, 400.0), seed=1)
    env.new_round(3)
    for i in range(12):
        light = env.outcome(i, "light")
        prio = env.outcome(i, "priority")
        if light["reason"] == "unavailable":
            assert prio["reason"] == "unavailable"
        else:
            assert light["latency"] <= prio["latency"] + 1e-9


def test_oracle_eligibility_implies_some_mode_succeeds():
    cfg = _cfg()
    env = ClientEnv(cfg, np.full(12, 300.0), seed=2)
    env.new_round(1)
    for i in range(12):
        if env.oracle_eligible(i):
            assert any(env.outcome(i, m)["success"] for m in cfg.system.modes)


# ------------------------------------------------------------------- ARL

def test_arl_only_counts_avoidable_failures():
    values = {0: 1.0, 1: 2.0, 2: 4.0}
    eligible = {0: 1, 1: 0, 2: 1}
    success = {0: 0, 1: 0, 2: 1}
    # 0 was eligible and failed (avoidable); 1 was not eligible (unavoidable);
    # 2 succeeded. Only 0 counts.
    assert M.arl_online(values, eligible, success) == 1.0


# ----------------------------------------------------------- aggregation

def test_aggregators_differ_under_unequal_local_steps():
    ups = [torch.ones(4), torch.ones(4) * 2]
    n_i, steps = [100.0, 100.0], [2, 8]
    avg = aggregate("fedavg", ups, n_i, steps)
    nova = aggregate("fednova", ups, n_i, steps)
    assert not torch.allclose(avg, nova), "FedNova must renormalise by local steps"


def test_importance_reweighting_favours_unlikely_clients():
    ups = [torch.ones(3), torch.zeros(3)]
    n_i, steps = [100.0, 100.0], [4, 4]
    plain = aggregate("fedavg", ups, n_i, steps)
    corrected = aggregate("fedavg", ups, n_i, steps, success_probs=[0.1, 0.9])
    assert corrected[0] > plain[0], "the rarely-succeeding client must gain weight"


# ------------------------------------------------------------- predictor

def test_predictor_learns_a_separable_signal():
    cfg = _cfg()
    pred = SuccessPredictor(cfg)
    fast = {"speed_hat": 3.0, "bw_hat": 3.0, "energy_hat": 1.0, "avail_hint": 0.9}
    slow = {"speed_hat": 0.2, "bw_hat": 0.2, "energy_hat": 0.1, "avail_hint": 0.2}
    hist = {"success_rate": 0.5, "rounds_since_success": 0.0, "selected_frac": 0.5}
    for _ in range(400):
        pred.update(fast, hist, "standard", 1)
        pred.update(slow, hist, "standard", 0)
    assert pred.predict(fast, hist, "standard") > pred.predict(slow, hist, "standard") + 0.3


def test_calibration_error_is_zero_for_a_perfect_predictor():
    probs = [0.0] * 50 + [1.0] * 50
    out = [0] * 50 + [1] * 50
    assert M.calibration_error(probs, out) < 1e-9


# ---------------------------------------------------------------- engine

def test_every_policy_completes_a_run():
    for name in ("random", "fedcs", "powerofchoice", "oort", "rl"):
        cfg = _cfg(policy={"name": name})
        rows = Experiment(cfg).run(verbose=False)
        assert len(rows) == cfg.fl.rounds
        assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)


def test_run_is_reproducible_under_a_fixed_seed():
    a = Experiment(_cfg()).run(verbose=False)
    b = Experiment(_cfg()).run(verbose=False)
    assert [r["accuracy"] for r in a] == [r["accuracy"] for r in b]


def test_different_seeds_give_different_runs():
    a = Experiment(_cfg()).run(verbose=False)
    cfg = _cfg()
    cfg.seed = 5
    cfg.data.seed = 5
    b = Experiment(cfg).run(verbose=False)
    assert [r["accuracy"] for r in a] != [r["accuracy"] for r in b]


def test_selection_respects_the_round_budget():
    cfg = _cfg()
    rows = Experiment(cfg).run(verbose=False)
    assert all(r["n_selected"] == cfg.fl.clients_per_round for r in rows)
    assert all(r["n_success"] <= r["n_selected"] for r in rows)
