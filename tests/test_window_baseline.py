"""Multi-round REINFORCE window with policy.prior_baseline (rl.py::_update_window).

The baseline used for an action must not depend on that action. Folding an
earlier return of the same window into the baseline first does exactly that,
because the earlier return contains the later rewards. These tests compute
the expected implemented gradient by exact enumeration and compare it with
the true policy gradient.

Note: every s2 run used horizon 1, so this path does not touch those results.
"""
from __future__ import annotations

import itertools
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from afl.policies.rl import RLPolicy  # noqa: E402


def _fake_policy(prior_baseline=True, gamma=1.0, baseline=0.0, baseline_n=0):
    net = nn.Module()
    net.theta = nn.Parameter(torch.zeros(()))
    return SimpleNamespace(
        cfg=SimpleNamespace(reward=SimpleNamespace(gamma=gamma),
                            policy=SimpleNamespace(prior_baseline=prior_baseline,
                                                   entropy_coef=0.0)),
        net=net, opt=torch.optim.SGD(net.parameters(), lr=1.0),
        baseline=baseline, baseline_n=baseline_n, _window=[])


def _expected_gradient(reward_fn, steps=2, **kw):
    """E over all action sequences of the implemented update direction for a
    Bernoulli(sigmoid(theta)) action per step at theta = 0."""
    total = 0.0
    for acts in itertools.product((0, 1), repeat=steps):
        pol = _fake_policy(**kw)
        p = torch.sigmoid(pol.net.theta)
        for t, a in enumerate(acts):
            logp = torch.log(p) if a else torch.log(1 - p)
            pol._window.append((logp, torch.zeros(()), reward_fn(t, acts)))
        RLPolicy._update_window(pol)
        total += float(pol.net.theta.detach()) * 0.5 ** steps   # SGD lr 1: theta = -dloss
    return total


@pytest.mark.parametrize("start", [(0.0, 0), (0.7, 5)])
def test_reward_only_at_the_second_step(start):
    # r1 = 0, r2 = a2. True gradient d/dtheta E[a2] = p(1-p) = 0.25. The old
    # in-window update gave 0.0 (the second advantage became g2 - g1 = 0).
    g = _expected_gradient(lambda t, a: float(a[1]) if t == 1 else 0.0,
                           baseline=start[0], baseline_n=start[1])
    assert g == pytest.approx(0.25, abs=1e-6)


def test_reward_at_every_step():
    # r_t = a_t: true gradient d/dtheta E[a1 + a2] = 2 * 0.25 = 0.5.
    assert _expected_gradient(lambda t, a: float(a[t])) == pytest.approx(0.5, abs=1e-6)


def test_baseline_statistics_update_after_the_window():
    pol = _fake_policy(gamma=0.5, baseline=0.1, baseline_n=1)
    for r in (0.1, 0.2, 0.4):                     # a partial final window of 3
        pol._window.append((pol.net.theta * 1.0, torch.zeros(()), r))
    RLPolicy._update_window(pol)
    # returns 0.3, 0.4, 0.4 (gamma 0.5); every advantage is against 0.1
    # (small numbers, so the gradient clip at norm 5 does not act)
    assert float(pol.net.theta.detach()) == pytest.approx(0.2 + 0.3 + 0.3, abs=1e-6)
    assert pol.baseline_n == 4
    assert pol.baseline == pytest.approx((0.1 + 0.3 + 0.4 + 0.4) / 4)
    assert pol._window == []


def test_without_prior_baseline_the_original_rule_is_unchanged():
    pol = _fake_policy(prior_baseline=False, gamma=1.0)
    for r in (1.0, 3.0):
        pol._window.append((pol.net.theta * 1.0, torch.zeros(()), r))
    RLPolicy._update_window(pol)
    # returns 4, 3; running mean then advantage: 4-4 = 0, 3-3.5 = -0.5
    assert float(pol.net.theta.detach()) == pytest.approx(-0.5)
    assert pol.baseline == pytest.approx(3.5) and pol.baseline_n == 2


def test_horizon_one_never_uses_the_window():
    from afl.config import Cfg
    from afl.engine import Experiment
    from afl.run import _load
    from afl.conditions import CONDITIONS, apply
    raw = _load(os.path.join(ROOT, "configs", "quick.yaml"))
    apply(raw, CONDITIONS["c2_rl_fedprox"])
    raw.setdefault("policy", {})["prior_baseline"] = True
    raw["fl"]["rounds"] = 3
    exp = Experiment(Cfg.from_dict(raw))
    assert exp.cfg.policy.horizon == 1
    exp.policy._update_window = lambda: (_ for _ in ()).throw(AssertionError("window used"))
    exp.run(verbose=False)
    assert 1 <= exp.policy.baseline_n <= 3 and math.isfinite(exp.policy.baseline)
