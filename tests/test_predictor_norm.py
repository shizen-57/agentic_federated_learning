"""policy.predictor_norm (docs/DIAGNOSTIC_FINDINGS.md): standardising the
success predictor's client features must stop failure-only training from
ranking slow, low-battery clients highest, and must change nothing when off."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from afl.config import Cfg  # noqa: E402
from afl.predictor import SuccessPredictor  # noqa: E402


def _client(speed, energy, avail, rate=0.0, since=5.0):
    return ({"speed_hat": speed, "bw_hat": speed, "energy_hat": energy, "avail_hint": avail},
            {"success_rate": rate, "rounds_since_success": since})


def _trained(norm, failing):
    cfg = Cfg()
    cfg.policy.predictor_norm = norm
    p = SuccessPredictor(cfg)
    rng = np.random.default_rng(0)
    pop = [_client(rng.choice([0.35, 1.0, 2.2]) * rng.uniform(0.8, 1.2),
                   rng.uniform(0.6, 1.0), rng.uniform(0.7, 1.0)) for _ in range(100)]
    if norm == "round":
        p.set_reference(np.array([p.client_features(*c) for c in pop]))
    for _ in range(40):
        p.update(*failing(pop, rng), "standard", 0)
    return p


@pytest.mark.parametrize("who", ["random clients", "slow clients"])
def test_failures_do_not_rank_slow_low_battery_clients_first(who):
    pick = ((lambda pop, rng: pop[rng.integers(len(pop))]) if who == "random clients"
            else (lambda pop, rng: _client(0.35 * rng.uniform(0.8, 1.2), 0.6, 0.75)))
    fast, slow = _client(2.2, 1.0, 0.9), _client(0.35, 0.6, 0.75)
    old, new = _trained("none", pick), _trained("round", pick)
    # the original model inverts the ranking ...
    assert old.predict(*slow, "standard") > old.predict(*fast, "standard")
    # ... the standardised one does not
    assert new.predict(*fast, "standard") >= new.predict(*slow, "standard")


def test_off_is_the_original_feature_vector():
    cfg = Cfg()
    p = SuccessPredictor(cfg)
    obs, hist = _client(1.3, 0.8, 0.9, rate=0.25, since=3.0)
    f = p.features(obs, hist, "priority")
    expect = np.array([np.log1p(1.3), np.log1p(1.3), 0.8, 0.9, 0.25, np.tanh(0.3), 0, 0, 1.0])
    assert np.array_equal(f, expect)


def test_round_mode_needs_a_reference_and_rejects_unknown_values():
    cfg = Cfg()
    cfg.policy.predictor_norm = "round"
    with pytest.raises(RuntimeError):
        SuccessPredictor(cfg).features(*_client(1, 1, 1), "light")
    cfg.policy.predictor_norm = "bogus"
    with pytest.raises(ValueError):
        SuccessPredictor(cfg)


def test_engine_standardises_by_the_round_population(tmp_path):
    from afl.engine import Experiment
    raw = {"data": {"dataset": "synthetic", "num_clients": 12}, "out": str(tmp_path),
           "fl": {"rounds": 3, "clients_per_round": 4, "model": "mlp"},
           "policy": {"name": "rl", "predictor_norm": "round"}}
    exp = Experiment(Cfg.from_dict(raw))
    exp.run(verbose=False)
    st = exp._state(4)
    rows = np.array([exp.predictor.client_features(st["obs"][i], st["hist"][i])
                     for i in range(exp.N)])
    assert np.allclose(exp.predictor.ref_mean, rows.mean(axis=0))
    z = np.array([exp.predictor.features(st["obs"][i], st["hist"][i], "light")[:6]
                  for i in range(exp.N)])
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9)
