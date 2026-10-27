"""RL2 M1: predictor v2 (smooth cold start, per-client correction, uncertainty,
no ranking inversion) and the Thompson gate (no absorbing fallback)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from afl.config import Cfg  # noqa: E402
from afl.predictor import SuccessPredictorV2  # noqa: E402


def _client(speed, energy=0.9, avail=0.85):
    return ({"speed_hat": speed, "bw_hat": speed, "energy_hat": energy, "avail_hint": avail},
            {"success_rate": 0.0, "rounds_since_success": 5.0})


def _pred():
    p = SuccessPredictorV2(Cfg())
    rng = np.random.default_rng(0)
    pop = [_client(rng.choice([0.35, 1.0, 2.2]) * rng.uniform(0.8, 1.2), rng.uniform(0.6, 1.0))
           for _ in range(100)]
    p.set_reference(np.array([p.client_features(*c) for c in pop]))
    return p, pop, rng


def test_cold_start_is_smooth():
    p, pop, rng = _pred()
    obs, hist = _client(1.0)
    prev = p.predict_client(0, obs, hist, "standard")[0]
    for n in range(80):                      # mostly failures, as in the collapse
        p.update_client(int(rng.integers(1, 100)), *pop[rng.integers(100)], "standard",
                        int(rng.random() < 0.1))
        cur = p.predict_client(0, obs, hist, "standard")[0]
        assert abs(cur - prev) < 0.06, n      # no jump (v1 fell ~0.3 at n = 20)
        prev = cur


def test_no_inversion_after_failures():
    p, pop, rng = _pred()
    for _ in range(40):
        p.update_client(int(rng.integers(100)), *_client(0.35 * rng.uniform(0.8, 1.2), 0.6, 0.75),
                        "standard", 0)
    fast = p.predict_client(500, *_client(2.2, 1.0, 0.9), "standard")[0]
    slow = p.predict_client(501, *_client(0.35, 0.6, 0.75), "standard")[0]
    assert fast >= slow


def test_own_record_corrects_the_shared_model_and_shrinks_uncertainty():
    p, _, _ = _pred()
    obs, hist = _client(1.0)
    p_a0, sd0 = p.predict_client(1, obs, hist, "light")
    for _ in range(10):
        p.update_client(1, obs, hist, "light", 1)
        p.update_client(2, obs, hist, "light", 0)
    p_a, sd_a = p.predict_client(1, obs, hist, "light")
    p_b, _ = p.predict_client(2, obs, hist, "light")
    assert p_a > p_b and sd_a < sd0
    # a mode's record does not leak into another mode of the same client
    assert p.posterior(1, obs, hist, "priority") != p.posterior(1, obs, hist, "light")


def test_thompson_fallback_varies_and_default_gate_does_not():
    from afl.policies.base import gate
    cfg = Cfg()
    cfg.policy.strict_gate = True
    N = 30
    state = {"eligible": {i: 0 for i in range(N)},
             "p_succ": {i: {"light": 0.3 - i * 0.001, "standard": 0.2, "priority": 0.1}
                        for i in range(N)},
             "p_post": {i: {m: (2.0, 5.0) for m in ("light", "standard", "priority")}
                        for i in range(N)},
             "gate_rng": np.random.default_rng(0)}
    top = {tuple(gate(cfg, state, list(range(N)), 10)[1]) for _ in range(5)}
    assert len(top) == 1                                  # the absorbing choice
    cfg.policy.gate_mode = "thompson"
    th = {tuple(sorted(gate(cfg, state, list(range(N)), 10)[1])) for _ in range(20)}
    assert len(th) > 10                                   # explores


def test_config_guards():
    from afl.engine import Experiment
    with pytest.raises(ValueError):
        Experiment(Cfg.from_dict({"policy": {"gate_mode": "thompson"}}))
    with pytest.raises(ValueError):
        Experiment(Cfg.from_dict({"policy": {"predictor": "v3"}}))


def test_engine_runs_with_v2_and_thompson(tmp_path):
    from afl.engine import Experiment
    raw = {"data": {"dataset": "synthetic", "num_clients": 16}, "out": str(tmp_path),
           "fl": {"rounds": 6, "clients_per_round": 4, "model": "mlp"},
           "policy": {"name": "rl", "predictor": "v2", "gate_mode": "thompson",
                      "strict_gate": True}}
    exp = Experiment(Cfg.from_dict(raw))
    rows = exp.run(verbose=False)
    assert len(rows) == 6 and all(np.isfinite(r["accuracy"]) for r in rows)
    st = exp._state(7)
    assert all(0 < st["p_sd"][i]["light"] < 0.5 for i in range(16))
