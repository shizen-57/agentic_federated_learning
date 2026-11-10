"""Second RL2 review: empty rounds stay in PPO's trajectory (finding 6), an
explicit curriculum x feasibility rule (4), ARL-v4's set-level gain and joint
budget (5), the hard cost cap, and predictor evidence decay under drift."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from test_safe_ppo import _ppo
from test_sixth_review import _state


def test_empty_rounds_keep_their_reward_and_time():
    pol = _ppo(4, 2, **{"policy.ppo_batch": 100})
    st = _state(4)
    empty = dict(st, eligible={i: 0 for i in range(4)}, candidates=[])
    for s, r in ((st, 1.0), (empty, -100.0), (st, 2.0)):
        sel, assign = pol.act(s, 1)
        pol.observe_outcome(s, (sel, assign), r, {"success": {}})
    assert [b["reward"] for b in pol.buf] == [1.0, -100.0, 2.0]
    assert [b["act"] for b in pol.buf] == [True, False, True]
    stats = pol._update(0.0)                      # actor loss only on acting rounds
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])


def _sets(pol, p, sd):
    st = _state(1)
    st["p_succ"], st["p_sd"] = {0: p}, {0: sd}
    return pol._mode_sets(st)


def test_curriculum_never_overrides_feasibility_silently():
    pol = _ppo(1, 1, **{"policy.mode_mask": "feasible", "policy.curriculum": True,
                        "policy.gate_z": 1.0})
    assert pol.level == 0
    only_std = {"light": 0.30, "standard": 0.80, "priority": 0.10}   # the reviewer's example
    pm, ab = _sets(pol, only_std, {m: 0.05 for m in only_std})
    assert bool(ab[0]) and pol.mask_stats == {"light_override": 0, "abstain": 1}
    pm, ab = _sets(pol, {"light": 0.45, "standard": 0.80, "priority": 0.1},
                   {"light": 0.10, "standard": 0.05, "priority": 0.05})
    assert not ab[0] and pm[0].tolist() == [True, False, False]
    assert pol.mask_stats["light_override"] == 1      # recorded, not silent
    pol.level = 1
    pm, ab = _sets(pol, only_std, {m: 0.05 for m in only_std})
    assert pm[0].tolist() == [False, True, False]     # feasible AND unlocked


def test_abstaining_clients_are_never_invited():
    pol = _ppo(4, 2, **{"policy.mode_mask": "feasible", "policy.curriculum": True})
    st = _state(4)
    st["p_sd"] = {i: {m: 0.01 for m in ("light", "standard", "priority")} for i in range(4)}
    for i in (0, 1):
        st["p_succ"][i] = {"light": 0.1, "standard": 0.9, "priority": 0.1}
    for _ in range(10):
        sel, _ = pol.act(st, 1)
        assert not {0, 1} & set(sel)
        pol._last = None


def test_predictor_decay_tracks_drift_and_uncertainty_regrows():
    from afl.config import Cfg
    from afl.predictor import SuccessPredictorV2
    obs = {"speed_hat": 1.0, "bw_hat": 1.0, "energy_hat": 1.0, "avail_hint": 1.0}
    hist = {"success_rate": 0.0, "rounds_since_success": 0.0}
    errs = {}
    for decay in (1.0, 0.9):
        c = Cfg()
        c.policy.pred_decay = decay
        p = SuccessPredictorV2(c)
        p.set_reference(np.ones((5, 6)) + np.arange(5)[:, None] * 0.1)
        rng = np.random.default_rng(0)
        err = []
        for t in range(200):
            p.t = t
            prob = 0.9 if t < 100 else 0.1                  # the client's battery fails at t=100
            est = p.predict_client(0, obs, hist, "light")[0]
            if t >= 120:
                err.append(abs(est - prob))
            p.update_client(0, obs, hist, "light", int(rng.random() < prob))
        errs[decay] = np.mean(err)
        if decay < 1:
            sd_now = p.predict_client(0, obs, hist, "light")[1]
            p.t += 60                                       # 60 rounds without news
            assert p.predict_client(0, obs, hist, "light")[1] > sd_now
    assert errs[0.9] < errs[1.0] / 2


def _engine(tmp_path, **over):
    from afl.config import Cfg
    from afl.engine import Experiment
    raw = {"data": {"dataset": "synthetic", "num_clients": 16}, "out": str(tmp_path),
           "fl": {"rounds": 6, "clients_per_round": 5, "model": "mlp", "clean_rng": True,
                  "val_frac": 0.3},
           "policy": {"name": "safe_ppo", "predictor": "v2", "per_client_credit": True,
                      "credit_total": True, "cost_target_mult": 1.0, **over},
           "reward": {"damage_metric": "arl_v4", "batch_client_norm": True, "robust_norm": True},
           "diag": {"enabled": True}}
    return Experiment(Cfg.from_dict(raw))


def test_arl4_joint_reward_uses_delivered_coverage_not_summed_credits(tmp_path, monkeypatch):
    from afl import metrics as M
    from afl.diagnostics import read
    monkeypatch.setattr(M, "difference_credit",
                        lambda *a, **k: pytest.fail("difference credits summed into the joint reward"))
    exp = _engine(tmp_path)
    exp.run(verbose=False)
    exp.save()
    _, rounds = read(str(tmp_path / "run.diag_round.jsonl"))
    sk = exp.d["sketch"]
    for r in rounds:
        ok = r["succeeded"]
        assert r["reward"]["damage_raw"]["arl_v4_set_gain"] >= 0
        if not ok:
            assert r["reward"]["damage_raw"]["arl_v4_set_gain"] == 0.0
    # adding a successful client never lowers the gain f(D)
    w = 1.0 + np.random.default_rng(0).random(sk.shape[1])
    f = lambda D: float((sk[D].max(axis=0) * w).sum()) if D else 0.0  # noqa: E731
    for D in ([0], [0, 1], [2, 3, 4]):
        for extra in range(sk.shape[0]):
            assert f(D + [extra]) >= f(D) - 1e-12


def test_arl4_counterfactual_fits_the_joint_budget(tmp_path, monkeypatch):
    from afl import metrics as M
    seen = []
    real = M.arl4_expected_gap

    def spy(p_conf, chosen, epochs, thr, affordable=None):
        seen.append(affordable)
        return real(p_conf, chosen, epochs, thr, affordable)
    monkeypatch.setattr(M, "arl4_expected_gap", spy)
    exp = _engine(tmp_path, cost_target_mult=0.2)        # a tight joint budget
    exp.run(verbose=False)
    assert seen and all(a is not None for a in seen)
    assert any(len(a) < 3 for a in seen)                 # heavy modes ruled out when the room is gone


def test_hard_budget_caps_every_round(tmp_path):
    exp = _engine(tmp_path, hard_budget_mult=0.6, cost_target_mult=0.0)
    rows = exp.run(verbose=False)
    cap = 0.6 * exp._state(7)["cost_ref"]
    assert all(r["cost"] <= cap + 1e-9 for r in rows)
