"""Regressions for the fifth review: credit assignment, FAVOR's target,
FedCER's timing, reuse in the older runner, the last learning window, and
curve reporting."""
from __future__ import annotations

import csv
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_advanced import ROOT, Cfg, Experiment, _tiny


def _three_client_policy(split: bool):
    from afl.policies.rl import RLPolicy
    cfg = _tiny({"arl3_full": None, "policy.credit_split": split,
                 "fl.clean_rng": True, "policy.entropy_coef": 0.0})
    cfg.data.num_clients, cfg.fl.clients_per_round = 3, 3
    return cfg, RLPolicy(cfg)


def _state3():
    obs = {i: {"speed_hat": 1.0, "bw_hat": 1.0, "energy_hat": 1.0, "avail_hint": 1.0}
           for i in range(3)}
    hist = {i: {"success_rate": 0.5, "rounds_since_success": 0.0, "selected_frac": 0.5}
            for i in range(3)}
    p = {i: {"light": 0.9, "standard": 0.7, "priority": 0.5} for i in range(3)}
    return {"obs": obs, "hist": hist, "p_succ": p, "eligible": {i: 1 for i in range(3)},
            "repr_value": np.ones(3) / 3, "n_i": np.ones(3) * 100.0,
            "local_loss": np.ones(3), "sketch": np.eye(3), "cov_deficit": np.ones(3),
            "candidates": None, "debt": np.zeros(3)}


@pytest.mark.parametrize("split", [False, True])
def test_selection_gets_no_coverage_gradient_when_every_set_covers_the_same(split):
    # All three clients are always picked, so the round's total coverage is the
    # same whatever the order: the correct selection gradient is zero. The
    # per-client terms still differ (each client's own loss and gain).
    cfg, pol = _three_client_policy(split)
    pol.act(_state3(), 1)
    info = {"client_adv": {0: 1.0, 1: -1.0, 2: 0.5},
            "set_adv": 0.0 if split else None}          # constant total -> 0
    pol.observe_outcome(None, None, 0.0, info)          # shared advantage is 0 too
    g_pick = float(pol.net.score.weight.grad.abs().sum())
    g_mode = float(pol.net.mode.weight.grad.abs().sum())
    assert g_mode > 0.0                                  # modes still learn their own terms
    assert (g_pick == 0.0) if split else (g_pick > 0.0)  # old rule: a spurious pick gradient


def test_engine_passes_the_round_total_to_the_picks_under_credit_split():
    exp = Experiment(_tiny({"arl3_full": None, "policy.credit_split": True,
                            "reward.robust_norm": True}, rounds=3))
    seen = []
    orig = exp.policy.observe_outcome
    def spy(state, action, reward, info, orig=orig):
        seen.append(info["set_adv"])
        orig(state, action, reward, info)
    exp.policy.observe_outcome = spy
    exp.run(verbose=False)
    assert all(v is not None and np.isfinite(v) for v in seen)
    off = Experiment(_tiny({"arl3_full": None}, rounds=1))
    off.policy.observe_outcome = lambda s, a, r, info: seen.append(("off", info["set_adv"]))
    off.run(verbose=False)
    assert seen[-1] == ("off", None)


def test_favor_target_uses_only_clients_selectable_next_round():
    from afl.policies.published import FAVORPolicy
    cfg = _tiny({"b_favor": None})
    cfg.data.num_clients = 3
    pol = FAVORPolicy(cfg)
    flat = torch.zeros(6)
    pol.init_probe({0: torch.ones(6), 1: -torch.ones(6), 2: torch.arange(6.0)}, flat)
    s2 = pol._state_vec(flat)[None]
    with torch.no_grad():
        for net in (pol.net, pol.tgt):
            net[-1].weight.zero_()
            net[-1].bias.copy_(torch.tensor([1.0, 5.0, 2.0]))   # client 1 looks best
    r = torch.zeros(1)
    only0 = torch.tensor([[True, False, False]])
    assert float(pol._targets(r, s2, only0, 0.9)) == pytest.approx(0.9 * 1.0)
    every = torch.ones(1, 3, dtype=torch.bool)
    assert float(pol._targets(r, s2, every, 0.9)) == pytest.approx(0.9 * 5.0)
    none = torch.zeros(1, 3, dtype=torch.bool)
    assert float(pol._targets(r, s2, none, 0.9)) == 0.0     # nothing to bootstrap from


@pytest.mark.parametrize("shared,expected", [(True, "priority"), (False, "light")])
def test_fedcer_timing_includes_the_workload(shared, expected):
    from afl.policies.baselines import FedCERPolicy
    from afl.policies.base import est_time
    cfg = _tiny({"b_fedcer": None, "policy.shared_time_estimate": shared})
    cfg.data.num_clients, cfg.fl.clients_per_round = 1, 1
    st = {"obs": {0: {"speed_hat": 1.0, "bw_hat": 1.0}}, "n_work": np.array([120.0]),
          "sketch": np.ones((1, 10)) / 10, "candidates": None}
    assert est_time(cfg, st, 0, "priority") == pytest.approx(0.65)
    _, assign = FedCERPolicy(cfg).act(st, 1)
    assert assign[0] == expected


def test_older_runner_reruns_a_result_it_cannot_verify(tmp_path):
    out = tmp_path / "res"
    out.mkdir()
    (out / "b_random_s0.csv").write_text("round,accuracy\n1,0.5\n")     # no config, one row
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_experiments.py"),
           "--config", "configs/quick.yaml", "--only", "b_random", "--seeds", "0",
           "--out", str(out)]
    first = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    assert "re-run b_random_s0: no saved config" in first and "done b_random_s0" in first
    again = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    assert "skip b_random_s0 (done)" in again


def test_last_unfinished_learning_window_is_used():
    exp = Experiment(_tiny({"c2_gamma": None}, rounds=3))     # horizon 10 > 3 rounds
    before = [p.detach().clone() for p in exp.policy.net.parameters()]
    exp.run(verbose=False)
    moved = sum(float((p.detach() - q).abs().sum())
                for p, q in zip(exp.policy.net.parameters(), before))
    assert moved > 0 and exp.policy._window == []


def test_curve_average_uses_the_fixed_test_series(tmp_path):
    def write(name, fixed, acc):
        with open(tmp_path / f"{name}.csv", "w", newline="") as fh:
            cols = ["round", "accuracy", "accuracy_fixed", "acc_worst10", "cv", "fail_ratio",
                    "arl_oracle", "cost", "latency"]
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for t, (f, a) in enumerate(zip(fixed, acc), 1):
                wr.writerow({"round": t, "accuracy": a, "accuracy_fixed": f, "acc_worst10": 0,
                             "cv": 0, "fail_ratio": 0, "arl_oracle": 0, "cost": 0, "latency": 0})
    for s in range(3):
        write(f"c2_rl_fedprox_s{s}", [0.1, 0.2], [0.1, 0.9])
        write(f"b_random_s{s}", [0.1, 0.2], [0.1, 0.2])
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "stats.py"),
                          "--dir", str(tmp_path)], capture_output=True, text=True).stdout
    assert "mean_accuracy_fixed" in out and "mean_accuracy (" not in out
