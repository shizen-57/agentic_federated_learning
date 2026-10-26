"""Regressions for the sixth review: one consistent round objective for
ARL-v3, FAVOR and availability, shared timing, resume validation, the last
learning window, and curve metrics."""
from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_advanced import ROOT, Experiment, _tiny


def _state(N):
    obs = {i: {"speed_hat": 1.0 + 0.3 * i, "bw_hat": 1.0, "energy_hat": 1.0, "avail_hint": 1.0}
           for i in range(N)}
    hist = {i: {"success_rate": 0.2 * i, "rounds_since_success": float(i), "selected_frac": 0.5}
            for i in range(N)}
    p = {i: {"light": 0.9, "standard": 0.7 - 0.1 * i, "priority": 0.5} for i in range(N)}
    sk = np.eye(N)
    return {"obs": obs, "hist": hist, "p_succ": p, "eligible": {i: 1 for i in range(N)},
            "repr_value": np.linspace(0.2, 0.8, N), "n_i": np.ones(N) * 100.0,
            "local_loss": np.ones(N), "sketch": sk, "cov_deficit": np.ones(N),
            "candidates": None, "debt": np.zeros(N)}


def _policy(N, K):
    from afl.policies.rl import RLPolicy
    cfg = _tiny({"arl3_full": None, "policy.credit_total": True, "policy.prior_baseline": True,
                 "fl.clean_rng": True, "policy.coverage_features": True,
                 "policy.entropy_coef": 0.0, "policy.rl_lr": 0.0})
    cfg.data.num_clients, cfg.fl.clients_per_round = N, K
    return RLPolicy(cfg)


def _forced(pol, seq):
    """Make the policy draw exactly `seq` (picks, then modes), with exact log-probs."""
    it = iter(seq)
    pol._draw = lambda logits: (lambda i: (i, torch.log_softmax(logits, -1)[i]))(next(it))


def _expected_vs_exact(N, K, R, b=0.3):
    pol = _policy(N, K)
    st = _state(N)
    params = list(pol.net.parameters())
    actions = [picks + modes for picks in itertools.permutations(range(N), K)
               for modes in itertools.product(range(3), repeat=K)]
    exact = 0.0                                   # E[R] as a function of the parameters
    for a in actions:
        _forced(pol, a)
        pol.act(st, 1)
        exact = exact + torch.exp(pol._pending["logp"]) * R(a)
    grads = torch.autograd.grad(exact, params, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    expected = [torch.zeros_like(p) for p in params]   # E[the policy's own update]
    for a in actions:
        _forced(pol, a)
        pol.act(st, 1)
        prob = float(torch.exp(pol._pending["logp"].detach()))
        pol.baseline, pol.baseline_n = b, 5               # baseline from earlier rounds
        pol.observe_outcome(None, None, R(a), {"client_adv": None, "set_adv": None})
        for e, p in zip(expected, params):
            e += prob * p.grad
    return expected, grads


def test_policy_update_is_an_unbiased_gradient_of_the_round_objective():
    # Every pick order and mode for 3 clients, 2 picked: E[update] = -grad E[R].
    table = {frozenset(s): v for s, v in zip(itertools.combinations(range(3), 2), (0.01, 0.03, 0.02))}
    R = lambda a: table[frozenset(a[:2])] + 0.004 * (a[2] - a[3])
    expected, exact = _expected_vs_exact(3, 2, R)
    for e, g in zip(expected, exact):
        assert torch.allclose(e, -g, atol=1e-7)
    assert sum(float(g.abs().sum()) for g in exact) > 0


def test_constant_coverage_gives_no_expected_learning_signal():
    # All 3 of 3 clients picked: the round total cannot change with the order,
    # so the exact gradient is 0, and so is the expected update.
    expected, exact = _expected_vs_exact(3, 3, lambda a: 0.02)
    for e, g in zip(expected, exact):
        assert float(g.abs().max()) < 1e-9 and float(e.abs().max()) < 1e-7


def test_round_reward_is_shared_reward_plus_the_sum_of_client_credits():
    def run(total):
        exp = Experiment(_tiny({"arl3_full": None, "policy.credit_total": total,
                                "reward.batch_client_norm": True, "policy.rl_lr": 0.0,
                                "fl.clean_rng": True}, rounds=4))
        seen = []
        orig = exp.policy.observe_outcome
        def spy(state, action, reward, info, orig=orig):
            seen.append((reward, info["client_adv"], info["set_adv"]))
            orig(state, action, reward, info)
        exp.policy.observe_outcome = spy
        exp.run(verbose=False)
        return seen
    with_total, without = run(True), run(False)
    for (r_tot, adv_t, set_t), (r_sh, adv, _) in zip(with_total, without):
        assert adv_t is None and set_t is None
        assert r_tot == pytest.approx(r_sh + sum(adv.values()))
    assert any(abs(sum(adv.values())) > 0 for _, adv, _ in without)


def test_credit_total_and_credit_split_cannot_both_be_on():
    with pytest.raises(ValueError):
        Experiment(_tiny({"arl3_full": None, "policy.credit_total": True,
                          "policy.credit_split": True}))


def _favor(N=3):
    from afl.policies.published import FAVORPolicy
    cfg = _tiny({"b_favor": None})
    cfg.data.num_clients = N
    pol = FAVORPolicy(cfg)
    flat = torch.zeros(6)
    pol.init_probe({0: torch.ones(6), 1: -torch.ones(6), 2: torch.arange(6.0)}, flat)
    return pol, flat


def test_offline_client_with_a_huge_value_does_not_enter_the_target():
    pol, flat = _favor()
    s2 = pol._state_vec(flat)[None]
    with torch.no_grad():
        for net in (pol.net, pol.tgt):
            net[-1].weight.zero_()
            net[-1].bias.copy_(torch.tensor([1.0, 1e6, 2.0]))     # client 1: huge, offline
    only_online = torch.tensor([[True, False, True]])
    assert float(pol._targets(torch.zeros(1), s2, only_online, 0.9)) == pytest.approx(0.9 * 2.0)


def test_favor_skips_rounds_with_nobody_online_and_discounts_them():
    pol, flat = _favor()
    g = pol.cfg.policy.favor_gamma
    st = {"global_flat": flat, "candidates": [0, 1]}
    info = lambda acc: {"acc": acc, "updates": {}, "base_flat": flat}
    sel, _ = pol.act(st, 1)                            # decision at round 1
    pol.observe_outcome(None, None, 0.0, info(0.5))
    for t in (2, 3):                                   # two rounds with nobody online
        assert pol.act({"global_flat": flat, "candidates": []}, t) == ([], {})
        pol.observe_outcome(None, None, 0.0, info(0.6))
    pol.act({"global_flat": flat, "candidates": [2]}, 4)
    r = lambda acc: 64.0 ** (acc - pol.target) - 1.0
    for (_, a, R, _, mask, disc) in pol.buffer:
        assert R == pytest.approx(r(0.5) + g * r(0.6) + g * g * r(0.6))
        assert disc == pytest.approx(g ** 3) and mask.tolist() == [False, False, True]
    assert sorted(a for _, a, *_ in pol.buffer) == sorted(sel)


@pytest.mark.parametrize("work,mode", [(120, "priority"), (400, "standard"), (900, "light")])
def test_shared_timing_scales_with_workload_and_drives_fedcer(work, mode):
    from afl.policies.baselines import FedCERPolicy
    from afl.policies.base import est_time
    cfg = _tiny({"b_fedcer": None, "policy.shared_time_estimate": True})
    cfg.data.num_clients, cfg.fl.clients_per_round = 1, 1
    st = {"obs": {0: {"speed_hat": 1.0, "bw_hat": 1.0}}, "n_work": np.array([float(work)]),
          "sketch": np.ones((1, 10)) / 10, "candidates": None}
    times = [est_time(cfg, st, 0, m) for m in ("light", "standard", "priority")]
    assert times == sorted(times)
    assert FedCERPolicy(cfg).act(st, 1)[1][0] == mode
    exp = Experiment(_tiny({"b_fedcer": None}))
    s2 = {"obs": {0: {"speed_hat": 1.3, "bw_hat": 0.7}}}
    assert exp._est_time(0, "standard", s2, None) == pytest.approx(
        est_time(cfg, {"obs": s2["obs"], "n_work": exp.env.n_work}, 0, "standard"))


@pytest.mark.parametrize("damage", ["truncated", "no_config", "changed_setting", "changed_code"])
def test_older_runner_reruns_unverifiable_results(tmp_path, damage):
    out = tmp_path / "res"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_experiments.py"),
           "--config", "configs/quick.yaml", "--only", "b_random", "--seeds", "0", "--out", str(out)]
    run = lambda: subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    run()
    csv_p, meta_p = out / "b_random_s0.csv", out / "b_random_s0.config.json"
    if damage == "truncated":
        lines = csv_p.read_text().splitlines()
        csv_p.write_text("\n".join(lines[:4]) + "\n")
    elif damage == "no_config":
        meta_p.unlink()
    else:
        meta = json.loads(meta_p.read_text())
        if damage == "changed_setting":
            meta["reward"]["lambda_fair"] = 9.0
        else:
            meta["code_hash"] = "0" * 16
        meta_p.write_text(json.dumps(meta))
    reason = {"truncated": "incomplete", "no_config": "no saved config",
              "changed_setting": "different config", "changed_code": "made by different code"}
    out2 = run()
    assert f"re-run b_random_s0: {reason[damage]}" in out2 and "done b_random_s0" in out2


@pytest.mark.parametrize("rounds,updates", [(23, 3), (3, 1)])
def test_unfinished_learning_window_is_trained_on(rounds, updates):
    exp = Experiment(_tiny({"c2_gamma": None}, rounds=rounds))      # horizon 10
    calls = []
    orig = exp.policy._update_window
    def count():
        calls.append(len(exp.policy._window))
        orig()
    exp.policy._update_window = count
    exp.run(verbose=False)
    assert len(calls) == updates
    assert calls == ([10, 10, 3] if rounds == 23 else [3])


def _curve_runs(tmp_path, legacy=False, with_meta=True):
    for cond, fixed, final in (("c2_rl_fedprox", [0.2, 0.3, 0.4], 0.99),
                               ("b_random", [0.2, 0.3, 0.4], 0.10)):
        for s in range(3):
            p = tmp_path / f"{cond}_s{s}.csv"
            cols = ["round", "accuracy", "acc_worst10", "cv", "fail_ratio", "arl_oracle",
                    "cost", "latency", "jain", "ece", "arl_online", "reward"]
            if not legacy:
                cols.append("accuracy_fixed")
            with open(p, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=cols)
                wr.writeheader()
                for t, f in enumerate(fixed, 1):
                    row = {c: 0 for c in cols}
                    row.update(round=t, accuracy=final if t == 3 else f)
                    if not legacy:
                        row["accuracy_fixed"] = f
                    wr.writerow(row)
            if with_meta:
                (tmp_path / f"{cond}_s{s}.config.json").write_text(
                    json.dumps({"fl": {"rounds": 3, "eval_subset": 2000, "val_frac": 0.0}}))


def test_curve_statistics_ignore_a_different_final_full_test_score(tmp_path):
    from report import accuracy_curve, rounds_to_target
    _curve_runs(tmp_path)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "stats.py"),
                          "--dir", str(tmp_path)], capture_output=True, text=True).stdout
    block = out.split("## mean_accuracy_fixed")[1].split("##")[0]
    row = next(l for l in block.splitlines() if l.startswith("| b_random "))
    assert "+0.0000" in row                          # curves are identical on the fixed set
    rows = [{"round": 1, "accuracy": 0.2, "accuracy_fixed": 0.2},
            {"round": 2, "accuracy": 0.99, "accuracy_fixed": 0.3}]
    assert np.isnan(rounds_to_target(rows, 0.9))      # the full-test 0.99 does not count
    assert accuracy_curve(rows) == [(1, 0.2), (2, 0.3)]


def test_legacy_curves_drop_a_switched_final_round_or_are_refused(tmp_path):
    from report import accuracy_curve
    rows = [{"round": 1, "accuracy": 0.2}, {"round": 2, "accuracy": 0.3}, {"round": 3, "accuracy": 0.99}]
    meta = {"fl": {"rounds": 3, "eval_subset": 2000, "val_frac": 0.0}}
    assert accuracy_curve(rows, meta) == [(1, 0.2), (2, 0.3)]
    assert accuracy_curve(rows, {"fl": {"rounds": 3, "eval_subset": 0}}) == [(1, 0.2), (2, 0.3), (3, 0.99)]
    assert accuracy_curve(rows, None) is None
    _curve_runs(tmp_path, legacy=True, with_meta=False)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "stats.py"),
                          "--dir", str(tmp_path)], capture_output=True, text=True).stdout
    assert "curve average skipped" in out and "## mean_accuracy" not in out
    rep = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "report.py"),
                          "--dir", str(tmp_path), "--target", "0.9"], capture_output=True, text=True).stdout
    assert "unsupported" in rep
