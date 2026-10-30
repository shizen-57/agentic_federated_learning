"""The repairs checked together, and a fixed validation matrix of edge cases
(fourth external review: findings 1 and 7). Each scenario states its
acceptance criteria; the rest of the matrix is covered by the tests named in
docs/VALIDATION.md."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]


def _exp(tmp_path, rounds=12, policy=None, system=None):
    from afl.config import Cfg
    from afl.engine import Experiment
    raw = {"data": {"dataset": "synthetic", "num_clients": 16}, "out": str(tmp_path),
           "fl": {"rounds": rounds, "clients_per_round": 5, "model": "mlp", "clean_rng": True, "val_frac": 0.3},
           "system": {"charge_actual": True, "checkin_pool": True, **(system or {})},
           "policy": {"name": "safe_ppo", "predictor": "v2", "gate_mode": "thompson", "per_client_credit": True,
                      "credit_total": True, "mode_mask": "feasible", "mode_mask_z": 0.25, "pred_decay": 0.98,
                      "cost_target_mult": 1.0, "ppo_batch": 4, **(policy or {})},
           "reward": {"damage_metric": "raw_fail", "batch_client_norm": True, "robust_norm": True}}
    return Experiment(Cfg.from_dict(raw))


class _Transitions:
    """Counts every transition PPO keeps: those consumed by updates plus the
    ones still in the buffer at the end."""
    def __init__(self, pol):
        self.pol, self.used, real = pol, 0, pol._update

        def upd(last_value):
            self.used += len(pol.buf)
            return real(last_value)
        pol._update = upd

    def total(self):
        return self.used + len(self.pol.buf)


def test_select_project_learn_hold_together(tmp_path, monkeypatch):
    """Acceptance (finding 1): executed jobs satisfy the cap and the mask;
    calibration uses the executed mode; dropped jobs do no work, cost nothing
    and train nothing; the PPO trajectory keeps every round with the sampled
    action."""
    import afl.engine as E
    exp = _exp(tmp_path, rounds=14, policy={"hard_budget_mult": 0.6, "curriculum": True})
    pol, env = exp.policy, exp.env
    rounds, cur = [], {}
    real_act, real_budget, real_outcome, real_train = pol.act, exp._enforce_budget, env.outcome, E.local_train
    real_cur, real_spend, real_oracle = pol._curriculum, env.spend, env.oracle_eligible
    in_oracle = [False]

    def act(state, t):
        sel, assign = real_act(state, t)
        cur.clear()
        cur.update(t=t, sampled=dict(assign), allowed=dict(pol.allowed_modes or {}), p=state["p_succ"],
                   ref=state["cost_ref"], outcomes=[], trains=0, spends=[])
        rounds.append(cur.copy())
        return sel, assign

    def budget(sel, assign, state, fracs, cap=None, allowed=None):
        keep, out, dropped = real_budget(sel, assign, state, fracs, cap, allowed)
        rounds[-1].update(executed=dict(out), cap=cap, dropped=[i for i in sel if i not in out])
        return keep, out, dropped

    def outcome(i, mode, frac=None):
        # the ARL* oracle diagnostic also asks outcome() about modes nobody
        # ran (it is pure: reads this round's draws, changes nothing)
        if not in_oracle[0]:
            rounds[-1]["outcomes"].append((i, mode))
        return real_outcome(i, mode, frac)

    def oracle(i):
        e0 = env.energy.copy()
        in_oracle[0] = True
        try:
            return real_oracle(i)
        finally:
            in_oracle[0] = False
            assert np.array_equal(env.energy, e0)          # a look-up, not work

    def spend(i, mode, frac=None):
        rounds[-1]["spends"].append((i, mode))
        return real_spend(i, mode, frac)

    def train(*a, **k):
        rounds[-1]["trains"] += 1
        return real_train(*a, **k)

    def curriculum(action, info):
        n0, level0 = len(pol.cal), pol.level
        real_cur(action, info)
        sel, executed = action
        want = [(float(pol._p_round[i][executed[i]]), int(info["success"].get(i, 0))) for i in sel]
        if pol.level == level0 and want:                  # (a level change resets the window)
            assert pol.cal[-len(want):] == want[-len(pol.cal):]
        rounds[-1]["cal_checked"] = n0 >= 0

    pol.act, exp._enforce_budget, env.outcome, pol._curriculum = act, budget, outcome, curriculum
    env.spend, env.oracle_eligible = spend, oracle
    monkeypatch.setattr(E, "local_train", train)
    tr = _Transitions(pol)
    rows = exp.run(verbose=False)

    assert tr.total() == len(rows) == 14                  # every round is a transition
    changed = 0
    for r, row in zip(rounds, rows):
        ex = r["executed"]
        assert set(ex) <= set(r["sampled"])
        assert all(ex[i] in r["allowed"][i] for i in ex)                       # the mask holds
        assert sum(env.cost(i, ex[i]) for i in ex) <= r["cap"] + 1e-9          # the cap holds
        assert sorted(r["outcomes"]) == sorted(ex.items())                     # only executed jobs ran
        assert not {i for i, _ in r["outcomes"]} & set(r["dropped"])
        assert set(r["spends"]) <= set(ex.items())                              # energy: executed jobs only
        assert r["trains"] <= len(ex) and row["cost"] <= r["cap"] + 1e-9
        assert row["n_selected"] == len(ex)
        changed += sum(ex[i] != r["sampled"][i] for i in ex) + len(r["dropped"])
    assert changed > 0                                    # the projection actually acted
    # PPO learns from what it sampled, not from the projection
    for b in pol.buf:
        if b["act"]:
            chosen = b["picks"] + b["inp"]["fill"]
            assert len(b["modes"]) == len(chosen)


def test_nobody_online(tmp_path):
    """Acceptance: rounds with no client online run (no crash, finite
    values), invite nobody, are logged as offline, are not counted as
    starvation, and stay in the PPO trajectory."""
    import report_dev_screen as rep
    exp = _exp(tmp_path, rounds=12)
    real = exp.env.new_round

    def new_round(t):
        real(t)
        if 3 <= t <= 8:
            exp.env._draw["available"][:] = False
    exp.env.new_round = new_round
    tr = _Transitions(exp.policy)
    rows = exp.run(verbose=False)
    off = [r for r in rows if 3 <= r["round"] <= 8]
    assert all(r["n_online"] == 0 and r["n_selected"] == 0 and r["cost"] == 0 for r in off)
    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, float) and v == v)
    assert tr.total() == 12
    path = os.path.join(str(tmp_path), "rows.csv")
    _write(path, rows)
    m = rep.measures(path)
    assert m["offline"] == 6 and not m["starved"]


def test_everyone_abstains_then_recovers(tmp_path):
    """Acceptance: while every client is ruled out, rounds are empty but kept
    (PPO) and counted as no-contribution rounds; once the rule lifts, clients
    are invited again."""
    import report_dev_screen as rep
    import torch
    exp = _exp(tmp_path, rounds=60)
    pol = exp.policy
    real = pol._mode_sets

    def mode_sets(state):
        pm, ab = real(state)
        if pol._t <= 54:
            return torch.zeros_like(pm), torch.ones_like(ab)
        return pm, ab
    pol._mode_sets = mode_sets
    tr = _Transitions(pol)
    rows = exp.run(verbose=False)
    assert all(r["n_selected"] == 0 for r in rows[:54])
    assert sum(r["n_selected"] for r in rows[54:]) > 0
    assert tr.total() == 60
    path = os.path.join(str(tmp_path), "rows.csv")
    _write(path, rows)
    m = rep.measures(path)
    assert m["starved"] and m["longest_empty"] >= 50 and m["empty"] >= 54


@pytest.mark.parametrize("how", ["fail", "reject"])
def test_every_update_fails_or_is_rejected(tmp_path, monkeypatch, how):
    """Acceptance: with no update arriving, the global model is unchanged,
    nothing is aggregated, the run finishes with finite values, and the
    starvation measure sees it."""
    import afl.engine as E
    import report_dev_screen as rep
    exp = _exp(tmp_path, rounds=55)
    if how == "fail":
        real = exp.env.outcome

        def outcome(i, mode, frac=None):
            out = real(i, mode, frac)
            return {**out, "success": 0, "reason": "deadline"}
        exp.env.outcome = outcome
    else:
        monkeypatch.setattr(E, "accepted", lambda *a, **k: False)
    w0 = exp.global_flat.clone()
    rows = exp.run(verbose=False)
    assert sum(r["n_success"] for r in rows) == 0
    assert np.allclose(exp.global_flat.cpu().numpy(), w0.cpu().numpy())
    assert all(math.isfinite(r["reward"]) for r in rows)
    path = os.path.join(str(tmp_path), "rows.csv")
    _write(path, rows)
    assert rep.measures(path)["starved"]


def test_a_cap_too_tight_for_any_job(tmp_path):
    """Acceptance: if no allowed job fits, every invitation is dropped: no
    work, no cost, an empty round kept in the trajectory."""
    exp = _exp(tmp_path, rounds=6, policy={"hard_budget_mult": 1e-6})
    tr = _Transitions(exp.policy)
    rows = exp.run(verbose=False)
    assert all(r["n_selected"] == 0 and r["cost"] == 0 and r["budget_dropped"] > 0 for r in rows)
    assert tr.total() == 6


def test_sudden_bandwidth_and_battery_drop_and_recovery(tmp_path):
    """Acceptance: when half the devices suddenly lose bandwidth and battery,
    their failures rise; after the devices recover, they are invited and
    succeed again (the predictor's ageing lets old failures fade) -- no
    client is avoided for good."""
    exp = _exp(tmp_path, rounds=120, system={"availability_p": 1.0})
    env = exp.env
    hit = np.arange(0, env.N, 2)
    bw0, cap0 = env.bandwidth.copy(), env.energy_cap.copy()
    log = {"invited": [], "ok": []}
    real_round, real_out, real_oracle = env.new_round, env.outcome, env.oracle_eligible
    in_oracle = [False]

    def new_round(t):
        if t == 30:
            env.bandwidth[hit] *= 0.02
            env.energy_cap[hit] *= 0.05
            env.energy[hit] = np.minimum(env.energy[hit], env.energy_cap[hit])
        if t == 60:
            env.bandwidth[:], env.energy_cap[:] = bw0, cap0
        real_round(t)

    def oracle_eligible(i):
        # the evaluation-only ARL* oracle asks "what if" for every client;
        # those are not invitations and must not count
        in_oracle[0] = True
        try:
            return real_oracle(i)
        finally:
            in_oracle[0] = False

    def outcome(i, mode, frac=None):
        out = real_out(i, mode, frac)
        if not in_oracle[0]:
            log["invited"].append((exp.t_now, i))
            log["ok"].append((exp.t_now, i, out["success"]))
        return out
    env.new_round, env.outcome, env.oracle_eligible = new_round, outcome, oracle_eligible
    real_run_round = exp.run_round

    def run_round(t):
        exp.t_now = t
        return real_run_round(t)
    exp.run_round = run_round
    exp.run(verbose=False)
    rate = lambda a, b: np.mean([s for t, i, s in log["ok"] if a <= t < b and i in set(hit)] or [np.nan])  # noqa: E731
    assert rate(35, 60) < rate(5, 30)                     # the change hurt them
    late = {i for t, i in log["invited"] if t >= 90}
    assert len(set(hit.tolist()) & late) >= len(hit) // 2  # most invited again after recovery
    assert rate(90, 121) > rate(35, 60)                   # and succeeding again


def _write(path, rows):
    import csv

    from afl.engine import LOG_FIELDS
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 0.0) for k in LOG_FIELDS})


def test_cost_time_and_energy_follow_the_documented_equations(tmp_path):
    """docs/SIMULATOR_MODEL.md: cost C = E n / 1000 + f, linear in examples,
    epochs and upload fraction; t_train = E n / 1200 / s x jitter;
    t_comm = 0.25 f / b x channel; energy = 0.08 E / s + 0.05 f."""
    exp = _exp(tmp_path)
    env, sysc = exp.env, exp.cfg.system
    env.new_round(1)
    d = env._draw
    for i in range(env.N):
        for m in ("light", "standard", "priority"):
            E, f, n = sysc.local_epochs[m], sysc.upload_frac[m], env.n_work[i]
            assert env.cost(i, m) == pytest.approx(E * n / 1000 + f)
            t_train, t_comm, energy = env.workload(i, m, f)
            assert t_train == pytest.approx(E * n / 1200 / env.speed[i] * d["compute_jitter"][i])
            assert t_comm == pytest.approx(0.25 * f / env.bandwidth[i] * d["channel"][i])
            assert energy == pytest.approx(0.08 * E / env.speed[i] + 0.05 * f)
        # doubling the work doubles the compute part; the upload part is fixed
        assert env.cost(i, "priority") - env.cost(i, "standard") == pytest.approx(
            2 * env.n_work[i] / 1000 + 0.5)


def test_sensitivity_scales_multiply_battery_and_bandwidth(tmp_path):
    base = _exp(tmp_path).env
    scaled = _exp(tmp_path, system={"energy_scale": 0.5, "bandwidth_scale": 2.0}).env
    assert np.allclose(scaled.energy_cap, 0.5 * base.energy_cap)
    assert np.allclose(scaled.bandwidth, 2.0 * base.bandwidth)
    assert np.allclose(scaled.speed, base.speed)                    # nothing else moves
