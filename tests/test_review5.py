"""Review 5 (docs/REVIEW_5.md): the opt-in simulator mechanics (F2), the
workload-aware predictor (F1), step-normalised MimiC (F3), the oracle bounds
(F4), the whole-curve measures (F5), the wasted-cost damage term (F6) and
the second development seed block (F7)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

MODES = ("light", "standard", "priority")


def _cfg(policy: dict | None = None, rounds: int = 6, system: dict | None = None, data: dict | None = None,
         reward: dict | None = None, **fl):
    from afl.config import Cfg
    return Cfg.from_dict({"data": {"dataset": "synthetic", "num_clients": 16, **(data or {})}, "out": "/tmp/afl_r5",
                          "fl": {"rounds": rounds, "clients_per_round": 4, "model": "mlp", "clean_rng": True,
                                 "val_frac": 0.3, **fl},
                          "system": {"charge_actual": True, "checkin_pool": True, **(system or {})},
                          "reward": reward or {},
                          "policy": {"strict_gate": True, **(policy or {})}})


def _env(system: dict | None = None, n: int = 40, seed: int = 3):
    from afl.sim import ClientEnv
    cfg = _cfg(system=system, data={"num_clients": n})
    n_i = np.linspace(100, 900, n)
    env = ClientEnv(cfg, n_i, seed)
    return cfg, env


# ------------------------------------------------------------ F2 simulator

def test_defaults_keep_the_original_mechanics():
    cfg, env = _env()
    env.new_round(1)
    for i in range(env.N):
        assert env._survive_prob(i, 0.1) == env._survive_prob(i, 5.0) == env.reliability[i]
        assert env.cost(i, "light") == 1 * env.n_work[i] / 1000.0 + 0.25
        t_train, t_comm, energy = env.workload(i, "standard")
        assert energy == 0.08 * 2 / env.speed[i] + 0.05 * 0.5
        o = env.outcome(i, "priority")
        if o["reason"] == "deadline":
            assert o["latency"] == o["t_train"] + o["t_comm"] > cfg.system.deadline


def test_exposure_drop_grows_with_job_time_and_keeps_common_random_numbers():
    cfg, env = _env({"drop_model": "exposure", "drop_ref_time": 1.0})
    worse = better = 0
    for t in range(1, 40):
        env.new_round(t)
        for i in np.where(env._draw["available"])[0]:
            i = int(i)
            assert env._survive_prob(i, 0.2) > env._survive_prob(i, 0.9) > 0
            # a job as long as the reference has the device's nominal reliability;
            # time is capped at the deadline
            assert env._survive_prob(i, 1.0) == pytest.approx(env.reliability[i])
            assert env._survive_prob(i, 7.0) == pytest.approx(env._survive_prob(i, cfg.system.deadline))
            ok = [env.outcome(i, m)["success"] for m in MODES]
            # same rolls for every mode: a heavier job never succeeds where a lighter one fails
            assert ok[0] >= ok[1] >= ok[2]
            worse += ok[0] > ok[2]
            better += ok[0]
    assert better > 0 and worse > 0
    with pytest.raises(ValueError):
        _env({"drop_model": "nope"})


def test_job_overhead_is_paid_by_every_job_in_cost_and_time():
    from afl.policies.base import est_time
    cfg0, env0 = _env()
    cfg1, env1 = _env({"job_overhead": 1.0, "job_overhead_time": 0.05})
    env0.new_round(1), env1.new_round(1)
    for i in range(env0.N):
        obs = {"speed_hat": 1.0, "bw_hat": 0.5}
        st0 = {"obs": {i: obs}, "n_work": env0.n_work}
        for m in MODES:
            assert env1.cost(i, m) == pytest.approx(env0.cost(i, m) + 1.0)
            a, b = env0.workload(i, m), env1.workload(i, m)
            assert b[0] == a[0] and b[2] == a[2]
            assert b[1] == pytest.approx(a[1] + 0.05 / env0.bandwidth[i] * env0._draw["channel"][i])
            assert est_time(cfg1, st0, i, m) == pytest.approx(est_time(cfg0, st0, i, m) + 0.05 / 0.5)
    # the mode costs are no longer 1 : 2 : 4
    i = int(np.argmin(np.abs(env0.n_work - np.mean(env0.n_work))))
    assert env0.cost(i, "priority") / env0.cost(i, "light") > 3.0
    assert env1.cost(i, "priority") / env1.cost(i, "light") < 2.6


def test_energy_follows_the_amount_of_work_when_asked():
    _, env0 = _env()
    _, env1 = _env({"energy_by_work": True})
    env0.new_round(1), env1.new_round(1)
    small, big = int(np.argmin(env1.n_work)), int(np.argmax(env1.n_work))
    e = lambda env, i: env.workload(i, "standard")[2] - 0.05 * 0.5      # the training part  # noqa: E731
    assert e(env0, small) * env0.speed[small] == pytest.approx(e(env0, big) * env0.speed[big])
    ratio = (e(env1, big) * env1.speed[big]) / (e(env1, small) * env1.speed[small])
    assert ratio == pytest.approx(env1.n_work[big] / env1.n_work[small])
    mid = np.mean(env1.n_work)
    for i in range(env1.N):          # the mean client is unchanged
        assert e(env1, i) == pytest.approx(e(env0, i) * env1.n_work[i] / mid)


def test_latency_cap_reports_at_most_one_deadline_and_keeps_late_delivery():
    cfg0, env0 = _env()
    cfg1, env1 = _env({"cap_latency": True})
    seen = 0
    for t in range(1, 30):
        env0.new_round(t), env1.new_round(t)
        for i in np.where(env0._draw["available"])[0]:
            i = int(i)
            a, b = env0.outcome(i, "priority"), env1.outcome(i, "priority")
            assert (a["success"], a["reason"]) == (b["success"], b["reason"])
            assert b["latency"] == min(a["latency"], cfg1.system.deadline)
            assert env0.late_ok(i, "priority") == env1.late_ok(i, "priority")
            seen += a["latency"] > cfg0.system.deadline
    assert seen > 0


# ------------------------------------------------------------ F1 predictor

def test_predictor_v3_learns_what_the_job_inputs_decide():
    """Same device for everyone; only the data volume decides whether the
    deadline is met. v2's shared model cannot see it; v3 can."""
    from afl.predictor import SuccessPredictorV2, SuccessPredictorV3
    cfg = _cfg({"predictor": "v3"})
    rng = np.random.default_rng(0)
    n = 60
    n_work = rng.uniform(100, 2000, n)
    obs = {i: {"speed_hat": 1.0, "bw_hat": 1.0, "energy_hat": 1.0, "avail_hint": 1.0} for i in range(n)}
    hist = {i: {"success_rate": 0.5, "rounds_since_success": 0.0} for i in range(n)}
    t_est = {i: {m: cfg.system.local_epochs[m] * n_work[i] / 1200.0 + 0.1 for m in MODES} for i in range(n)}
    v3, v2 = SuccessPredictorV3(cfg), SuccessPredictorV2(cfg)
    ref = np.array([v2.client_features(obs[i], hist[i]) for i in range(n)])
    for p in (v2, v3):
        p.set_reference(ref)
    v3.set_jobs(t_est, obs, n_work)
    truth = lambda i, m: int(t_est[i][m] <= cfg.system.deadline)  # noqa: E731
    half = n // 2                                   # train on half the clients, test on the rest
    for _ in range(40):
        for i in range(half):
            m = MODES[rng.integers(3)]
            v3._cur = i
            v3.update(obs[i], hist[i], m, truth(i, m))
            v2.update(obs[i], hist[i], m, truth(i, m))

    def score(p, v3_):
        err = []
        for i in range(half, n):
            for m in MODES:
                if v3_:
                    p._cur = i
                err.append((p.shared(obs[i], hist[i], m) - truth(i, m)) ** 2)
        return float(np.mean(err))
    assert v3.features(obs[0], hist[0], "light").shape == (11,)
    assert score(v3, True) < 0.5 * score(v2, False)
    with pytest.raises(RuntimeError):
        SuccessPredictorV3(cfg).features(obs[0], hist[0], "light")


@pytest.mark.parametrize("policy", [{"name": "rl", "predictor": "v3", "gate_mode": "thompson"},
                                    {"name": "safe_ppo", "predictor": "v3", "gate_mode": "thompson",
                                     "mode_mask": "feasible", "mode_mask_z": 0.25, "encoder": "attention",
                                     "memory": True, "pred_decay": 0.98},
                                    {"name": "oort", "predictor": "v3", "mode_rule": "expected_work"}])
def test_engine_runs_with_predictor_v3(policy, tmp_path):
    from afl.engine import Experiment
    cfg = _cfg(policy, system={"energy_by_work": True, "job_overhead_time": 0.05})
    cfg.out = str(tmp_path)
    rows = Experiment(cfg).run(verbose=False)
    assert len(rows) == 6 and all(np.isfinite(r["reward"]) for r in rows)


# --------------------------------------------------------- F3 aggregation

def test_mimic_nova_equals_mimic_when_every_update_has_the_same_length(tmp_path):
    """With equal local work per client and no stored correction yet (round
    1), step normalisation changes nothing; later rounds stay finite."""
    from afl.engine import Experiment
    flats = {}
    for agg in ("mimic", "mimic_nova"):
        cfg = _cfg({"name": "random", "mode_rule": "light"}, rounds=1, aggregator=agg,
                   data={"min_samples": 64, "max_samples": 64})
        cfg.out = str(tmp_path)
        exp = Experiment(cfg)
        exp.run(verbose=False)
        flats[agg] = exp.global_flat.clone()
    assert float((flats["mimic"] - flats["mimic_nova"]).abs().max()) < 1e-5
    cfg = _cfg({"name": "random", "mode_rule": "own"}, rounds=8, aggregator="mimic_nova")
    cfg.out = str(tmp_path)
    exp = Experiment(cfg)
    rows = exp.run(verbose=False)
    assert bool(exp.global_flat.isfinite().all()) and all(np.isfinite(r["accuracy"]) for r in rows)


def test_mimic_nova_and_oracle_conditions_exist():
    from afl.conditions import CONDITIONS
    for c in ("ppo_full", "rl_light", "b_oort_ew", "b_fairq_ew", "o_light", "o_work"):
        a, b = CONDITIONS[f"{c}_mimic"], CONDITIONS[f"{c}_mnova"]
        assert b["fl.aggregator"] == "mimic_nova" and {k: v for k, v in a.items() if k != "fl.aggregator"} == \
            {k: v for k, v in b.items() if k != "fl.aggregator"}
    for k in ("light", "work", "fair", "cov"):
        assert CONDITIONS[f"o_{k}_mimic"]["policy.name"] == f"oracle_{k}"


# ------------------------------------------------------------- F4 oracles

@pytest.mark.parametrize("kind", ["oracle_light", "oracle_work", "oracle_fair", "oracle_cov"])
def test_oracles_never_waste_a_job(kind, tmp_path):
    from afl.engine import Experiment
    cfg = _cfg({"name": kind}, rounds=12, system={"norm_clip": 1e9})
    cfg.out = str(tmp_path)
    exp = Experiment(cfg)
    rows = exp.run(verbose=False)
    assert sum(r["n_selected"] for r in rows) > 0
    assert all(r["n_success"] == r["n_selected"] for r in rows)
    heavy = sum(r["mode_standard"] + r["mode_priority"] for r in rows)
    assert (heavy > 0) == (kind == "oracle_work")


def test_only_oracles_are_shown_the_truth_and_the_heaviest_finishing_job_is_given():
    from afl.engine import Experiment
    exp = Experiment(_cfg({"name": "random"}))
    exp.env.new_round(1)
    assert "truth_ok" not in exp._state(1)
    exp = Experiment(_cfg({"name": "oracle_work"}))
    exp.env.new_round(1)
    state = exp._state(1)
    sel, assign = exp.policy.act(state, 1)
    for i in sel:
        ok = state["truth_ok"][i]
        assert ok[assign[i]] == 1
        assert all(ok[m] == 0 for m in MODES[MODES.index(assign[i]) + 1:])
    with pytest.raises(ValueError):
        Experiment(_cfg({"name": "oracle_light", "mode_rule": "expected_work"}))


def test_fairness_oracle_spreads_contributions_more_evenly(tmp_path):
    from afl.engine import Experiment
    jain = {}
    for kind in ("oracle_light", "oracle_fair"):
        cfg = _cfg({"name": kind}, rounds=40)
        cfg.out = str(tmp_path)
        jain[kind] = Experiment(cfg).run(verbose=False)[-1]["jain"]
    assert jain["oracle_fair"] >= jain["oracle_light"]


# -------------------------------------------------------------- F6 reward

def test_wasted_cost_damage_counts_what_failures_cost(tmp_path):
    from afl.engine import Experiment
    rewards = {}
    for dm in ("raw_fail", "waste"):
        cfg = _cfg({"name": "random", "mode_rule": "own"}, rounds=20, reward={"damage_metric": dm})
        cfg.out = str(tmp_path)
        rewards[dm] = [r["reward"] for r in Experiment(cfg).run(verbose=False)]
    assert all(np.isfinite(rewards["waste"])) and rewards["waste"] != rewards["raw_fail"]


# -------------------------------------------------------------- F5 report

def test_efficiency_measures_on_a_hand_made_curve():
    import report_dev_screen as rep
    acc = [0.1, 0.3, 0.5, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8]
    rows = [{"accuracy_fixed": a, "cost": 10.0, "latency": 3.0 if k == 0 else 0.5} for k, a in enumerate(acc)]
    e = rep.efficiency(rows, budget=50.0)
    assert e["auc"] == pytest.approx(sum(acc[:5]) * 10.0 / 50.0)
    # the 5-round mean first reaches 0.6 at round 7 (0.5,0.7,0.7,0.7,0.8 -> 0.68)
    assert e["rounds_to"][0.60] == 7 and e["cost_to"][0.60] == pytest.approx(70.0)
    assert e["time_to"][0.60] == pytest.approx(1.0 + 6 * 0.5)          # the straggler counts one deadline
    assert e["cost_to"][0.75] is None or e["rounds_to"][0.75] >= 9
    # a run that stops spending keeps its last accuracy for the rest of the budget
    e2 = rep.efficiency(rows[:3], budget=50.0)
    assert e2["auc"] == pytest.approx((0.1 + 0.3 + 0.5) * 10 / 50 + 0.5 * 20 / 50)


def test_measures_accept_a_world_budget(tmp_path):
    import csv
    import report_dev_screen as rep
    from afl.engine import Experiment, LOG_FIELDS
    cfg = _cfg({"name": "random", "mode_rule": "light"}, rounds=10)
    cfg.out, cfg.tag = str(tmp_path), "r"
    exp = Experiment(cfg)
    exp.run(verbose=False)
    path = exp.save()
    assert next(csv.reader(open(path))) == LOG_FIELDS
    total = sum(float(r["cost"]) for r in csv.DictReader(open(path)))
    a, b = rep.measures(path), rep.measures(path, budget=total / 2)
    # the standard budget covers the whole short run; half the spend stops earlier
    assert a["acc_budget"][1] == 10 and b["acc_budget"][1] < 10 and b["spend_eq5"] <= total / 2 + 1e-9
    assert 0 <= b["auc"] <= 1 and 0 <= a["fail"] <= 1 and set(a["cost_to"]) == set(rep.TARGETS)


def test_describe_rule_reports_arms_against_the_reference():
    import screen_rl2 as sc
    m = lambda e, w, a, c: {"eq5": e, "w10b": w, "auc": a, "cost_to": {0.7: c}, "fail": 0.4,  # noqa: E731
                            "weak_share": 0.3, "modes": "80/15/5", "spend": 1500.0}
    R = {("femnist", None, "light"): {s: m(0.78, 0.50, 0.60, 500.0) for s in (400, 401)},
         ("femnist", None, "o_work"): {s: m(0.80, 0.53, 0.66, 300.0 if s == 400 else None) for s in (400, 401)}}
    spec = {"datasets": {"femnist": "g"}, "arms": {"light": {}, "o_work": {}},
            "rule": {"kind": "describe", "reference": "light", "bounds": ["o_work"], "targets": [0.7]}}
    lines, verdict = sc.rule_describe(spec, R)
    text = "\n".join(lines)
    assert "o_work (bound)" in text and "+0.020" in text and "300 (1/2)" in text
    assert verdict["arms"]["femnist/-/o_work"]["diff"]["auc"] == pytest.approx(0.06)
    assert verdict["arms"]["femnist/-/light"]["diff"]["eq5"] is None


# ---------------------------------------------------------------- F7 seeds

def test_second_development_block_is_registered():
    import seeds
    assert seeds.kind(400) == seeds.kind(599) == "dev" and seeds.kind(299) == "dev"
    assert seeds.kind(300) == "confirmation" and seeds.kind(600) == "unregistered"
    seeds.check([400, 401, 402], "dev")
    with pytest.raises(ValueError):
        seeds.check([399], "dev")


# ------------------------------------------------- the two checking scripts

def test_sim_truth_and_predictor_check_reports_render():
    import predictor_check as pc
    import sim_truth as stt
    assert pc.ece([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1]) == pytest.approx(0.1)
    mode = lambda ok: {"ok": ok, "energy": 0.0, "dropped": 0.3, "deadline": 0.1, "rejected": 0.0}  # noqa: E731
    res = {"dataset": "femnist", "seeds": [403], "set": [], "rounds": 5, "groups": {
        g: {"share": 0.25, "online": 0.8, "modes": {m: mode(0.5) for m in MODES}, "any_mode": 0.5,
            "light_below_0.2": 0.0, "light_below_0.5": 0.4, "light_median": 0.5, "cost_light": 0.7}
        for g in stt.GROUPS}}
    text = stt.render(res)
    assert "| weak | 25% | 0.80 | 0.50 | 0.50 | 0.50 | 0.50 | 0.30 / 0.10 / 0.00 | 40% | 0.70 |" in text
    ver = {"brier": 0.2, "ece": 0.05, "by_mode": {m: [0.5, 0.4] for m in MODES},
           "by_group": {g: [0.3, 0.2] for g in pc.GROUPS}, "rule_share": {m: 1 / 3 for m in MODES},
           "rule_ok": 0.5, "rule_epochs_per_cost": 0.7,
           "rules": {r: {"share": {m: 1 / 3 for m in MODES}, "ok": 0.5, "epochs_per_cost": 0.7,
                         "jobs_per_cost": 0.6} for r in pc.RULES}}
    out = pc.render({"dataset": "femnist", "seeds": [403], "set": [], "rounds": 5,
                     "how": {h: {v: ver for v in pc.VERSIONS} for h in ("random", "rule")}})
    assert "| v3 | 0.200 | 0.050 | 0.50 / 0.40 |" in out and "| always Light | 33% / 33% / 33% | 50% | 0.70 | 0.60 |" in out
