"""Tests for the advanced-sweep additions: new datasets, models, the persistent
weak-client regime, the sweep plan and the paired statistics.

Same priority as test_core.py: invariants that, if broken, would still produce
plausible-looking CSVs -- above all, that the defaults leave the original
experiments untouched and that common random numbers survive the new regime.
"""
from __future__ import annotations

import io
import os
import pickle
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from afl.config import Cfg                       # noqa: E402
from afl.data import _ArrayOnlyUnpickler, class_sketch, dirichlet_partition, load_femnist  # noqa: E402
from afl.engine import Experiment                # noqa: E402
from afl.local import local_train                # noqa: E402
from afl.models import build_model, get_flat     # noqa: E402
from afl.sim import ClientEnv, rarity_score      # noqa: E402


def _sketch(n=30, seed=0):
    y = np.tile(np.arange(10), 600)
    parts = dirichlet_partition(y, n, 0.3, seed=seed, min_samples=40, max_samples=200)
    return class_sketch(y, parts, 10)


# ------------------------------------------------------- weak-client regime

def test_defaults_leave_the_device_profile_unchanged():
    cfg = Cfg.from_dict({"data": {"num_clients": 30}})
    a = ClientEnv(cfg, np.full(30, 300.0), seed=3)
    b = ClientEnv(cfg, np.full(30, 300.0), seed=3, sketch=_sketch())
    assert not b.weak.any()
    for k in ("speed", "bandwidth", "reliability", "energy_cap", "tier"):
        assert np.array_equal(getattr(a, k), getattr(b, k))


def test_weak_clients_are_the_rare_data_holders():
    sk = _sketch()
    cfg = Cfg.from_dict({"data": {"num_clients": 30},
                         "system": {"weak_frac": 0.3, "weak_select": "rarity",
                                    "diurnal_amplitude": 0.0}})
    env = ClientEnv(cfg, np.full(30, 300.0), seed=0, sketch=sk)
    assert env.weak.sum() == 9
    score = rarity_score(sk)
    assert score[env.weak].min() >= score[~env.weak].max()
    assert env.speed[env.weak].max() < 1.0               # slower than the mid tier
    assert env.reliability[env.weak].mean() < 0.65          # below the mid tier (0.70)
    p = env.availability_prob(5)
    assert np.allclose(p[env.weak], 0.85 - 0.1) and np.allclose(p[~env.weak], 0.85)


def test_weak_clients_are_mode_sensitive_not_dead():
    """The regime only tests ARL if the configuration choice decides the outcome.

    A weak client must usually succeed under Light and almost never under
    Priority; if it failed under every mode (as low-tier clients mostly do),
    no scheduler could rescue it and ARL would have nothing to reward.
    """
    sk = _sketch(100)
    cfg = Cfg.from_dict({"data": {"num_clients": 100},
                         "system": {"weak_frac": 0.3, "avail_persistence": 0.8,
                                    "work_ref_samples": 500}})
    env = ClientEnv(cfg, np.full(100, 300.0), seed=0, sketch=sk)
    ok = {m: [] for m in ("light", "priority")}
    for t in range(1, 101):
        env.new_round(t)
        for i in np.where(env.weak)[0]:
            for m in ok:
                ok[m].append(env.outcome(i, m)["success"])
    light, prio = np.mean(ok["light"]), np.mean(ok["priority"])
    assert light > 0.3 and prio < 0.1 and light > 4 * prio


def test_random_weak_selection_is_not_value_correlated_by_construction():
    cfg = Cfg.from_dict({"data": {"num_clients": 30},
                         "system": {"weak_frac": 0.3, "weak_select": "random"}})
    a = ClientEnv(cfg, np.full(30, 300.0), seed=0, sketch=_sketch())
    b = ClientEnv(cfg, np.full(30, 300.0), seed=1, sketch=_sketch())
    assert a.weak.sum() == 9 and not np.array_equal(a.weak, b.weak)


def test_sticky_availability_persists_and_keeps_common_random_numbers():
    def autocorr(rho):
        cfg = Cfg.from_dict({"data": {"num_clients": 40},
                             "system": {"avail_persistence": rho, "availability_p": 0.6,
                                        "diurnal_amplitude": 0.0}})
        env = ClientEnv(cfg, np.full(40, 300.0), seed=0)
        prev, same = None, []
        for t in range(1, 80):
            env.new_round(t)
            # Within a round the outcome is still a pure function of the draws.
            first = [env.outcome(i, "light")["success"] for i in range(40)]
            assert first == [env.outcome(i, "light")["success"] for i in range(40)]
            cur = env._draw["available"].copy()
            if prev is not None:
                same.append((cur == prev).mean())
            prev = cur
        return float(np.mean(same))
    assert autocorr(0.8) > autocorr(0.0) + 0.2


def test_workload_scaling_only_changes_the_simulated_work():
    n_i = np.array([100.0, 200.0, 300.0])
    cfg = Cfg.from_dict({"data": {"num_clients": 3}, "system": {"work_ref_samples": 500}})
    env = ClientEnv(cfg, n_i, seed=0)
    assert np.isclose(env.n_work.mean(), 500) and np.array_equal(env.n_i, n_i)
    off = ClientEnv(Cfg.from_dict({"data": {"num_clients": 3}}), n_i, seed=0)
    assert np.array_equal(off.n_work, n_i)


# ----------------------------------------------------------------- datasets

def test_femnist_partition_is_one_writer_per_client(tmp_path):
    rng = np.random.default_rng(0)
    writers = np.repeat(np.arange(12), rng.integers(30, 80, size=12)).astype(np.int32)
    n = len(writers)
    np.save(tmp_path / "femnist_x.npy", rng.integers(0, 256, (n, 784), dtype=np.uint8))
    np.save(tmp_path / "femnist_y.npy", (writers % 62).astype(np.int16))  # label = writer id
    np.save(tmp_path / "femnist_writer.npy", writers)
    xtr, ytr, xte, yte, parts = load_femnist(str(tmp_path), 8, seed=0,
                                             min_samples=30, max_samples=50)
    flat = np.concatenate(parts)
    assert len(flat) == len(np.unique(flat)) == len(ytr) == len(xtr)
    for p in parts:
        assert len(np.unique(ytr[p])) == 1          # every client is a single writer
        assert len(p) <= 50
    assert len(yte) > 0 and set(np.unique(yte)) <= set(np.unique(ytr))
    assert abs(float(xtr.mean())) < 1e-3              # normalised


def test_femnist_needs_preparation(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_femnist"):
        load_femnist(str(tmp_path), 4, 0, 10, 50)


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_cifar_unpickler_refuses_non_array_globals():
    blob = pickle.dumps({b"data": _Evil()})
    with pytest.raises(pickle.UnpicklingError):
        _ArrayOnlyUnpickler(io.BytesIO(blob), encoding="bytes").load()
    ok = pickle.dumps({b"data": np.zeros((2, 3), dtype=np.uint8), b"labels": [1, 2]})
    out = _ArrayOnlyUnpickler(io.BytesIO(ok), encoding="bytes").load()
    assert out[b"data"].shape == (2, 3)


# ------------------------------------------------------------------- models

@pytest.mark.parametrize("name,dim,classes,lo,hi", [
    ("cnn", 784, 10, 10_000, 15_000),
    ("cnn_femnist", 784, 62, 150_000, 300_000),
    ("cnn_cifar", 3072, 10, 150_000, 250_000),
])
def test_models_have_the_documented_size_and_shape(name, dim, classes, lo, hi):
    cfg = Cfg.from_dict({"fl": {"model": name}})
    m = build_model(cfg, dim, classes)
    assert lo <= sum(p.numel() for p in m.parameters()) <= hi
    assert m(torch.zeros(4, dim)).shape == (4, classes)


def test_local_train_accepts_numpy_or_tensors_identically():
    cfg = Cfg.from_dict({"fl": {"model": "mlp", "aggregator": "fedprox"}})
    torch.manual_seed(0)
    flat = get_flat(build_model(cfg, 12, 10))
    x = np.random.default_rng(0).normal(size=(40, 12)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 10, 40)
    a, sa = local_train(cfg, flat, x, y, 2, 12, 10, torch.Generator().manual_seed(1))
    b, sb = local_train(cfg, flat, torch.from_numpy(x), torch.from_numpy(y), 2, 12, 10,
                        torch.Generator().manual_seed(1))
    assert sa == sb and torch.allclose(a, b)


# ------------------------------------------------------------------- engine

def test_engine_logs_new_columns_and_uses_full_test_set_last():
    cfg = Cfg.from_dict({"data": {"num_clients": 12}, "fl": {"rounds": 4, "clients_per_round": 4,
                                                             "eval_subset": 200},
                         "system": {"weak_frac": 0.25}})
    exp = Experiment(cfg)
    rows = exp.run(verbose=False)
    for r in rows:
        assert 0.0 <= r["acc_worst10"] <= r["accuracy"] + 1e-9
        assert r["weak_success"] <= r["weak_selected"] <= r["n_selected"]
    full_acc, _, _ = exp._evaluate(full=True)
    assert rows[-1]["accuracy"] == pytest.approx(full_acc)


# ------------------------------------------------------------ sweep + stats

def test_sweep_plan_expands_and_applies_overrides():
    from sweep import build_jobs
    from afl.run import _load
    plan = _load(os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    jobs = build_jobs(plan, ["weak_mnist"], [0, 1], "out", ["fl.rounds=3"])
    assert len(jobs) == 2 * len(plan["sets"]["focus"])
    j = next(j for j in jobs if j["cond"] == "ablation_rawfail" and j["seed"] == 1)
    raw = j["raw"]
    assert raw["system"]["weak_frac"] == 0.3 and raw["system"]["weak_select"] == "rarity"
    assert raw["reward"]["damage_metric"] == "raw_fail" and raw["fl"]["rounds"] == 3
    assert raw["tag"] == "ablation_rawfail_s1" and raw["data"]["seed"] == 1
    assert raw["out"] == os.path.join("out", "weak_mnist")
    # Overrides must not leak between jobs through a shared nested dict.
    other = next(j for j in jobs if j["cond"] == "c2_rl_fedprox")
    assert other["raw"]["reward"].get("damage_metric", "arl") == "arl"


def test_sign_flip_p_is_exact_and_holm_is_monotone():
    from stats import holm, sign_flip_p
    assert sign_flip_p(np.ones(5)) == pytest.approx(2 / 32)
    assert sign_flip_p(np.ones(10)) == pytest.approx(2 / 1024)
    assert sign_flip_p(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(1.0)
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


# ------------------------------------------------------- CIFAR local copy

def _write_fake_cifar(d, n=8, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(d, exist_ok=True)
    for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
        with open(os.path.join(d, name), "wb") as fh:
            pickle.dump({b"data": rng.integers(0, 256, (n, 3072), dtype=np.uint8),
                         b"labels": [int(v) for v in rng.integers(0, 10, n)]}, fh, protocol=2)


def test_cifar_is_built_from_a_local_batch_directory(tmp_path):
    from afl.data import _cifar_cache, _find_cifar_dir, load_cifar10
    root = str(tmp_path)
    _write_fake_cifar(os.path.join(root, "cifar-10-batches-py"))
    assert _find_cifar_dir(root) == os.path.join(root, "cifar-10-batches-py")
    z = np.load(_cifar_cache(root))
    assert z["xtr"].shape == (40, 3072) and z["ytr"].shape == (40,)
    assert z["xte"].shape == (8, 3072) and z["xtr"].dtype == np.uint8
    xtr, ytr, xte, yte = load_cifar10(root)
    assert xtr.shape == (40, 3072) and xtr.dtype == np.float32 and yte.shape == (8,)


def test_cifar_local_search_ignores_incomplete_directories(tmp_path):
    from afl.data import _find_cifar_dir
    os.makedirs(tmp_path / "cifar-10-batches-py")
    (tmp_path / "cifar-10-batches-py" / "data_batch_1").write_bytes(b"x")
    assert _find_cifar_dir(str(tmp_path)) is not None   # presence of batch 1 in root is accepted...
    # ...but loading it must fail loudly rather than silently using bad data
    from afl.data import _cifar_from_dir
    with pytest.raises(Exception):
        _cifar_from_dir(str(tmp_path / "cifar-10-batches-py"))


def test_one_sided_sign_flip_and_group_seeds():
    from stats import sign_flip_p
    d = np.ones(5)
    assert sign_flip_p(d, alternative="greater") == pytest.approx(1 / 32)
    assert sign_flip_p(d, alternative="less") == pytest.approx(1.0)
    assert sign_flip_p(-d, alternative="less") == pytest.approx(1 / 32)
    from sweep import build_jobs
    from afl.run import _load
    plan = _load(os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    jobs = build_jobs(plan, ["confirm_cifar"], None, "out", [])
    assert sorted({j["seed"] for j in jobs}) == list(range(10, 40))   # fresh seeds only
    assert {j["cond"] for j in jobs} == {"c2_rl_fedprox", "ablation_rawfail"}
    assert jobs[0]["raw"]["system"]["weak_select"] == "rarity"


# ------------------------------------- Chapter 3 pieces and published baselines

def _small(**over):
    base = {"data": {"num_clients": 12}, "fl": {"rounds": 6, "clients_per_round": 4}}
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return Cfg.from_dict(base)


def test_budget_respects_cost_and_time_limits():
    exp = Experiment(_small(system={"cost_max": 1.5, "latency_max": 1.0}))
    for t in range(1, 5):
        exp.env.new_round(t)
        state = exp._state(t)
        sel, assign = exp.policy.act(state, t)
        keep, assign2, dropped = exp._enforce_budget(sel, assign, state, {})
        assert sum(exp.env.cost(i, assign2[i]) for i in keep) <= 1.5 + 1e-9
        assert all(exp._est_time(i, assign2[i], state, None) <= 1.0 for i in keep)
        assert dropped == len(sel) - len(keep)
    rows = Experiment(_small(system={"cost_max": 1.5, "latency_max": 1.0})).run(verbose=False)
    assert all(r["cost"] <= 1.5 + 1e-9 for r in rows)


def test_budget_is_a_no_op_by_default():
    exp = Experiment(_small())
    exp.env.new_round(1)
    state = exp._state(1)
    sel, assign = exp.policy.act(state, 1)
    assert exp._enforce_budget(sel, assign, state, {}) == (sel, assign, 0)


def test_semi_async_delivers_late_updates_only_within_the_staleness_bound():
    over = dict(fl={"async_late": True, "rounds": 12}, system={"deadline": 0.4})
    rows = Experiment(_small(**over)).run(verbose=False)
    assert sum(r["late_delivered"] for r in rows) > 0
    over["system"]["staleness_max"] = 0          # no lateness allowed -> nothing late
    rows = Experiment(_small(**over)).run(verbose=False)
    assert sum(r["late_delivered"] for r in rows) == 0


def test_importance_reweighting_changes_the_aggregate_when_switched_on():
    a = Experiment(_small()).run(verbose=False)
    b = Experiment(_small(fl={"importance_reweight": True})).run(verbose=False)
    assert [r["accuracy"] for r in a] != [r["accuracy"] for r in b]


def test_multi_round_returns_update_the_policy_once_per_window():
    exp = Experiment(_small(policy={"horizon": 3}, fl={"rounds": 6}))
    before = [p.detach().clone() for p in exp.policy.net.parameters()]
    exp.run_round(1); exp.run_round(2)
    assert all(torch.equal(a, b) for a, b in zip(before, exp.policy.net.parameters()))
    exp.run_round(3)
    assert not all(torch.equal(a, b) for a, b in zip(before, exp.policy.net.parameters()))


def test_fairness_debt_accrues_for_unserved_eligible_clients_and_is_a_feature():
    exp = Experiment(_small(reward={"lambda_debt": 0.3}))
    assert exp.policy.net.trunk[0].in_features == 13      # 12 features + debt
    rows = exp.run(verbose=False)
    assert all(r["debt_mean"] >= 0 for r in rows) and rows[-1]["debt_mean"] > 0
    assert Experiment(_small()).policy.net.trunk[0].in_features == 12   # off by default


def test_mifa_steps_by_the_mean_of_every_clients_latest_update():
    exp = Experiment(_small(policy={"name": "random"}, fl={"aggregator": "mifa"}))
    exp.run_round(1); exp.run_round(2)
    before = exp.global_flat.clone()
    exp.run_round(3)
    if exp.mem:
        expected = sum(exp.mem.values()) / exp.N        # never-seen clients count as zero
        assert torch.allclose(exp.global_flat - before, expected, atol=1e-6)


def test_mimic_corrects_received_updates_with_their_last_correction():
    exp = Experiment(_small(policy={"name": "random"}, fl={"aggregator": "mimic", "rounds": 8}))
    for t in range(1, 5):
        exp.run_round(t)
    prev = {i: c.clone() for i, c in exp.mimic_c.items()}
    before = exp.global_flat.clone()
    exp.run_round(5)
    delta = exp.global_flat - before
    recv = [i for i, c in exp.mimic_c.items() if i not in prev or not torch.equal(c, prev[i])]
    if recv:
        raw = {i: delta - exp.mimic_c[i] for i in recv}               # c_i = v - g_i
        corrected = [raw[i] + prev[i] if i in prev else raw[i] for i in recv]
        assert torch.allclose(sum(corrected) / len(corrected), delta, atol=1e-5)


def test_fedcg_selects_k_clients_and_fits_compression_to_the_deadline():
    exp = Experiment(_small(policy={"name": "fedcg"}, fl={"aggregator": "fedcg"}))
    exp.env.new_round(1)
    state = exp._state(1)
    assert exp.has_hist.all()                                # history probed for everyone
    sel, assign = exp.policy.act(state, 1)
    assert len(sel) == len(set(sel)) == 4 and set(assign.values()) == {"standard"}
    for i in sel:
        theta = exp.policy.upload_frac[i]
        assert 0.01 <= theta <= 1.0
        if theta < 1.0 and theta > 0.01:                     # binding: estimated time == deadline
            assert exp._est_time(i, "standard", state, theta) == pytest.approx(1.0, rel=1e-6)
    rows = exp.run(verbose=False)
    assert len(rows) == 6


def test_fedcg_compression_keeps_top_k_and_carries_the_error():
    exp = Experiment(_small(policy={"name": "fedcg"}, fl={"aggregator": "fedcg"}))
    u = torch.tensor([0.5, -3.0, 0.1, 2.0, -0.2, 0.0])
    sent = exp._compress(7, u, 2 / 6)
    assert torch.equal(sent, torch.tensor([0.0, -3.0, 0.0, 2.0, 0.0, 0.0]))
    assert torch.allclose(sent + exp.ef_err[7], u)
    sent2 = exp._compress(7, torch.zeros(6), 2 / 6)          # error is sent next time
    assert torch.equal(sent2, torch.tensor([0.5, 0.0, 0.0, 0.0, -0.2, 0.0]))


def test_fedscale_trace_sets_real_device_profiles_without_moving_availability():
    a = ClientEnv(Cfg.from_dict({"data": {"num_clients": 50}}), np.full(50, 300.0), seed=4)
    b = ClientEnv(Cfg.from_dict({"data": {"num_clients": 50}, "system": {"device_trace": "fedscale"}}),
                  np.full(50, 300.0), seed=4)
    z = np.load(os.path.join(ROOT, "src", "afl", "traces", "fedscale_device_sample.npz"))
    comp = z["computation_ms"].astype(float)
    med = np.median(comp)
    assert b.speed.min() >= med / comp.max() - 1e-9
    assert b.speed.max() <= med / comp.min() + 1e-9
    assert not np.allclose(a.speed, b.speed)
    assert np.allclose(a.availability_prob(7), b.availability_prob(7))
    c = ClientEnv(Cfg.from_dict({"data": {"num_clients": 50}, "system": {"device_trace": "fedscale"}}),
                  np.full(50, 300.0), seed=4)
    assert np.array_equal(b.speed, c.speed)                  # deterministic per seed


def test_new_conditions_and_groups_are_all_runnable():
    from afl.conditions import CONDITIONS
    from sweep import build_jobs
    from afl.run import _load
    plan = _load(os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    for g in ("ext_mnist", "ext_cifar", "sens_cifar", "trace_mnist", "trace_cifar"):
        assert build_jobs(plan, [g], [0], "out", [])
    for name in ("c2_gamma", "c2_reweight", "c2_async", "c2_debt", "c2_budget",
                 "b_random_budget", "b_oort_budget", "b_mifa", "b_mimic", "b_fedcg", "sens_arl0"):
        raw = {"data": {"num_clients": 12}, "fl": {"rounds": 2, "clients_per_round": 4}}
        from afl.conditions import apply
        apply(raw, CONDITIONS[name])
        assert len(Experiment(Cfg.from_dict(raw)).run(verbose=False)) == 2, name


def test_sweep_runs_groups_in_the_order_asked():
    from sweep import build_jobs
    from afl.run import _load
    plan = _load(os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    jobs = build_jobs(plan, ["confirm_cifar", "main_mnist"], [0], "out", [])
    order = list(dict.fromkeys(j["group"] for j in jobs))
    assert order == ["confirm_cifar", "main_mnist"]


def test_fedcg_margin_shrinks_the_compression_ratio_and_defaults_to_the_paper():
    base = Experiment(_small(policy={"name": "fedcg"}, fl={"aggregator": "fedcg"}))
    slack = Experiment(_small(policy={"name": "fedcg", "fedcg_margin": 0.7}, fl={"aggregator": "fedcg"}))
    for e in (base, slack):
        e.env.new_round(1)
    sb, ss = base._state(1), slack._state(1)
    i = 0
    assert slack.policy._theta(ss, i) <= base.policy._theta(sb, i)
    assert Cfg().policy.fedcg_margin == 1.0


def test_deterministic_flag_does_not_change_cpu_results_and_sets_the_flags():
    a = Experiment(_small()).run(verbose=False)
    b = Experiment(_small(fl={"deterministic": True})).run(verbose=False)
    assert [r["accuracy"] for r in a] == [r["accuracy"] for r in b]
    assert torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    assert Cfg().fl.deterministic is False


def test_follow_up_group_uses_fresh_seeds_and_the_new_conditions():
    from sweep import build_jobs
    from afl.run import _load
    plan = _load(os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    jobs = build_jobs(plan, ["follow_cifar"], None, "out", [])
    assert sorted({j["seed"] for j in jobs}) == list(range(10, 30))
    assert {j["cond"] for j in jobs} == {"c2_rl_fedprox", "b_mimic", "c2_mimic", "b_fedcg", "b_fedcg_m07"}
    mim = next(j for j in jobs if j["cond"] == "c2_mimic")["raw"]
    assert mim["policy"]["name"] == "rl" and mim["fl"]["aggregator"] == "mimic"
    assert mim["system"]["weak_select"] == "rarity"


def test_marginal_value_input_matches_what_arl_rewards_and_is_opt_in():
    from afl import metrics as M
    sk = _sketch(60)
    v = M.expected_marginal_value(sk, 10, n_samples=100, seed=0)
    rng = np.random.default_rng(1)
    V = np.mean([M.per_client_representation_value(sk, list(rng.choice(60, 10, replace=False)))
                 for _ in range(100)], axis=0)
    assert np.corrcoef(v, V)[0, 1] > 0.9 and v.max() == pytest.approx(1.0)
    assert np.corrcoef(M.standalone_value(sk), V)[0, 1] < 0      # the original input points the other way
    off = Experiment(_small())
    on = Experiment(_small(policy={"value_feature": "marginal"}))
    assert np.allclose(off.value_input, M.standalone_value(off.d["sketch"]))
    assert not np.allclose(on.value_input, off.value_input)
    from sweep import build_jobs
    from afl.run import _load
    jobs = build_jobs(_load(os.path.join(ROOT, "configs", "sweep_advanced.yaml")), ["value_cifar"], None, "o", [])
    assert sorted({j["seed"] for j in jobs}) == list(range(40, 60))
    assert all(j["raw"]["policy"]["value_feature"] == "marginal" for j in jobs)


# ------------------------------------------------------------ ARL-v2 and churn

def test_churn_is_off_by_default_and_departures_are_permanent():
    env = ClientEnv(Cfg.from_dict({"data": {"num_clients": 20}}), np.full(20, 300.0), seed=0)
    assert np.isinf(env.patience).all()
    cfg = Cfg.from_dict({"data": {"num_clients": 20}, "system": {"churn_after": 2, "churn_prob": 1.0}})
    env = ClientEnv(cfg, np.full(20, 300.0), seed=0)
    env.record(3, "unavailable"); env.record(3, "unavailable")
    assert not env.departed[3]                       # offline rounds do not count
    env.record(3, "deadline"); env.record(3, "ok"); env.record(3, "deadline")
    assert not env.departed[3]                       # a success resets the streak
    env.record(3, "energy")
    assert env.departed[3]
    for t in range(1, 6):
        env.new_round(t)
        assert not env._draw["available"][3]
        assert env.outcome(3, "light")["reason"] == "unavailable"


def test_churn_patience_is_the_same_for_every_policy():
    cfg = Cfg.from_dict({"data": {"num_clients": 40}, "system": {"churn_after": 3}})
    a = ClientEnv(cfg, np.full(40, 300.0), seed=5)
    b = ClientEnv(cfg, np.full(40, 300.0), seed=5)
    assert np.array_equal(a.patience, b.patience) and 0 < np.isfinite(a.patience).sum() < 40


def test_arl_v2_counts_only_avoidable_failures_of_invited_clients():
    rows = Experiment(_small(reward={"damage_metric": "arl_v2"},
                             policy={"value_feature": "marginal"})).run(verbose=False)
    assert all(r["arl_v2"] >= 0 for r in rows)
    exp = Experiment(_small())
    exp.env.new_round(1)
    state = exp._state(1)
    # The formula: gap of the assigned mode vs the best mode, times value and coverage factor.
    i = 0
    p = state["p_succ"][i]
    best = max(p, key=p.get)
    assert max(p.values()) - p[best] == 0.0            # choosing the safest mode forfeits nothing


def test_per_client_credit_changes_learning_only_when_switched_on():
    a = Experiment(_small(reward={"damage_metric": "arl_v2"}, policy={"value_feature": "marginal"}))
    b = Experiment(_small(reward={"damage_metric": "arl_v2"},
                          policy={"value_feature": "marginal", "per_client_credit": True}))
    ra, rb = a.run(verbose=False), b.run(verbose=False)
    pa = torch.cat([p.detach().flatten() for p in a.policy.net.parameters()])
    pb = torch.cat([p.detach().flatten() for p in b.policy.net.parameters()])
    assert not torch.allclose(pa, pb)
    assert all("arl_v2" in r and "departed" in r for r in rb)
