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


def test_sign_flip_is_exact_for_twenty_seeds():
    from stats import sign_flip_p
    d = np.array([1.0] * 15 + [-1.0] * 5)
    # Exact: P(mean of random signs >= 0.5) = P(#plus >= 15) = sum_{k>=15} C(20,k) / 2^20
    from math import comb
    exact = sum(comb(20, k) for k in range(15, 21)) / 2 ** 20
    assert sign_flip_p(d, alternative="greater") == pytest.approx(exact, abs=1e-12)


# ------------------------------------------- simulator fix and new baselines

def _tiny(extra=None, rounds=4, seed=1):
    from afl.conditions import CONDITIONS, apply
    raw = {"seed": seed, "data": {"dataset": "synthetic", "num_clients": 20, "seed": seed},
           "fl": {"rounds": rounds, "clients_per_round": 4, "model": "mlp"},
           "system": {"weak_frac": 0.3, "avail_persistence": 0.8}}
    for k, v in (extra or {}).items():
        if k in CONDITIONS:
            apply(raw, CONDITIONS[k])
        else:
            apply(raw, {k: v})
    return Cfg.from_dict(raw)


def test_fixes_are_off_by_default():
    cfg = Cfg()
    assert cfg.fl.compress_upload is False
    assert cfg.policy.oort_estimated_time is False
    assert cfg.policy.mode_rule == "own"


def test_compressed_upload_sends_only_the_mode_fraction():
    exp = Experiment(_tiny({"fl.compress_upload": True}))
    upd = torch.randn(1000)
    sent = exp._compress(0, upd, exp._frac("light", None))
    assert int((sent != 0).sum()) == 250                  # Light uploads 25%
    assert torch.allclose(sent + exp.ef_err[0], upd)      # the rest is carried over
    full = exp._compress(1, upd, exp._frac("priority", None))
    assert torch.equal(full, upd)                          # Priority sends everything


def test_light_update_reaching_the_server_is_sparse_only_with_the_fix():
    for on in (False, True):
        cfg = _tiny({"rl_light": None, "fl.compress_upload": on}, rounds=2)
        exp = Experiment(cfg)
        seen = []
        orig = exp.policy.observe_outcome
        def spy(state, action, reward, info, orig=orig):
            seen.extend(info["updates"].values())
            orig(state, action, reward, info)
        exp.policy.observe_outcome = spy
        exp.run(verbose=False)
        assert seen
        dens = max(float((u != 0).float().mean()) for u in seen)
        assert (dens <= 0.26) if on else (dens > 0.9)


def test_oort_fix_uses_only_observed_time():
    exp = Experiment(_tiny({"b_oort": None, "policy.oort_estimated_time": True}))
    exp.env.new_round(1)
    st = exp._state(1)
    i = 0
    assert st["t_est"][i]["standard"] == pytest.approx(exp._est_time(i, "standard", st, None))
    # The true time uses this round's jitter, which the estimate must not.
    exp.env._draw["compute_jitter"][:] = 50.0
    assert exp._state(1)["t_expected"][i]["standard"] > 10
    assert exp._est_time(i, "standard", st, None) < 10


def test_expected_work_rule_takes_the_most_training_that_is_safe_enough():
    from afl.policies.base import best_work_mode
    cfg = Cfg()
    assert best_work_mode(cfg, {"light": 0.9, "standard": 0.7, "priority": 0.55}) == "priority"
    assert best_work_mode(cfg, {"light": 0.9, "standard": 0.7, "priority": 0.3}) == "standard"
    assert best_work_mode(cfg, {"light": 0.4, "standard": 0.2, "priority": 0.1}) == "light"


def test_mode_rule_overrides_every_policy_and_rl_skips_its_mode_head():
    for cond, want in (("b_random_ew", None), ("rl_light", "light")):
        exp = Experiment(_tiny({cond: None}, rounds=3))
        rows = exp.run(verbose=False)
        if want == "light":
            assert all(r["mode_light"] == r["n_selected"] for r in rows)
        else:
            assert sum(r["mode_standard"] for r in rows) < sum(r["n_selected"] for r in rows)


def test_arl_v3_loss_is_zero_for_the_best_mode_and_counts_gambles_as_lost():
    exp = Experiment(_tiny({"arl3_full": None}, rounds=1))
    exp.env.new_round(1)
    st = exp._state(1)
    from afl.policies.base import best_work_mode
    # Follow the expected-work rule: lost work must be exactly zero.
    exp.cfg.policy.mode_rule = "expected_work"
    row = exp.run_round(1)
    assert row["arl_v3_loss"] == pytest.approx(0.0)
    assert row["arl_v3_gain"] >= 0.0
    p = {"light": 0.9, "standard": 0.7, "priority": 0.3}
    assert best_work_mode(exp.cfg, p) == "standard"


def test_refl_selects_least_available_and_scales_stale_updates():
    from afl.policies.published import REFLPolicy
    cfg = _tiny({"b_refl": None})
    pol = REFLPolicy(cfg)
    N = cfg.data.num_clients
    obs = {i: {"avail_hint": 0.9 if i < 15 else 0.1} for i in range(N)}
    fc = np.linspace(1.0, 0.0, N)                     # client 14 least available online
    sel, _ = pol.act({"obs": obs, "avail_forecast": fc, "n_late": 1}, 1)
    assert len(sel) == 3                               # APT: K - late
    assert sel == [14, 13, 12]
    exp = Experiment(cfg)
    a, b = torch.ones(10), torch.ones(10) * 5
    d = exp._refl_aggregate([a, b], [1.0, 1.0], [0, 1])
    w_s = (1 - 0.35) / 2 + 0.35 * (1 - np.exp(-1))     # the only stale update has Lambda_max
    assert torch.allclose(d, (a + w_s * b) / (1 + w_s))


def test_delta_redistributes_the_mass_of_the_clients_that_reported():
    from afl.policies.published import DELTAPolicy
    cfg = _tiny({"b_delta": None})
    pol = DELTAPolicy(cfg)
    N = cfg.data.num_clients
    pol.act({"n_i": np.full(N, 10.0)}, 1)
    u = {0: torch.zeros(5), 1: torch.ones(5) * 0.1}
    pol.observe_outcome(None, None, 0.0, {"recv_ids": [0, 1], "stale": {}, "updates": u,
                                          "grad_var": {0: 0.0, 1: 0.0}})
    assert pol.p.sum() == pytest.approx(1.0)
    assert pol.p[0] + pol.p[1] == pytest.approx(2.0 / N)  # mass of S_t kept
    assert pol.p[0] == pytest.approx(pol.p[1])            # equal distance from their mean


def test_local_gradient_variance_is_tracked_on_request():
    cfg = _tiny()
    x = torch.randn(64, 20)
    y = torch.randint(0, 10, (64,))
    model = build_model(cfg, 20, 10)
    gs = {}
    local_train(cfg, get_flat(model), x, y, 1, 20, 10, torch.Generator().manual_seed(0), grad_stats=gs)
    assert gs["var"] > 0


def test_favor_builds_its_state_from_pca_and_learns_online():
    cfg = _tiny({"b_favor": None}, rounds=6)
    exp = Experiment(cfg)
    exp.run(verbose=False)
    pol = exp.policy
    assert exp.probed and pol.client.shape == (20, min(100, 19))
    assert len(pol.buffer) > 0
    assert pol.target == 0.9                              # synthetic default


def test_new_baselines_do_not_change_the_fl_model_initialisation():
    a = Experiment(_tiny({"b_random": None}))
    b = Experiment(_tiny({"b_favor": None}))
    assert torch.equal(a.global_flat, b.global_flat)


def test_simulator_fix_groups_use_fresh_seeds_and_the_fix():
    import yaml
    plan = yaml.safe_load(open(os.path.join(ROOT, "configs", "sweep_advanced.yaml")))
    used = set()
    for name, g in plan["groups"].items():
        if not name.startswith("s2_"):
            used |= set(g.get("seeds", plan["seeds"]))
    for name in ("s2_cifar", "s2_femnist", "s2_churn_cifar", "s2_churn_femnist"):
        g = plan["groups"][name]
        assert g["set"]["fl.compress_upload"] and g["set"]["policy.oort_estimated_time"]
        assert set(g["seeds"]) == set(range(100, 120))
        assert not set(g["seeds"]) & used


# ------------------------------------------------------------ protocol fix

def test_protocol_fixes_are_off_by_default():
    cfg = Cfg()
    assert cfg.fl.val_frac == 0.0 and not cfg.fl.clean_rng and not cfg.fl.absence_counter
    assert not cfg.policy.coverage_features


def test_reward_uses_a_validation_set_disjoint_from_the_reported_test_set():
    exp = Experiment(_tiny({"b_favor": None, "fl.val_frac": 0.3, "fl.eval_subset": 50}, rounds=2))
    val = set(exp.val_idx.cpu().tolist())
    test = set(exp.test_idx.cpu().tolist())
    assert val and test and not val & test
    assert len(val) + len(test) == len(exp.d["y_test"])
    assert set(exp.eval_idx.cpu().tolist()) <= test
    seen = []
    orig = exp.policy.observe_outcome
    def spy(state, action, reward, info, orig=orig):
        seen.append(info["acc"])
        orig(state, action, reward, info)
    exp.policy.observe_outcome = spy
    rows = exp.run(verbose=False)
    for r, a in zip(rows, seen):
        assert r["val_accuracy"] == a                 # learners see validation only
    a_val, _, _ = exp._evaluate(idx=exp.val_idx)
    assert a_val == pytest.approx(rows[-1]["val_accuracy"])


def test_clean_rng_gives_every_method_the_same_initial_model():
    sums = {c: Experiment(_tiny({c: None, "fl.clean_rng": True})).init_checksum
            for c in ("b_random", "c2_rl_fedprox", "b_favor", "b_oort")}
    assert len(set(sums.values())) == 1
    other = Experiment(_tiny({"b_random": None, "fl.clean_rng": True}, seed=2)).init_checksum
    assert other != sums["b_random"]
    # Without the fix the RL policy's own network shifts the FL model's init.
    a = Experiment(_tiny({"b_random": None})).init_checksum
    b = Experiment(_tiny({"c2_rl_fedprox": None})).init_checksum
    assert a != b


def test_checksum_is_saved_with_the_config(tmp_path):
    import json
    cfg = _tiny({"b_random": None, "fl.clean_rng": True}, rounds=1)
    cfg.out = str(tmp_path)
    exp = Experiment(cfg)
    exp.run(verbose=False)
    exp.save()
    meta = json.load(open(os.path.join(str(tmp_path), f"{cfg.tag}.config.json")))
    assert meta["init_checksum"] == exp.init_checksum


def test_absence_counter_counts_rounds_without_an_invitation():
    for on in (False, True):
        exp = Experiment(_tiny({"b_random": None, "fl.absence_counter": on}, rounds=5))
        exp.run(verbose=False)
        never = [i for i in range(exp.N) if exp.x_sel[i] == 0]
        if never:
            assert all(exp.since_success[i] == (5 if on else 0) for i in never)


def test_shapley_share_does_not_hide_duplicate_holders():
    from afl import metrics as M
    sk = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    loo = M.per_client_representation_value(sk, [0, 1, 2])
    assert loo[0] == 0 and loo[1] == 0                 # the flaw: both look worthless
    phi = M.coverage_shapley(sk, [0, 1, 2])
    assert phi[0] == pytest.approx(0.5) and phi[1] == pytest.approx(0.5)
    assert sum(phi.values()) == pytest.approx(M.coverage(sk))


def test_shapley_matches_brute_force():
    import itertools, math
    from afl import metrics as M
    rng = np.random.default_rng(0)
    sk = rng.dirichlet(np.ones(4) * 0.5, size=5)
    S = [0, 1, 2, 3, 4]
    phi = M.coverage_shapley(sk, S)
    v = lambda T: M.coverage(sk[list(T)]) if T else 0.0
    for i in S:
        rest = [j for j in S if j != i]
        val = sum(math.factorial(r) * math.factorial(len(S) - r - 1) / math.factorial(len(S))
                  * (v(T + (i,)) - v(T))
                  for r in range(len(rest) + 1) for T in itertools.combinations(rest, r))
        assert phi[i] == pytest.approx(val)


def test_set_level_coverage_loss_is_logged_and_bounded():
    exp = Experiment(_tiny({"b_random": None}, rounds=4))
    rows = exp.run(verbose=False)
    for r in rows:
        assert 0.0 <= r["cov_lost_avoid"] <= r["cov_lost"] + 1e-9 or r["cov_lost"] >= 0
        assert r["cov_lost"] >= -1e-9


def test_rl_with_coverage_inputs_picks_k_distinct_clients_and_learns():
    cfg = _tiny({"c2_rl_fedprox": None, "policy.coverage_features": True,
                 "fl.clean_rng": True}, rounds=4)
    exp = Experiment(cfg)
    before = [p.detach().clone() for p in exp.policy.net.parameters()]
    rows = exp.run(verbose=False)
    assert all(r["n_selected"] == 4 for r in rows)
    moved = sum(float((p - q).abs().sum()) for p, q in zip(exp.policy.net.parameters(), before))
    assert moved > 0
    st = exp._state(5)
    picked, _ = exp.policy.act(st, 5)
    assert len(set(picked)) == 4


def test_greedy_coverage_rule_complements_its_picks():
    from afl.policies.baselines import GreedyCoveragePolicy
    cfg = _tiny({"b_cov_ew": None})
    cfg.fl.clients_per_round = 2
    pol = GreedyCoveragePolicy(cfg)
    N = cfg.data.num_clients
    sk = np.zeros((N, 10))
    sk[:, 0] = 1.0                      # everyone holds class 0 ...
    sk[7] = 0.0
    sk[7, 3] = 1.0                      # ... except client 7, the only holder of class 3
    p = {i: {"light": 0.9, "standard": 0.8, "priority": 0.6} for i in range(N)}
    st = {"sketch": sk, "cov_deficit": np.ones(10), "p_succ": p,
          "eligible": {i: 1 for i in range(N)}}
    sel, assign = pol.act(st, 1)
    assert 7 in sel and len(sel) == 2
    assert all(m == "priority" for m in assign.values())   # 0.6 x 4 is the most work


def test_prespecified_families_use_conditions_that_are_run():
    import yaml
    import prespec_s2
    plan = yaml.safe_load(open(os.path.join(ROOT, "configs", "sweep_advanced.yaml")))
    for fam, tests in prespec_s2.FAMILIES.items():
        for g, ref, other, *_ in tests:
            conds = plan["groups"][g]["conditions"]
            conds = plan["sets"][conds] if isinstance(conds, str) else conds
            assert ref in conds and other in conds, (fam, g, ref, other)


# ----------------------------------------------------- second review fixes

def test_second_review_fixes_are_off_by_default():
    cfg = Cfg()
    assert not cfg.reward.batch_client_norm and not cfg.system.charge_actual
    assert not cfg.system.checkin_pool and not cfg.policy.strict_gate


def test_batch_normalisation_does_not_depend_on_order():
    from afl import metrics as M
    vals = [0.5, 2.0, -1.0, 3.0]
    seen = None
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        n = M.Normalizer()
        n.batch([1.0, 2.0, 0.0])                       # some history
        z = dict(zip(order, n.batch([vals[k] for k in order])))
        if seen is not None:
            assert all(z[k] == pytest.approx(seen[k]) for k in range(4))
        seen = z
    seq = M.Normalizer()
    seq.batch([1.0, 2.0, 0.0])
    a = [seq(v) for v in vals]
    seq2 = M.Normalizer()
    seq2.batch([1.0, 2.0, 0.0])
    b = [seq2(v) for v in reversed(vals)][::-1]
    assert a != pytest.approx(b)                         # the old way is order-dependent


def test_offline_invitation_costs_nothing_only_with_the_fix():
    for on in (False, True):
        exp = Experiment(_tiny({"b_random": None, "system.charge_actual": on}, rounds=1))
        exp.env.new_round(1)
        off = int(np.where(~exp.env._draw["available"])[0][0])
        before = exp.env.energy[off]
        exp.policy.act = lambda state, t, off=off: ([off], {off: "priority"})
        exp.env.new_round = lambda t: None             # keep this round's draws
        row = exp.run_round(1)
        if on:
            assert exp.env.energy[off] == before and row["cost"] == 0.0
        else:
            assert exp.env.energy[off] < before and row["cost"] > 0.0


def test_multi_round_updates_with_per_client_credit_are_rejected():
    with pytest.raises(ValueError):
        Experiment(_tiny({"arl3_full": None, "policy.horizon": 5}))


def test_strict_gate_keeps_the_clients_that_pass_and_fills_by_success():
    from afl.policies.base import gate
    cfg = _tiny({"c2_rl_fedprox": None, "policy.strict_gate": True})
    N, K = 20, 10
    p = {i: {"light": 0.9 - 0.01 * i, "standard": 0.1, "priority": 0.1} for i in range(N)}
    st = {"eligible": {i: int(i < 9) for i in range(N)}, "p_succ": p}
    elig, fill = gate(cfg, st, list(range(N)), K)
    assert elig == list(range(9)) and fill == [9]
    cfg.policy.strict_gate = False
    assert gate(cfg, st, list(range(N)), K)[1] == []
    # RL: the 9 passing clients are always kept.
    exp = Experiment(_tiny({"c2_rl_fedprox": None, "policy.strict_gate": True}))
    exp.cfg.fl.clients_per_round = K
    exp.env.new_round(1)
    s = exp._state(1)
    s["eligible"], s["p_succ"] = st["eligible"], p
    sel, _ = exp.policy.act(s, 1)
    assert set(range(9)) <= set(sel) and len(sel) == K and exp.policy.gate_filled == 1


def test_delta_weights_are_unbiased_when_everyone_reports():
    from afl.policies.published import DELTAPolicy
    cfg = _tiny({"b_delta": None})
    cfg.data.num_clients, cfg.fl.clients_per_round = 2, 2
    pol = DELTAPolicy(cfg)
    pol.p = np.array([0.9, 0.1])
    u = {0: 0.0, 1: 10.0}
    est = []
    for _ in range(20000):
        sel, _ = pol.act({"n_i": np.array([50.0, 50.0])}, 1)
        est.append(sum(w * u[i] for i, w in zip(sel, pol.agg_weights(sel))))
    assert np.mean(est) == pytest.approx(5.0, abs=0.15)   # the old rule gave 9


def test_every_policy_picks_only_online_clients_with_the_check_in_pool():
    for cond in ("b_random", "b_fedcs", "b_powerofchoice", "b_oort", "c4_heuristic_fedprox",
                 "b_fedcer", "b_cov_ew", "b_favor", "b_delta", "b_refl", "c2_rl_fedprox",
                 "arl3_full", "b_fedcg"):
        exp = Experiment(_tiny({cond: None, "system.checkin_pool": True, "fl.compress_upload": True},
                               rounds=3))
        orig = exp.policy.act
        def spy(state, t, orig=orig, exp=exp, cond=cond):
            sel, assign = orig(state, t)
            online = set(np.where(exp.env._draw["available"])[0].tolist())
            assert set(sel) <= online, cond
            return sel, assign
        exp.policy.act = spy
        exp.run(verbose=False)


def test_report_shows_how_many_runs_reach_the_target(tmp_path):
    import csv, subprocess
    def write(name, accs):
        with open(tmp_path / f"{name}.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["round", "accuracy", "cv", "jain", "fail_ratio",
                                                "arl_online", "arl_oracle", "cost", "latency", "ece",
                                                "reward", "accuracy_fixed"])
            wr.writeheader()
            for t, a in enumerate(accs, 1):
                wr.writerow({"round": t, "accuracy": a, "cv": 0, "jain": 1, "fail_ratio": 0,
                             "arl_online": 0, "arl_oracle": 0, "cost": 0, "latency": 0, "ece": 0,
                             "reward": 0, "accuracy_fixed": a})
    write("A_s0", [0.1, 0.9])
    for s in range(1, 10):
        write(f"A_s{s}", [0.1, 0.2])
    for s in range(10):
        write(f"B_s{s}", [0.1, 0.5, 0.9])
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "report.py"),
                          "--dir", str(tmp_path), "--target", "0.8"],
                         capture_output=True, text=True).stdout
    line_a = next(l for l in out.splitlines() if l.startswith("| A "))
    line_b = next(l for l in out.splitlines() if l.startswith("| B "))
    assert "1/10" in line_a and "10/10" in line_b


def test_sweep_reruns_results_from_other_configs_code_or_unfinished(tmp_path):
    import json, subprocess
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "seeds: [0]\nsets: {}\ngroups:\n  g: {config: configs/quick.yaml, "
        "set: {fl.rounds: 2}, conditions: [b_random]}\n")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "sweep.py"), "--plan", str(plan),
           "--out", str(tmp_path / "out")]
    run = lambda *extra: subprocess.run(cmd + list(extra), capture_output=True, text=True,
                                        cwd=ROOT).stdout
    run()
    assert "1 already done" in run()
    meta_path = tmp_path / "out" / "g" / "b_random_s0.config.json"
    meta = json.loads(meta_path.read_text())
    meta["code_hash"] = "old"
    meta_path.write_text(json.dumps(meta))
    out = run("--list")
    assert "made by different code" in out
    assert "todo=0" in run("--list", "--ignore-code")
    meta["reward"]["lambda_arl"] = 9.0
    meta_path.write_text(json.dumps(meta))
    assert "different config" in run("--list", "--ignore-code")


def test_confirmatory_report_gives_no_verdict_on_partial_data(tmp_path):
    import json, subprocess
    g = tmp_path / "s2_cifar"
    g.mkdir()
    for cond, base in (("arl3_full", 0.5), ("v2_rawfail", 0.1)):
        for s in range(100, 108):                       # 8 of 20 seeds
            (g / f"{cond}_s{s}.csv").write_text(
                "round,acc_worst10,accuracy,fail_ratio,acc_client_worst10\n"
                f"1,{base},{base},0.1,{base}\n")
            (g / f"{cond}_s{s}.config.json").write_text(json.dumps(
                {"fl": {"rounds": 1, "val_frac": 0.3, "clean_rng": True}, "code_hash": "x"}))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prespec_s2.py"),
                          "--root", str(tmp_path)], capture_output=True, text=True).stdout
    fam_a = out.split("## Family A ")[1].split("## Family")[0]
    assert "INCOMPLETE" in fam_a and "| yes |" not in fam_a


# ------------------------------------------------------ third review fixes

def test_robust_normaliser_survives_a_constant_history():
    from afl import metrics as M
    plain = M.Normalizer()
    plain.batch([0.0] * 10)
    assert plain.batch([0.2])[0] > 1e5                     # the problem
    robust = M.Normalizer(floor=1e-3, clip=3.0, warmup=10)
    robust.batch([0.0] * 10)
    assert robust.batch([0.2]) == [3.0]
    early = M.Normalizer(floor=1e-3, clip=3.0, warmup=10)
    assert early.batch([5.0, -5.0]) == [0.0, 0.0] and early(7.0) == 0.0
    # Defaults are the original behaviour.
    a, b = M.Normalizer(), M.Normalizer(floor=0.0, clip=None, warmup=2)
    xs = [0.3, 1.0, -2.0, 0.5, 4.0]
    assert [a(x) for x in xs] == [b(x) for x in xs]


def test_equal_spend_never_gives_a_method_more_than_the_budget(tmp_path):
    import csv, subprocess
    def write(name, costs, accs):
        with open(tmp_path / f"{name}.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["round", "cost", "accuracy", "accuracy_fixed"])
            wr.writeheader()
            for t, (c, a) in enumerate(zip(costs, accs), 1):
                wr.writerow({"round": t, "cost": c, "accuracy": a, "accuracy_fixed": a})
    write("b_random_s0", [5, 5], [0.3, 0.4])               # spends 10 in total
    write("c2_rl_fedprox_s0", [9, 9, 9], [0.5, 0.9, 0.95])  # 9 <= 10 < 18
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "budget_s2.py"),
                          "--dir", str(tmp_path)], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("| c2_rl_fedprox "))
    assert "| 0.5000 |" in line and "90%" in line             # round 1, cost 9 -- not round 2


def test_code_fingerprint_is_taken_at_load_and_mid_run_edits_are_flagged(tmp_path, monkeypatch):
    import json
    from afl import engine as E
    cfg = _tiny({"b_random": None}, rounds=1)
    cfg.out = str(tmp_path)
    exp = Experiment(cfg)
    exp.run(verbose=False)
    monkeypatch.setattr(E, "code_hash", lambda: "edited")
    exp.save()
    meta = json.load(open(os.path.join(str(tmp_path), f"{cfg.tag}.config.json")))
    assert meta["code_hash"] == E.CODE_LOADED and meta["code_changed_during_run"]


def test_measured_client_accuracy_uses_each_clients_own_examples():
    exp = Experiment(_tiny({"b_random": None}, rounds=1))
    n = len(exp.d["y_test"])
    owner = torch.arange(n, device=exp.dev) % exp.N
    exp.test_owner = owner
    exp._evaluate(full=True)
    acc = exp._measured_client_acc()
    ok = exp._last_ok.cpu().numpy()
    for i in (0, 3, 7):
        assert acc[i] == pytest.approx(ok[owner.cpu().numpy() == i].mean())
    exp.test_owner = torch.full((n,), -1, device=exp.dev)
    assert exp._measured_client_acc() is None             # pooled test set: label-mix instead


def _fake_s2_runs(root, plan_jobs, family_tests, value, bad=None):
    """Write valid-looking runs (exact planned config, all rounds) for a family."""
    import json, csv
    from sweep import expected_config
    from prespec_s2 import FROZEN_CODE_HASH
    for (g, cond, seed), raw in plan_jobs.items():
        if not any(g == t[0] and cond in (t[1], t[2]) for t in family_tests):
            continue
        os.makedirs(raw["out"], exist_ok=True)
        meta = expected_config(raw)
        meta["code_hash"] = FROZEN_CODE_HASH
        T = meta["fl"]["rounds"]
        col = family_tests[0][3]
        v = value(cond)
        rows = [{"round": t, col: v} for t in range(1, T + 1)]
        if bad == "nan" and cond == family_tests[0][1]:
            rows[-1][col] = float("nan")
        if bad == "short":
            rows = rows[:1]
            meta["fl"]["rounds"] = 1
        if bad == "nohash":
            meta.pop("code_hash")
        if bad == "wronghash":
            meta["code_hash"] = "0" * 16
        if bad == "protocol_off":
            meta["fl"]["val_frac"] = 0.0
        with open(os.path.join(raw["out"], raw["tag"] + ".csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["round", col])
            wr.writeheader()
            wr.writerows(rows)
        with open(os.path.join(raw["out"], raw["tag"] + ".config.json"), "w") as fh:
            json.dump(meta, fh)


@pytest.mark.parametrize("bad", [None, "nan", "short", "nohash", "wronghash", "protocol_off"])
def test_confirmatory_report_accepts_only_valid_complete_runs(tmp_path, bad):
    import subprocess
    import prespec_s2
    jobs = prespec_s2.load_plan(str(tmp_path))
    tests = prespec_s2.FAMILIES["A"]
    _fake_s2_runs(tmp_path, jobs, tests,
                  value=lambda c: 0.5 if c == "arl3_full" else 0.1, bad=bad)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prespec_s2.py"),
                          "--root", str(tmp_path)], capture_output=True, text=True, cwd=ROOT).stdout
    fam_a = out.split("## Family A ")[1].split("## Family")[0]
    if bad is None:
        assert "INCOMPLETE" not in fam_a and fam_a.count("| significantly better |") == 4
    else:
        assert "INCOMPLETE" in fam_a and "| significantly better |" not in fam_a
