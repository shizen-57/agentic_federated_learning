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
