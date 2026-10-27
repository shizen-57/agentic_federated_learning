"""RL2 review finding 1: with data.split = persistent, no example changes role
across run seeds, and FEMNIST practice and evaluation use disjoint writers.
Checked by original example ids and writer ids."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from afl import data as D  # noqa: E402
from afl.config import Cfg  # noqa: E402

HAVE_CIFAR = os.path.exists(os.path.join(ROOT, "data", "cifar10_uint8.npz"))
HAVE_FEMNIST = all(os.path.exists(os.path.join(ROOT, "data", f)) for f in D.FEMNIST_FILES)


def _cfg(dataset, seed, practice=False, split="persistent", **data):
    c = Cfg()
    c.data.dataset, c.data.seed, c.data.split = dataset, seed, split
    c.fl.val_frac, c.fl.practice = 0.3, practice
    if dataset == "femnist":
        c.data.num_clients, c.data.min_samples, c.data.max_samples = 100, 150, 400
    for k, v in data.items():
        setattr(c.data, k, v)
    return c


def test_pooled_roles_do_not_depend_on_the_run_seed():
    a = D.pooled_roles(10000, 20261027, 0.3)
    assert a.mean() == pytest.approx(0.3, abs=0.02)
    assert np.array_equal(a, D.pooled_roles(10000, 20261027, 0.3))
    assert not np.array_equal(a, D.pooled_roles(10000, 1, 0.3))


def test_reviewer_check_per_seed_split_leaks_and_persistent_does_not():
    """The per-seed split puts every test example of seed 300 into some practice
    seed's validation set (the reported contamination); persistent puts none."""
    n = 10000
    test300 = set(np.random.default_rng(300 + 6).permutation(n)[3000:])
    val_practice = set()
    for s in range(1000, 1160):
        val_practice |= set(np.random.default_rng(s + 6).permutation(n)[:3000])
    assert len(test300 & val_practice) == len(test300) == 7000          # old behaviour
    roles = D.pooled_roles(n, 20261027, 0.3)
    assert not set(np.where(~roles)[0]) & set(np.where(roles)[0])        # persistent


@pytest.mark.skipif(not HAVE_CIFAR, reason="CIFAR-10 not prepared")
def test_cifar_build_uses_the_fixed_roles():
    a, b = D.build(_cfg("cifar10", 300)), D.build(_cfg("cifar10", 1000, practice=True))
    assert np.array_equal(a["split"]["val_role"], b["split"]["val_role"])
    assert a["split"]["fingerprint"] == b["split"]["fingerprint"]


@pytest.fixture(scope="module")
def femnist_runs():
    if not HAVE_FEMNIST:
        pytest.skip("FEMNIST not prepared")
    runs = {}
    for s, prac in ((300, False), (301, False), (1000, True), (1001, True), (1002, True)):
        d = D.build(_cfg("femnist", s, practice=prac))
        sp = d["split"]
        runs[s] = {"writers": set(sp["writers"].tolist()),
                   "train": set(sp["train_global"].tolist()),
                   "val": set(sp["test_global"][sp["val_role"]].tolist()),
                   "test": set(sp["test_global"][~sp["val_role"]].tolist()),
                   "fp": sp["fingerprint"]}
    return runs


def test_femnist_practice_and_evaluation_writers_are_disjoint(femnist_runs):
    w = np.load(os.path.join(ROOT, "data", "femnist_writer.npy"))
    prac, evl = D.femnist_writer_pools(int(w.max()) + 1, 20261027, 0.5)
    assert not set(prac) & set(evl)
    for s, r in femnist_runs.items():
        assert r["writers"] <= set(prac if s >= 1000 else evl)
    practice_ids = set().union(*(femnist_runs[s][k] for s in (1000, 1001, 1002)
                                 for k in ("train", "val", "test")))
    for s in (300, 301):
        assert not practice_ids & (femnist_runs[s]["train"] | femnist_runs[s]["val"]
                                   | femnist_runs[s]["test"])


def test_femnist_roles_are_fixed_across_evaluation_seeds(femnist_runs):
    a, b = femnist_runs[300], femnist_runs[301]
    assert not a["test"] & (b["train"] | b["val"])
    assert not a["val"] & (b["train"] | b["test"])
    assert not a["train"] & (b["val"] | b["test"])
    assert len({r["fp"] for r in femnist_runs.values()}) == 1


def test_split_needs_validation_and_rejects_unknown_modes():
    c = _cfg("synthetic", 0)
    c.fl.val_frac = 0.0
    with pytest.raises(ValueError):
        D.build(c)
    with pytest.raises(ValueError):
        D.build(_cfg("synthetic", 0, split="bogus"))


def test_engine_records_the_fingerprint_and_reuses_the_run(tmp_path):
    import json
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from afl.engine import Experiment
    from sweep import build_jobs, run_status
    from afl.provenance import code_hash
    plan = {"seeds": [0], "sets": {}, "groups": {"g": {
        "config": "configs/quick.yaml", "conditions": ["rl_light"],
        "set": {"fl.rounds": 2, "fl.val_frac": 0.3, "data.split": "persistent"}}}}
    raw = build_jobs(plan, None, None, str(tmp_path), [])[0]["raw"]
    exp = Experiment(Cfg.from_dict(raw))
    exp.run(verbose=False)
    exp.save()
    meta = json.load(open(os.path.join(raw["out"], raw["tag"] + ".config.json")))
    assert len(meta["split_fingerprint"]) == 64
    assert run_status(raw, code_hash()) == "ok"
    roles = exp.d["split"]["val_role"]
    assert set(exp.val_idx.tolist()) == set(np.where(roles)[0])
    assert set(exp.test_idx.tolist()) == set(np.where(~roles)[0])
