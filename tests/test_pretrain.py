"""RL2 M6: pre-training -- practice episodes never touch the test set, the
checkpoint carries its provenance, and a policy only loads matching ones."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PLAN = ("seeds: [0]\nsets: {}\ngroups:\n  g: {config: configs/quick.yaml, set: {fl.val_frac: 0.3, "
        "fl.clean_rng: true, policy.strict_gate: true}, conditions: [ppo_full]}\n")


def test_practice_episode_never_evaluates_on_test_data(tmp_path):
    from afl.config import Cfg
    from afl.engine import Experiment
    raw = {"data": {"dataset": "synthetic", "num_clients": 12}, "out": str(tmp_path),
           "fl": {"rounds": 4, "clients_per_round": 3, "model": "mlp", "val_frac": 0.3,
                  "practice": True},
           "policy": {"name": "safe_ppo"}}
    exp = Experiment(Cfg.from_dict(raw))
    seen = []
    orig = exp._evaluate
    exp._evaluate = lambda full=True, idx=None: (seen.append(idx is exp.val_idx), orig(full, idx))[1]
    exp._measured_client_acc = lambda: pytest.fail("practice touched per-client test accuracy")
    rows = exp.run(verbose=False)
    assert seen and all(seen)
    assert all(r["accuracy"] != r["accuracy"] for r in rows)        # NaN: not measured
    with pytest.raises(RuntimeError):
        exp.save()
    raw["fl"]["val_frac"] = 0.0
    with pytest.raises(ValueError):
        Experiment(Cfg.from_dict(raw))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    d = tmp_path_factory.mktemp("ck")
    plan = d / "plan.yaml"
    plan.write_text(PLAN)
    out = d / "ck.pt"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "pretrain.py"),
                        "--plan", str(plan), "--group", "g", "--condition", "ppo_full",
                        "--iterations", "2", "--workers", "2", "--rounds", "4", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return plan, out


def test_checkpoint_has_provenance_and_loads(checkpoint, tmp_path):
    from afl.config import Cfg
    from afl.engine import Experiment
    from afl.provenance import code_hash
    plan, out = checkpoint
    ck = torch.load(out, weights_only=False)
    m = ck["meta"]
    assert m["code_hash"] == code_hash() and m["practice_only"] and m["seeds"] == [1000, 1003]
    assert (out.parent / "ck.json").exists()
    raw = {"data": {"dataset": "synthetic", "num_clients": 20}, "out": str(tmp_path),
           "fl": {"rounds": 2, "clients_per_round": 5, "model": "mlp", "val_frac": 0.3,
                  "clean_rng": True},
           "policy": {"name": "safe_ppo", "encoder": "attention", "memory": True,
                      "predictor": "v2", "gate_mode": "thompson", "strict_gate": True,
                      "mode_mask": "feasible", "mode_mask_z": 0.5, "cost_target_mult": 1.25,
                      "curriculum": True, "per_client_credit": True,
                      "init_checkpoint": str(out)},
           "reward": {"damage_metric": "arl_v4", "protect_weight": 0.5, "beta": 0.0}}
    exp = Experiment(Cfg.from_dict(raw))
    for k, v in exp.policy.net.state_dict().items():
        assert torch.equal(v, ck["state_dict"][k])
    exp.run(verbose=False)
    exp.save()
    import json
    meta = json.load(open(tmp_path / "run.config.json"))
    assert len(meta["init_checkpoint_sha256"]) == 64


def test_foreign_checkpoints_are_refused(checkpoint, tmp_path):
    from afl.policies.safe_rl import load_checkpoint
    _, out = checkpoint
    ck = torch.load(out, weights_only=False)
    for change in ({"code_hash": "0" * 16}, {"practice_only": False}):
        bad = tmp_path / "bad.pt"
        torch.save({"state_dict": ck["state_dict"], "meta": {**ck["meta"], **change}}, bad)
        with pytest.raises(ValueError):
            load_checkpoint(torch.nn.Linear(1, 1), str(bad))


def test_pretraining_refuses_non_practice_seeds(checkpoint):
    plan, out = checkpoint
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "pretrain.py"),
                        "--plan", str(plan), "--group", "g", "--condition", "ppo_full",
                        "--iterations", "1", "--workers", "1", "--rounds", "2",
                        "--seed-start", "210", "--out", str(out.parent / "x.pt")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "practice" in r.stderr
