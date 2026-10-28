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
        "fl.clean_rng: true, policy.strict_gate: true}, conditions: [rl_light]}\n")   # ppo_full NOT listed
HAVE_FEMNIST = all(os.path.exists(os.path.join(ROOT, "data", f))
                   for f in ("femnist_x.npy", "femnist_y.npy", "femnist_writer.npy"))


def _pretrain(*args, cwd=ROOT):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "pretrain.py"), *args],
                          capture_output=True, text=True, cwd=cwd)


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
    r = _pretrain("--plan", str(plan), "--group", "g", "--condition", "ppo_full",
                  "--iterations", "2", "--workers", "2", "--rounds", "4", "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    return plan, out


def test_every_worker_starts_from_one_shared_network(checkpoint):
    import json
    _, out = checkpoint
    log = json.load(open(out.parent / "ck.json"))
    starts = {e["start_checksum"] for it in log["log"][:1] for e in it["episodes"]}
    assert starts == {log["meta"]["init_checksum"]}          # iteration 0 too
    assert log["meta"]["mode"] == "average"
    tot = log["meta"]["practice_totals"]
    assert tot["episodes"] == 4 and tot["rounds"] == 16 and tot["cost"] > 0


def test_sequential_mode(tmp_path, checkpoint):
    plan, _ = checkpoint
    r = _pretrain("--plan", str(plan), "--group", "g", "--condition", "ppo_full", "--mode",
                  "sequential", "--iterations", "2", "--rounds", "3", "--out", str(tmp_path / "s.pt"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert torch.load(tmp_path / "s.pt", weights_only=False)["meta"]["practice_totals"]["episodes"] == 2


@pytest.mark.skipif(not HAVE_FEMNIST, reason="FEMNIST not prepared")
def test_the_documented_server_command_runs(tmp_path):
    # exactly what scripts/dept_gpu.sh runs, shortened: s2_femnist does not list ppo_full
    r = _pretrain("--group", "s2_femnist", "--condition", "ppo_full", "--iterations", "1",
                  "--workers", "2", "--rounds", "2", "--out", str(tmp_path / "f.pt"))
    assert r.returncode == 0, r.stdout + r.stderr
    meta = torch.load(tmp_path / "f.pt", weights_only=False)["meta"]
    assert meta["group"] == "s2_femnist" and len(meta["split_fingerprint"]) == 64


def test_checkpoint_has_provenance_and_loads(checkpoint, tmp_path):
    from afl.config import Cfg
    from afl.engine import Experiment
    from afl.provenance import code_hash
    plan, out = checkpoint
    ck = torch.load(out, weights_only=False)
    m = ck["meta"]
    assert m["code_hash"] == code_hash() and m["practice_only"] and m["seeds"] == [1000, 1003]
    assert (out.parent / "ck.json").exists()
    raw = {"data": {"dataset": "synthetic", "num_clients": 20, "split": "persistent"},
           "out": str(tmp_path),
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
    assert meta["split_fingerprint"] == m["split_fingerprint"]
    raw["data"]["split_seed"] = 1                       # another split: refused
    with pytest.raises(ValueError):
        Experiment(Cfg.from_dict(raw))
    raw["data"]["split"], raw["data"]["split_seed"] = "per_seed", 20261027
    with pytest.raises(ValueError):                    # per-seed split: refused
        Experiment(Cfg.from_dict(raw))


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


def test_fewer_processes_give_the_same_checkpoint(checkpoint, tmp_path):
    """--procs only limits how many episodes run at once (GPU memory); every
    episode starts from the same weights, so the averaged result is identical."""
    import torch
    plan, out = checkpoint
    one = tmp_path / "one.pt"
    r = _pretrain("--plan", str(plan), "--group", "g", "--condition", "ppo_full", "--iterations", "2",
                  "--workers", "2", "--procs", "1", "--rounds", "4", "--out", str(one))
    assert r.returncode == 0, r.stdout + r.stderr
    a, b = torch.load(out, weights_only=False), torch.load(one, weights_only=False)
    assert b["meta"]["procs"] == 1
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
