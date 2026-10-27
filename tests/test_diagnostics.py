"""Diagnostics (diag.enabled) must observe a run without changing it, and its
records must agree with the run's own CSV rows."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from afl.config import Cfg  # noqa: E402
from afl.diagnostics import read  # noqa: E402
from afl.engine import Experiment  # noqa: E402
from afl.run import _load  # noqa: E402

# The s2 protocol switches and weak-client regime, on the small synthetic task.
CONDS = ["arl3_full", "c2_rl_fedprox", "v2_rawfail", "rl_light", "rl_ew", "b_random_ew", "b_cov_ew",
         "ppo_v3", "arl3_full_pg", "ppo_att"]


def _raw(cond, out, diag, rounds=8, seed=3):
    from sweep import build_jobs
    plan = _load(os.path.join(ROOT, "configs", "sweep_advanced.yaml"))
    s2set = dict(plan["groups"]["s2_femnist"]["set"])
    s2set.update({"fl.rounds": rounds})
    mini = {"seeds": [seed], "sets": {}, "groups": {"g": {
        "config": "configs/quick.yaml", "set": s2set, "conditions": [cond]}}}
    extra = ["diag.enabled=true"] if diag else []
    return build_jobs(mini, None, None, str(out), extra)[0]["raw"]


def _run(raw):
    torch.manual_seed(1234)
    np.random.seed(1234)
    exp = Experiment(Cfg.from_dict(raw))
    exp.run(verbose=False)
    exp.save()
    return exp


def _fingerprint(exp):
    rows = [{k: v for k, v in r.items() if k != "wall_s"} for r in exp.rows]
    pol = exp.policy
    params = ([p.detach().clone() for p in pol.net.parameters()]
              if hasattr(pol, "net") and hasattr(pol.net, "parameters") else [])
    gens = [exp.gen.get_state()] + ([pol.gen.get_state()] if hasattr(pol, "gen") and
                                    isinstance(pol.gen, torch.Generator) else [])
    return {"rows": rows, "params": params, "gens": gens,
            "global_rng": torch.get_rng_state(), "np_rng": np.random.get_state()[1].copy(),
            "pred": (exp.predictor.w.copy(), exp.predictor.b, exp.predictor.seen),
            "model": exp.global_flat.detach().clone(),
            "baseline": (getattr(pol, "baseline", None), getattr(pol, "baseline_n", None))}


@pytest.fixture(scope="module", params=CONDS)
def pair(request, tmp_path_factory):
    d = tmp_path_factory.mktemp(request.param)
    off = _run(_raw(request.param, d / "off", diag=False))
    off_fp = _fingerprint(off)
    on_raw = _raw(request.param, d / "on", diag=True)
    on = _run(on_raw)
    return request.param, off_fp, _fingerprint(on), on, on_raw


def test_diagnostics_do_not_change_the_run(pair):
    cond, a, b, *_ = pair
    assert a["rows"] == b["rows"], cond                     # every metric, every round
    assert len(a["params"]) == len(b["params"])
    for p, q in zip(a["params"], b["params"]):
        assert torch.equal(p, q)                            # learned policy
    assert torch.equal(a["model"], b["model"])              # global model
    for g, h in zip(a["gens"], b["gens"]):
        assert torch.equal(g, h)                            # RNG streams
    assert torch.equal(a["global_rng"], b["global_rng"])
    assert np.array_equal(a["np_rng"], b["np_rng"])
    assert np.array_equal(a["pred"][0], b["pred"][0]) and a["pred"][1:] == b["pred"][1:]
    assert a["baseline"] == b["baseline"]


def test_sidecars_reconcile_with_the_csv(pair):
    cond, _, _, exp, raw = pair
    base = os.path.join(raw["out"], raw["tag"])
    _, rounds = read(base + ".diag_round.jsonl")
    _, invites = read(base + ".diag_invite.jsonl")
    _, updates = read(base + ".diag_update.jsonl")
    assert [r["t"] for r in rounds] == [row["round"] for row in exp.rows]
    for rec, row in zip(rounds, exp.rows):
        inv = [x for x in invites if x["t"] == rec["t"]]
        assert len(rec["invited"]) == len(inv) == row["n_selected"]
        assert len(rec["succeeded"]) == sum(x["success"] for x in inv) == row["n_success"]
        assert rec["mode_counts"] == {"light": row["mode_light"], "standard": row["mode_standard"],
                                      "priority": row["mode_priority"]}
        assert rec["gate_filled"] == row["gate_filled"] == len(rec["fill"])
        assert rec["reward"]["total"] == pytest.approx(row["reward"], abs=0, rel=0)
        assert sum(x["charged"] for x in inv) == pytest.approx(row["cost"], rel=1e-12, abs=1e-12)
        assert sum(rec["reason_counts"].values()) == row["n_selected"]
        assert rec["reason_counts"].get("ok", 0) == row["n_success"]
        assert rec["empty_round"] == row["empty_round"]
        for x in inv:
            assert x["mode"] == rec["modes"][str(x["client"])]
            assert x["how"] in ("sampled", "fallback", "chosen")
            if x["how"] == "fallback":
                assert x["logp_pick"] is None and x["client"] in rec["fill"]
    if cond.startswith("b_"):
        assert updates == []
    else:
        assert {u["t"] for u in updates} == {row["round"] for row in exp.rows}
        for u in updates:
            if u["updated"] and not u.get("ppo"):
                assert u["advantage"] == pytest.approx(u["reward"] - u["baseline_before"])


def test_sampled_picks_carry_their_probabilities(pair):
    cond, _, _, _, raw = pair
    if cond.startswith("b_"):
        return
    _, invites = read(os.path.join(raw["out"], raw["tag"]) + ".diag_invite.jsonl")
    sampled = [x for x in invites if x["how"] == "sampled"]
    for x in sampled:
        assert x["logp_pick"] is not None and x["logp_pick"] <= 0
    learns_modes = cond not in ("rl_light", "rl_ew")
    for x in invites:
        if learns_modes:
            assert abs(sum(x["mode_probs"].values()) - 1) < 1e-5
            assert x["proposed_mode"] == x["mode"] or x["rule_mode"] != x["proposed_mode"]


def test_disabled_diagnostics_write_nothing_and_keep_the_config(tmp_path):
    raw = _raw("arl3_full", tmp_path, diag=False, rounds=2)
    _run(raw)
    files = os.listdir(raw["out"])
    assert not [f for f in files if "diag" in f]
    meta = json.load(open(os.path.join(raw["out"], raw["tag"] + ".config.json")))
    assert "diag" not in meta                    # earlier runs' configs still match


def test_sidecars_are_parts_until_the_run_is_saved(tmp_path):
    raw = _raw("rl_light", tmp_path, diag=True, rounds=2)
    exp = Experiment(Cfg.from_dict(raw))
    exp.run(verbose=False)
    files = os.listdir(raw["out"])
    assert files and all(f.endswith(".part") for f in files if "diag" in f)
    exp.save()
    files = os.listdir(raw["out"])
    assert sum(f.endswith(".jsonl") for f in files) == 3 and not [f for f in files if f.endswith(".part")]
    meta = json.load(open(os.path.join(raw["out"], raw["tag"] + ".config.json")))
    assert meta["diag"] == {"enabled": True}


def test_diagnostic_runs_differ_from_the_s2_runs_only_where_intended():
    import subprocess
    for variant, diff in (("short", "diag.enabled, fl.rounds, out"), ("full", "diag.enabled, out")):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "diagnose_collapse.py"),
                            "--dry-run", "--variant", variant, "--out", "/nonexistent"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        assert r.stdout.count(f"differs from s2 in: {diff}\n") == 8
        assert "UNEXPECTED" not in r.stdout
