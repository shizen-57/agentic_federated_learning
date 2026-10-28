"""RL2 pipeline after screen B: M7 tuning (tune_rl2.py), M8 screen C
(screen_c_rl2.py), M9 freeze (freeze_rl2.py), M10 confirmation
(confirm_rl2.py, prespec_rl2.py), and their shared library (rl2lib.py)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import rl2lib  # noqa: E402

SPEC_T = os.path.join(ROOT, "configs", "rl2_tune.yaml")
SPEC_C = os.path.join(ROOT, "configs", "rl2_screen_c.yaml")


# ------------------------------------------------------------ shared pieces

def test_group_override_wins_over_the_condition():
    from sweep import build_jobs
    src = rl2lib.source_group("configs/sweep_advanced.yaml", "s2_femnist")
    plan = {"seeds": [215], "sets": {}, "groups": {
        "g": rl2lib.group_entry(src, {}, 3, ["ppo_full_mimic"], [215],
                                {"policy.encoder": "mlp", "reward.damage_metric": "raw_fail"})}}
    raw = build_jobs(plan, None, None, "/tmp/x", [])[0]["raw"]
    assert raw["policy"]["encoder"] == "mlp" and raw["reward"]["damage_metric"] == "raw_fail"
    assert raw["fl"]["aggregator"] == "mimic"          # the condition's other settings stay


def test_group_value_restores_the_groups_own_setting():
    src = rl2lib.source_group("configs/sweep_advanced.yaml", "s2_femnist")
    e = rl2lib.group_entry(src, {}, 3, ["ppo_full_mimic"], [215])
    assert rl2lib.resolve(e, {"reward.beta": rl2lib.GROUP_VALUE, "policy.ppo_lr": 0.01}) == \
        {"reward.beta": rl2lib.group_defaults(e)["reward"]["beta"], "policy.ppo_lr": 0.01}


def test_sign_flip_holm_and_seed_count():
    d = [0.1] * 6
    assert rl2lib.signflip_p(d, "greater") == pytest.approx(1 / 64)
    assert rl2lib.signflip_p(d, "two-sided") == pytest.approx(2 / 64)
    assert rl2lib.signflip_p([-0.1] * 6, "greater") == 1.0
    mixed = [0.3, -0.1, 0.2, 0.05, -0.02, 0.15, 0.1, -0.05, 0.12, 0.08, 0.2, -0.01]
    exact = rl2lib.signflip_p(mixed, "greater")
    mc = rl2lib.signflip_p(mixed * 2, "greater")         # n = 24 -> Monte Carlo
    assert 0 < exact < 0.05 and 0 < mc < exact
    assert rl2lib.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    # n = ((z_{1-0.05/8} + z_0.8) * sd / delta)^2: sd = delta -> 12
    assert rl2lib.seeds_needed(0.01, 0.01, 8) == 12
    assert rl2lib.seeds_needed(0.02, 0.01, 8) == 45


def _csv(path, rows):
    from afl.engine import LOG_FIELDS
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LOG_FIELDS)
        for t, (acc, w10, cost) in enumerate(rows, 1):
            r = dict.fromkeys(LOG_FIELDS, 0.0)
            r.update(round=t, n_selected=10, n_success=6, n_online=40, cost=cost, mode_light=10,
                     accuracy=acc, accuracy_fixed=acc, acc_client_worst10=w10)
            w.writerow([r[k] for k in LOG_FIELDS])
    return path


def test_steadier_endpoints(tmp_path):
    import report_dev_screen as rep
    rows = [(0.5, 0.2, 100.0)] * 10 + [(0.8, 0.5, 100.0)] * 10      # cost 1,400 reached at round 14
    rows[-1] = (0.2, 0.0, 100.0)                                   # a one-round dip at the end
    m = rep.measures(_csv(tmp_path / "r.csv", rows))
    assert m["acc"] == 0.2 and m["acc10"] == pytest.approx((9 * 0.8 + 0.2) / 10)
    assert m["w10"] == pytest.approx((9 * 0.5 + 0.0) / 10)
    # rounds 10-14 are the last 5 within the budget: (0.5 + 4 x 0.8) / 5
    assert m["eq5"] == pytest.approx(0.74) and m["acc_budget"] == (0.8, 14)


# --------------------------------------------------------------------- M7

def test_tuning_candidates_are_fixed_distinct_and_valid():
    import tune_rl2 as t
    from afl.config import Cfg
    from sweep import build_jobs
    spec = yaml.safe_load(open(SPEC_T))
    a, b = t.sample_configs(spec), t.sample_configs(spec)
    assert a == b and a[0] == {} and len(a) == spec["n_configs"]
    assert len({json.dumps(c, sort_keys=True) for c in a}) == len(a)
    plan = t.stage_plan(spec, 1, list(range(len(a))), a)
    jobs = build_jobs(plan, None, None, "/tmp/x", [])
    assert len(jobs) == len(a) * 5 + 5
    for j in jobs:
        Cfg.from_dict(j["raw"])
        assert j["raw"]["data"]["split"] == "persistent"
    import seeds as seedreg
    seedreg.check(spec["stage1_seeds"] + spec["stage2_seeds"], spec["purpose"])
    assert not set(spec["stage1_seeds"]) & set(spec["stage2_seeds"])


def test_tuning_rank_disqualifies_starvation_and_breaks_ties_by_id():
    import tune_rl2 as t
    m = lambda eq, w, starved=False: {"eq5": eq, "w10": w, "acc10": eq, "starved": starved}  # noqa: E731
    by = {"c00": {"b": {1: m(0.7, 0.5)}}, "c01": {"b": {1: m(0.9, 0.6, True)}},
          "c02": {"b": {1: m(0.8, 0.5)}}, "c03": {"b": {1: m(0.8, 0.5)}}, "ref": {"rl_light": {1: m(0, 0)}}}
    assert [r["id"] for r in t.rank(by, "b")] == ["c02", "c03", "c00", "c01"]


def test_recorded_ranking_is_compared_not_rewritten():
    import tune_rl2 as t
    row = {"id": "c03", "n": 5, "score": 0.6412035920158898, "eq5": 0.77, "w10": 0.5, "acc10": 0.78,
           "starved": 0}
    rec = {"ranking": [row], "references": [], "top": ["c03"]}
    py313 = {"ranking": [{**row, "score": 0.64120359201589}], "references": [], "top": ["c03"]}
    assert t.same_ranking(rec, py313) is None                 # last-digit float noise (Python 3.11 vs 3.13)
    assert "score" in t.same_ranking(rec, {**py313, "ranking": [{**row, "score": 0.65}]})
    assert "top" in t.same_ranking(rec, {**py313, "top": ["c04"]})
    assert "starved" in t.same_ranking(rec, {**py313, "ranking": [{**row, "starved": 1}]})


def test_tune_end_to_end_tiny(tmp_path):
    """The real runner at tiny size: smoke, stage 1, check, rank, stage 2."""
    if not os.path.exists(os.path.join(ROOT, "data", "femnist_x.npy")):
        pytest.skip("needs the FEMNIST data")
    s = yaml.safe_load(open(SPEC_T))
    s.update(purpose="diagnostic", n_configs=2, stage1_seeds=[900], stage2_seeds=[900], top_k=1, rounds=2)
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(s))
    out = os.path.relpath(tmp_path / "out", ROOT)
    r = subprocess.run([sys.executable, "scripts/tune_rl2.py", "--spec", str(spec), "--out", out,
                        "--jobs", "2", "--set", "fl.device=cpu"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True, cwd=ROOT).stdout.strip()
    if dirty:      # during development only the clean-checkout check may fail
        assert r.returncode == 2 and "uncommitted changes" in r.stdout, r.stdout[-3000:]
    else:
        assert r.returncode == 0 and "Selected:" in r.stdout, r.stdout[-3000:]
        assert json.load(open(tmp_path / "out" / "SELECTED.json"))["winner"] in ("c00", "c01")


# --------------------------------------------------------------------- M8

def _screen_c(tmp_path, tuned_overrides=None, refs=("rl_light", "rl_light_mimic", "arl3_full_pg", "b_oort_ew")):
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"base": "ppo_full_mimic", "overrides": tuned_overrides or {}}))
    s = yaml.safe_load(open(SPEC_C))
    rel = lambda p: os.path.relpath(p, ROOT)  # noqa: E731
    s.update(datasets={"femnist": "s2_femnist"}, tuned=rel(tuned), rounds=3, references=list(refs))
    s["pretrain"]["checkpoints"] = rel(tmp_path / "ckpt")
    path = tmp_path / "spec_c.yaml"
    path.write_text(yaml.safe_dump(s))
    return str(path), s


def test_screen_c_methods_alias_and_plan(tmp_path):
    import screen_c_rl2 as sc
    _, s = _screen_c(tmp_path)                    # the base already uses ARL-v4
    tuned = {"base": "ppo_full_mimic", "overrides": {}}
    ms = sc.methods(s, tuned, "femnist")
    assert ms["tuned_arl4"]["alias_of"] == "tuned" and ms["tuned_noarl"]["alias_of"] is None
    plan = sc.build_plan(s, tuned)
    assert set(plan["groups"]) == {"femnist_tuned_pt", "femnist_tuned_noarl_pt", "femnist_tuned_scratch",
                                   "femnist_ref"}
    pt = plan["groups"]["femnist_tuned_pt"]["override"]
    assert pt["policy.init_checkpoint"].endswith(os.path.join("ckpt", "tuned_femnist.pt"))
    assert not os.path.isabs(pt["policy.init_checkpoint"])        # no home folder in saved configs
    assert "policy.init_checkpoint" not in plan["groups"]["femnist_tuned_scratch"].get("override", {})
    raw = tmp_path / "t2.json"
    ms2 = sc.methods(s, {"base": "ppo_full_mimic", "overrides": {"reward.damage_metric": "raw_fail",
                                                                  "policy.per_client_credit": False}}, "femnist")
    assert ms2["tuned_noarl"]["alias_of"] == "tuned" and ms2["tuned_arl4"]["alias_of"] is None
    assert not raw.exists()


def _m(eq, w):
    return {"eq5": eq, "w10": w, "acc10": eq, "starved": False}


def test_screen_c_decision_rule(tmp_path, monkeypatch):
    import screen_c_rl2 as sc
    _, s = _screen_c(tmp_path)
    seeds = range(240, 250)
    runs = {"femnist": {
        "tuned_pt": {k: _m(0.80, 0.55) for k in seeds}, "tuned_arl4_pt": {k: _m(0.80, 0.55) for k in seeds},
        "tuned_noarl_pt": {k: _m(0.79, 0.52) for k in seeds},
        "tuned_scratch": {k: _m(0.78 + (k % 2) * 0.03, 0.55) for k in seeds},
        "rl_light": {k: _m(0.77, 0.50) for k in seeds}, "rl_light_mimic": {k: _m(0.785, 0.51) for k in seeds},
        "arl3_full_pg": {k: _m(0.76, 0.52) for k in seeds}, "b_oort_ew": {k: _m(0.70, 0.40) for k in seeds}}}
    monkeypatch.setattr(sc, "methods", lambda spec, tuned, d: {})
    import torch
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"meta": {"practice_totals": {"episodes": 1}}})
    dec, lines = sc.decide(s, {"overrides": {}}, runs, {"femnist_tuned": ("p", "g", {})})
    # pre-trained vs scratch: score diffs +0.01 / -0.005 alternate -> median 0.0025 >= 0
    assert dec["final"]["name"] == "tuned_pt"
    assert dec["best_baseline"] == "rl_light_mimic"          # highest equal-spend acc after rl_light
    tests = {(t["family"], t["vs"]): t for t in dec["tests"]}
    assert set(tests) == {("P1", "rl_light"), ("P1", "rl_light_mimic"), ("P2", "rl_light"),
                          ("P2", "arl3_full_pg")}
    assert tests[("P1", "rl_light")]["mean_diff"] == pytest.approx(0.03) and tests[("P1", "rl_light")]["sd"] == 0
    for k in seeds:
        runs["femnist"]["tuned_scratch"][k] = _m(0.85, 0.6)
    assert sc.decide(s, {"overrides": {}}, runs, {"femnist_tuned": ("p", "g", {})})[0]["final"]["name"] == \
        "tuned_scratch"


# --------------------------------------------------------------- M9 / M10

def _frozen(tmp_path, monkeypatch):
    """A freeze made from a fabricated screen C decision (one dataset), with
    the pins written into a COPY of prespec_rl2.py."""
    import freeze_rl2
    from afl.provenance import code_hash
    spec_path, s = _screen_c(tmp_path)
    ck = {}
    os.makedirs(tmp_path / "ckpt")
    for m in ("tuned", "tuned_noarl"):
        p = tmp_path / "ckpt" / f"{m}_femnist.pt"
        p.write_bytes(m.encode())
        ck[f"femnist_{m}"] = {"path": os.path.relpath(p, ROOT), "sha256": rl2lib._sha(str(p))}
    tests = [{"family": f, "dataset": "femnist", "vs": v, "endpoint": e, "sd": 0.02, "mean_diff": 0.01, "n": 10}
             for f, e, vs in (("P1", "eq5", ("rl_light", "b_oort_ew")), ("P2", "w10", ("rl_light", "arl3_full_pg")))
             for v in vs]
    dec = {"code_hash": code_hash(), "spec": os.path.relpath(spec_path, ROOT), "spec_sha256": rl2lib._sha(spec_path),
           "tuned": s["tuned"], "tuned_sha256": rl2lib._sha(os.path.join(ROOT, s["tuned"])),
           "final": {"name": "tuned_pt", "method": "tuned", "pretrained": True}, "best_baseline": "b_oort_ew",
           "p2_reference": "arl3_full_pg", "tests": tests, "checkpoints": ck}
    (tmp_path / "DECISION.json").write_text(json.dumps(dec))
    pre_copy = tmp_path / "prespec_copy.py"
    pre_copy.write_text(open(os.path.join(ROOT, "scripts", "prespec_rl2.py")).read())
    monkeypatch.setattr(rl2lib, "git", lambda *a: "" if a[0] == "status" else "c0ffee")
    monkeypatch.setattr(sys, "argv", ["freeze_rl2.py", "--decision", str(tmp_path / "DECISION.json"),
                                      "--out", str(tmp_path / "frozen.json"), "--prespec", str(pre_copy)])
    assert freeze_rl2.main() == 0
    return str(tmp_path / "frozen.json"), json.load(open(tmp_path / "frozen.json")), pre_copy.read_text()


def test_freeze_pins_seeds_power_and_tests(tmp_path, monkeypatch):
    path, frozen, pinned = _frozen(tmp_path, monkeypatch)
    assert f'FROZEN_SHA256 = "{rl2lib._sha(path)}"' in pinned and f'FROZEN_CODE = "{frozen["code_hash"]}"' in pinned
    # sd 0.02: P1 needs ((2.50 + 0.84) * 0.02 / 0.01)^2 = 45 seeds; P2 12 -> 45
    assert frozen["power"]["n"] == 45 and frozen["seeds"] == list(range(300, 345))
    assert len(frozen["primary_tests"]) == 4 and {t["alternative"] for t in frozen["primary_tests"]} == {"greater"}
    refs = frozen["plan"]["groups"]["femnist_ref"]["conditions"]
    assert refs == ["rl_light", "b_oort_ew", "arl3_full_pg", "rl_light_mimic"]
    import seeds as seedreg
    seedreg.check(frozen["seeds"], "confirmation", path)
    with pytest.raises(ValueError):
        seedreg.check(frozen["seeds"], "confirmation", None)


def test_freeze_refuses_a_second_freeze_and_other_code(tmp_path, monkeypatch):
    import freeze_rl2
    import prespec_rl2 as pre
    monkeypatch.setattr(pre, "FROZEN_SHA256", "f" * 64)
    monkeypatch.setattr(sys, "argv", ["freeze_rl2.py", "--decision", "x"])
    with pytest.raises(SystemExit, match="already frozen"):
        freeze_rl2.main()


def _fake_confirmation(tmp_path, frozen, frozen_path, gap):
    """Runs for every job of the frozen plan: the final method gap points
    ahead of everything else on both endpoints, on every seed."""
    from sweep import build_jobs, expected_config
    from afl.provenance import code_hash
    import prespec_rl2 as pre
    out = tmp_path / "confirm"
    rl2lib.write_plan(frozen["plan"], str(out))
    for j in build_jobs(frozen["plan"], None, None, str(out), []):
        os.makedirs(j["raw"]["out"], exist_ok=True)
        base = os.path.join(j["raw"]["out"], j["raw"]["tag"])
        json.dump({**expected_config(j["raw"]), "code_hash": code_hash()}, open(base + ".config.json", "w"))
        top = j["group"] == "femnist_tuned_pt"
        acc = 0.7 + (gap if top else 0.0) + 0.001 * (j["seed"] % 7)
        _csv(base + ".csv", [(acc, acc - 0.25, 50.0)] * frozen["spec"]["rounds"])
    man = rl2lib.write_manifest(str(out), pre.inputs_of(frozen, frozen_path), ["fl.device=auto"])
    man["git_dirty"] = False
    json.dump(man, open(out / "MANIFEST.json", "w"))
    return str(out)


@pytest.mark.parametrize("gap, supported", [(0.05, 4), (0.0, 0)])
def test_prespec_verdicts_on_complete_evidence(tmp_path, monkeypatch, gap, supported):
    import prespec_rl2 as pre
    path, frozen, _ = _frozen(tmp_path, monkeypatch)
    root = _fake_confirmation(tmp_path, frozen, path, gap)
    code, report, res = pre.evaluate(frozen, path, root)
    assert code == 0, report
    assert sum(r["supported"] for r in res["primary"]) == supported
    assert all(r["n"] == len(frozen["seeds"]) for r in res["primary"])
    os.remove(os.path.join(root, "femnist_ref", "rl_light_s300.csv"))
    code, report, _ = pre.evaluate(frozen, path, root)
    assert code == 2 and "INCOMPLETE" in report and "rl_light seed 300: missing" in report


def test_confirmation_needs_the_pins_and_go(tmp_path, monkeypatch):
    import confirm_rl2
    import prespec_rl2 as pre
    path, frozen, _ = _frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["confirm_rl2.py", "--frozen", path, "--out", str(tmp_path / "c")])
    with pytest.raises(SystemExit, match="no RL2 confirmation has been frozen"):
        confirm_rl2.main()
    monkeypatch.setattr(pre, "FROZEN_SHA256", rl2lib._sha(path))
    monkeypatch.setattr(pre, "FROZEN_CODE", frozen["code_hash"])
    assert confirm_rl2.main() == 0                      # all checks pass, nothing run without --go
    assert not (tmp_path / "c").exists()
    ck = os.path.join(ROOT, next(p for p in frozen["inputs"] if p.endswith(".pt")))
    open(ck, "ab").write(b"!")                          # a changed checkpoint blocks it
    assert confirm_rl2.main() == 1
