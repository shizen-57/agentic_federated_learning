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
    # P2 at equal spend: the last 10 rounds within the budget are 5-14
    assert m["w10b"] == pytest.approx((6 * 0.2 + 4 * 0.5) / 10)
    assert m["spend_eq5"] == pytest.approx(1200.0) and m["spend"] == 2000.0
    # cost curves: accuracy and actual spend at the last round within each budget
    assert m["curve"][500] == (0.5, 500.0) and m["curve"][1400] == (0.8, 1400.0)
    assert m["curve"][2000] == (0.2, 2000.0)
    assert rl2lib.score_eq(m) == pytest.approx((0.74 + 0.32) / 2) and rl2lib.score(m) != rl2lib.score_eq(m)


def test_small_search_spaces_are_searched_exhaustively():
    import tune_rl2 as t
    spec = {"sampler_seed": 1, "n_configs": 40, "space": {
        "predictor": [{"policy.predictor": "v1"}, {"policy.predictor": "v2", "policy.gate_mode": "thompson"}],
        "policy.elig_threshold": [0.4, 0.5, 0.6]}}
    cs = t.sample_configs(spec)
    assert cs[0] == {} and len(cs) == 7 and len({json.dumps(c, sort_keys=True) for c in cs}) == 7
    assert {"policy.predictor": "v2", "policy.gate_mode": "thompson", "policy.elig_threshold": 0.6} in cs
    for p in ("rl2_tune_b_cov_ew_mimic", "rl2_tune_b_delta_ew_mimic", "rl2_tune_b_oort_ew_mimic"):
        big = yaml.safe_load(open(os.path.join(ROOT, "configs", p + ".yaml")))
        assert 1 < len(t.sample_configs(big)) <= 40


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

TREFS = ("rl_light_mimic", "arl3_full_pg_mimic", "b_oort_ew_mimic")


def _screen_c(tmp_path, tuned_overrides=None, refs=("rl_light",), trefs=TREFS):
    """A one-dataset screen C spec with tuned comparators (each with default
    settings in its own tuned-settings file); returns (path, resolved spec)."""
    import screen_c_rl2 as sc
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"base": "ppo_full_mimic", "overrides": tuned_overrides or {}}))
    s = yaml.safe_load(open(SPEC_C))
    rel = lambda p: os.path.relpath(p, ROOT)  # noqa: E731
    files = {}
    for c in trefs:
        f = tmp_path / f"tuned_{c}.json"
        f.write_text(json.dumps({"base": c, "overrides": {}}))
        files[f"{c}_tuned"] = rel(f)
    s.update(datasets={"femnist": "s2_femnist"}, tuned=rel(tuned), rounds=3, references=list(refs),
             tuned_references=files, control="rl_light_mimic_tuned")
    s["pretrain"]["checkpoints"] = rel(tmp_path / "ckpt")
    path = tmp_path / "spec_c.yaml"
    path.write_text(yaml.safe_dump(s))
    s["tuned_references"] = sc.resolve_trefs(files)
    return str(path), s


def test_screen_c_methods_alias_and_plan(tmp_path):
    import screen_c_rl2 as sc
    _, s = _screen_c(tmp_path)                    # the base already uses ARL-v4
    tuned = {"base": "ppo_full_mimic", "overrides": {}}
    ms = sc.methods(s, tuned, "femnist")
    assert ms["tuned_arl4"]["alias_of"] == "tuned" and ms["tuned_noarl"]["alias_of"] is None
    plan = sc.build_plan(s, tuned)
    assert set(plan["groups"]) == {"femnist_tuned_pt", "femnist_tuned_noarl_pt", "femnist_tuned_scratch",
                                   "femnist_ref", *(f"femnist_tref_{c}_tuned" for c in TREFS)}
    # comparators aggregate like the tuned method (gap review A1)
    from sweep import build_jobs
    jobs = build_jobs(plan, None, None, "/x", [])
    agg = {j["group"]: j["raw"]["fl"]["aggregator"] for j in jobs}
    assert {agg[f"femnist_tref_{c}_tuned"] for c in TREFS} == {"mimic"} == {agg["femnist_tuned_pt"]}
    pt = plan["groups"]["femnist_tuned_pt"]["override"]
    assert pt["policy.init_checkpoint"].endswith(os.path.join("ckpt", "tuned_femnist.pt"))
    assert not os.path.isabs(pt["policy.init_checkpoint"])        # no home folder in saved configs
    assert "policy.init_checkpoint" not in plan["groups"]["femnist_tuned_scratch"].get("override", {})
    raw = tmp_path / "t2.json"
    ms2 = sc.methods(s, {"base": "ppo_full_mimic", "overrides": {"reward.damage_metric": "raw_fail",
                                                                  "policy.per_client_credit": False}}, "femnist")
    assert ms2["tuned_noarl"]["alias_of"] == "tuned" and ms2["tuned_arl4"]["alias_of"] is None
    assert not raw.exists()


def _m(eq, w, wb=None):
    return {"eq5": eq, "w10": w, "w10b": w if wb is None else wb, "acc10": eq, "starved": False,
            "spend": 1400.0, "n_never": 3.0, "jain": 0.8}


def test_screen_c_decision_rule(tmp_path, monkeypatch):
    import screen_c_rl2 as sc
    _, s = _screen_c(tmp_path)
    seeds = range(240, 250)
    runs = {"femnist": {
        "tuned_pt": {k: _m(0.80, 0.55) for k in seeds}, "tuned_arl4_pt": {k: _m(0.80, 0.55) for k in seeds},
        "tuned_noarl_pt": {k: _m(0.79, 0.52) for k in seeds},
        "tuned_scratch": {k: _m(0.78 + (k % 2) * 0.03, 0.55) for k in seeds},
        "rl_light": {k: _m(0.77, 0.50) for k in seeds},
        "rl_light_mimic_tuned": {k: _m(0.785, 0.51) for k in seeds},
        "arl3_full_pg_mimic_tuned": {k: _m(0.76, 0.52, 0.50) for k in seeds},
        "b_oort_ew_mimic_tuned": {k: _m(0.70, 0.40, 0.53) for k in seeds}}}
    monkeypatch.setattr(sc, "methods", lambda spec, tuned, d: {})
    import torch
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"meta": {"practice_totals": {"episodes": 1}}})
    dec, lines = sc.decide(s, {"overrides": {}}, runs, {"femnist_tuned": ("p", "g", {})})
    # pre-trained vs scratch: score diffs +0.01 / -0.005 alternate -> median 0.0025 >= 0
    assert dec["final"]["name"] == "tuned_pt"
    # the control is the tuned always-Light; the second comparator is the best
    # tuned comparator on each endpoint (gap review A6)
    assert dec["control"] == "rl_light_mimic_tuned"
    assert dec["best_baseline"] == "arl3_full_pg_mimic_tuned"        # best eq5 among the others
    assert dec["best_baseline_p2"] == "b_oort_ew_mimic_tuned"        # best w10b among the others
    tests = {(t["family"], t["vs"]): t for t in dec["tests"]}
    assert set(tests) == {("P1", "rl_light_mimic_tuned"), ("P1", "arl3_full_pg_mimic_tuned"),
                          ("P2", "rl_light_mimic_tuned"), ("P2", "b_oort_ew_mimic_tuned")}
    assert {t["endpoint"] for t in dec["tests"] if t["family"] == "P2"} == {"w10b"}
    c = tests[("P1", "rl_light_mimic_tuned")]
    assert c["mean_diff"] == pytest.approx(0.015) and c["sd"] == 0
    for k in seeds:
        runs["femnist"]["tuned_scratch"][k] = _m(0.85, 0.6)
    assert sc.decide(s, {"overrides": {}}, runs, {"femnist_tuned": ("p", "g", {})})[0]["final"]["name"] == \
        "tuned_scratch"


# --------------------------------------------------------------- M9 / M10

def _frozen(tmp_path, monkeypatch, dec_extra=None):
    """A freeze made from a fabricated screen C decision (one dataset), with
    the pins written into a COPY of prespec_rl2.py."""
    if not os.path.exists(os.path.join(ROOT, "data", "femnist_writer.npy")):
        pytest.skip("the recorded data roles need the FEMNIST data")
    import freeze_rl2
    spec_path, s = _screen_c(tmp_path)
    ck = {"femnist_tuned": {"path": "x", "sha256": "0" * 64}}      # screen C's: not used by the freeze
    ctl, b1, b2 = "rl_light_mimic_tuned", "b_oort_ew_mimic_tuned", "arl3_full_pg_mimic_tuned"
    tests = [{"family": f, "dataset": "femnist", "vs": v, "endpoint": e, "sd": 0.02, "mean_diff": 0.01, "n": 10}
             for f, e, vs in (("P1", "eq5", (ctl, b1)), ("P2", "w10b", (ctl, b2))) for v in vs]
    dec = {"code_hash": "screen-C-code", "spec": os.path.relpath(spec_path, ROOT), "spec_sha256": rl2lib._sha(spec_path),
           "tuned": s["tuned"], "tuned_sha256": rl2lib._sha(os.path.join(ROOT, s["tuned"])),
           "final": {"name": "tuned_pt", "method": "tuned", "pretrained": True}, "control": ctl,
           "best_baseline": b1, "best_baseline_p2": b2, "tuned_references": s["tuned_references"],
           "tests": tests, "checkpoints": ck,
           "eval_pool": "dev", "data_roles_sha256": rl2lib._sha(os.path.join(ROOT, "configs", "rl2_data_roles.json")),
           **(dec_extra or {})}
    (tmp_path / "DECISION.json").write_text(json.dumps(dec))
    pre_copy = tmp_path / "prespec_copy.py"
    pre_copy.write_text(open(os.path.join(ROOT, "scripts", "prespec_rl2.py")).read())
    monkeypatch.setattr(rl2lib, "git", lambda *a: "" if a[0] == "status" else "c0ffee")
    monkeypatch.setattr(sys, "argv", ["freeze_rl2.py", "--decision", str(tmp_path / "DECISION.json"),
                                      "--out", str(tmp_path / "frozen.json"), "--prespec", str(pre_copy)])
    assert freeze_rl2.main() == 0
    return str(tmp_path / "frozen.json"), json.load(open(tmp_path / "frozen.json")), pre_copy.read_text()


def test_freeze_pins_seeds_power_and_tests(tmp_path, monkeypatch):
    import freeze_rl2
    path, frozen, pinned = _frozen(tmp_path, monkeypatch)
    assert f'FROZEN_SHA256 = "{rl2lib._sha(path)}"' in pinned and f'FROZEN_CODE = "{frozen["code_hash"]}"' in pinned
    # sd 0.02 from 10 seeds, taken at its 80% upper confidence limit (0.0258;
    # gap review B8): P1 needs ((2.50 + 0.84) * 0.0258 / 0.01)^2 = 75 seeds
    sd = freeze_rl2.sd_upper(0.02, 10)
    assert sd == pytest.approx(0.0258, abs=2e-4)
    n = rl2lib.seeds_needed(sd, 0.01, 8)
    assert frozen["power"]["n"] == n == 75 and frozen["seeds"] == list(range(300, 300 + n))
    assert len(frozen["primary_tests"]) == 4 and {t["alternative"] for t in frozen["primary_tests"]} == {"greater"}
    assert {(t["family"], t["vs"], t["endpoint"]) for t in frozen["primary_tests"]} == {
        ("P1", "rl_light_mimic_tuned", "eq5"), ("P1", "b_oort_ew_mimic_tuned", "eq5"),
        ("P2", "rl_light_mimic_tuned", "w10b"), ("P2", "arl3_full_pg_mimic_tuned", "w10b")}
    groups = frozen["plan"]["groups"]
    assert groups["femnist_ref"]["conditions"] == ["rl_light"]
    assert {g for g in groups if "_tref_" in g} == {"femnist_tref_rl_light_mimic_tuned",
                                                     "femnist_tref_b_oort_ew_mimic_tuned",
                                                     "femnist_tref_arl3_full_pg_mimic_tuned"}
    # secondary worlds: the final method (pre-trained in the usual world) and the control, on 20 seeds
    worlds = {g for g in groups if "_world_" in g}
    assert worlds == {f"femnist_world_{w}_{k}" for w in freeze_rl2.WORLDS for k in ("final", "control")}
    fs = groups["femnist_world_fedscale_devices_final"]
    assert fs["seeds"] == list(range(300, 320)) and fs["set"]["system.device_trace"] == "fedscale"
    assert fs["override"]["policy.init_checkpoint"].endswith(os.path.join("confirm", "tuned_femnist.pt"))
    # measured on the reserved confirmation pool, pre-trained afresh in M10
    assert all(g["set"]["data.eval_pool"] == "confirm" for g in frozen["plan"]["groups"].values())
    assert frozen["spec"]["pretrain"]["checkpoints"].endswith(os.path.join("ckpt", "confirm"))
    assert "configs/rl2_data_roles.json" in frozen["inputs"] and frozen["screen_c_code_hash"] == "screen-C-code"
    import seeds as seedreg
    seedreg.check(frozen["seeds"], "confirmation", path)
    with pytest.raises(ValueError):
        seedreg.check(frozen["seeds"], "confirmation", None)


def test_freeze_refuses_a_screen_c_from_before_the_reserved_pools(tmp_path, monkeypatch):
    # an old screen C decision has no pool record: its runs read the whole
    # evaluation pool, so it may choose nothing for the confirmation
    for extra in ({"eval_pool": None}, {"data_roles_sha256": "0" * 64}):
        with pytest.raises(SystemExit, match="development pool"):
            _frozen(tmp_path, monkeypatch, extra)


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
    import screen_c_rl2 as sc
    import torch
    out = tmp_path / "confirm"
    rl2lib.write_plan(frozen["plan"], str(out))
    ck = sc.checkpoints(frozen["spec"], pre.tuned_of(frozen))
    for name, (p, group, ov) in ck.items():             # checkpoints as M10's pre-training makes them
        os.makedirs(os.path.dirname(os.path.join(ROOT, p)), exist_ok=True)
        torch.save({"state_dict": {}, "meta": {"code_hash": code_hash(), "practice_only": True, "group": group,
                                               "condition": frozen["spec"]["base"], "overrides": ov}},
                   os.path.join(ROOT, p))
    for j in build_jobs(frozen["plan"], None, None, str(out), []):
        os.makedirs(j["raw"]["out"], exist_ok=True)
        base = os.path.join(j["raw"]["out"], j["raw"]["tag"])
        ck_path = (j["raw"].get("policy") or {}).get("init_checkpoint")
        sha = rl2lib._sha(os.path.join(ROOT, ck_path)) if ck_path else None
        json.dump({**expected_config(j["raw"]), "code_hash": code_hash(), "init_checkpoint_sha256": sha},
                  open(base + ".config.json", "w"))
        top = j["group"] == "femnist_tuned_pt"
        acc = 0.7 + (gap if top else 0.0) + 0.001 * (j["seed"] % 7)
        _csv(base + ".csv", [(acc, acc - 0.25, 50.0)] * frozen["spec"]["rounds"])
    man = rl2lib.write_manifest(str(out), {**pre.inputs_of(frozen, frozen_path),
                                           **sc.checkpoint_inputs(frozen["spec"], pre.tuned_of(frozen))},
                                ["fl.device=auto"])
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
    # the disclosures (gap review A5) and the secondary worlds are in the report
    for text in ("## Resources, fairness and safety", "Accuracy at common budget checkpoints",
                 "Pre-training cost per checkpoint", "## Secondary worlds"):
        assert text in report
    assert "world femnist churn w10b" in res["disclosures"]
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
    spec = os.path.join(ROOT, next(p for p in frozen["inputs"] if p.endswith("spec_c.yaml")))
    open(spec, "a").write("# changed\n")               # a changed pinned input blocks it
    assert confirm_rl2.main() == 1


def test_development_runs_cannot_touch_the_confirmation_pool(tmp_path):
    import screen_c_rl2 as sc
    import tune_rl2 as t
    spec = yaml.safe_load(open(SPEC_T))
    spec["set"] = {"data.split": "persistent", "data.eval_pool": "confirm"}
    cs = t.sample_configs(spec)
    with pytest.raises(SystemExit, match="may not use the confirmation pool"):
        rl2lib.check_pool(t.stage_plan(spec, 1, [0], cs), "dev")
    _, s = _screen_c(tmp_path)
    plan = sc.build_plan(s, {"base": "ppo_full_mimic", "overrides": {}})
    rl2lib.check_pool(plan, "dev")                       # the default: development pool
    with pytest.raises(SystemExit, match="must use data.eval_pool=confirm"):
        rl2lib.check_pool(plan, "confirmation")


# ------------------------------------------ controlled screens (ARL, sensitivity)

def _m2(eq, w, ratio=1.0):
    return {"eq5": eq, "w10": w, "w10b": w, "acc10": eq, "starved": False, "spend_ratio_late": ratio}


def test_arl_rule_verdicts():
    import screen_rl2 as sc2
    spec = {"datasets": {"femnist": "s2_femnist"}, "rule": {"kind": "arl", "plain": "plain",
                                                             "contrasts": ["arl3", "arl4"], "alpha": 0.05}}
    seeds = range(250, 270)
    R = {("femnist", None, "plain"): {s: _m2(0.70, 0.50) for s in seeds},
         ("femnist", None, "arl3"): {s: _m2(0.70 + 0.001 * (s % 3 - 1), 0.50) for s in seeds},   # no effect
         ("femnist", None, "arl4"): {s: _m2(0.72 + 0.001 * (s % 3), 0.52) for s in seeds}}       # clearly better
    spec["arms"] = {"plain": {}, "arl3": {"override": {"reward.damage_metric": "arl_v3"}},
                    "arl4": {"override": {"reward.damage_metric": "arl_v4"}}}
    lines, verdict = sc2.rule_arl(spec, R)
    text = "\n".join(lines)
    assert "**arl3: no evidence that it adds value**" in text and "**arl4: adds value**" in text
    assert verdict["arms"]["arl4"]["verdict"] == "adds value"
    assert verdict["arms"]["arl4"]["override"] == {"reward.damage_metric": "arl_v4"}
    assert {t["endpoint"] for t in verdict["tests"]} == {"eq5", "w10b"}        # both at equal spend
    R[("femnist", None, "arl4")] = {s: _m2(0.68 - 0.001 * (s % 3), 0.48) for s in seeds}
    assert "**arl4: makes it worse**" in "\n".join(sc2.rule_arl(spec, R)[0])


def test_sensitivity_rule_reports_reversals():
    import screen_rl2 as sc2
    spec = {"datasets": {"femnist": "s2_femnist"}, "settings": {"base": {}, "lo": {}, "hi": {}},
            "rule": {"kind": "sensitivity", "arm": "tuned", "reference": "rl_light",
                     "factors": {"deadline": ["lo", "base", "hi"]}}}
    seeds = range(270, 280)
    R = {("femnist", s, "rl_light"): {k: _m2(0.70, 0.50) for k in seeds} for s in ("base", "lo", "hi")}
    R.update({("femnist", s, "tuned"): {k: _m2(0.73, 0.55) for k in seeds} for s in ("base", "hi")})
    R[("femnist", "lo", "tuned")] = {k: _m2(0.69, 0.55) for k in seeds}       # loses at a short deadline
    text = "\n".join(sc2.rule_sensitivity(spec, R)[0])
    assert "**deadline: NOT robust** -- reversed at femnist lo eq5" in text


def test_cost_rule_adopts_a_variant_only_if_it_meets_the_target_cheaply():
    import screen_rl2 as sc2
    spec = {"datasets": {"femnist": "s2_femnist"},
            "arms": {"base": {"tuned": True}, "fast": {"tuned": True, "override": {"policy.lagrange_lr": 0.5}},
                     "cap": {"tuned": True, "override": {"policy.hard_budget_mult": 1.0}}},
            "rule": {"kind": "cost", "base": "base", "variants": ["fast", "cap"], "slack": 1.05, "max_loss": 0.01}}
    tuned = {"overrides": {"policy.cost_target_mult": 1.0}}
    seeds = range(280, 285)
    R = {("femnist", None, "base"): {k: _m2(0.76, 0.52, 1.25) for k in seeds},     # 25% over its target
         ("femnist", None, "fast"): {k: _m2(0.755, 0.52, 1.02) for k in seeds},    # meets it, small loss
         ("femnist", None, "cap"): {k: _m2(0.70, 0.45, 0.98) for k in seeds}}      # meets it, big loss
    lines, v = sc2.rule_cost(spec, tuned, R)
    assert v["adopt"] == "fast" and v["override"] == {"policy.lagrange_lr": 0.5}
    R[("femnist", None, "fast")] = {k: _m2(0.70, 0.45, 1.02) for k in seeds}
    lines, v = sc2.rule_cost(spec, tuned, R)
    assert v["adopt"] is None and "keep the base" in "\n".join(lines)
    R[("femnist", None, "base")] = {k: _m2(0.76, 0.52, 1.0) for k in seeds}     # base already meets it
    assert sc2.rule_cost(spec, tuned, R)[1]["adopt"] is None


def test_transfer_rule():
    import screen_rl2 as sc2
    spec = {"datasets": {"cifar": "s2_cifar"}, "arms": {"c00": {}, "c17": {}, "c34": {}},
            "rule": {"kind": "transfer", "selected": "c34", "base_candidate": "c00"}}
    seeds = range(285, 290)
    R = {("cifar", None, "c00"): {k: _m2(0.50, 0.20) for k in seeds},
         ("cifar", None, "c17"): {k: _m2(0.53, 0.22) for k in seeds},
         ("cifar", None, "c34"): {k: _m2(0.51, 0.21) for k in seeds}}
    lines, v = sc2.rule_transfer(spec, R)
    assert v["holds"] and v["rank"] == 2 and "holds" in "\n".join(lines)
    R[("cifar", None, "c34")] = {k: _m2(0.48, 0.19) for k in seeds}
    assert not sc2.rule_transfer(spec, R)[1]["holds"]


def test_finalize_carries_verdicts_into_the_tuned_settings(tmp_path):
    """The ARL route (gap review A4): an ARL version that adds value becomes
    part of the final settings; nothing changes otherwise; a stale verdict is
    refused."""
    import finalize_tuned as ft
    root = tmp_path
    (root / "configs").mkdir()
    (root / "configs" / "rl2_tuned_cost.json").write_text(json.dumps(
        {"base": "ppo_full_mimic", "overrides": {"reward.damage_metric": "raw_fail", "policy.lagrange_lr": 0.5}}))
    spec = root / "arl.yaml"
    spec.write_text(yaml.safe_dump({"tuned": "configs/rl2_tuned_cost.json"}))
    verdict = {"kind": "arl", "_sha256": "v" * 64, "spec_sha256": rl2lib._sha(str(spec)),
               "tuned_sha256": rl2lib._sha(str(root / "configs" / "rl2_tuned_cost.json")),
               "arms": {"arl3": {"verdict": "adds value", "override": {"reward.damage_metric": "arl_v3"},
                                 "mean_score_gain": 0.004},
                        "arl4": {"verdict": "adds value", "override": {"reward.damage_metric": "arl_v4"},
                                 "mean_score_gain": 0.009}}}
    out = ft.finalize("arl", str(spec), verdict, root=str(root))
    assert out["overrides"]["reward.damage_metric"] == "arl_v4" and out["overrides"]["policy.lagrange_lr"] == 0.5
    assert out["provenance"][-1]["adopted"] == "arl4"
    assert json.load(open(root / "configs" / "rl2_tuned_final.json")) == out
    verdict["arms"]["arl4"]["verdict"] = verdict["arms"]["arl3"]["verdict"] = "no evidence that it adds value"
    out = ft.finalize("arl", str(spec), verdict, root=str(root))
    assert out["overrides"]["reward.damage_metric"] == "raw_fail" and out["provenance"][-1]["adopted"] is None
    (root / "configs" / "rl2_tuned_cost.json").write_text("{}")
    with pytest.raises(SystemExit, match="changed after the screen ran"):
        ft.finalize("arl", str(spec), verdict, root=str(root))
    with pytest.raises(SystemExit, match="needs a screen run on"):
        ft.finalize("cost", str(spec), verdict, root=str(root))


def test_controlled_screen_end_to_end_tiny(tmp_path):
    """The real runner at tiny size: smoke, runs, evidence, rule."""
    if not os.path.exists(os.path.join(ROOT, "data", "femnist_x.npy")):
        pytest.skip("needs the FEMNIST data")
    s = yaml.safe_load(open(os.path.join(ROOT, "configs", "rl2_arl_screen.yaml")))
    s.update(purpose="diagnostic", seeds=[900], rounds=2, datasets={"femnist": "s2_femnist"},
             tuned="configs/rl2_tuned.json")      # the real one comes from the cost check
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(s))
    out = os.path.relpath(tmp_path / "out", ROOT)
    r = subprocess.run([sys.executable, "scripts/screen_rl2.py", "run", "--spec", str(spec), "--out", out,
                        "--jobs", "2", "--set", "fl.device=cpu"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True, cwd=ROOT).stdout.strip()
    if dirty:
        assert r.returncode == 2 and "uncommitted changes" in r.stdout, r.stdout[-3000:]
    else:
        assert r.returncode == 0 and "Verdict" in r.stdout, r.stdout[-3000:]
