"""The development-screen report decides only on complete, valid evidence
(second external review, finding 7), and screen B's spec is unbundled: each
ablation removes exactly one component of ppo_full."""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import report_dev_screen as rep  # noqa: E402
from afl.conditions import CONDITIONS  # noqa: E402
from afl.engine import LOG_FIELDS  # noqa: E402
from afl.provenance import code_hash  # noqa: E402
from diagnose_collapse import _sha, plan_for  # noqa: E402
from sweep import build_jobs, expected_config  # noqa: E402

ROUNDS = 3
SPEC = {"version": 1, "title": "test screen", "purpose": "dev",
        "source_plan": "configs/sweep_advanced.yaml", "source_group": "s2_femnist",
        "seeds": [215, 216], "set": {"data.split": "persistent"},
        "conditions": ["rl_light", "arl3_full", "ppo_full", "ppo_full_mlp"],
        "reference": "arl3_full", "new_conditions": ["ppo_full", "ppo_full_mlp"],
        "variants": {"full": ROUNDS}, "out": "unused",
        "rule": {"kind": "screen_b", "gate": ["ppo_full"], "equal_spend_ref": "rl_light",
                 "accuracy_ref": "arl3_full", "margin": 0.02, "full": "ppo_full",
                 "flag_delta": 0.01, "compare": ["rl_light", "arl3_full"],
                 "ablations": {"ppo_full_mlp": "attention set encoder"}}}


def _screen(tmp_path, spec=SPEC, acc=0.5, overrides=("fl.device=auto",)):
    """Fake, valid results for every run the spec asks for (made with the
    given runner overrides), as diagnose_collapse.py writes them. Returns
    (spec path, group dir, jobs)."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    variant_dir = tmp_path / "out" / "full"
    plan, _ = plan_for(spec, "full")
    os.makedirs(variant_dir, exist_ok=True)
    (variant_dir / "plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=True))
    jobs = build_jobs(plan, None, None, str(variant_dir), list(overrides))
    for j in jobs:
        os.makedirs(j["raw"]["out"], exist_ok=True)
        base = os.path.join(j["raw"]["out"], j["raw"]["tag"])
        with open(base + ".config.json", "w") as fh:
            json.dump({**expected_config(j["raw"]), "code_hash": code_hash(),
                       "split_fingerprint": "fp"}, fh)
        with open(base + ".csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(LOG_FIELDS)
            for r in range(1, ROUNDS + 1):
                row = dict.fromkeys(LOG_FIELDS, 0.0)
                row.update(round=r, n_selected=10, n_success=6, n_online=40, cost=10.0, mode_light=10,
                           accuracy=acc, accuracy_fixed=acc, acc_client_worst10=acc / 2)
                w.writerow([row[k] for k in LOG_FIELDS])
    (variant_dir / "MANIFEST.json").write_text(json.dumps(
        {"variant": "full", "rounds": ROUNDS, "code_hash": code_hash(), "git_commit": "c0ffee",
         "git_dirty": False,
         "plan_sha256": {"diagnostic": _sha(str(spec_path)), "generated": _sha(str(variant_dir / "plan.yaml")),
                         "source": _sha(os.path.join(ROOT, spec["source_plan"]))},
         "overrides_vs_source": ["fl.rounds", "diag.enabled", "out", "data.split", *overrides]}))
    return str(spec_path), str(variant_dir / "s2_femnist"), jobs


def _report(monkeypatch, capsys, group_dir, spec_path=None):
    argv = ["report_dev_screen.py", group_dir] + (["--spec", spec_path] if spec_path else [])
    monkeypatch.setattr(sys, "argv", argv)
    code = rep.main()
    return code, capsys.readouterr().out


def _csv(jobs, cond, seed):
    j = next(j for j in jobs if j["cond"] == cond and j["seed"] == seed)
    return os.path.join(j["raw"]["out"], j["raw"]["tag"])


def test_complete_evidence_gets_a_decision(tmp_path, monkeypatch, capsys):
    spec, group, _ = _screen(tmp_path)
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 0
    assert "Evidence complete: all 8 runs" in out
    assert "Carried forward to pre-training" in out and "INCOMPLETE" not in out


@pytest.mark.parametrize("damage, why", [
    ("delete", "ppo_full seed 216: missing"),
    ("truncate", "ppo_full seed 216: incomplete"),
    ("config", "ppo_full seed 216: different config"),
    ("code", "different code versions"),
    ("nan", "ppo_full seed 216: non-finite value"),
])
def test_missing_or_invalid_runs_give_no_decision(tmp_path, monkeypatch, capsys, damage, why):
    spec, group, jobs = _screen(tmp_path)
    base = _csv(jobs, "ppo_full", 216)
    if damage == "delete":
        os.remove(base + ".csv")
    elif damage in ("truncate", "nan"):
        lines = open(base + ".csv").read().splitlines()
        if damage == "truncate":
            lines = lines[:-1]
        else:
            lines[-1] = lines[-1].replace("0.5", "nan", 1)
        open(base + ".csv", "w").write("\n".join(lines) + "\n")
    else:
        meta = json.load(open(base + ".config.json"))
        if damage == "config":
            meta["policy"]["ppo_lr"] = 1.0
        else:
            meta["code_hash"] = "0" * 16
        json.dump(meta, open(base + ".config.json", "w"))
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 2
    assert "INCOMPLETE -- no decision" in out and why in out
    assert "Carried forward" not in out and "PASS" not in out


@pytest.mark.parametrize("field, value, why", [
    ("git_dirty", True, "uncommitted changes"),
    ("code_hash", "f" * 16, "run the report from that commit"),
    ("rounds", ROUNDS + 1, "MANIFEST rounds"),
])
def test_manifest_problems_give_no_decision(tmp_path, monkeypatch, capsys, field, value, why):
    spec, group, _ = _screen(tmp_path)
    path = os.path.join(os.path.dirname(group), "MANIFEST.json")
    man = json.load(open(path))
    man[field] = value
    json.dump(man, open(path, "w"))
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 2 and why in out and "Carried forward" not in out


def test_runs_with_other_rounds_or_settings_are_invalid(tmp_path, monkeypatch, capsys):
    # a run made without the spec's shared persistent split is not the run asked for
    spec, group, jobs = _screen(tmp_path)
    base = _csv(jobs, "rl_light", 215)
    meta = json.load(open(base + ".config.json"))
    meta["data"]["split"] = "per_seed"
    json.dump(meta, open(base + ".config.json", "w"))
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 2 and "rl_light seed 215: different config" in out


def test_stray_runs_are_listed_but_ignored(tmp_path, monkeypatch, capsys):
    spec, group, jobs = _screen(tmp_path)
    base = _csv(jobs, "rl_light", 215)
    for ext in (".csv", ".config.json"):
        os.link(base + ext, os.path.join(group, "b_oort_ew_s215" + ext))
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 0 and "Ignored (not part of the spec): b_oort_ew_s215.csv" in out


def test_no_spec_means_no_decision(tmp_path, monkeypatch, capsys):
    _, group, _ = _screen(tmp_path)
    code, out = _report(monkeypatch, capsys, group)
    assert code == 0 and "no decision" in out and "Carried forward" not in out


def test_rule_may_only_name_conditions_the_screen_runs(tmp_path):
    spec = json.loads(json.dumps(SPEC))
    spec["rule"]["compare"].append("ppo_full_mimic")
    with pytest.raises(SystemExit, match="ppo_full_mimic"):
        rep.check_rule(spec)


def test_failing_gate_and_leave_one_out_verdicts(tmp_path, monkeypatch, capsys):
    spec, group, jobs = _screen(tmp_path)
    # ppo_full worse than the references by 0.1; removing attention helps it by 0.05
    for (cond, seed) in [(j["cond"], j["seed"]) for j in jobs]:
        acc = {"ppo_full": 0.4, "ppo_full_mlp": 0.45}.get(cond, 0.5)
        base = _csv(jobs, cond, seed)
        lines = open(base + ".csv").read().splitlines()
        hdr = lines[0].split(",")
        rows = [ln.split(",") for ln in lines[1:]]
        for r in rows:
            for k, v in (("accuracy", acc), ("accuracy_fixed", acc), ("acc_client_worst10", acc / 2)):
                r[hdr.index(k)] = str(v)
        open(base + ".csv", "w").write("\n".join([lines[0]] + [",".join(r) for r in rows]) + "\n")
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 0
    assert "- ppo_full: starvation pass, equal spend FAIL, accuracy FAIL -> **fail**" in out
    assert "Carried forward to pre-training: **ppo_full** (none passed" in out
    assert "| attention set encoder | ppo_full_mlp | +0.050 | +0.025 | 0/2 | drop candidate |" in out


def test_screen_b_spec_is_unbundled_and_runnable():
    with open(os.path.join(ROOT, "configs", "rl2_screen_b.yaml")) as fh:
        spec = yaml.safe_load(fh)
    rep.check_rule(spec)
    rule = spec["rule"]
    full = CONDITIONS[rule["full"]]
    seen = []
    for cond in rule["ablations"]:
        c = CONDITIONS[cond]
        changed = {k for k in set(c) | set(full) if c.get(k) != full.get(k)}
        # one component each (predictor v1 also drops what only v2 supports)
        assert changed and len(changed) <= 4, (cond, changed)
        assert changed not in seen, cond
        seen.append(changed)
    assert spec["set"] == {"data.split": "persistent"}
    assert {"rl_light_pg", "ppo_full_rawfail", "ppo_full_arl3"} <= set(spec["conditions"])
    import seeds as seedreg
    seedreg.check(spec["seeds"], spec["purpose"])


def test_a_runner_override_cannot_replace_a_spec_setting(tmp_path, monkeypatch, capsys):
    # third review, finding 2: spec data.split=persistent, manifest override
    # data.split=per_seed, configs saved with per_seed -> must not validate
    spec, group, _ = _screen(tmp_path, overrides=("fl.device=auto", "data.split=per_seed"))
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 2 and "overrides that change the experiment: ['data.split=per_seed']" in out
    assert "rl_light seed 215: different config" in out and "Carried forward" not in out


@pytest.mark.parametrize("what", ["spec", "plan"])
def test_spec_and_plan_must_be_the_recorded_ones(tmp_path, monkeypatch, capsys, what):
    spec, group, _ = _screen(tmp_path)
    if what == "spec":          # the rule edited after the runs
        d = yaml.safe_load(open(spec))
        d["rule"]["margin"] = 0.5
        open(spec, "w").write(yaml.safe_dump(d))
        why = "the spec"
    else:
        path = os.path.join(os.path.dirname(group), "plan.yaml")
        open(path, "a").write("# edited\n")
        why = "the generated plan"
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 2 and why in out and "does not match the hash" in out


def test_where_the_files_are_recorded_does_not_matter(tmp_path, monkeypatch, capsys):
    # screen B on the server: the runner saved "out" relative to its working
    # folder, the report rebuilt it absolute -> every run "different config"
    spec, group, jobs = _screen(tmp_path)
    for j in jobs:
        base = os.path.join(j["raw"]["out"], j["raw"]["tag"])
        meta = json.load(open(base + ".config.json"))
        meta["out"] = os.path.join("runs", "screen", "full", "s2_femnist")
        json.dump(meta, open(base + ".config.json", "w"))
    code, out = _report(monkeypatch, capsys, group, spec)
    assert code == 0 and "Evidence complete" in out
