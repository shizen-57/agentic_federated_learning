#!/usr/bin/env python3
"""RL2 M8, screen C (docs/RL2_SCREEN_C.md).

    python scripts/screen_c_rl2.py plan                                   # what would run
    python scripts/screen_c_rl2.py pretrain --set fl.device=auto          # the checkpoints
    python scripts/screen_c_rl2.py run --out runs/rl2_screen_c --jobs 6 --set fl.device=auto
    python scripts/screen_c_rl2.py report --out runs/rl2_screen_c         # re-check, decide again

Methods are M7's tuned settings (configs/rl2_tuned.json) with the spec's
method settings on top; a method equal to an earlier one is run once (an
alias). Each pre-trained method has its own checkpoint per dataset, made on
practice seeds only (scripts/pretrain.py); a checkpoint is accepted only if
it was made by this code, from practice episodes, for exactly that method,
condition and data group. `run` checks every run and checkpoint before the
rule is applied; anything missing or invalid prints INCOMPLETE and stops.
Writes <out>/DECISION.json and <out>/report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import yaml  # noqa: E402

import rl2lib  # noqa: E402

PRIMARY = (("P1", "eq5"), ("P2", "w10"))


def load(spec_path: str) -> tuple[dict, dict]:
    with open(spec_path) as fh:
        spec = yaml.safe_load(fh)
    with open(os.path.join(ROOT, spec["tuned"])) as fh:
        tuned = json.load(fh)
    if tuned.get("base") != spec["base"]:
        raise SystemExit(f"{spec['tuned']} tunes {tuned.get('base')}, the spec's base is {spec['base']}")
    return spec, tuned


def entry_for(spec: dict, dataset: str, conditions, seeds, override=None) -> dict:
    src = rl2lib.source_group(spec["source_plan"], spec["datasets"][dataset])
    return rl2lib.group_entry(src, spec.get("set"), spec["rounds"], conditions, seeds, override)


def effective(spec: dict, dataset: str, overrides: dict) -> dict:
    """The full config a run of the base condition with these settings gets
    (seed, output folder and tag aside): two methods are the same method
    when these are equal, however their settings are written."""
    from sweep import build_jobs, expected_config
    e = entry_for(spec, dataset, [spec["base"]], [0], overrides)
    raw = build_jobs({"seeds": [0], "sets": {}, "groups": {"g": e}}, None, None, "/x", [])[0]["raw"]
    cfg = expected_config(raw)
    for k in ("out", "tag"):
        cfg.pop(k, None)
    return cfg


def methods(spec: dict, tuned: dict, dataset: str) -> dict[str, dict]:
    """{method: {"overrides": resolved settings, "alias_of": earlier method
    with the same effective config, or None}}."""
    entry = entry_for(spec, dataset, [spec["base"]], [])
    out: dict[str, dict] = {}
    eff: dict[str, dict] = {}
    for m, extra in spec["methods"].items():
        ov = rl2lib.resolve(entry, {**tuned["overrides"], **(extra or {})})
        eff[m] = effective(spec, dataset, ov)
        alias = next((k for k, v in out.items() if v["alias_of"] is None and eff[k] == eff[m]), None)
        out[m] = {"overrides": ov, "alias_of": alias}
    return out


def canonical(ms: dict, m: str) -> str:
    return ms[m]["alias_of"] or m


def ckpt(spec: dict, method: str, dataset: str) -> str:
    """Relative to the repository (no home folder ends up in a saved config)."""
    return os.path.join(spec["pretrain"]["checkpoints"], f"{method}_{dataset}.pt")


def build_plan(spec: dict, tuned: dict) -> dict:
    groups = {}
    for d in spec["datasets"]:
        ms = methods(spec, tuned, d)
        for m in dict.fromkeys(canonical(ms, m) for m in spec["pretrained"]):
            groups[f"{d}_{m}_pt"] = entry_for(spec, d, [spec["base"]], spec["seeds"],
                                              {**ms[m]["overrides"], "policy.init_checkpoint": ckpt(spec, m, d)})
        for m in dict.fromkeys(canonical(ms, m) for m in spec["scratch"]):
            groups[f"{d}_{m}_scratch"] = entry_for(spec, d, [spec["base"]], spec["seeds"], ms[m]["overrides"])
        groups[f"{d}_ref"] = entry_for(spec, d, spec["references"], spec["seeds"])
    return {"seeds": list(spec["seeds"]), "sets": {}, "groups": groups}


def checkpoints(spec: dict, tuned: dict) -> dict[str, tuple[str, str, dict]]:
    """{"<dataset>_<method>": (path, data group, method settings)} for every
    canonical pre-trained method."""
    out = {}
    for d, group in spec["datasets"].items():
        ms = methods(spec, tuned, d)
        for m in dict.fromkeys(canonical(ms, m) for m in spec["pretrained"]):
            out[f"{d}_{m}"] = (ckpt(spec, m, d), group, ms[m]["overrides"])
    return out


def checkpoint_problem(path: str, group: str, base: str, overrides: dict) -> str | None:
    """Why a checkpoint cannot be used for this method, or None."""
    import torch
    from afl.provenance import code_hash
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return "missing"
    meta = torch.load(full, map_location="cpu", weights_only=False).get("meta", {})
    for key, want in (("code_hash", code_hash()), ("practice_only", True), ("group", group),
                      ("condition", base), ("overrides", overrides)):
        if meta.get(key) != want:
            return f"made with a different {key}"
    return None


def cmd_pretrain(spec: dict, tuned: dict, args) -> int:
    return pretrain_methods(spec, tuned, args.max_procs, args.set)


def pretrain_methods(spec: dict, tuned: dict, max_procs: int, sets: list[str]) -> int:
    """Make every checkpoint the spec's pre-trained methods need (a valid one
    is kept); also used by the confirmation, under the frozen code."""
    pt = spec["pretrain"]
    todo = []
    for name, (path, group, ov) in checkpoints(spec, tuned).items():
        why = checkpoint_problem(path, group, spec["base"], ov)
        print(f"{name:28s} {path}: {'ok, kept' if why is None else why}")
        if why is not None:
            todo.append((name, path, group, ov))
    for name, path, group, ov in todo:
        d = name.split("_", 1)[0]
        os.makedirs(os.path.join(ROOT, os.path.dirname(path)), exist_ok=True)
        ov_path = os.path.join(ROOT, os.path.splitext(path)[0] + ".overrides.json")
        with open(ov_path, "w") as fh:
            json.dump(ov, fh, indent=1, sort_keys=True)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "pretrain.py"), "--group", group,
               "--condition", spec["base"], "--overrides", ov_path,
               "--iterations", str(pt["iterations"]), "--workers", str(pt["workers"]),
               "--procs", str(min((pt.get("procs") or {}).get(d, pt["workers"]), max_procs or 99)),
               "--rounds", str(pt["rounds"]), "--seed-start", str(pt["seed_start"][d]),
               "--mode", pt["mode"], "--out", path] + [x for s in sets for x in ("--set", s)]
        print(f"\n== pre-training {name}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc:
            print(f"pre-training {name} FAILED (exit {rc})")
            return rc
        if checkpoint_problem(path, group, spec["base"], ov):
            print(f"{name}: the new checkpoint does not check out")
            return 1
    print("all checkpoints present and valid")
    return 0


def checkpoint_inputs(spec: dict, tuned: dict) -> dict[str, str]:
    return {f"checkpoint {name}": os.path.join(ROOT, path) for name, (path, _, _) in checkpoints(spec, tuned).items()}


def evidence(spec: dict, tuned: dict, out: str, plan: dict, inputs: dict) -> tuple[list[str], dict]:
    """Every run (rl2lib.check_evidence, with `inputs` and every checkpoint
    by hash) and every checkpoint (made by this code, for exactly its
    method), and every pre-trained run started from its checkpoint."""
    inputs = {**inputs, **checkpoint_inputs(spec, tuned)}
    problems, jobs, valid = rl2lib.check_evidence(out, plan, inputs)
    for name, (path, group, ov) in checkpoints(spec, tuned).items():
        why = checkpoint_problem(path, group, spec["base"], ov)
        if why:
            problems.append(f"checkpoint {name} ({path}): {why}")
    sha = {name: rl2lib._sha(os.path.join(ROOT, path)) for name, (path, _, _) in checkpoints(spec, tuned).items()}
    ok = set(valid)
    for j in jobs:
        base = os.path.join(j["raw"]["out"], j["raw"]["tag"])
        if base + ".csv" in ok and j["group"].endswith("_pt"):
            with open(base + ".config.json") as fh:
                used = json.load(fh).get("init_checkpoint_sha256")
            if used != sha[j["group"][:-3]]:
                problems.append(f"{j['group']}/{j['cond']} seed {j['seed']}: started from another checkpoint")
    if problems:
        return problems, {}
    return [], rl2lib.measures_by_group([j for j in jobs if os.path.join(
        j["raw"]["out"], j["raw"]["tag"] + ".csv") in ok])


def by_condition(spec: dict, tuned: dict, mg: dict) -> dict[str, dict[str, dict[int, dict]]]:
    """{dataset: {name: {seed: measures}}}: "<method>_pt" (aliases too),
    "<method>_scratch", and every reference by its condition name."""
    out: dict = {}
    for d in spec["datasets"]:
        ms = methods(spec, tuned, d)
        cur = out.setdefault(d, {})
        for m in spec["pretrained"]:
            cur[f"{m}_pt"] = mg[f"{d}_{canonical(ms, m)}_pt"][spec["base"]]
        for m in spec["scratch"]:
            cur[f"{m}_scratch"] = mg[f"{d}_{canonical(ms, m)}_scratch"][spec["base"]]
        cur.update(mg[f"{d}_ref"])
    return out


def paired(a: dict, b: dict, k: str) -> list[float]:
    return [a[s][k] - b[s][k] for s in sorted(a) if s in b]


def decide(spec: dict, tuned: dict, runs: dict, ck: dict) -> tuple[dict, list[str]]:
    import torch
    sc = lambda a, b: [rl2lib.score(a[s]) - rl2lib.score(b[s]) for s in sorted(a) if s in b]  # noqa: E731
    pt_vs = [x for d in runs for x in sc(runs[d]["tuned_pt"], runs[d]["tuned_scratch"])]
    final = "tuned_pt" if st.median(pt_vs) >= 0 else "tuned_scratch"
    cands = [r for r in spec["references"] if r != "rl_light"]
    eq = {r: st.mean(st.median(m["eq5"] for m in runs[d][r].values()) for d in runs) for r in cands}
    best = max(cands, key=lambda r: (eq[r], -cands.index(r)))
    tests = []
    for d in runs:
        for fam, k in PRIMARY:
            for vs in ("rl_light", best if fam == "P1" else spec["p2_reference"]):
                diff = paired(runs[d][final], runs[d][vs], k)
                tests.append({"family": fam, "dataset": d, "vs": vs, "endpoint": k, "n": len(diff),
                              "mean_diff": st.mean(diff), "sd": st.stdev(diff) if len(diff) > 1 else 0.0,
                              "wins": sum(x > 0 for x in diff)})
    cost = {}
    for name, (path, _, _) in ck.items():
        meta = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)["meta"]
        cost[name] = meta.get("practice_totals")
    from afl.provenance import code_hash
    import data_roles
    decision = {"final": {"name": final, "method": "tuned", "pretrained": final == "tuned_pt"},
                "pt_minus_scratch_median_score": st.median(pt_vs), "best_baseline": best,
                "best_baseline_eq5": eq, "p2_reference": spec["p2_reference"], "tests": tests,
                "methods": {d: methods(spec, tuned, d) for d in spec["datasets"]},
                "checkpoints": {n: {"path": p, "sha256": rl2lib._sha(os.path.join(ROOT, p))}
                                for n, (p, _, _) in ck.items()},
                "pretrain_cost": cost, "code_hash": code_hash(),
                # measured on the development pool of these recorded data roles
                # (the freeze accepts no screen C run before the pools existed)
                "eval_pool": "dev", "data_roles_sha256": rl2lib._sha(data_roles.PATH),
                "starved_final_seeds": {d: sum(m["starved"] for m in runs[d][final].values()) for d in runs}}
    lines = [f"# {spec['title']}", "", "Evidence complete: every run and checkpoint checked.", "",
             f"**Final method: {final}** (median paired score, pre-trained minus scratch, over "
             f"{len(pt_vs)} dataset-seed pairs: {st.median(pt_vs):+.4f}).",
             f"**Best baseline for P1: {best}** (mean over datasets of median equal-spend acc).", "",
             "## Per dataset, median over seeds", ""]
    for d in runs:
        lines += [f"### {d}", "", "| condition | eq5 | w10 | acc10 | score | starved |", "|---|---|---|---|---|---|"]
        for c, r in sorted(runs[d].items()):
            v = r.values()
            lines.append(f"| {c} | {st.median(m['eq5'] for m in v):.3f} | {st.median(m['w10'] for m in v):.3f} | "
                         f"{st.median(m['acc10'] for m in v):.3f} | {st.median(rl2lib.score(m) for m in v):.4f} | "
                         f"{sum(m['starved'] for m in v)}/{len(r)} |")
        lines.append("")
    lines += ["## The confirmation's primary tests, as estimated here (no test is run here)", "",
              "| family | dataset | final vs | endpoint | mean paired diff | sd | wins |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {t['family']} | {t['dataset']} | {t['vs']} | {t['endpoint']} | {t['mean_diff']:+.4f} | "
              f"{t['sd']:.4f} | {t['wins']}/{t['n']} |" for t in tests]
    lines += ["", "## Secondary (descriptive): final method minus each condition, median paired", "",
              "| dataset | condition | eq5 | w10 | acc10 |", "|---|---|---|---|---|"]
    for d in runs:
        for c in sorted(runs[d]):
            if c != final:
                lines.append(f"| {d} | {c} | {st.median(paired(runs[d][final], runs[d][c], 'eq5')):+.3f} | "
                             f"{st.median(paired(runs[d][final], runs[d][c], 'w10')):+.3f} | "
                             f"{st.median(paired(runs[d][final], runs[d][c], 'acc10')):+.3f} |")
    lines += ["", "ARL contrast (with ARL-v4 minus without, pre-trained, median paired):", ""]
    for d in runs:
        a, b = runs[d]["tuned_arl4_pt"], runs[d]["tuned_noarl_pt"]
        lines.append(f"- {d}: eq5 {st.median(paired(a, b, 'eq5')):+.3f}, w10 {st.median(paired(a, b, 'w10')):+.3f}")
    lines += ["", "Pre-training cost per checkpoint (practice only, never evaluated):", ""]
    lines += [f"- {n}: {c}" for n, c in cost.items()]
    return decision, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("plan", "pretrain", "run", "report"))
    ap.add_argument("--spec", default=os.path.join(ROOT, "configs", "rl2_screen_c.yaml"))
    ap.add_argument("--out", default=os.path.join("runs", "rl2_screen_c"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="operational override, e.g. fl.device=auto")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--max-procs", type=int, default=0,
                    help="pretrain: at most this many episodes at once (GPU memory; same result)")
    args = ap.parse_args()
    if rl2lib.non_operational(args.set):
        raise SystemExit(f"--set may only be operational; put {rl2lib.non_operational(args.set)} in the spec")
    spec, tuned = load(args.spec)
    import seeds as seedreg
    seedreg.check(spec["seeds"], spec.get("purpose", "dev"))
    plan = build_plan(spec, tuned)
    rl2lib.check_pool(plan, spec.get("purpose", "dev"))
    n = sum(len(g["conditions"]) * len(g["seeds"]) for g in plan["groups"].values())
    if args.command == "plan":
        for g, e in plan["groups"].items():
            print(f"  {g:28s} {', '.join(e['conditions'])}  override: {e.get('override', {})}")
        print(f"{len(plan['groups'])} groups, {n} runs")
        return 0
    if args.command == "pretrain":
        return cmd_pretrain(spec, tuned, args)
    ck = checkpoints(spec, tuned)
    inputs = {"spec": args.spec, "source": os.path.join(ROOT, spec["source_plan"]),
              "tuned": os.path.join(ROOT, spec["tuned"])}
    if args.command == "run":
        bad = {k: checkpoint_problem(p, g, spec["base"], ov) for k, (p, g, ov) in ck.items()}
        if any(bad.values()):
            raise SystemExit(f"checkpoints not ready: {bad}; run the pretrain step first")
        cwd = os.getcwd()
        os.chdir(ROOT)                      # checkpoints are found relative to the repository
        try:
            if not args.no_smoke and rl2lib.smoke(plan, args.set):
                return 1
        finally:
            os.chdir(cwd)
        rl2lib.write_plan(plan, args.out)
        rl2lib.write_manifest(args.out, {**inputs, **checkpoint_inputs(spec, tuned)},
                              ["fl.rounds", "diag.enabled", "out", *sorted(spec.get("set") or {}), *args.set],
                              title=spec.get("title"))
        rl2lib.run_sweep(os.path.join(args.out, "plan.yaml"), args.out, args.jobs, args.set)
    problems, mg = evidence(spec, tuned, args.out, plan, inputs)
    if problems:
        print("\n**INCOMPLETE -- no decision.**\n\n" + "\n".join(f"- {p}" for p in problems))
        return 2
    decision, lines = decide(spec, tuned, by_condition(spec, tuned, mg), ck)
    decision.update(spec=os.path.relpath(args.spec, ROOT), spec_sha256=rl2lib._sha(args.spec),
                    tuned=spec["tuned"], tuned_sha256=rl2lib._sha(os.path.join(ROOT, spec["tuned"])))
    with open(os.path.join(args.out, "DECISION.json"), "w") as fh:
        json.dump(decision, fh, indent=1, sort_keys=True)
    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
