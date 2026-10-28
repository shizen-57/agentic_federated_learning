#!/usr/bin/env python3
"""RL2 M7: tune the method carried forward by screen B (docs/RL2_M7_TUNING.md).

    python scripts/tune_rl2.py --spec configs/rl2_tune.yaml --out runs/rl2_tune --jobs 6 --set fl.device=auto
    python scripts/tune_rl2.py --spec configs/rl2_tune.yaml --out runs/rl2_tune --dry-run
    python scripts/tune_rl2.py --spec configs/rl2_tune.yaml --out runs/rl2_tune --report-only

Stage 1: c00 (the base condition unchanged) and n_configs-1 random candidates
from the spec's search space, on stage1_seeds. Stage 2: the top_k of stage 1
(and c00 for comparison) on fresh stage2_seeds; the winner is the best
stage-2 score. Every run of a stage is checked (rl2lib.check_evidence) before
anything is ranked; anything missing or invalid prints INCOMPLETE and stops.
Writes <out>/SELECTED.json and <out>/report.md. Running again resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import yaml  # noqa: E402

import rl2lib  # noqa: E402


def sample_configs(spec: dict) -> list[dict]:
    """c00 = {} (the base), then random draws: one option per dimension, in
    sorted dimension order, from a fixed seed -- the same list every time."""
    import numpy as np
    rng = np.random.default_rng(spec["sampler_seed"])
    dims = sorted(spec["space"])
    configs: list[dict] = [{}]
    while len(configs) < spec["n_configs"]:
        c: dict = {}
        for d in dims:
            opts = spec["space"][d]
            o = opts[int(rng.integers(len(opts)))]
            c.update(o if isinstance(o, dict) else {d: o})
        if c not in configs:
            configs.append(c)
    return configs


def name(k: int) -> str:
    return f"c{k:02d}"


def stage_plan(spec: dict, stage: int, ids: list[int], configs: list[dict]) -> dict:
    src = rl2lib.source_group(spec["source_plan"], spec["source_group"])
    seeds = spec[f"stage{stage}_seeds"]
    groups = {}
    for k in ids:
        entry = rl2lib.group_entry(src, spec.get("set"), spec["rounds"], [spec["base"]], seeds)
        ov = rl2lib.resolve(entry, configs[k])
        if ov:
            entry["override"] = ov
        groups[name(k)] = entry
    if spec.get("references"):
        groups["ref"] = rl2lib.group_entry(src, spec.get("set"), spec["rounds"], spec["references"], seeds)
    return {"seeds": list(seeds), "sets": {}, "groups": groups}


def rank(by_group: dict, base: str) -> list[dict]:
    """Candidates by mean score over seeds; a candidate with any starved seed
    is disqualified (ranked after every qualified one). Ties: lower id first."""
    rows = []
    for g, conds in by_group.items():
        if g == "ref":
            continue
        runs = conds[base]
        rows.append({"id": g, "n": len(runs), "score": st.mean(rl2lib.score(m) for m in runs.values()),
                     "eq5": st.mean(m["eq5"] for m in runs.values()),
                     "w10": st.mean(m["w10"] for m in runs.values()),
                     "acc10": st.mean(m["acc10"] for m in runs.values()),
                     "starved": sum(m["starved"] for m in runs.values())})
    rows.sort(key=lambda r: (r["starved"] > 0, -r["score"], r["id"]))
    return rows


def refs(by_group: dict) -> list[dict]:
    return [{"id": c, "score": st.mean(rl2lib.score(m) for m in runs.values()),
             "eq5": st.mean(m["eq5"] for m in runs.values()), "w10": st.mean(m["w10"] for m in runs.values()),
             "acc10": st.mean(m["acc10"] for m in runs.values()),
             "starved": sum(m["starved"] for m in runs.values()), "n": len(runs)}
            for c, runs in sorted(by_group.get("ref", {}).items())]


def same_ranking(recorded: dict, now: dict, tol: float = 1e-9) -> str | None:
    """None if the recorded ranking is the one recomputed now (same order,
    same top, every value within tol); else what differs."""
    if recorded.get("top") != now["top"]:
        return f"top {recorded.get('top')} vs {now['top']}"
    for key in ("ranking", "references"):
        a, b = recorded.get(key, []), now[key]
        if [r["id"] for r in a] != [r["id"] for r in b]:
            return f"{key} order"
        for x, y in zip(a, b):
            for k, v in y.items():
                if isinstance(v, float) and not abs(x.get(k, float("inf")) - v) <= tol:
                    return f"{key} {y['id']} {k}: {x.get(k)} vs {v}"
                if not isinstance(v, float) and x.get(k) != v:
                    return f"{key} {y['id']} {k}: {x.get(k)} vs {v}"
    return None


def table(rows: list[dict]) -> list[str]:
    out = ["| candidate | seeds | score | equal-spend acc (eq5) | worst-10% (last 10) | acc (last 10) "
           "| starved seeds |", "|---|---|---|---|---|---|---|"]
    out += [f"| {r['id']} | {r['n']} | {r['score']:.4f} | {r['eq5']:.3f} | {r['w10']:.3f} | "
            f"{r['acc10']:.3f} | {r['starved']} |" for r in rows]
    return out


def stage(spec, args, k: int, plan: dict, inputs: dict) -> tuple[list[str], dict]:
    """Run (unless --report-only) and check one stage; (problems, measures)."""
    out = os.path.join(args.out, f"stage{k}")
    if not args.report_only:
        rl2lib.write_plan(plan, out)
        rl2lib.write_manifest(out, inputs, ["fl.rounds", "diag.enabled", "out",
                                            *sorted(spec.get("set") or {}), *args.set],
                              stage=k, title=spec.get("title"))
        rl2lib.run_sweep(os.path.join(out, "plan.yaml"), out, args.jobs, args.set)
    problems, _, valid = rl2lib.check_evidence(out, plan, inputs)
    if problems:
        return problems, {}
    from sweep import build_jobs
    jobs = [j for j in build_jobs(plan, None, None, out, [])
            if os.path.join(j["raw"]["out"], j["raw"]["tag"] + ".csv") in set(valid)]
    return [], rl2lib.measures_by_group(jobs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(ROOT, "configs", "rl2_tune.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="operational override, e.g. fl.device=auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="re-check and re-rank, run nothing")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    if rl2lib.non_operational(args.set):
        raise SystemExit(f"--set may only be operational; put {rl2lib.non_operational(args.set)} in the spec")
    with open(args.spec) as fh:
        spec = yaml.safe_load(fh)
    import seeds as seedreg
    seedreg.check(spec["stage1_seeds"] + spec["stage2_seeds"], spec.get("purpose", "dev"))
    configs = sample_configs(spec)
    plan1 = stage_plan(spec, 1, list(range(len(configs))), configs)
    rl2lib.check_pool(plan1, spec.get("purpose", "dev"))
    n2 = (spec["top_k"] + 1) * len(spec["stage2_seeds"]) + len(spec.get("references", [])) * len(spec["stage2_seeds"])
    n1 = sum(len(g["conditions"]) * len(g["seeds"]) for g in plan1["groups"].values())
    print(f"{spec.get('title')}: stage 1 {n1} runs ({len(configs)} candidates x {len(spec['stage1_seeds'])} seeds"
          f" + references), stage 2 up to {n2} runs; {spec['rounds']} rounds each -> {args.out}")
    if args.dry_run:
        for k, c in enumerate(configs[:3]):
            print(f"  {name(k)}: {c or '(the base, unchanged)'}")
        print(f"  ... {len(configs)} candidates in all")
        return 0

    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, "configs.json")
    if not args.report_only:
        with open(cfg_path, "w") as fh:
            json.dump({name(k): c for k, c in enumerate(configs)}, fh, indent=1, sort_keys=True)
        if not args.no_smoke and rl2lib.smoke(plan1, args.set):
            return 1
    inputs = {"spec": args.spec, "source": os.path.join(ROOT, spec["source_plan"]), "configs": cfg_path}

    problems, m1 = stage(spec, args, 1, plan1, inputs)
    if problems:
        print("\n**INCOMPLETE -- no decision (stage 1).**\n\n" + "\n".join(f"- {p}" for p in problems))
        return 2
    r1 = rank(m1, spec["base"])
    qualified = [r for r in r1 if not r["starved"]]
    top = [int(r["id"][1:]) for r in qualified[:spec["top_k"]]]
    ids2 = sorted(set(top) | {0})
    rank_path = os.path.join(args.out, "stage1", "ranking.json")
    ranking = {"ranking": r1, "references": refs(m1), "top": [name(k) for k in top]}
    if args.report_only and os.path.exists(rank_path):
        # The recorded ranking is an input of stage 2 (by hash): recompute and
        # compare it, never rewrite it. Last-digit float differences between
        # Python versions (3.12 changed float sum) are not a different ranking.
        with open(rank_path) as fh:
            why = same_ranking(json.load(fh), ranking)
        if why:
            print(f"\n**INCOMPLETE -- no decision.** The recorded stage-1 ranking differs: {why}")
            return 2
    else:
        with open(rank_path, "w") as fh:
            json.dump(ranking, fh, indent=1)

    plan2 = stage_plan(spec, 2, ids2, configs)
    problems, m2 = stage(spec, args, 2, plan2, {**inputs, "stage1_ranking": rank_path})
    if problems:
        print("\n**INCOMPLETE -- no decision (stage 2).**\n\n" + "\n".join(f"- {p}" for p in problems))
        return 2
    r2 = rank(m2, spec["base"])
    eligible = [r for r in r2 if int(r["id"][1:]) in top and not r["starved"]]
    winner = eligible[0]["id"] if eligible else name(0)
    k = int(winner[1:])
    src = rl2lib.source_group(spec["source_plan"], spec["source_group"])
    entry = rl2lib.group_entry(src, spec.get("set"), spec["rounds"], [spec["base"]], [])
    from afl.provenance import code_hash
    selected = {"winner": winner, "base": spec["base"], "overrides": configs[k],
                "resolved_" + spec["source_group"]: rl2lib.resolve(entry, configs[k]),
                "stage1": next(r for r in r1 if r["id"] == winner),
                "stage2": next(r for r in r2 if r["id"] == winner),
                "spec": os.path.relpath(args.spec, ROOT), "spec_sha256": rl2lib._sha(args.spec),
                "code_hash": code_hash(), "no_eligible_candidate": not eligible}
    with open(os.path.join(args.out, "SELECTED.json"), "w") as fh:
        json.dump(selected, fh, indent=1, sort_keys=True)

    lines = [f"# {spec.get('title')}", "", "Evidence complete in both stages (every run checked).", "",
             f"## Stage 1 (seeds {spec['stage1_seeds'][0]}-{spec['stage1_seeds'][-1]})", "",
             *table(r1), "", "References:", "", *table(refs(m1)), "",
             f"Top {spec['top_k']} (qualified): {', '.join(name(k) for k in top) or 'none'}", "",
             f"## Stage 2 (seeds {spec['stage2_seeds'][0]}-{spec['stage2_seeds'][-1]})", "",
             *table(r2), "", "References:", "", *table(refs(m2)), "",
             f"**Selected: {winner}**" + (" (no top-k candidate qualified: the base is kept)" if not eligible else ""),
             "", f"Settings: `{json.dumps(configs[k], sort_keys=True) if configs[k] else 'the base, unchanged'}`",
             "", "## Exploratory (no decision): stage-1 mean score by option, c01-c39", ""]
    for d in sorted(spec["space"]):
        opts = spec["space"][d]
        cells = []
        for o in opts:
            ov = o if isinstance(o, dict) else {d: o}
            ks = [j for j in range(1, len(configs)) if all(configs[j].get(a) == b for a, b in ov.items())]
            sc = [next(r["score"] for r in r1 if r["id"] == name(j)) for j in ks]
            label = (", ".join(f"{a.split('.')[-1]}={b}" for a, b in sorted(o.items()))
                     if isinstance(o, dict) else str(o))
            cells.append(f"{label}: {st.mean(sc):.4f} (n={len(sc)})" if sc else f"{label}: -")
        lines.append(f"- {d}: " + "; ".join(cells))
    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
