#!/usr/bin/env python3
"""Controlled development screens on the tuned method (fourth external
review, findings 3 and 4; gap review A2-A4, B2, B5): one thing changes,
everything else is fixed.

    python scripts/screen_rl2.py plan --spec configs/rl2_arl_screen.yaml
    python scripts/screen_rl2.py run  --spec configs/rl2_arl_screen.yaml --out runs/rl2_arl --jobs 3 --set fl.device=auto
    python scripts/screen_rl2.py report --spec configs/rl2_arl_screen.yaml --out runs/rl2_arl

A spec names datasets, seeds, `arms` and optionally `settings` (named changes
of the shared world, for sensitivity). An arm is the tuned settings (the
spec's `tuned` file) with an override on top (`tuned: true`), another
method's tuned settings (`tuned_file: ...`, e.g. the tuned control), an M7
candidate by id (`candidate: c17`, from `candidates_from`), or a plain
condition (`condition: rl_light`). Every arm runs in every dataset x
setting. Evidence first (rl2lib.check_evidence: every run, one code version,
clean checkout, recorded hashes); then the spec's rule. The endpoints are
both at equal spend: eq5 and w10b (worst-10% client accuracy over the last
10 rounds within the same budget; gap review A2).

- kind `arl`: per dataset, each ARL arm minus the plain arm, paired by seed,
  on eq5 and w10b (two-sided sign-flip, Holm over every contrast x endpoint x
  dataset), with a 95% bootstrap interval. An ARL version "adds value" only
  if it is Holm-significantly better than plain on some endpoint and dataset
  and significantly worse on none.
- kind `sensitivity`: per setting, the tuned arm minus the reference arm,
  paired by seed, on eq5 and w10b. The conclusion "tuned beats the
  reference" is robust to a factor if the mean difference stays positive at
  every level of it; every reversal is reported.
- kind `cost`: does the soft cost constraint meet its target, and does a
  tighter variant do better? An arm meets the target if its median spend per
  round in the second half is at most `slack` x the target. Among the
  variants that meet it, the one with the highest mean equal-spend score is
  adopted if it loses at most `max_loss` of that score against the base
  (paired mean); otherwise the base is kept.
- kind `transfer`: do M7's settings, tuned on FEMNIST, hold on another
  dataset? The selected candidate's score against the untuned base and the
  other top candidates, paired by seed.

`arl` and `cost` also write <out>/VERDICT.json, which
scripts/finalize_tuned.py turns into the tuned settings the next steps use.
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

ENDPOINTS = ("eq5", "w10b")


def load(path: str) -> tuple[dict, dict]:
    with open(path) as fh:
        spec = yaml.safe_load(fh)
    tuned_path = os.path.join(ROOT, spec["tuned"])
    if not os.path.exists(tuned_path):
        raise SystemExit(f"{spec['tuned']} does not exist yet: make it first with scripts/finalize_tuned.py "
                         "(docs/RL2_ARL_SCREEN.md, 'Order')")
    with open(tuned_path) as fh:
        tuned = json.load(fh)
    return spec, tuned


def gname(dataset: str, setting: str | None, arm: str) -> str:
    return "_".join(x for x in (dataset, setting, arm) if x)


def arm_entry(spec: dict, tuned: dict, src: dict, shared: dict, a: dict) -> dict:
    if a.get("tuned_file"):                 # another method's tuned settings (e.g. the tuned control)
        path = os.path.join(ROOT, a["tuned_file"])
        if not os.path.exists(path):
            raise SystemExit(f"{a['tuned_file']} does not exist yet: run its matched tuning first "
                             "(docs/RL2_BASELINE_TUNING.md)")
        with open(path) as fh:
            t = json.load(fh)
        e = rl2lib.group_entry(src, shared, spec["rounds"], [t["base"]], spec["seeds"])
        if t["overrides"]:
            e["override"] = rl2lib.resolve(e, t["overrides"])
        return e
    if a.get("tuned") or a.get("candidate"):
        e = rl2lib.group_entry(src, shared, spec["rounds"], [tuned["base"]], spec["seeds"])
        if a.get("candidate"):
            with open(os.path.join(ROOT, spec["candidates_from"])) as fh:
                ov = json.load(fh)[a["candidate"]]
        else:
            ov = {**tuned["overrides"], **(a.get("override") or {})}
        if ov:
            e["override"] = rl2lib.resolve(e, ov)
        return e
    return rl2lib.group_entry(src, shared, spec["rounds"], [a["condition"]], spec["seeds"])


def build_plan(spec: dict, tuned: dict) -> dict:
    settings = spec.get("settings") or {None: {}}
    groups = {}
    for d, g in spec["datasets"].items():
        src = rl2lib.source_group(spec["source_plan"], g)
        for sname, sset in settings.items():
            shared = {**(spec.get("set") or {}), **(sset or {})}
            for arm, a in spec["arms"].items():
                groups[gname(d, sname, arm)] = arm_entry(spec, tuned, src, shared, a)
    return {"seeds": list(spec["seeds"]), "sets": {}, "groups": groups}


def runs(spec: dict, mg: dict) -> dict:
    """{(dataset, setting, arm): {seed: measures}}."""
    settings = spec.get("settings") or {None: {}}
    out = {}
    for d in spec["datasets"]:
        for s in settings:
            for arm in spec["arms"]:
                (cond_runs,) = mg[gname(d, s, arm)].values()
                out[(d, s, arm)] = cond_runs
    return out


def paired(a: dict, b: dict, k) -> list[float]:
    get = (lambda m: m[k]) if isinstance(k, str) else k
    return [get(a[s]) - get(b[s]) for s in sorted(a) if s in b]


def bootstrap_ci(d: list[float], n: int = 10000, seed: int = 20261028) -> tuple[float, float]:
    import numpy as np
    x = np.asarray(d)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rule_arl(spec: dict, R: dict) -> tuple[list[str], dict]:
    rule, plain = spec["rule"], spec["rule"]["plain"]
    rows = []
    for d in spec["datasets"]:
        for arm in rule["contrasts"]:
            for k in ENDPOINTS:
                diff = paired(R[(d, None, arm)], R[(d, None, plain)], k)
                rows.append({"dataset": d, "arm": arm, "endpoint": k, "n": len(diff), "mean": st.mean(diff),
                             "ci": bootstrap_ci(diff), "wins": sum(x > 0 for x in diff),
                             "p": rl2lib.signflip_p(diff, "two-sided")})
    for r, adj in zip(rows, rl2lib.holm([r["p"] for r in rows])):
        r["p_holm"] = adj
    L = ["## ARL vs the plain failure count (all else fixed)", "",
         f"Two-sided paired sign-flip, Holm over {len(rows)} tests; 95% bootstrap interval of the mean difference.", "",
         "| dataset | arm - plain | endpoint | n | mean diff | 95% interval | wins | p | Holm p |",
         "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['dataset']} | {r['arm']} | {r['endpoint']} | {r['n']} | {r['mean']:+.4f} | "
          f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['wins']}/{r['n']} | {r['p']:.4f} | {r['p_holm']:.4f} |"
          for r in rows]
    L += ["", "Verdict (rule fixed in the spec before the runs):", ""]
    verdict = {}
    for arm in rule["contrasts"]:
        mine = [r for r in rows if r["arm"] == arm]
        better = [r for r in mine if r["p_holm"] < rule["alpha"] and r["mean"] > 0]
        worse = [r for r in mine if r["p_holm"] < rule["alpha"] and r["mean"] < 0]
        v = ("adds value" if better and not worse else
             "makes it worse" if worse and not better else
             "mixed (better somewhere, worse elsewhere)" if better and worse else
             "no evidence that it adds value")
        gain = st.mean([x for d in spec["datasets"]
                        for x in paired(R[(d, None, arm)], R[(d, None, plain)], rl2lib.score_eq)])
        verdict[arm] = {"verdict": v, "override": spec["arms"][arm].get("override") or {},
                        "mean_score_gain": gain}
        L.append(f"- **{arm}: {v}**" + (f" (better: {', '.join(r['dataset'] + ' ' + r['endpoint'] for r in better)})" if better else "")
                 + (f" (worse: {', '.join(r['dataset'] + ' ' + r['endpoint'] for r in worse)})" if worse else ""))
    return L, {"kind": "arl", "plain": plain, "arms": verdict, "tests": rows}


def rule_sensitivity(spec: dict, R: dict) -> tuple[list[str], dict]:
    rule = spec["rule"]
    a, b = rule["arm"], rule["reference"]
    L = [f"## {a} minus {b}, per setting (paired by seed; unadjusted two-sided p)", "",
         "| dataset | setting | endpoint | n | mean diff | wins | p |", "|---|---|---|---|---|---|---|"]
    sign: dict = {}
    for d in spec["datasets"]:
        for s in spec["settings"]:
            for k in ENDPOINTS:
                diff = paired(R[(d, s, a)], R[(d, s, b)], k)
                m = st.mean(diff)
                sign[(d, s, k)] = m
                L.append(f"| {d} | {s} | {k} | {len(diff)} | {m:+.4f} | {sum(x > 0 for x in diff)}/{len(diff)} | "
                         f"{rl2lib.signflip_p(diff, 'two-sided'):.4f} |")
    L += ["", "Robustness by factor (rule fixed in the spec before the runs):", ""]
    for factor, levels in rule["factors"].items():
        flips = [f"{d} {s} {k} ({sign[(d, s, k)]:+.4f})" for d in spec["datasets"] for s in levels
                 for k in ENDPOINTS if sign[(d, s, k)] <= 0]
        L.append(f"- **{factor}: {'robust' if not flips else 'NOT robust'}**"
                 + (f" -- reversed at {', '.join(flips)}" if flips else " -- the difference stays positive at every level"))
    return L, {"kind": "sensitivity", "mean_diff": {f"{d}|{s}|{k}": v for (d, s, k), v in sign.items()}}


def rule_cost(spec: dict, tuned: dict, R: dict) -> tuple[list[str], dict]:
    """Soft cost constraint: target = cost_target_mult x (K x mean Light cost)."""
    rule = spec["rule"]
    mult = tuned["overrides"].get("policy.cost_target_mult", 0.0)
    base, slack = rule["base"], rule["slack"]
    (d,) = spec["datasets"]
    L = [f"## Spend against the soft target ({mult} x K x mean Light cost), second half of the run", "",
         "| arm | median spend / target | meets target (<= " + f"{slack})" + " | eq5 | w10b | score_eq | "
         "score_eq minus base (paired mean) |", "|---|---|---|---|---|---|---|"]
    rows = {}
    for arm in spec["arms"]:
        r = R[(d, None, arm)]
        ratio = st.median(m["spend_ratio_late"] for m in r.values()) / max(mult, 1e-9)
        loss = st.mean(paired(r, R[(d, None, base)], rl2lib.score_eq)) if arm != base else 0.0
        rows[arm] = {"ratio": ratio, "meets": ratio <= slack,
                     "score_eq": st.mean(rl2lib.score_eq(m) for m in r.values()), "vs_base": loss}
        L.append(f"| {arm} | {ratio:.3f} | {'yes' if ratio <= slack else 'no'} | "
                 f"{st.median(m['eq5'] for m in r.values()):.3f} | {st.median(m['w10b'] for m in r.values()):.3f} | "
                 f"{rows[arm]['score_eq']:.4f} | {loss:+.4f} |")
    ok = [a for a in rule["variants"] if rows[a]["meets"] and rows[a]["vs_base"] >= -rule["max_loss"]]
    pick = max(ok, key=lambda a: (rows[a]["score_eq"], -rule["variants"].index(a))) if ok else None
    if rows[base]["meets"]:
        L += ["", "**Verdict: the base already meets its target; it is kept.**"]
        pick = None
    elif pick:
        L += ["", f"**Verdict: adopt {pick}** (meets the target and loses at most {rule['max_loss']} of the score)."]
    else:
        L += ["", "**Verdict: keep the base.** No variant meets the target without losing more than "
              f"{rule['max_loss']} of the score; the thesis states that the soft constraint does not bind."]
    return L, {"kind": "cost", "base": base, "adopt": pick,
               "override": (spec["arms"][pick].get("override") or {}) if pick else {}, "arms": rows}


def rule_transfer(spec: dict, R: dict) -> tuple[list[str], dict]:
    rule = spec["rule"]
    (d,) = spec["datasets"]
    sel, base = rule["selected"], rule["base_candidate"]
    L = [f"## Do M7's settings (tuned on FEMNIST) hold on {d}?", "",
         "| candidate | M7 score | score_eq | eq5 | w10b |", "|---|---|---|---|---|"]
    mean = {a: st.mean(rl2lib.score(m) for m in R[(d, None, a)].values()) for a in spec["arms"]}
    for a in sorted(spec["arms"], key=lambda a: -mean[a]):
        r = R[(d, None, a)].values()
        L.append(f"| {a}{' (selected)' if a == sel else ''} | {mean[a]:.4f} | "
                 f"{st.mean(rl2lib.score_eq(m) for m in r):.4f} | {st.median(m['eq5'] for m in r):.3f} | "
                 f"{st.median(m['w10b'] for m in r):.3f} |")
    diff = paired(R[(d, None, sel)], R[(d, None, base)], rl2lib.score)
    rank = sorted(mean, key=lambda a: -mean[a]).index(sel) + 1
    holds = st.mean(diff) >= 0
    L += ["", f"Selected minus untuned base, paired: {st.mean(diff):+.4f} ({sum(x > 0 for x in diff)}/{len(diff)} "
          f"seeds); rank {rank} of {len(mean)}.", "",
          f"**Verdict: the tuning {'holds' if holds else 'does NOT hold'} on {d}** (rule: the selected settings are "
          "at least as good as the untuned base). Nothing is re-selected: the baselines are also tuned on FEMNIST "
          "only, so both sides are treated alike."]
    return L, {"kind": "transfer", "holds": holds, "rank": rank, "vs_base": st.mean(diff)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("plan", "run", "report"))
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="operational override, e.g. fl.device=auto")
    ap.add_argument("--no-smoke", action="store_true")
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
            print(f"  {g:36s} {e['conditions'][0]}{' + settings' if e.get('override') else ''}")
        print(f"{len(plan['groups'])} groups, {n} runs")
        return 0
    out = args.out or os.path.join("runs", os.path.splitext(os.path.basename(args.spec))[0])
    inputs = {"spec": args.spec, "source": os.path.join(ROOT, spec["source_plan"]),
              "tuned": os.path.join(ROOT, spec["tuned"])}
    if spec.get("candidates_from"):
        inputs["candidates"] = os.path.join(ROOT, spec["candidates_from"])
    for arm, a in spec["arms"].items():
        if a.get("tuned_file"):
            inputs[f"tuned settings of {arm}"] = os.path.join(ROOT, a["tuned_file"])
    if args.command == "run":
        if not args.no_smoke and rl2lib.smoke(plan, args.set):
            return 1
        rl2lib.write_plan(plan, out)
        rl2lib.write_manifest(out, inputs, ["fl.rounds", "diag.enabled", "out", *sorted(spec.get("set") or {}), *args.set],
                              title=spec.get("title"))
        rl2lib.run_sweep(os.path.join(out, "plan.yaml"), out, args.jobs, args.set)
    problems, jobs, valid = rl2lib.check_evidence(out, plan, inputs)
    if problems:
        print("\n**INCOMPLETE -- no decision.**\n\n" + "\n".join(f"- {p}" for p in problems))
        return 2
    ok = set(valid)
    mg = rl2lib.measures_by_group([j for j in jobs if os.path.join(j["raw"]["out"], j["raw"]["tag"] + ".csv") in ok])
    R = runs(spec, mg)
    L = [f"# {spec['title']}", "", f"Evidence complete: {n} runs, one code version.", "",
         "| dataset | setting | arm | eq5 | w10b | w10 (whole run) | acc10 | spend | starved |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (d, s, arm), r in R.items():
        v = r.values()
        L.append(f"| {d} | {s or '-'} | {arm} | {st.median(m['eq5'] for m in v):.3f} | "
                 f"{st.median(m['w10b'] for m in v):.3f} | {st.median(m['w10'] for m in v):.3f} | "
                 f"{st.median(m['acc10'] for m in v):.3f} | {st.median(m['spend'] for m in v):.0f} | "
                 f"{sum(m['starved'] for m in v)}/{len(r)} |")
    kind = spec["rule"]["kind"]
    body, verdict = {"arl": lambda: rule_arl(spec, R), "sensitivity": lambda: rule_sensitivity(spec, R),
                     "cost": lambda: rule_cost(spec, tuned, R), "transfer": lambda: rule_transfer(spec, R)}[kind]()
    L += [""] + body
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(text)
    verdict.update(spec=os.path.relpath(args.spec, ROOT), spec_sha256=rl2lib._sha(args.spec),
                   tuned=spec["tuned"], tuned_sha256=rl2lib._sha(os.path.join(ROOT, spec["tuned"])),
                   manifest_sha256=rl2lib._sha(os.path.join(out, "MANIFEST.json")))
    with open(os.path.join(out, "VERDICT.json"), "w") as fh:
        json.dump(verdict, fh, indent=1, sort_keys=True, default=float)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
