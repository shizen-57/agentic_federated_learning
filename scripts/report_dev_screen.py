#!/usr/bin/env python3
"""Report a development screen and apply its pre-registered rule -- but only
to complete, valid evidence.

    python scripts/report_dev_screen.py runs/rl2_screen_b/full/s2_femnist \
        --spec configs/rl2_screen_b.yaml

The spec (configs/*.yaml) fixes the conditions, seeds, rounds and the rule.
Before any decision, every run it asks for (each condition x seed) must exist
and pass sweep.run_status: exactly the expected config (rounds and shared
settings included), every round present with finite values, and no source
change while it ran. All runs must come from one code version -- the one in
MANIFEST.json, made from a clean checkout -- and this report must run with
that same simulator code, because it rebuilds the expected configs. Anything
else prints "INCOMPLETE -- no decision" and exits with 2.

Without --spec the tables are printed and no decision is made. A spec with no
`rule` block (the screens before screen B, whose decisions are recorded in
their docs) is validated and tabulated only. Every seed of every expected run
is reported; nothing is dropped.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

STARVE = 50          # rounds with invitations and zero successes, in a row
BUDGET = 1400.0      # simulated cost units, fixed before the runs
NAN = float("nan")


def measures(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k])  # noqa: E731
    streak = longest = 0
    spend = 0.0
    at_budget = None
    for r in rows:
        streak = streak + 1 if f(r, "n_selected") and not f(r, "n_success") else 0
        longest = max(longest, streak)
        if spend + f(r, "cost") <= BUDGET:
            spend += f(r, "cost")
            at_budget = (f(r, "accuracy_fixed"), int(f(r, "round")))
        else:
            spend = float("inf")
    inv = sum(f(r, "n_selected") for r in rows)
    modes = [sum(f(r, f"mode_{m}") for r in rows) for m in ("light", "standard", "priority")]
    return {"succ": int(sum(f(r, "n_success") for r in rows)), "inv": int(inv),
            "fallback": sum(f(r, "gate_filled") for r in rows) / max(inv, 1),
            "longest": longest, "starved": longest >= STARVE,
            "modes": "/".join(f"{100 * m / max(sum(modes), 1):.0f}" for m in modes),
            "acc": f(rows[-1], "accuracy"), "spend": sum(f(r, "cost") for r in rows),
            "worst10": f(rows[-1], "acc_client_worst10") if "acc_client_worst10" in rows[-1] else NAN,
            "acc_budget": at_budget, "eq": at_budget[0] if at_budget else NAN}


def validate(spec: dict, variant_dir: str) -> tuple[list[str], list[dict], list[str]]:
    """(problems, expected jobs, CSV paths of the valid ones). No problems =
    complete, valid evidence."""
    from afl.provenance import code_hash
    from diagnose_collapse import plan_for
    from sweep import build_jobs, run_status
    problems: list[str] = []
    try:
        with open(os.path.join(variant_dir, "MANIFEST.json")) as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        return [f"no readable MANIFEST.json in {variant_dir}"], [], []
    variant = man.get("variant")
    if variant not in spec["variants"]:
        return [f"MANIFEST variant {variant!r} is not one of the spec's {list(spec['variants'])}"], [], []
    if man.get("rounds") != spec["variants"][variant]:
        problems.append(f"MANIFEST rounds {man.get('rounds')} != spec {spec['variants'][variant]}")
    if man.get("git_dirty") is not False:
        problems.append("the runs were made from a checkout with uncommitted changes (or unknown)")
    if man.get("code_hash") != code_hash():
        # the expected configs are rebuilt with this checkout's code
        return problems + [f"this checkout's simulator code ({code_hash()}) is not the code that "
                           f"made the runs ({man.get('code_hash')}, commit {man.get('git_commit')}); "
                           "run the report from that commit"], [], []
    plan, _ = plan_for(spec, variant)
    extra = [o for o in man.get("overrides_vs_source", []) if "=" in o]   # --set given to the runner
    jobs = build_jobs(plan, None, None, variant_dir, extra)
    hashes, valid = set(), []
    for j in jobs:
        status = run_status(j["raw"])
        if status != "ok":
            problems.append(f"{j['cond']} seed {j['seed']}: {status}")
            continue
        base = os.path.join(j["raw"]["out"], j["raw"]["tag"])
        valid.append(base + ".csv")
        with open(base + ".config.json") as fh:
            hashes.add(json.load(fh).get("code_hash"))
    if len(hashes) > 1:
        problems.append(f"runs made by different code versions: {sorted(map(str, hashes))}")
    elif hashes and hashes != {man.get("code_hash")}:
        problems.append(f"runs' code {hashes.pop()} != MANIFEST code {man.get('code_hash')}")
    return problems, jobs, valid


def check_rule(spec: dict) -> None:
    """A spec's rule may only name conditions the spec runs."""
    rule = spec.get("rule")
    if not rule:
        return
    named = set(rule.get("gate", [])) | set(rule.get("compare", [])) | set(rule.get("ablations", {}))
    named |= {rule[k] for k in ("full", "equal_spend_ref", "accuracy_ref") if k in rule}
    missing = sorted(named - set(spec["conditions"]))
    if missing:
        raise SystemExit(f"the spec's rule names conditions it does not run: {missing}")
    if rule.get("kind") != "screen_b":
        raise SystemExit(f"unknown rule kind {rule.get('kind')!r}")


def med(runs, c, k):
    v = [m[k] for m in runs[c].values() if m[k] == m[k]]
    return st.median(v) if v else NAN


def paired(runs, a, b, k):
    """Median over seeds of (a - b), seed by seed."""
    v = [runs[a][s][k] - runs[b][s][k] for s in runs[a] if s in runs[b]]
    v = [x for x in v if x == x]
    return st.median(v) if v else NAN


def screen_b(runs: dict, rule: dict) -> None:
    eq_ref, acc_ref, margin = rule["equal_spend_ref"], rule["accuracy_ref"], rule["margin"]
    full, flag = rule["full"], rule["flag_delta"]
    print(f"\n## Rule (docs/RL2_SCREEN_B.md)\n\nGate: no seed starves; median equal-spend acc >= "
          f"{eq_ref} - {margin}; median final acc >= {acc_ref} - {margin}.\n")
    passing = []
    for c in rule["gate"]:
        c1 = not any(m["starved"] for m in runs[c].values())
        c2 = med(runs, c, "eq") >= med(runs, eq_ref, "eq") - margin
        c3 = med(runs, c, "acc") >= med(runs, acc_ref, "acc") - margin
        passing += [c] if c1 and c2 and c3 else []
        print(f"- {c}: starvation {'pass' if c1 else 'FAIL'}, equal spend {'pass' if c2 else 'FAIL'}, "
              f"accuracy {'pass' if c3 else 'FAIL'} -> **{'PASS' if c in passing else 'fail'}**")
    best = max(passing, key=lambda c: med(runs, c, "eq")) if passing else None
    print(f"\nCarried forward to pre-training: **{best or full}**"
          + ("" if best else " (none passed; the full stack still goes forward, per the rule, "
             "and screen C judges the pre-trained version on new seeds)"))
    print("\n## Paired differences vs " + full + " (median over seeds of condition - full; "
          "reported, no decision)\n")
    print("| condition | equal-spend acc | final acc | worst-10% client acc | starved seeds |")
    print("|---|---|---|---|---|")
    for c in rule["compare"]:
        print(f"| {c} | {paired(runs, c, full, 'eq'):+.3f} | {paired(runs, c, full, 'acc'):+.3f} | "
              f"{paired(runs, c, full, 'worst10'):+.3f} | {sum(m['starved'] for m in runs[c].values())}"
              f"/{len(runs[c])} |")
    print(f"\n## Leave-one-out (each removes one component from {full})\n\nA component is a "
          f"*drop candidate* if removing it starves no seed and raises one endpoint (median paired "
          f"equal-spend acc or worst-10% acc) by >= {flag} while lowering neither by more than "
          f"{flag}. Drop candidates are decided in tuning (M7) on separate dev seeds, not here.\n")
    print("| removed | condition | equal-spend acc | worst-10% client acc | starved | verdict |")
    print("|---|---|---|---|---|---|")
    for c, what in rule["ablations"].items():
        d_eq, d_w = paired(runs, c, full, "eq"), paired(runs, c, full, "worst10")
        starved = sum(m["starved"] for m in runs[c].values())
        drop = (not starved and min(d_eq, d_w) >= -flag and max(d_eq, d_w) >= flag)
        print(f"| {what} | {c} | {d_eq:+.3f} | {d_w:+.3f} | {starved}/{len(runs[c])} | "
              f"{'drop candidate' if drop else 'kept'} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="<out>/<variant>/<group> of a screen")
    ap.add_argument("--spec", help="the screen's spec; without it no decision is made")
    args = ap.parse_args()
    group_dir = os.path.abspath(args.dir)
    spec, problems = None, []
    if args.spec:
        import yaml
        with open(args.spec) as fh:
            spec = yaml.safe_load(fh)
        check_rule(spec)
        if os.path.basename(group_dir) != spec["source_group"]:
            raise SystemExit(f"{args.dir} is not a {spec['source_group']} folder")
        problems, jobs, paths = validate(spec, os.path.dirname(group_dir))
        extra = sorted(set(glob.glob(os.path.join(group_dir, "*_s*.csv")))
                       - {os.path.join(j["raw"]["out"], j["raw"]["tag"] + ".csv") for j in jobs}
                       ) if jobs else []
    else:
        paths, extra = sorted(glob.glob(os.path.join(group_dir, "*_s*.csv"))), []
    runs: dict[str, dict[int, dict]] = {}
    for p in paths:
        cond, seed = os.path.basename(p)[:-4].rsplit("_s", 1)
        runs.setdefault(cond, {})[int(seed)] = measures(p)

    title = spec.get("title", os.path.basename(args.spec)) if spec else os.path.basename(group_dir)
    print(f"# Development screen: {title} (exploratory)\n")
    print(f"Starved = {STARVE}+ rounds in a row with invitations and no success. "
          f"Equal spend: fixed-set accuracy at the last round with cumulative cost <= {BUDGET:.0f}.\n")
    print("| condition | seed | successes/invites | fallback share | longest zero streak | starved "
          "| modes % L/S/P | final acc | worst-10% acc | total spend | acc at equal spend (round) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in sorted(runs):
        for seed, m in sorted(runs[cond].items()):
            ab = m["acc_budget"]
            print(f"| {cond} | {seed} | {m['succ']}/{m['inv']} | {m['fallback']:.2f} | {m['longest']} | "
                  f"{'YES' if m['starved'] else 'no'} | {m['modes']} | {m['acc']:.3f} | "
                  f"{m['worst10']:.3f} | {m['spend']:.0f} | "
                  + (f"{ab[0]:.3f} (r{ab[1]})" if ab else "-") + " |")
    print("\n| condition | median successes | median final acc | median acc at equal spend "
          "| median worst-10% client acc | starved seeds |")
    print("|---|---|---|---|---|---|")
    for cond in sorted(runs):
        v = runs[cond].values()
        print(f"| {cond} | {st.median(m['succ'] for m in v):.0f} | {med(runs, cond, 'acc'):.3f} | "
              f"{med(runs, cond, 'eq'):.3f} | {med(runs, cond, 'worst10'):.3f} | "
              f"{sum(m['starved'] for m in v)}/{len(v)} |")
    if extra:
        print("\nIgnored (not part of the spec): " + ", ".join(os.path.basename(p) for p in extra))

    if spec is None:
        print("\nNo --spec given: the runs were not validated and no decision is made.")
        return 0
    if problems:
        print("\n**INCOMPLETE -- no decision.** The evidence the spec asks for is missing or "
              "invalid:\n")
        for p in problems:
            print(f"- {p}")
        return 2
    print(f"\nEvidence complete: all {len(jobs)} runs present, valid, one code version.")
    rule = spec.get("rule")
    if not rule:
        print("The spec has no `rule` block: no decision here (see the screen's doc).")
        return 0
    screen_b(runs, rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
