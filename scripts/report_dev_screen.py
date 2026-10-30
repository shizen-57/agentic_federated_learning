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

STARVE = 50          # rounds in a row with no successful contribution (see measures)
BUDGET = 1400.0      # simulated cost units, fixed before the runs
CHECKPOINTS = (500, 800, 1100, 1400, 1700, 2000)   # common budgets for the cost curves
NAN = float("nan")


def _longest(flags: list[bool | None]) -> int:
    """Longest run of True; None (a round that cannot count) neither extends
    nor breaks a run."""
    run = best = 0
    for x in flags:
        if x is None:
            continue
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def measures(path: str) -> dict:
    """Starvation (review 3, finding 3) is the longest run of rounds with NO
    successful contribution -- invitations that all failed and rounds with no
    invitation (abstention) alike. A round in which no client was online
    (`n_online` = 0) cannot be the scheduler's fault: it is skipped, neither
    extending nor breaking a run. The two parts are also reported apart."""
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k])  # noqa: E731
    online = [f(r, "n_online") > 0 if "n_online" in r else True for r in rows]
    dry = [None if not on else not f(r, "n_success") for r, on in zip(rows, online)]
    inv_fail = [None if not on else bool(f(r, "n_selected")) and not f(r, "n_success")
                for r, on in zip(rows, online)]
    empty = [None if not on else not f(r, "n_selected") for r, on in zip(rows, online)]
    longest = _longest(dry)
    spend = 0.0
    at_budget = None
    within = []                       # fixed-set accuracy of every round within the budget
    within_w = []                     # worst-10% client accuracy of those rounds
    for r in rows:
        if spend + f(r, "cost") <= BUDGET:
            spend += f(r, "cost")
            at_budget = (f(r, "accuracy_fixed"), int(f(r, "round")))
            within.append(f(r, "accuracy_fixed"))
            if "acc_client_worst10" in r:
                within_w.append(f(r, "acc_client_worst10"))
        else:
            spend = float("inf")
    # accuracy and actual spend at the last round within each common budget
    # (gap review A5: the cost curves next to the endpoints)
    cum, curve = 0.0, {}
    for r in rows:
        cum += f(r, "cost")
        for b in CHECKPOINTS:
            if cum <= b:
                curve[b] = (f(r, "accuracy_fixed"), cum)
    last = rows[-1] if rows else {}
    has = lambda k: k in last and last[k] not in ("", None)  # noqa: E731
    tail = lambda k, n: (sum(f(r, k) for r in rows[-n:]) / len(rows[-n:])  # noqa: E731
                         if rows and k in rows[-1] else NAN)
    inv = sum(f(r, "n_selected") for r in rows)
    modes = [sum(f(r, f"mode_{m}") for r in rows) for m in ("light", "standard", "priority")]
    return {"succ": int(sum(f(r, "n_success") for r in rows)), "inv": int(inv),
            "fallback": sum(f(r, "gate_filled") for r in rows) / max(inv, 1),
            "longest": longest, "starved": longest >= STARVE,
            "longest_fail": _longest(inv_fail), "longest_empty": _longest(empty),
            "empty": int(sum(1 for x in empty if x)), "offline": int(sum(1 for on in online if not on)),
            "modes": "/".join(f"{100 * m / max(sum(modes), 1):.0f}" for m in modes),
            "acc": f(rows[-1], "accuracy"), "spend": sum(f(r, "cost") for r in rows),
            "worst10": f(rows[-1], "acc_client_worst10") if "acc_client_worst10" in rows[-1] else NAN,
            "acc_budget": at_budget, "eq": at_budget[0] if at_budget else NAN,
            # steadier versions (from M7 on; screen B showed one round can
            # decide a last-round value): equal spend = mean of the last 5
            # rounds within the budget; final = mean of the last 10 rounds
            "eq5": sum(within[-5:]) / len(within[-5:]) if within else NAN,
            "acc10": tail("accuracy", 10), "w10": tail("acc_client_worst10", 10),
            # P2 at equal spend (gap review A2): worst-10% client accuracy over
            # the last 10 rounds within the same budget as eq5
            "w10b": sum(within_w[-10:]) / len(within_w[-10:]) if within_w else NAN,
            "spend_eq5": None if not within else spend_at(rows, len(within)),
            "curve": curve,
            # participation fairness and projections (gap review B6, B10), at the end
            "jain": f(last, "jain") if has("jain") else NAN,
            "cv": f(last, "cv") if has("cv") else NAN,
            "n_never": f(last, "n_never") if has("n_never") else NAN,
            "n_gap50": f(last, "n_gap50") if has("n_gap50") else NAN,
            "acc_weak10": tail("acc_client_weak", 10),
            "gap_max_weak": f(last, "gap_max_weak") if has("gap_max_weak") else NAN,
            # spend per round in the second half, relative to K x mean Light cost
            "spend_ratio_late": (sum(f(r, "cost") for r in rows[len(rows) // 2:])
                                 / max(len(rows) - len(rows) // 2, 1) / f(last, "cost_ref")
                                 if has("cost_ref") and f(last, "cost_ref") > 0 else NAN),
            "budget_dropped": sum(f(r, "budget_dropped") for r in rows) if has("budget_dropped") else NAN,
            "budget_lighter": sum(f(r, "budget_lighter") for r in rows) if has("budget_lighter") else NAN}


def spend_at(rows: list[dict], k: int) -> float:
    """Mean cumulative spend over the last 5 of the first k rounds (eq5's rounds)."""
    cum, out = 0.0, []
    for r in rows[:k]:
        cum += float(r["cost"])
        out.append(cum)
    return sum(out[-5:]) / len(out[-5:])


def validate(spec: dict, variant_dir: str, spec_path: str) -> tuple[list[str], list[dict], list[str]]:
    """(problems, expected jobs, CSV paths of the valid ones). No problems =
    complete, valid evidence (the check itself: rl2lib.check_evidence)."""
    import rl2lib
    from diagnose_collapse import plan_for
    try:
        with open(os.path.join(variant_dir, "MANIFEST.json")) as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        return [f"no readable MANIFEST.json in {variant_dir}"], [], []
    variant = man.get("variant")
    if variant not in spec["variants"]:
        return [f"MANIFEST variant {variant!r} is not one of the spec's {list(spec['variants'])}"], [], []
    problems = []
    if man.get("rounds") != spec["variants"][variant]:
        problems.append(f"MANIFEST rounds {man.get('rounds')} != spec {spec['variants'][variant]}")
    plan, _ = plan_for(spec, variant)
    more, jobs, valid = rl2lib.check_evidence(
        variant_dir, plan, {"diagnostic": spec_path, "source": os.path.join(ROOT, spec["source_plan"])},
        manifest_key="plan_sha256")
    return problems + more, jobs, valid


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
    print(f"\n## Rule (docs/RL2_SCREEN_B.md)\n\nGate: no seed starves ({STARVE}+ rounds in a row "
          f"with no successful contribution, invited or not); median equal-spend acc >= "
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
        problems, jobs, paths = validate(spec, os.path.dirname(group_dir), args.spec)
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
    print(f"Starved = {STARVE}+ rounds in a row with no successful contribution (all invitations "
          "failed, or nobody invited); rounds with no client online are skipped. "
          f"Equal spend: fixed-set accuracy at the last round with cumulative cost <= {BUDGET:.0f}.\n")
    print("| condition | seed | successes/invites | fallback share | longest no-contribution run "
          "(failed invites / no invites) | empty rounds | offline rounds | starved "
          "| modes % L/S/P | final acc | worst-10% acc | total spend | acc at equal spend (round) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in sorted(runs):
        for seed, m in sorted(runs[cond].items()):
            ab = m["acc_budget"]
            print(f"| {cond} | {seed} | {m['succ']}/{m['inv']} | {m['fallback']:.2f} | {m['longest']} "
                  f"({m['longest_fail']} / {m['longest_empty']}) | {m['empty']} | {m['offline']} | "
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
