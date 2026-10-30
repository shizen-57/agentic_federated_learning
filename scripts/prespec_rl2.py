#!/usr/bin/env python3
"""RL2 confirmation (M10): the pre-specified tests, exactly as frozen in
configs/rl2_frozen.json by scripts/freeze_rl2.py (docs/RL2_CONFIRMATION.md).

    python scripts/prespec_rl2.py --frozen configs/rl2_frozen.json --root runs/rl2_confirm

Primary: 8 paired, one-sided sign-flip tests (the final method better than
the comparator), Holm-adjusted together at alpha 0.05 -- per dataset, P1
equal-spend accuracy (eq5) and P2 equal-spend worst-10% client accuracy
(w10b), each vs the tuned always-Light with MimiC and vs screen C's best
tuned comparator on that endpoint. The report also discloses spend, cost
curves, starvation, participation fairness, budget projections, pre-training
cost and the secondary worlds (descriptive).
A test supports superiority only if its Holm-adjusted p < alpha (its mean
difference is then > 0). Secondary comparisons are descriptive, with
unadjusted two-sided p. No verdict is printed unless the frozen manifest
matches the pins below and every run is complete and valid.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import rl2lib  # noqa: E402

# Written by scripts/freeze_rl2.py, together, once; never edited by hand.
FROZEN_SHA256 = None
FROZEN_CODE = None


def verify_frozen(path: str) -> dict:
    """The frozen manifest, if it is exactly the pinned one."""
    if FROZEN_SHA256 is None:
        raise SystemExit("no RL2 confirmation has been frozen yet (scripts/freeze_rl2.py)")
    if rl2lib._sha(path) != FROZEN_SHA256:
        raise SystemExit(f"{path} is not the frozen manifest pinned in scripts/prespec_rl2.py")
    with open(path) as fh:
        frozen = json.load(fh)
    if frozen.get("code_hash") != FROZEN_CODE:
        raise SystemExit("the frozen manifest's code hash does not match the pin")
    return frozen


def inputs_of(frozen: dict, frozen_path: str) -> dict[str, str]:
    """Every input the frozen manifest pins (paths relative to the repo)."""
    out = {"frozen": frozen_path}
    out.update({p: os.path.join(ROOT, p) for p in frozen["inputs"]})
    return out


def runs_by_condition(frozen: dict, mg: dict) -> dict:
    import screen_c_rl2 as sc
    return sc.by_condition(frozen["spec"], tuned_of(frozen), mg)


def tuned_of(frozen: dict) -> dict:
    return {"overrides": frozen["tuned_overrides"], "base": frozen["spec"]["base"]}


def evaluate(frozen: dict, frozen_path: str, root: str) -> tuple[int, str, dict]:
    import screen_c_rl2 as sc
    # every run, every checkpoint (made by this code, for its method; each
    # pre-trained run started from it) and every pinned input
    problems, mg = sc.evidence(frozen["spec"], tuned_of(frozen), root, frozen["plan"],
                               inputs_of(frozen, frozen_path))
    for p, sha in frozen["inputs"].items():
        if rl2lib._sha(os.path.join(ROOT, p)) != sha:
            problems.append(f"{p} has changed since the freeze")
    if problems:
        return 2, "**INCOMPLETE -- no verdict.**\n\n" + "\n".join(f"- {p}" for p in problems) + "\n", {}
    runs = runs_by_condition(frozen, mg)
    n_runs = sum(len(g["conditions"]) * len(g["seeds"]) for g in frozen["plan"]["groups"].values())
    final = frozen["final"]["name"]
    res = []
    for t in frozen["primary_tests"]:
        a, b = runs[t["dataset"]][final], runs[t["dataset"]][t["vs"]]
        d = [a[s][t["endpoint"]] - b[s][t["endpoint"]] for s in sorted(a) if s in b]
        res.append({**t, "n": len(d), "mean_diff": st.mean(d), "wins": sum(x > 0 for x in d),
                    "p": rl2lib.signflip_p(d, "greater")})
    for r, adj in zip(res, rl2lib.holm([r["p"] for r in res])):
        r["p_holm"] = adj
        r["supported"] = adj < frozen["alpha"] and r["mean_diff"] > 0
    sec = []
    for t in frozen["secondary_tests"]:
        a, b = runs[t["dataset"]][t["a"]], runs[t["dataset"]][t["b"]]
        d = [a[s][t["endpoint"]] - b[s][t["endpoint"]] for s in sorted(a) if s in b]
        sec.append({**t, "n": len(d), "mean_diff": st.mean(d), "p_two_sided": rl2lib.signflip_p(d, "two-sided")})
    lines = ["# RL2 confirmation: pre-specified tests", "",
             f"Evidence complete: {n_runs} runs, frozen manifest verified. Final method: **{final}**.", "",
             "## Primary (one-sided, Holm over all 8)", "",
             "| family | dataset | vs | endpoint | n | mean diff | wins | p | Holm p | supported |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['family']} | {r['dataset']} | {r['vs']} | {r['endpoint']} | {r['n']} | {r['mean_diff']:+.4f} | "
              f"{r['wins']}/{r['n']} | {r['p']:.4f} | {r['p_holm']:.4f} | {'YES' if r['supported'] else 'no'} |"
              for r in res]
    lines += ["", "## Secondary (descriptive; unadjusted two-sided p)", "",
              "| dataset | comparison | endpoint | n | mean diff | p |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['dataset']} | {r['a']} - {r['b']} | {r['endpoint']} | {r['n']} | {r['mean_diff']:+.4f} | "
              f"{r['p_two_sided']:.4f} |" for r in sec]
    more, extra = disclosures(frozen, runs)
    return 0, "\n".join(lines + more) + "\n", {"primary": res, "secondary": sec, **extra}


def _med(v):
    v = [x for x in v if x == x]
    return st.median(v) if v else float("nan")


def disclosures(frozen: dict, runs: dict) -> tuple[list[str], dict]:
    """What the endpoints alone do not show (gap review A5, B6, B10, B3-B5):
    actual spend and accuracy at common budgets, starvation, participation
    fairness, how often the budget step changed jobs, the soft cost target,
    pre-training cost, and the secondary worlds. Descriptive; medians over
    seeds."""
    import report_dev_screen as rep
    import screen_c_rl2 as sc
    L, out = ["", "## Resources, fairness and safety (descriptive)", "",
              "Medians over seeds. Spend in simulated cost units; budget checkpoints show accuracy and "
              "(actual spend) at the last round within each budget.", ""], {"disclosures": {}}
    for d, conds in runs.items():
        main = {c: r for c, r in conds.items() if not c.startswith("world:")}
        L += [f"### {d}", "", "| condition | total spend | spend at eq5 | spend / soft target, 2nd half | "
              "starved | never contributed | idle 50+ rounds | Jain | weak-client acc | jobs made lighter / "
              "dropped by the budget step |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c, r in sorted(main.items()):
            v = list(r.values())
            inv = _med([m["inv"] for m in v]) or 1
            L.append(f"| {c} | {_med([m['spend'] for m in v]):.0f} | {_med([m['spend_eq5'] for m in v]):.0f} | "
                     f"{_med([m['spend_ratio_late'] for m in v]):.2f} | {sum(m['starved'] for m in v)}/{len(v)} | "
                     f"{_med([m['n_never'] for m in v]):.0f} | {_med([m['n_gap50'] for m in v]):.0f} | "
                     f"{_med([m['jain'] for m in v]):.3f} | {_med([m['acc_weak10'] for m in v]):.3f} | "
                     f"{_med([m['budget_lighter'] for m in v]) / inv:.1%} / "
                     f"{_med([m['budget_dropped'] for m in v]) / inv:.1%} |")
        L += ["", "Accuracy at common budget checkpoints, median (median actual spend) [seeds that got there]:", "",
              "| condition | " + " | ".join(f"B={b}" for b in rep.CHECKPOINTS) + " |",
              "|---|" + "---|" * len(rep.CHECKPOINTS)]
        for c, r in sorted(main.items()):
            cells = []
            for b in rep.CHECKPOINTS:
                hits = [m["curve"][b] for m in r.values() if b in m["curve"]]
                cells.append(f"{_med([h[0] for h in hits]):.3f} ({_med([h[1] for h in hits]):.0f}) [{len(hits)}]"
                             if hits else "-")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")
    tuned = tuned_of(frozen)
    L += ["Pre-training cost per checkpoint (practice episodes only; not inside the 1,400 budget):", ""]
    import torch
    for name, (path, _, _) in sc.checkpoints(frozen["spec"], tuned).items():
        meta = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False).get("meta", {})
        L.append(f"- {name}: {meta.get('practice_totals')}")
        out["disclosures"][f"pretrain {name}"] = meta.get("practice_totals")
    w = frozen["spec"].get("worlds")
    if w:
        L += ["", f"## Secondary worlds: final method minus the control ({w['control']}), "
              f"seeds {w['seeds'][0]}-{w['seeds'][-1]}", "",
              "| dataset | world | eq5 mean diff | wins | w10b mean diff | wins |", "|---|---|---|---|---|---|"]
        for d, conds in runs.items():
            for wname in w["settings"]:
                a, b = conds[f"world:{wname}:final"], conds[f"world:{wname}:control"]
                row = [f"| {d} | {wname} "]
                for k in ("eq5", "w10b"):
                    diff = [a[s][k] - b[s][k] for s in sorted(a) if s in b]
                    row.append(f"| {st.mean(diff):+.4f} | {sum(x > 0 for x in diff)}/{len(diff)} ")
                    out["disclosures"][f"world {d} {wname} {k}"] = st.mean(diff)
                L.append("".join(row) + "|")
    return L, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=os.path.join(ROOT, "configs", "rl2_frozen.json"))
    ap.add_argument("--root", required=True, help="the confirmation's output folder")
    args = ap.parse_args()
    frozen = verify_frozen(args.frozen)
    code, report, results = evaluate(frozen, args.frozen, args.root)
    print(report)
    if not code:
        with open(os.path.join(args.root, "RESULTS.json"), "w") as fh:
            json.dump(results, fh, indent=1, sort_keys=True)
        with open(os.path.join(args.root, "report.md"), "w") as fh:
            fh.write(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
