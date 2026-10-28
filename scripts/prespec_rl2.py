#!/usr/bin/env python3
"""RL2 confirmation (M10): the pre-specified tests, exactly as frozen in
configs/rl2_frozen.json by scripts/freeze_rl2.py (docs/RL2_CONFIRMATION.md).

    python scripts/prespec_rl2.py --frozen configs/rl2_frozen.json --root runs/rl2_confirm

Primary: 8 paired, one-sided sign-flip tests (the final method better than
the comparator), Holm-adjusted together at alpha 0.05 -- per dataset, P1
equal-spend accuracy vs always-Light and vs the best baseline of screen C,
P2 worst-10% client accuracy vs always-Light and vs the repaired original RL.
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
    return 0, "\n".join(lines) + "\n", {"primary": res, "secondary": sec}


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
