#!/usr/bin/env python3
"""Apply the pre-registered rule of docs/DEV_SCREEN_PREDICTOR_V1.md (and of
later screens that reuse it with --revision).

    python scripts/report_dev_screen.py results/dev_predictor_v1/full/s2_femnist

Every seed of every condition is reported; nothing is dropped.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st

STARVE = 50          # rounds with invitations and zero successes, in a row
BUDGET = 1400.0      # simulated cost units, fixed before the runs
REVISION, BASE = "arl3_full_pn", "arl3_full"


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
            "acc_budget": at_budget}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()
    rev = args.revision
    runs: dict[str, dict[int, dict]] = {}
    for p in sorted(glob.glob(os.path.join(args.dir, "*_s*.csv"))):
        cond, seed = os.path.basename(p)[:-4].rsplit("_s", 1)
        runs.setdefault(cond, {})[int(seed)] = measures(p)
    print("# Development screen v1 (exploratory; seeds 200-204)\n")
    print(f"Starved = {STARVE}+ rounds in a row with invitations and no success. "
          f"Equal spend: fixed-set accuracy at the last round with cumulative cost <= {BUDGET:.0f}.\n")
    print("| condition | seed | successes/invites | fallback share | longest zero streak | starved "
          "| modes % L/S/P | final acc | total spend | acc at equal spend (round) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cond in sorted(runs):
        for seed, m in sorted(runs[cond].items()):
            ab = m["acc_budget"]
            print(f"| {cond} | {seed} | {m['succ']}/{m['inv']} | {m['fallback']:.2f} | {m['longest']} | "
                  f"{'YES' if m['starved'] else 'no'} | {m['modes']} | {m['acc']:.3f} | {m['spend']:.0f} | "
                  + (f"{ab[0]:.3f} (r{ab[1]})" if ab else "-") + " |")
    print("\n| condition | median successes | median final acc | median acc at equal spend | starved seeds |")
    print("|---|---|---|---|---|")
    med = {}
    for cond in sorted(runs):
        v = runs[cond].values()
        med[cond] = (st.median(m["succ"] for m in v), st.median(m["acc"] for m in v))
        eq = [m["acc_budget"][0] for m in v if m["acc_budget"]]
        print(f"| {cond} | {med[cond][0]:.0f} | {med[cond][1]:.3f} | "
              f"{st.median(eq) if eq else float('nan'):.3f} | {sum(m['starved'] for m in v)}/{len(v)} |")
    if rev in runs and BASE in runs:
        c1 = not any(m["starved"] for m in runs[rev].values())
        c2 = med[rev][0] >= med[BASE][0]
        c3 = med[rev][1] >= med[BASE][1] - 0.02
        print(f"\nDecision rule: (1) no revision seed starves: {'pass' if c1 else 'FAIL'}; "
              f"(2) median successes >= unchanged ARL-v3: {'pass' if c2 else 'FAIL'}; "
              f"(3) median final accuracy within 0.02 of unchanged ARL-v3: {'pass' if c3 else 'FAIL'}.")
        print(f"**Screen {'PASSED' if c1 and c2 and c3 else 'FAILED'}** "
              "(passing means worth a confirmation study on fresh seeds, not better).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
