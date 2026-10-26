#!/usr/bin/env python3
"""Summarise the CSVs in results/ into the table Chapter 5 needs.

Groups runs by tag (seed suffix stripped), reports mean +- std across seeds for
each headline metric, and writes results/summary.csv. Prints a markdown table so
it can be pasted straight into the thesis draft.

Reporting a single seed is not a result: with 5 clients sampled from 30 per
round, run-to-run spread is large enough to invent an improvement that is not
there. This script refuses to hide that -- it always prints the std.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics as st
from collections import defaultdict

FINAL = ["accuracy", "cv", "jain", "ece"]
MEANS = ["cost", "latency", "fail_ratio", "arl_online", "arl_oracle", "reward"]


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def rounds_to_target(rows: list[dict], target: float) -> float:
    col = "accuracy_fixed" if all("accuracy_fixed" in r for r in rows) else "accuracy"
    for r in rows:
        if r[col] >= target:
            return r["round"]
    return float("nan")


def invitation_failure_rate(rows: list[dict]) -> float:
    """Fraction of invitations that failed on time; empty runs have no rate."""
    invited = sum(r["n_selected"] for r in rows)
    return (sum(r["n_selected"] - r["n_success"] for r in rows) / invited
            if invited else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--target", type=float, default=0.80,
                    help="accuracy threshold for the rounds-to-target column")
    args = ap.parse_args()

    groups: dict[str, list[list[dict]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        name = os.path.basename(path)[:-4]
        if name == "summary" or name.startswith("stats_"):
            continue
        groups[re.sub(r"_s\d+$", "", name)].append(load(path))

    if not groups:
        print(f"no run CSVs found in {args.dir}/")
        return 1

    out_rows = []

    # acc_worst10 only exists in CSVs written after the advanced-sweep upgrade.
    final = FINAL + (["acc_worst10"] if all("acc_worst10" in r[0][0] for r in groups.values())
                     else [])
    counts = all("n_selected" in row and "n_success" in row
                 for runs in groups.values() for run in runs for row in run)
    cols = ["run", "seeds"] + [f"final_{k}" for k in final] + \
           [f"mean_{k}" for k in MEANS] + [f"reached_{args.target:g}", f"rounds_to_{args.target:g}"]
    if counts:
        cols += ["invitation_fail_ratio", "invitation_rate_seeds", "empty_rounds"]

    for tag in sorted(groups):
        runs = groups[tag]
        rec = {"run": tag, "seeds": len(runs)}
        for k in final:
            vals = [r[-1][k] for r in runs]
            rec[f"final_{k}"] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
        for k in MEANS:
            vals = [st.mean([row[k] for row in r]) for r in runs]
            rec[f"mean_{k}"] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
        # Rounds to target is only defined for runs that reach it, so always
        # report how many did: otherwise one lucky run out of ten would look
        # faster than ten runs that all get there a little later.
        vals = [rounds_to_target(r, args.target) for r in runs]
        good = [v for v in vals if not math.isnan(v)]
        rec[f"reached_{args.target:g}"] = f"{len(good)}/{len(vals)}"
        rec[f"rounds_to_{args.target:g}"] = (
            (st.mean(good), st.stdev(good) if len(good) > 1 else 0.0) if good else (float("nan"), 0.0))
        if counts:
            rates = [invitation_failure_rate(r) for r in runs]
            defined = [v for v in rates if math.isfinite(v)]
            rec["invitation_fail_ratio"] = (
                (st.mean(defined), st.stdev(defined) if len(defined) > 1 else 0.0)
                if defined else (float("nan"), 0.0))
            rec["invitation_rate_seeds"] = f"{len(defined)}/{len(runs)}"
            empty = [sum(row["n_selected"] == 0 for row in r) for r in runs]
            rec["empty_rounds"] = (st.mean(empty), st.stdev(empty) if len(empty) > 1 else 0.0)
        out_rows.append(rec)

    def fmt(v):
        if isinstance(v, tuple):
            return "n/a" if v[0] != v[0] else f"{v[0]:.3f} ± {v[1]:.3f}"
        return str(v)

    if any("accuracy_fixed" not in row for runs in groups.values() for run in runs for row in run):
        print("Legacy target times use accuracy, which may switch test sets at the last round; "
              "new runs use accuracy_fixed.\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for rec in out_rows:
        print("| " + " | ".join(fmt(rec[c]) for c in cols) + " |")

    with open(os.path.join(args.dir, "summary.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for rec in out_rows:
            wr.writerow([rec[c][0] if isinstance(rec[c], tuple) else rec[c] for c in cols])
    print(f"\nwrote {os.path.join(args.dir, 'summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
