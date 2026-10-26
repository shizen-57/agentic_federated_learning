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
import json
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


def load_meta(path: str) -> dict | None:
    """The run's saved config (with provenance), or None if there is none."""
    meta = path[:-4] + ".config.json"
    if not os.path.exists(meta):
        return None
    with open(meta) as fh:
        return json.load(fh)


def accuracy_curve(rows: list[dict], meta: dict | None = None) -> list[tuple[float, float]] | None:
    """(round, accuracy) pairs measured on one fixed test population, for curves,
    curve averages and rounds-to-target -- or None if the file cannot provide one.

    Newer CSVs log accuracy_fixed. In older ones, "accuracy" is the same test set
    in every round unless a per-round subset was used (fl.eval_subset > 0), in
    which case the final round switched to the full set and is left out; with
    no saved config that cannot be checked, so the curve is refused.
    """
    if rows and all("accuracy_fixed" in r for r in rows):
        return [(r["round"], r["accuracy_fixed"]) for r in rows]
    if meta is None:
        return None
    fl = meta.get("fl", {})
    if fl.get("val_frac", 0) > 0:
        return None
    pts = [(r["round"], r["accuracy"]) for r in rows]
    if fl.get("eval_subset", 0) > 0 and rows and rows[-1]["round"] == fl.get("rounds"):
        pts = pts[:-1]
    return pts


def rounds_to_target(rows: list[dict], target: float, meta: dict | None = None) -> float | None:
    """First round whose fixed-population accuracy reaches target; NaN if never;
    None if the file has no fixed-population series (see accuracy_curve)."""
    curve = accuracy_curve(rows, meta)
    if curve is None:
        return None
    for rnd, acc in curve:
        if acc >= target:
            return rnd
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
    metas: dict[str, list[dict | None]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        name = os.path.basename(path)[:-4]
        if name == "summary" or name.startswith("stats_"):
            continue
        groups[re.sub(r"_s\d+$", "", name)].append(load(path))
        metas[re.sub(r"_s\d+$", "", name)].append(load_meta(path))

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
        vals = [rounds_to_target(r, args.target, m) for r, m in zip(runs, metas[tag])]
        if any(v is None for v in vals):
            # No fixed test population in some run: refuse rather than mix sets.
            rec[f"reached_{args.target:g}"] = "unsupported"
            rec[f"rounds_to_{args.target:g}"] = (float("nan"), 0.0)
        else:
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
        print("Older runs without accuracy_fixed: rounds to target use their per-round "
              "accuracy minus any final round that switched test sets, and are marked "
              "unsupported where that cannot be checked.\n")
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
