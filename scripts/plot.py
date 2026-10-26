#!/usr/bin/env python3
"""Chapter 5 figures, generated from the logged CSVs only.

Produces four panels -- accuracy, CV fairness, ARL' vs ARL*, and failed
participation -- with the mean across seeds and a shaded +-1 std band, because a
mean line without its spread is not a result at this sample size.

    python scripts/plot.py --dir results --out figures
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import accuracy_curve, load_meta  # noqa: E402

PANELS = [
    ("accuracy", "Global test accuracy (same held-out examples every round)", "accuracy"),
    ("cv", "Successful-contribution fairness (CV, lower is better)", "CV_t"),
    ("arl_online", "Avoidable representation loss (ARL', lower is better)", "ARL'_t"),
    ("fail_ratio", "Failed-participation ratio", "D^fail_t"),
]


def load_groups(d: str) -> dict[str, list[dict[str, np.ndarray]]]:
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(d, "*.csv"))):
        name = os.path.basename(path)[:-4]
        if name == "summary" or name.startswith("stats_"):
            continue
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        arr = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
        # Accuracy curve on one fixed test population (None: refused).
        curve = accuracy_curve([{k: float(v) for k, v in r.items()} for r in rows],
                               load_meta(path))
        arr["_curve"] = None if curve is None else np.array(curve)
        groups[re.sub(r"_s\d+$", "", name)].append(arr)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="*", help="restrict to these run tags")
    args = ap.parse_args()

    groups = load_groups(args.dir)
    if args.only:
        groups = {k: v for k, v in groups.items() if k in args.only}
    if not groups:
        print(f"no run CSVs found in {args.dir}/")
        return 1

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (key, title, ylab) in zip(axes.flat, PANELS):
        for tag in sorted(groups):
            runs = groups[tag]
            if key == "accuracy":
                # One test population in every round (report.accuracy_curve).
                if any(r["_curve"] is None for r in runs):
                    print(f"skip {tag} accuracy curve: no fixed-test-set series")
                    continue
                n = min(len(r["_curve"]) for r in runs)
                stack = np.stack([r["_curve"][:n, 1] for r in runs])
                x = runs[0]["_curve"][:n, 0]
            else:
                n = min(len(r[key]) for r in runs)
                stack = np.stack([r[key][:n] for r in runs])
                x = runs[0]["round"][:n]
            mu, sd = stack.mean(0), stack.std(0)
            ax.plot(x, mu, label=tag, linewidth=1.4)
            if len(runs) > 1:
                ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        # The oracle bound belongs on the ARL panel: the gap between ARL' and
        # ARL* is the quantity Sec. 3.23 asks to be measured, not assumed.
        if key == "arl_online":
            for tag in sorted(groups):
                runs = groups[tag]
                n = min(len(r["arl_oracle"]) for r in runs)
                stack = np.stack([r["arl_oracle"][:n] for r in runs])
                ax.plot(runs[0]["round"][:n], stack.mean(0), "--", linewidth=1.0,
                        alpha=0.6, label=f"{tag} (ARL* oracle)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("communication round")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)

    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = os.path.join(args.out, "chapter5_panels.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
