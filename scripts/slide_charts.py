#!/usr/bin/env python3
"""Presentation-ready charts for the results slide(s), distinct from the
per-round convergence panels scripts/plot.py already produces.

Chart types (see docs/CHART_GUIDE.md for which slide each is for):
1. headline_accuracy_bar  -- final accuracy, curated methods, bar + error bars
2. fairness_bar           -- final CV fairness, same method set
3. scorecard_grid         -- 2x2 small multiples: accuracy/cost/latency/fail-ratio
4. seed_variance_box      -- box plot of final accuracy across the 5 raw seeds
5. tradeoff_scatter       -- cost vs. accuracy, emphasis style (proposed highlighted)
6. ablation_bars          -- Task 1/2 ablation: ARL vs raw-fail, CV-succ vs CV-sel
7. aggregator_bars        -- FedProx vs FedNova vs Combined (c1/c2/c3/c5), 3 metrics

Uses the dataviz skill's validated 8-hue categorical palette (references/palette.md),
run through scripts/validate_palette.js before use. Every bar/point gets a direct
value label -- three of the eight hues fall under the light-surface contrast floor,
which the skill's relief rule requires visible labels for, so this isn't optional
polish here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

# Fixed categorical order -- never cycle, never reassign by rank (dataviz skill).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

LABELS = {
    "c1_rl_fedavg": "RL + FedAvg", "c2_rl_fedprox": "RL + FedProx (proposed)",
    "c3_rl_fednova": "RL + FedNova", "c4_heuristic_fedprox": "Round-robin (heuristic)",
    "c5_rl_fedprox_fednova": "RL + FedProx+FedNova",
    "b_random": "Random", "b_fedcs": "FedCS", "b_powerofchoice": "Power-of-Choice",
    "b_oort": "Oort", "b_fedcer": "FedCER-like", "b_mimic_impute": "Oort + imputation",
    "ablation_rawfail": "Raw failure-count reward", "ablation_cvsel": "CV-over-selection",
}


def _label(tag: str) -> str:
    return LABELS.get(tag, tag)


def load_final_rows(results_dir: str) -> dict[str, list[dict]]:
    """tag -> list of final-round dicts, one per seed CSV."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        name = os.path.basename(path)[:-4]
        if name in ("summary",):
            continue
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        tag = re.sub(r"_s\d+$", "", name)
        groups[tag].append({k: float(v) for k, v in rows[-1].items()})
    return groups


def _bar_with_labels(ax, x, heights, errs, colors, fmt="{:.3f}"):
    bars = ax.bar(x, heights, yerr=errs, color=colors, capsize=4,
                  edgecolor="#0b0b0b", linewidth=0.6)
    span = max(heights) - min(min(heights), 0) or 1.0
    for b, h, e in zip(bars, heights, errs):
        ax.text(b.get_x() + b.get_width() / 2, h + e + 0.02 * span, fmt.format(h),
                ha="center", va="bottom", fontsize=9, color="#0b0b0b")
    ax.margins(y=0.15)  # headroom so labels above error bars never clip the frame
    return bars


def headline_accuracy_bar(groups, tags, out):
    means = [st.mean(r["accuracy"] for r in groups[t]) for t in tags]
    stds = [st.stdev([r["accuracy"] for r in groups[t]]) if len(groups[t]) > 1 else 0 for t in tags]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _bar_with_labels(ax, range(len(tags)), means, stds, PALETTE[:len(tags)])
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([_label(t) for t in tags], rotation=20, ha="right")
    ax.set_ylabel("Final global accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Final accuracy by method (mean ± std, 5 seeds)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fairness_bar(groups, tags, out):
    means = [st.mean(r["cv"] for r in groups[t]) for t in tags]
    stds = [st.stdev([r["cv"] for r in groups[t]]) if len(groups[t]) > 1 else 0 for t in tags]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _bar_with_labels(ax, range(len(tags)), means, stds, PALETTE[:len(tags)])
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([_label(t) for t in tags], rotation=20, ha="right")
    ax.set_ylabel("CV of successful contributions (lower = fairer)")
    ax.set_title("Successful-contribution fairness by method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def scorecard_grid(groups, tags, out):
    metrics = [("accuracy", "Accuracy", "{:.3f}"), ("cost", "Cost", "{:.2f}"),
              ("latency", "Latency", "{:.2f}"), ("fail_ratio", "Failed-participation", "{:.2f}")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = PALETTE[:len(tags)]
    for ax, (key, title, fmt) in zip(axes.flat, metrics):
        means = [st.mean(r[key] for r in groups[t]) for t in tags]
        stds = [st.stdev([r[key] for r in groups[t]]) if len(groups[t]) > 1 else 0 for t in tags]
        _bar_with_labels(ax, range(len(tags)), means, stds, colors, fmt=fmt)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([_label(t) for t in tags], rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Method scorecard: four metrics at a glance")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def seed_variance_box(groups, tags, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [[r["accuracy"] for r in groups[t]] for t in tags]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showmeans=True)
    for patch, color in zip(bp["boxes"], PALETTE[:len(tags)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#0b0b0b")
    for i, vals in enumerate(data, start=1):
        ax.scatter([i] * len(vals), vals, color="#0b0b0b", s=14, zorder=3)
    ax.set_xticks(range(1, len(tags) + 1))
    ax.set_xticklabels([_label(t) for t in tags], rotation=20, ha="right")
    ax.set_ylabel("Final accuracy (each dot = one seed)")
    ax.set_title("Seed-to-seed variance, not just mean ± std")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def tradeoff_scatter(groups, tags, highlight, out):
    """Emphasis form (dataviz skill): one series is the point, rest are context."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in tags:
        acc = st.mean(r["accuracy"] for r in groups[t])
        cost = st.mean(r["cost"] for r in groups[t])
        if t == highlight:
            ax.scatter(cost, acc, color=PALETTE[0], s=180, zorder=5,
                      edgecolor="#0b0b0b", linewidth=1.2, label=_label(t))
            ax.annotate(_label(t), (cost, acc), textcoords="offset points",
                       xytext=(8, 8), fontsize=10, fontweight="bold", color="#0b0b0b")
        else:
            ax.scatter(cost, acc, color="#9a9a95", s=90, zorder=3,
                      edgecolor="#0b0b0b", linewidth=0.6)
            ax.annotate(_label(t), (cost, acc), textcoords="offset points",
                       xytext=(6, -10), fontsize=8, color="#52514e")
    ax.set_xlabel("Cost (lower is better)")
    ax.set_ylabel("Final accuracy (higher is better)")
    ax.set_title("Cost vs. accuracy trade-off -- top-left is best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def ablation_bars(groups, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    metrics = [("fail_ratio", "Failed-participation ratio"), ("cost", "Cost"), ("latency", "Latency")]
    tags = ["c2_rl_fedprox", "ablation_rawfail", "ablation_cvsel"]
    names = ["ARL' reward\n(proposed)", "Raw failure-count\nreward", "CV-over-selection\nfairness"]
    for ax, (key, title) in zip(axes, metrics):
        means = [st.mean(r[key] for r in groups[t]) for t in tags]
        stds = [st.stdev([r[key] for r in groups[t]]) if len(groups[t]) > 1 else 0 for t in tags]
        _bar_with_labels(ax, range(len(tags)), means, stds, PALETTE[:len(tags)], fmt="{:.2f}")
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Task 1/2 ablations: does the fancier metric earn its keep?")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def aggregator_bars(groups, out):
    tags = ["c1_rl_fedavg", "c2_rl_fedprox", "c3_rl_fednova", "c5_rl_fedprox_fednova"]
    metrics = [("accuracy", "Accuracy", "{:.3f}"), ("cv", "CV fairness", "{:.3f}"),
              ("cost", "Cost", "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    names = ["FedAvg", "FedProx", "FedNova", "FedProx+\nFedNova"]
    for ax, (key, title, fmt) in zip(axes, metrics):
        means = [st.mean(r[key] for r in groups[t]) for t in tags]
        stds = [st.stdev([r[key] for r in groups[t]]) if len(groups[t]) > 1 else 0 for t in tags]
        _bar_with_labels(ax, range(len(tags)), means, stds, PALETTE[:len(tags)], fmt=fmt)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Aggregation rule comparison, including the combined condition")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="figures/slides")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    groups = load_final_rows(args.dir)
    headline = ["c2_rl_fedprox", "b_random", "b_oort", "b_powerofchoice", "b_fedcer"]
    headline = [t for t in headline if t in groups]

    headline_accuracy_bar(groups, headline, os.path.join(args.out, "01_headline_accuracy.png"))
    fairness_bar(groups, headline, os.path.join(args.out, "02_fairness_bar.png"))
    scorecard_grid(groups, headline, os.path.join(args.out, "03_scorecard_grid.png"))
    seed_variance_box(groups, headline, os.path.join(args.out, "04_seed_variance_box.png"))
    tradeoff_scatter(groups, headline, "c2_rl_fedprox",
                     os.path.join(args.out, "05_tradeoff_scatter.png"))
    if "ablation_rawfail" in groups and "ablation_cvsel" in groups:
        ablation_bars(groups, os.path.join(args.out, "06_ablation_bars.png"))
    if "c5_rl_fedprox_fednova" in groups:
        aggregator_bars(groups, os.path.join(args.out, "07_aggregator_bars.png"))

    print(f"wrote charts to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
