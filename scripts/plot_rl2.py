#!/usr/bin/env python3
"""Thesis figures for RL2 (gap review C3).

    python scripts/plot_rl2.py --frozen configs/rl2_frozen.json --root runs/rl2_confirm --out figures/rl2
    python scripts/plot_rl2.py --spec configs/rl2_screen_c.yaml --root runs/rl2_screen_c --out figures/rl2_dev

From the confirmation (frozen manifest + its runs) or, for a development
look, screen C (its spec + runs). Reads every run's CSV through the plan;
the figures say nothing the pre-specified report does not: the report
(scripts/prespec_rl2.py) is the record, these are pictures of it.

- primary.png: each primary comparison, the mean paired difference (final
  minus comparator) with a 95% bootstrap interval, per dataset;
- seeds.png: the per-seed paired differences behind each primary comparison;
- cost_<dataset>.png: median fixed-set accuracy against cumulative cost for
  the final method and its comparators;
- worlds.png: final minus control in each secondary world (confirmation only).
Colours follow docs/CHART_GUIDE.md (validated categorical slots, ink and grid).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # validated categorical slots 1-4 (light)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def bootstrap_ci(d, n=10000, seed=20261028):
    import numpy as np
    x = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def comparisons(runs: dict, final: str, tests: list[dict]) -> list[dict]:
    """Each primary comparison with its per-seed differences and interval."""
    out = []
    for t in tests:
        a, b = runs[t["dataset"]][final], runs[t["dataset"]][t["vs"]]
        d = [a[s][t["endpoint"]] - b[s][t["endpoint"]] for s in sorted(a) if s in b]
        out.append({**t, "diffs": d, "mean": st.mean(d), "ci": bootstrap_ci(d)})
    return out


def plot_primary(rows: list[dict], path: str, supported: dict | None = None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    labels = []
    for k, r in enumerate(rows):
        y = len(rows) - 1 - k
        sup = (supported or {}).get(k)
        ax.plot(r["ci"], [y, y], color=COLORS[0], linewidth=2)
        ax.plot([r["mean"]], [y], marker="o", markersize=8, color=COLORS[0],
                markerfacecolor=COLORS[0] if sup else SURFACE, markeredgewidth=2)
        labels.append(f"{r['family']} {r['dataset']}: vs {r['vs']} ({r['endpoint']})"
                      + ("  -- supported" if sup else ""))
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1], fontsize=8, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("final method minus comparator (mean paired difference, 95% bootstrap interval)",
                  color=INK, fontsize=9)
    ax.set_title("Primary comparisons" + (" (filled = supported after Holm)" if supported else ""),
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_seeds(rows: list[dict], path: str) -> None:
    plt = _plt()
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    rng = np.random.default_rng(0)
    for k, r in enumerate(rows):
        y = len(rows) - 1 - k
        ax.scatter(r["diffs"], y + rng.uniform(-0.15, 0.15, len(r["diffs"])), s=14, color=COLORS[0],
                   alpha=0.6, linewidths=0)
        ax.plot([r["mean"]] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['family']} {r['dataset']}: vs {r['vs']} ({r['endpoint']})" for r in rows][::-1],
                       fontsize=8, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("final method minus comparator, one dot per seed (bar = mean)", color=INK, fontsize=9)
    ax.set_title("Per-seed paired differences", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_cost(curves: dict[str, list[tuple[float, float]]], path: str, title: str) -> None:
    """curves: {label: [(spend, median accuracy), ...]} (at most 4 labels)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for (label, pts), color in zip(curves.items(), COLORS):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2, label=label)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("cumulative cost (simulated units; actual spend)", color=INK, fontsize=9)
    ax.set_ylabel("fixed-set accuracy (median over seeds)", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_worlds(diffs: dict[str, dict[str, float]], path: str) -> None:
    """diffs: {"<dataset> <world>": {"eq5": d, "w10b": d}}."""
    plt = _plt()
    import numpy as np
    keys = list(diffs)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(keys) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    y = np.arange(len(keys))[::-1]
    for j, (ep, color) in enumerate(zip(("eq5", "w10b"), COLORS)):
        ax.barh(y + (0.2 if j == 0 else -0.2), [diffs[k][ep] for k in keys], height=0.36, color=color,
                edgecolor=SURFACE, linewidth=2, label=ep)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=8, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("final method minus control (mean paired difference)", color=INK, fontsize=9)
    ax.set_title("Secondary worlds", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def cost_points(conds: dict) -> list[tuple[float, float]]:
    """Median (over seeds) accuracy at the common budget checkpoints, at the
    median actual spend there; only checkpoints every seed reached."""
    import report_dev_screen as rep
    pts = []
    for b in rep.CHECKPOINTS:
        hits = [m["curve"][b] for m in conds.values() if b in m["curve"]]
        if len(hits) == len(conds):
            pts.append((st.median(h[1] for h in hits), st.median(h[0] for h in hits)))
    return pts


def load_runs(spec: dict, tuned: dict, plan: dict, root: str) -> dict:
    import report_dev_screen as rep
    import screen_c_rl2 as sc
    from sweep import build_jobs
    mg: dict = {}
    for j in build_jobs(plan, None, None, root, []):
        path = os.path.join(j["raw"]["out"], j["raw"]["tag"] + ".csv")
        if not os.path.exists(path):
            raise SystemExit(f"missing run {path}: plot only complete evidence")
        mg.setdefault(j["group"], {}).setdefault(j["cond"], {})[j["seed"]] = rep.measures(path)
    return sc.by_condition(spec, tuned, mg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen")
    ap.add_argument("--spec")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import screen_c_rl2 as sc
    supported = None
    if args.frozen:
        import prespec_rl2 as pre
        with open(args.frozen) as fh:
            frozen = json.load(fh)
        spec, tuned, plan = frozen["spec"], pre.tuned_of(frozen), frozen["plan"]
        final, tests = frozen["final"]["name"], frozen["primary_tests"]
        res_path = os.path.join(args.root, "RESULTS.json")
        if os.path.exists(res_path):
            with open(res_path) as fh:
                res = json.load(fh)["primary"]
            supported = {k: r["supported"] for k, r in enumerate(res)}
    elif args.spec:
        spec, tuned = sc.load(args.spec)
        plan = sc.build_plan(spec, tuned)
        with open(os.path.join(args.root, "DECISION.json")) as fh:
            dec = json.load(fh)
        final, tests = dec["final"]["name"], dec["tests"]
    else:
        raise SystemExit("give --frozen (confirmation) or --spec (screen C)")
    runs = load_runs(spec, tuned, plan, args.root)
    os.makedirs(args.out, exist_ok=True)
    rows = comparisons(runs, final, tests)
    plot_primary(rows, os.path.join(args.out, "primary.png"), supported)
    plot_seeds(rows, os.path.join(args.out, "seeds.png"))
    comps = list(dict.fromkeys(t["vs"] for t in tests))[:3]
    for d in runs:
        plot_cost({final: cost_points(runs[d][final]), **{c: cost_points(runs[d][c]) for c in comps}},
                  os.path.join(args.out, f"cost_{d}.png"), f"{d}: accuracy against what was actually spent")
    w = spec.get("worlds")
    if w:
        diffs = {}
        for d in runs:
            for wname in w["settings"]:
                a, b = runs[d][f"world:{wname}:final"], runs[d][f"world:{wname}:control"]
                diffs[f"{d} {wname}"] = {k: st.mean(a[s][k] - b[s][k] for s in sorted(a) if s in b)
                                         for k in ("eq5", "w10b")}
        plot_worlds(diffs, os.path.join(args.out, "worlds.png"))
    print(f"wrote figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
