#!/usr/bin/env python3
"""Figures for the s2_* runs (read-only: uses the run CSVs and the markdown
tables the analysis scripts already wrote; changes no frozen input).

    python scripts/plot_s2.py                    # writes figures/s2/*.png
    python scripts/plot_s2.py --root results/adv --out figures/s2
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Colour follows the kind of method, never its rank.
FAMILY_COLOR = {"learned": "#2a78d6", "rule": "#1baf7a", "ew": "#eb6834", "published": "#8a8984"}
FAMILY_LABEL = {"learned": "RL that learns its own modes (ours)",
                "rule": "RL / rule with a fixed mode rule (controls)",
                "ew": "baseline + shared mode rule (_ew)",
                "published": "baseline as published"}
LEARNED = {"c2_rl_fedprox", "arl2_full", "arl3_full", "arl3_shared", "v2_rawfail"}
RULE = {"rl_light", "rl_ew", "b_cov_ew"}
NICE = {"c2_rl_fedprox": "C2 RL (FedProx)", "arl2_full": "ARL-v2", "arl3_full": "ARL-v3",
        "arl3_shared": "ARL-v3 (shared)", "v2_rawfail": "RL raw-fail reward",
        "rl_light": "RL, always Light", "rl_ew": "RL + mode rule", "b_cov_ew": "coverage rule",
        "b_powerofchoice": "Power-of-Choice", "b_poc_ew": "Power-of-Choice_ew",
        "b_fedcer": "FedCER-like", "b_fedcs": "FedCS", "b_mimic": "MIMIC-like"}
SETTINGS = {"s2_cifar": "CIFAR-10", "s2_femnist": "FEMNIST",
            "s2_churn_cifar": "CIFAR-10 + churn", "s2_churn_femnist": "FEMNIST + churn",
            "s2_main_cifar": "CIFAR-10, no weak clients", "s2_main_femnist": "FEMNIST, no weak clients"}


def family(c: str) -> str:
    if c in LEARNED:
        return "learned"
    if c in RULE:
        return "rule"
    return "ew" if c.endswith("_ew") else "published"


def nice(c: str) -> str:
    if c in NICE:
        return NICE[c]
    base = c[2:] if c.startswith("b_") else c
    ew = base.endswith("_ew")
    base = base[:-3] if ew else base
    name = {"random": "Random", "oort": "Oort", "favor": "FAVOR", "refl": "REFL",
            "delta": "DELTA"}.get(base, base)
    return name + ("_ew" if ew else "")


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    })


def load_group(root: str, g: str) -> tuple[pd.DataFrame, dict]:
    """Per-run summary rows and the full per-round frames, keyed (cond, seed)."""
    rows, frames = [], {}
    for f in sorted(glob.glob(os.path.join(root, g, "*_s*.csv"))):
        c, s = os.path.basename(f)[:-4].rsplit("_s", 1)
        d = pd.read_csv(f)
        frames[(c, int(s))] = d
        rows.append(dict(cond=c, seed=int(s), acc=d.accuracy.iloc[-1],
                         fail=d.fail_ratio.mean(), cost=d.cost.mean(),
                         L=d.mode_light.mean(), S=d.mode_standard.mean(), P=d.mode_priority.mean()))
    return pd.DataFrame(rows), frames


def ci95(x: np.ndarray) -> float:
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def family_legend(fig, fams=("learned", "rule", "ew", "published"), y=0.0) -> None:
    hs = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], markersize=7) for f in fams]
    fig.legend(hs, [FAMILY_LABEL[f] for f in fams], loc="lower center", ncol=len(fams),
               bbox_to_anchor=(0.5, y), fontsize=8)


def fig_final_accuracy(data, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for ax, g in zip(axes, ("s2_cifar", "s2_femnist")):
        r = data[g][0]
        agg = r.groupby("cond").acc.agg(["mean", list]).sort_values("mean")
        for i, (c, row) in enumerate(agg.iterrows()):
            col = FAMILY_COLOR[family(c)]
            vals = np.array(row["list"])
            jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.35
            ax.scatter(vals, i + jit, s=9, color=col, alpha=0.35, lw=0)
            ax.errorbar(row["mean"], i, xerr=ci95(vals), fmt="o", color=col, ms=7,
                        mec="white", mew=1.5, elinewidth=2, capsize=0)
        ax.set_yticks(range(len(agg)))
        ax.set_yticklabels([nice(c) for c in agg.index])
        for t, c in zip(ax.get_yticklabels(), agg.index):
            if c == "c2_rl_fedprox":
                t.set_fontweight("bold")
        ax.set_title(f"{SETTINGS[g]}: final test accuracy")
        ax.set_xlabel("final accuracy (big dot = mean, bar = 95% CI, small dots = seeds)")
        ax.grid(axis="y", visible=False)
    family_legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "fig1_final_accuracy.png"), dpi=200)
    plt.close(fig)


def fig_curves(data, out) -> None:
    pick = ["c2_rl_fedprox", "arl3_full", "rl_light", "b_refl_ew", "b_oort_ew"]
    colors = ["#2a78d6", "#6250d6", "#1baf7a", "#eb6834", "#e34948"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, g in zip(axes, ("s2_cifar", "s2_femnist")):
        frames = data[g][1]
        for c, col in zip(pick, colors):
            m = np.array([frames[k].accuracy_fixed.to_numpy() for k in frames if k[0] == c])
            x = np.arange(1, m.shape[1] + 1)
            mu, lo, hi = m.mean(0), np.percentile(m, 25, 0), np.percentile(m, 75, 0)
            ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(x, mu, color=col, label=nice(c))
        ax.set_title(f"{SETTINGS[g]}: accuracy over rounds")
        ax.set_xlabel("round")
        ax.set_ylabel("accuracy on fixed test set")
        ax.legend(loc="lower right", fontsize=8)
    fig.text(0.5, -0.01, "line = mean of 20 seeds; band = middle half of seeds (25th-75th percentile)",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_accuracy_curves.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_fail_cost(data, out) -> None:
    # Only a few labels, each nudged so none collide; the rest are read by colour.
    label = {"c2_rl_fedprox": (6, -11), "arl3_full": (-8, -13), "rl_light": (6, 2),
             "b_refl_ew": (-8, 8), "b_oort_ew": (6, 4)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, g in zip(axes, ("s2_cifar", "s2_femnist")):
        agg = data[g][0].groupby("cond")[["fail", "cost", "acc"]].mean()
        for c, row in agg.iterrows():
            ax.scatter(row.cost, row.fail, s=50, color=FAMILY_COLOR[family(c)],
                       edgecolor="white", lw=1.5, zorder=3)
            if c in label:
                off = (6, 6) if (g, c) == ("s2_femnist", "c2_rl_fedprox") else label[c]
                ax.annotate(nice(c), (row.cost, row.fail), xytext=off, textcoords="offset points",
                            fontsize=7.5, color=INK2,
                            fontweight="bold" if c == "c2_rl_fedprox" else "normal")
        ax.set_title(f"{SETTINGS[g]}: failures vs cost")
        ax.set_xlabel("mean cost per round (lower = cheaper)")
        ax.set_ylabel("mean share of picked clients that failed")
        ax.text(0.98, 0.03, "best = bottom-left (cheap, few failures)", transform=ax.transAxes,
                fontsize=7.5, color=INK2, style="italic", ha="right")
    family_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out, "fig3_fail_vs_cost.png"), dpi=200)
    plt.close(fig)


def fig_modes(data, out) -> None:
    shades = ["#9ec5f4", "#3987e5", "#104281"]  # Light -> Priority: more work, darker
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=False)
    for ax, g in zip(axes, ("s2_cifar", "s2_femnist")):
        agg = data[g][0].groupby("cond")[["L", "S", "P"]].mean()
        agg = agg.div(agg.sum(axis=1), axis=0).sort_values(["P", "S"])
        y = np.arange(len(agg))
        left = np.zeros(len(agg))
        for k, col, name in zip("LSP", shades, ("Light", "Standard", "Priority")):
            ax.barh(y, agg[k], left=left, color=col, edgecolor="white", lw=1.5, height=0.75,
                    label=name)
            left += agg[k].to_numpy()
        ax.set_yticks(y)
        ax.set_yticklabels([nice(c) for c in agg.index])
        ax.set_xlim(0, 1)
        ax.set_title(f"{SETTINGS[g]}: which mode each method asks for")
        ax.set_xlabel("share of picked clients")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.1, -0.13), ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "fig4_mode_mix.png"), dpi=200)
    plt.close(fig)


STUCK_ROUNDS = 50


def zero_streak(n_success: pd.Series) -> int:
    """Longest run of consecutive rounds in which no client succeeded."""
    best = cur = 0
    for v in n_success:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def fig_lockup(data, out) -> None:
    r, frames = data["s2_femnist"]
    conds = ["c2_rl_fedprox", "arl3_shared", "arl3_full", "arl2_full", "v2_rawfail", "rl_ew",
             "rl_light"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    for i, c in enumerate(conds):
        sub = r[r.cond == c]
        lock = sub.seed.map(lambda s: zero_streak(frames[(c, s)].n_success) >= STUCK_ROUNDS)
        stuck, ok = sub[lock], sub[~lock]
        col = FAMILY_COLOR[family(c)]
        jit = lambda n: (np.random.default_rng(i).random(n) - 0.5) * 0.4  # noqa: E731
        a.scatter(ok.acc, i + jit(len(ok)), s=18, color=col, alpha=0.6, lw=0)
        a.scatter(stuck.acc, i + jit(len(stuck)), s=40, color="#e34948", marker="X", lw=0)
        a.annotate(f"{len(stuck)}/{len(sub)} stuck", (0.86, i), fontsize=8, va="center",
                   color="#e34948" if len(stuck) else INK2, xycoords=("axes fraction", "data"))
    a.set_yticks(range(len(conds)))
    a.set_yticklabels([nice(c) for c in conds])
    a.set_xlim(-0.02, 1.0)
    a.set_xlabel(f"final accuracy (one dot = one seed; red X = {STUCK_ROUNDS}+ rounds in a row "
                 "with no client succeeding)")
    a.set_title("FEMNIST: some RL runs get stuck")
    a.grid(axis="y", visible=False)

    c2 = r[r.cond == "c2_rl_fedprox"].sort_values("acc")
    bad, good = int(c2.seed.iloc[0]), int(c2.seed.iloc[-1])
    for s, col, txt in ((good, "#1baf7a", "healthy"), (bad, "#e34948", "stuck")):
        d = frames[("c2_rl_fedprox", s)]
        n = d.n_success.rolling(5, min_periods=1).mean()
        b.plot(d["round"], n, color=col, label=f"C2 RL, seed {s} ({txt})")
    b.set_xlabel("round")
    b.set_ylabel("clients that delivered an update (5-round mean)")
    b.set_title("Inside a stuck run: successes drop to zero and stay there")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_lockup.png"), dpi=200)
    plt.close(fig)


def fig_mode_rule(data, out) -> None:
    pairs = [("b_random", "b_random_ew"), ("b_oort", "b_oort_ew"),
             ("b_powerofchoice", "b_poc_ew"), ("b_favor", "b_favor_ew"),
             ("b_refl", "b_refl_ew"), ("b_delta", "b_delta_ew")]
    names = ["Random", "Oort", "Power-of-Choice", "FAVOR", "REFL", "DELTA"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, g in zip(axes, ("s2_cifar", "s2_femnist")):
        m = data[g][0].groupby("cond").acc.mean()
        for i, (p, e) in enumerate(pairs):
            ax.plot([m[p], m[e]], [i, i], color=GRID, lw=3, zorder=1)
            ax.scatter(m[p], i, color=FAMILY_COLOR["published"], s=55, zorder=3,
                       edgecolor="white", lw=1.5)
            ax.scatter(m[e], i, color=FAMILY_COLOR["ew"], s=55, zorder=3, edgecolor="white", lw=1.5)
            ax.annotate(f"{m[e] - m[p]:+.3f}", (max(m[p], m[e]), i), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.margins(x=0.15)
        ax.set_title(f"{SETTINGS[g]}: same picker, better mode rule")
        ax.set_xlabel("mean final accuracy")
        ax.grid(axis="y", visible=False)
    family_legend(fig, ("published", "ew"))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "fig6_mode_rule_effect.png"), dpi=200)
    plt.close(fig)


def md_table(path: str) -> list[list[str]]:
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.startswith("|") and not ln.startswith("|---")]
    return [[x.strip() for x in ln.strip("|").split("|")] for ln in lines]


def fig_equal_spend(root, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    for ax, g in zip(axes, ("s2_cifar", "s2_femnist")):
        t = md_table(os.path.join(root, g, "budget.md"))
        head, body = t[0], t[1:]
        rows = sorted(((r[0], float(r[1])) for r in body), key=lambda x: x[1])
        y = np.arange(len(rows))
        ax.barh(y, [v for _, v in rows], color=[FAMILY_COLOR[family(c)] for c, _ in rows],
                height=0.7, edgecolor="white", lw=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels([nice(c) for c, _ in rows])
        for tl, (c, _) in zip(ax.get_yticklabels(), rows):
            if c == "c2_rl_fedprox":
                tl.set_fontweight("bold")
        lo = min(v for _, v in rows)
        ax.set_xlim(lo - 0.05, max(v for _, v in rows) + 0.02)
        ax.set_title(f"{SETTINGS[g]}: accuracy when everyone spends the same")
        ax.set_xlabel("mean accuracy at equal total cost (axis does not start at 0)")
        ax.grid(axis="y", visible=False)
    family_legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "fig7_equal_spend.png"), dpi=200)
    plt.close(fig)


VERDICT = {"significantly better": ("#008300", "o"), "significantly worse": ("#e34948", "s"),
           "inconclusive": ("#8a8984", "D"), "no verdict": ("#8a8984", "x")}


def fig_tests(root, out) -> None:
    with open(os.path.join(root, "s2_prespecified_tests.md")) as fh:
        text = fh.read()
    fams = {}
    for block in re.split(r"\n## Family ", text)[1:]:
        name = block.split()[0]
        rows = [ln for ln in block.splitlines() if ln.startswith("| s2_")]
        fams[name] = [[x.strip() for x in ln.strip("|").split("|")] for ln in rows]
    order = ["A", "A2", "B", "C", "C2", "D", "D2"]
    what = {"A": "worst-10% accuracy", "A2": "worst-10% client accuracy", "B": "failure share",
            "C": "accuracy", "C2": "failure share", "D": "accuracy", "D2": "failure share"}
    fig, axes = plt.subplots(len(order), 1, figsize=(9, 16),
                             gridspec_kw={"height_ratios": [len(fams[f]) + 1.5 for f in order]})
    for ax, f in zip(axes, order):
        rows = fams[f]
        for i, r in enumerate(rows):
            col, mk = VERDICT[r[9]]
            ax.scatter(float(r[5]), i, color=col, marker=mk, s=36, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.margins(x=0.06, y=0.15)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{SETTINGS[r[0]]}: {nice(r[1])} vs {nice(r[2])}" for r in rows],
                           fontsize=7)
        ax.invert_yaxis()
        better = "higher is better" if "accuracy" in what[f] else "lower is better"
        ax.set_title(f"Family {f}: difference in {what[f]} ({better})", fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
    hs = [plt.Line2D([], [], marker=m, ls="", color=c) for c, m in list(VERDICT.values())[:3]]
    fig.legend(hs, list(VERDICT)[:3], loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(out, "fig8_prespecified_tests.png"), dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "adv"))
    ap.add_argument("--out", default=os.path.join("figures", "s2"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()
    data = {g: load_group(args.root, g) for g in ("s2_cifar", "s2_femnist")}
    fig_final_accuracy(data, args.out)
    fig_curves(data, args.out)
    fig_fail_cost(data, args.out)
    fig_modes(data, args.out)
    fig_lockup(data, args.out)
    fig_mode_rule(data, args.out)
    fig_equal_spend(args.root, args.out)
    fig_tests(args.root, args.out)
    print("wrote", sorted(os.listdir(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
