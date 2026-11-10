#!/usr/bin/env python3
"""Accuracy against what was actually spent (fourth external review, finding 6).

    python scripts/cost_curves.py --out results/rl2_screen_b/cost \
        --series rl_light=results/rl2_screen_b/full/s2_femnist:rl_light \
        --series ppo_full_mimic=results/rl2_screen_b/full/s2_femnist:ppo_full_mimic

For every series (a folder and a condition; all its seeds): the median, over
seeds, of fixed-set accuracy at the last round whose cumulative cost is at or
below each common budget, the median actual spend at that round, and how many
seeds got that far; plus the development endpoint eq5 with the actual spend
of its 5 rounds disclosed. Writes <out>.md (the table) and <out>.png (median
accuracy vs cumulative cost). Cost is in the simulator's units (see
docs/SIMULATOR_MODEL.md), not joules or dollars. Pre-training cost is not in
these curves; it is reported separately.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st

BUDGETS = (500, 800, 1100, 1400, 1700, 2000)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # validated categorical slots 1-4 (light)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def runs(spec: str) -> dict[int, tuple[list[float], list[float]]]:
    folder, cond = spec.rsplit(":", 1)
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, f"{cond}_s*.csv"))):
        rows = list(csv.DictReader(open(p)))
        cum, acc, c = [], [], 0.0
        for r in rows:
            c += float(r["cost"])
            cum.append(c)
            acc.append(float(r["accuracy_fixed"]))
        out[int(os.path.basename(p)[:-4].rsplit("_s", 1)[1])] = (cum, acc)
    if not out:
        raise SystemExit(f"no runs for {spec}")
    return out


def at_budget(cum, acc, b):
    """(accuracy, actual spend) at the last round with cumulative cost <= b, or None."""
    k = max((i for i, c in enumerate(cum) if c <= b), default=None)
    return None if k is None else (acc[k], cum[k])


def eq5(cum, acc, b=1400.0):
    ks = [i for i, c in enumerate(cum) if c <= b][-5:]
    return (st.mean(acc[i] for i in ks), st.mean(cum[i] for i in ks)) if ks else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True, help="LABEL=FOLDER:CONDITION (up to 4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Accuracy against cumulative cost")
    args = ap.parse_args()
    if len(args.series) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} series per chart")
    data = {s.split("=", 1)[0]: runs(s.split("=", 1)[1]) for s in args.series}
    lines = [f"# {args.title}", "",
             "Median over seeds of fixed-set accuracy at the last round with cumulative cost <= the budget "
             "(simulated cost units), the median actual spend at that round, and the seeds that reached it. "
             "eq5 (development endpoint): mean accuracy of the last 5 rounds within 1,400, with the mean "
             "actual spend of those rounds.", "",
             "| series | seeds | " + " | ".join(f"B={b}: acc (spend) [n]" for b in BUDGETS) + " | eq5 (spend) | total spend |",
             "|---|---|" + "---|" * (len(BUDGETS) + 2)]
    for label, rs in data.items():
        cells = []
        for b in BUDGETS:
            hits = [h for h in (at_budget(c, a, b) for c, a in rs.values()) if h]
            cells.append(f"{st.median(h[0] for h in hits):.3f} ({st.median(h[1] for h in hits):.0f}) [{len(hits)}]"
                         if hits else "-")
        e = [x for x in (eq5(c, a) for c, a in rs.values()) if x]
        lines.append(f"| {label} | {len(rs)} | " + " | ".join(cells) + f" | {st.median(x[0] for x in e):.3f} "
                     f"({st.median(x[1] for x in e):.0f}) | {st.median(c[-1] for c, _ in rs.values()):.0f} |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    top = min(max(c[-1] for c, _ in rs.values()) for rs in data.values())
    grid = [top * k / 200 for k in range(1, 201)]
    ends = []
    for (label, rs), color in zip(data.items(), COLORS):
        ys = []
        for g in grid:
            hits = [h[0] for h in (at_budget(c, a, g) for c, a in rs.values()) if h]
            ys.append(st.median(hits) if len(hits) == len(rs) else float("nan"))
        ax.plot(grid, ys, color=color, linewidth=2, label=label)
        last = max(i for i, y in enumerate(ys) if y == y)
        ends.append([ys[last], grid[last], label])
    # end labels, pushed apart so they never overlap
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y, x, label in ends:
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    for b in BUDGETS:
        if b <= top:
            ax.axvline(b, color=GRID, linewidth=1, linestyle="--", zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("cumulative cost (simulated units)", color=INK, fontsize=9)
    ax.set_ylabel("fixed-set accuracy (median over seeds)", color=INK, fontsize=9)
    ax.set_title(args.title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    ax.set_xlim(0, top * 1.12)
    fig.tight_layout()
    fig.savefig(args.out + ".png", facecolor=SURFACE)
    print(f"wrote {args.out}.md and {args.out}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
