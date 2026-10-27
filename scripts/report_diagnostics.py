#!/usr/bin/env python3
"""Read diagnostic sidecars and print, per run, how the round loop unfolded.

    python scripts/report_diagnostics.py results/diagnostic_v1/short
    python scripts/report_diagnostics.py results/diagnostic_v1/full --every 10 > full.md

For each run: a summary row, then a timeline that follows the chain
predictor -> gate/fallback -> client and mode choices -> failure reasons ->
reward and update -> next choices. Then the predictor's weights over time
and calibration by mode and client group. Calibration is computed on
invited clients only, so it is selection-biased and is not a population-wide
calibration.
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from afl.diagnostics import read  # noqa: E402

FEATURES = ["log1p(speed)", "log1p(bw)", "energy", "avail_hint", "success_rate",
            "tanh(since_success/10)", "mode=light", "mode=standard", "mode=priority"]


def load(base: str) -> dict:
    return {k: read(f"{base}.diag_{k}.jsonl")[1] for k in ("round", "invite", "update")}


def zero_streak(rounds) -> tuple[int, int | None]:
    """Longest run of rounds with invitations but no success, and where it starts."""
    best, cur, start, best_start = 0, 0, None, None
    for r in rounds:
        if r["invited"] and not r["succeeded"]:
            cur += 1
            start = r["t"] if cur == 1 else start
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def share(counts: dict) -> str:
    n = sum(counts.values()) or 1
    return "/".join(f"{100 * counts.get(m, 0) / n:.0f}" for m in ("light", "standard", "priority"))


def summary(name: str, d: dict) -> str:
    R, I = d["round"], d["invite"]
    K = max((len(r["invited"]) for r in R), default=0)
    succ = sum(len(r["succeeded"]) for r in R)
    inv = sum(len(r["invited"]) for r in R)
    streak, start = zero_streak(R)
    full_fill = next((r["t"] for r in R if K and r["gate_filled"] >= K), None)
    modes: dict = {}
    reasons: dict = {}
    for r in R:
        for m, c in r["mode_counts"].items():
            modes[m] = modes.get(m, 0) + c
        for m, c in r["reason_counts"].items():
            reasons[m] = reasons.get(m, 0) + c
    fb = sum(x["how"] == "fallback" for x in I)
    return (f"| {name} | {len(R)} | {succ}/{inv} | {fb}/{inv} | {full_fill or '-'} | "
            f"{streak}{f' (from r{start})' if streak else ''} | {share(modes)} | "
            f"{', '.join(f'{k} {v}' for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))} | "
            f"{R[-1]['accuracy']:.3f} |")


def timeline(d: dict, every: int) -> list[str]:
    out = ["| t | online | elig | fill | inv | succ | modes L/S/P | reasons | p_max med/max (cands) "
           "| pred seen, w | z_acc z_cost z_lat dmg | shared | total | adv | grad |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    upd = {u["t"]: u for u in d["update"]}
    for r in d["round"]:
        t = r["t"]
        if t > 30 and t % every:
            continue
        pm = r["p_max"]
        cands = r["candidates"]
        mc = r["mode_counts"]
        rw = r["reward"]
        z = rw["z"]
        u = upd.get(t, {})
        out.append(
            f"| {t} | {r['online']} | {r['eligible']}/{cands} | {r['gate_filled']} | {len(r['invited'])} | "
            f"{len(r['succeeded'])} | {mc.get('light', 0)}/{mc.get('standard', 0)}/{mc.get('priority', 0)} | "
            f"{' '.join(f'{k}:{v}' for k, v in sorted(r['reason_counts'].items()))} | "
            f"{st.median(pm):.2f}/{max(pm):.2f} | {r['predictor']['seen']}, {r['predictor']['w_norm']:.2f} | "
            f"{z['acc']:+.2f} {z['cost']:+.2f} {z['lat']:+.2f} {z['damage_term']:+.2f} | "
            f"{rw['shared_reward']:+.2f} | {rw['total']:+.2f} | "
            f"{u.get('advantage', float('nan')):+.2f} | {u.get('grad_norm', float('nan')):.2f} |")
    return out


def predictor_table(d: dict, every: int) -> list[str]:
    out = ["| t | seen | b | " + " | ".join(FEATURES) + " |",
           "|---|---|---|" + "---|" * len(FEATURES)]
    for r in d["round"]:
        if r["t"] in (1, 2, 3, 5, 10, 20, 30) or (r["t"] > 30 and r["t"] % (every * 5) == 0):
            p = r["predictor"]
            out.append(f"| {r['t']} | {p['seen']} | {p['b']:+.2f} | "
                       + " | ".join(f"{w:+.2f}" for w in p["w"]) + " |")
    return out


def calibration(d: dict) -> list[str]:
    """Mean predicted success for the executed mode vs observed success,
    on invited clients only (selection-biased)."""
    groups: dict = {}
    for x in d["invite"]:
        for key in (("mode", x["mode"]), ("group", "weak" if x["analysis_only"]["weak"] else "strong"),
                    ("how", x["how"])):
            g = groups.setdefault(key, [0, 0.0, 0])
            g[0] += 1
            g[1] += x["p_succ"][x["mode"]]
            g[2] += x["success"]
    out = ["| by | value | n | mean predicted | observed |", "|---|---|---|---|---|"]
    for (kind, val), (n, p, y) in sorted(groups.items()):
        out.append(f"| {kind} | {val} | {n} | {p / n:.3f} | {y / n:.3f} |")
    return out


def fallback_repeats(d: dict) -> str:
    """How often the same clients are filled in round after round."""
    prev, overlaps, ids = None, [], set()
    for r in d["round"]:
        f = set(r["fill"])
        ids |= f
        if prev is not None and f and prev:
            overlaps.append(len(f & prev) / len(f | prev))
        prev = f
    if not overlaps:
        return "no consecutive fallback rounds"
    return (f"distinct fallback clients over the run: {len(ids)}; mean Jaccard overlap of "
            f"consecutive fallback sets: {st.mean(overlaps):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="a diagnostic variant folder, e.g. results/diagnostic_v1/short")
    ap.add_argument("--every", type=int, default=10, help="after round 30, show every n-th round")
    args = ap.parse_args()
    bases = sorted(p[:-len(".diag_round.jsonl")]
                   for p in glob.glob(os.path.join(args.dir, "**", "*.diag_round.jsonl"), recursive=True))
    if not bases:
        print(f"no diagnostic sidecars under {args.dir}")
        return 1
    runs = {os.path.basename(b): load(b) for b in bases}
    print(f"# Diagnostics: {args.dir}\n")
    print("Labels: these are failure reproductions on already-inspected seeds, not tests. "
          "Calibration rows use invited clients only (selection-biased).\n")
    print("| run | rounds | successes/invites | fallback/invites | first all-fallback round "
          "| longest zero-success streak | modes % L/S/P | failure reasons | final acc |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, d in runs.items():
        print(summary(name, d))
    for name, d in runs.items():
        print(f"\n## {name}\n")
        print(fallback_repeats(d) + "\n")
        print("\n".join(timeline(d, args.every)))
        print("\nPredictor weights (shared logistic model):\n")
        print("\n".join(predictor_table(d, args.every)))
        print("\nCalibration on invited clients:\n")
        print("\n".join(calibration(d)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
