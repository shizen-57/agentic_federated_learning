#!/usr/bin/env python3
"""How good is each success predictor? (docs/REVIEW_5.md, 3.2 and P3)

No model is trained and no accuracy is measured: only the simulator's
device/availability/energy mechanics and the predictor run, so this takes
seconds. Each round K online clients are invited and their outcomes teach
the predictor, exactly as in a run. Two ways of inviting:

  random   random clients, random modes -- every predictor version sees the
           same invitations, so the comparison is exact;
  rule     random clients, each given the expected-work mode of the
           predictor under test (base.best_work_mode) -- how the `_ew`
           baselines use it.

Reported per predictor version: Brier score and calibration error of its
predictions for the invited jobs; predicted against observed success by mode
and by client group; and what its expected-work rule would do if applied to
EVERY online client each round (judged with the round's true outcomes): the
share of each mode, the share of jobs that would succeed, and the epochs of
training delivered per unit of cost.

    python scripts/predictor_check.py --dataset femnist
    python scripts/predictor_check.py --dataset cifar --set system.drop_model=exposure
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

VERSIONS = ("v1", "v2", "v3")
GROUPS = ("slow", "middle", "fast", "weak")
RULES = ("always Light", "expected work (the baselines' rule)", "expected work per unit cost",
         "oracle: the heaviest job that finishes")


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p), np.asarray(y, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def one_run(dataset: str, seed: int, version: str, how: str, sets: list[str], rounds: int) -> dict:
    from sim_truth import raw_config
    from afl.config import Cfg
    from afl.data import build as build_data
    from afl.policies.base import best_work_mode, est_time
    from afl.predictor import SuccessPredictor, SuccessPredictorV2, SuccessPredictorV3
    from afl.sim import ClientEnv
    extra = [f"policy.predictor={version}"] + (["policy.pred_decay=0.98"] if version != "v1" else [])
    cfg = Cfg.from_dict(raw_config(dataset, seed, [*extra, *sets]))
    d = build_data(cfg)
    env = ClientEnv(cfg, d["n_i"], cfg.seed, sketch=d["sketch"])
    N, K, modes = env.N, cfg.fl.clients_per_round, list(cfg.system.modes)
    pred = {"v1": SuccessPredictor, "v2": SuccessPredictorV2, "v3": SuccessPredictorV3}[version](cfg)
    v2 = version != "v1"
    grp = np.where(env.weak, "weak", np.array(GROUPS[:3])[env.tier])
    rng = np.random.default_rng(seed + 9100)          # the scheduler's own stream
    x_sel, x_succ, since = np.zeros(N), np.zeros(N), np.zeros(N)
    jobs = []                                          # (predicted, ok, mode, group)
    rule = {m: 0 for m in modes}
    rule_ok = rule_epochs = rule_cost = rule_n = 0.0
    # other ways of choosing the job, judged the same way (RULES)
    alt = {r: {"n": 0, "ok": 0, "epochs": 0.0, "cost": 0.0, **{m: 0 for m in modes}} for r in RULES}
    ep = cfg.system.local_epochs
    for t in range(1, rounds + 1):
        env.new_round(t)
        obs = {i: env.observe(i) for i in range(N)}
        hist = {i: {"success_rate": float(x_succ[i] / max(x_sel[i], 1.0)),
                    "rounds_since_success": float(since[i])} for i in range(N)}
        if v2:
            pred.t = t
            pred.set_reference(np.array([pred.client_features(obs[i], hist[i]) for i in range(N)]))
        if version == "v3":
            st = {"obs": obs, "n_work": env.n_work}
            pred.set_jobs({i: {m: est_time(cfg, st, i, m) for m in modes} for i in range(N)}, obs, env.n_work)
        p = {i: {m: (pred.predict_client(i, obs[i], hist[i], m)[0] if v2 else pred.predict(obs[i], hist[i], m))
                 for m in modes} for i in range(N)}
        online = [int(i) for i in np.where(env._draw["available"])[0]]
        # the expected-work rule on every online client, judged by the truth
        if t > rounds // 4:                            # after the predictor's warm-up
            for i in online:
                m = best_work_mode(cfg, p[i])
                ok = env.outcome(i, m)["success"]
                rule[m] += 1
                rule_n += 1
                rule_ok += ok
                rule_epochs += ok * cfg.system.local_epochs[m]
                rule_cost += env.cost(i, m)
                truth = {mm: env.outcome(i, mm)["success"] for mm in modes}
                choice = {
                    "always Light": modes[0],
                    "expected work (the baselines' rule)": m,
                    "expected work per unit cost": max(modes, key=lambda mm: p[i][mm] * ep[mm] / env.cost(i, mm)),
                    "oracle: the heaviest job that finishes": next(
                        (mm for mm in reversed(modes) if truth[mm]), None),
                }
                for r, mm in choice.items():
                    a = alt[r]
                    if mm is None:               # the oracle invites nobody who cannot finish
                        continue
                    a["n"] += 1
                    a[mm] += 1
                    a["ok"] += truth[mm]
                    a["epochs"] += truth[mm] * ep[mm]
                    a["cost"] += env.cost(i, mm)
        order = rng.permutation(len(online))
        picks = [online[k] for k in order[:K]]
        rand_modes = [modes[k] for k in rng.integers(0, len(modes), size=len(picks))]
        spend = []
        for i, rm in zip(picks, rand_modes):
            m = rm if how == "random" else best_work_mode(cfg, p[i])
            out = env.outcome(i, m)
            ok = int(out["success"])
            jobs.append((p[i][m], ok, m, grp[i], t))
            x_sel[i] += 1
            if out["reason"] not in ("unavailable", "energy"):
                spend.append((i, m))
            if v2:
                pred.update_client(i, obs[i], hist[i], m, ok)
            else:
                pred.update(obs[i], hist[i], m, ok)
            x_succ[i] += ok
            since[i] = 0 if ok else since[i]
        since += 1
        for i, m in spend:
            env.spend(i, m)
    late = [j for j in jobs if j[4] > rounds // 4]
    P, Y = np.array([j[0] for j in late]), np.array([j[1] for j in late])
    by_mode = {m: (float(P[[j[2] == m for j in late]].mean()), float(Y[[j[2] == m for j in late]].mean()),
                   int(sum(j[2] == m for j in late))) for m in modes if any(j[2] == m for j in late)}
    by_group = {g: (float(P[[j[3] == g for j in late]].mean()), float(Y[[j[3] == g for j in late]].mean()))
                for g in GROUPS if any(j[3] == g for j in late)}
    return {"brier": float(((P - Y) ** 2).mean()), "ece": ece(P, Y), "by_mode": by_mode, "by_group": by_group,
            "rule_share": {m: rule[m] / max(rule_n, 1) for m in modes}, "rule_ok": rule_ok / max(rule_n, 1),
            "rule_epochs_per_cost": rule_epochs / max(rule_cost, 1e-9),
            "rules": {r: {"share": {m: a[m] / max(a["n"], 1) for m in modes}, "ok": a["ok"] / max(a["n"], 1),
                          "epochs_per_cost": a["epochs"] / max(a["cost"], 1e-9),
                          "jobs_per_cost": a["ok"] / max(a["cost"], 1e-9)} for r, a in alt.items()}}


def check(dataset: str, seeds: list[int], sets: list[str], rounds: int) -> dict:
    out = {"dataset": dataset, "seeds": seeds, "set": sets, "rounds": rounds, "how": {}}
    for how in ("random", "rule"):
        out["how"][how] = {}
        for v in VERSIONS:
            runs = [one_run(dataset, s, v, how, sets, rounds) for s in seeds]
            mean = lambda f: float(np.mean([f(r) for r in runs]))  # noqa: E731
            modes = sorted({m for r in runs for m in r["by_mode"]}, key=("light", "standard", "priority").index)
            out["how"][how][v] = {
                "brier": mean(lambda r: r["brier"]), "ece": mean(lambda r: r["ece"]),
                "by_mode": {m: [float(np.mean([r["by_mode"][m][k] for r in runs if m in r["by_mode"]]))
                                for k in (0, 1)] for m in modes},
                "by_group": {g: [float(np.mean([r["by_group"][g][k] for r in runs if g in r["by_group"]]))
                                 for k in (0, 1)] for g in GROUPS},
                "rule_share": {m: mean(lambda r, m=m: r["rule_share"][m]) for m in runs[0]["rule_share"]},
                "rule_ok": mean(lambda r: r["rule_ok"]),
                "rule_epochs_per_cost": mean(lambda r: r["rule_epochs_per_cost"]),
                "rules": {name: {"share": {m: mean(lambda r, m=m: r["rules"][name]["share"][m])
                                           for m in runs[0]["rule_share"]},
                                 **{k: mean(lambda r, k=k: r["rules"][name][k])
                                    for k in ("ok", "epochs_per_cost", "jobs_per_cost")}} for name in RULES}}
    return out


def render(res: dict) -> str:
    L = [f"# Predictor check: {res['dataset']}, seeds {res['seeds']}, {res['rounds']} rounds"
         + (f", with {res['set']}" if res["set"] else ""), "",
         "Predicted / observed success of the invited jobs after the first quarter of the run. "
         "Lower Brier and calibration error are better.", ""]
    for how, label in (("random", "Random clients, random modes (same invitations for every version)"),
                       ("rule", "Random clients, the version's own expected-work mode")):
        L += [f"**{label}**", "",
              "| predictor | Brier | calib. error | Light pred / obs | Standard pred / obs | Priority pred / obs |"
              " weak clients pred / obs | slow devices pred / obs |",
              "|---|---|---|---|---|---|---|---|"]
        for v, r in res["how"][how].items():
            pm = lambda m: (f"{r['by_mode'][m][0]:.2f} / {r['by_mode'][m][1]:.2f}"  # noqa: E731
                            if m in r["by_mode"] else "-")
            pg = lambda g: f"{r['by_group'][g][0]:.2f} / {r['by_group'][g][1]:.2f}"  # noqa: E731
            L.append(f"| {v} | {r['brier']:.3f} | {r['ece']:.3f} | {pm('light')} | {pm('standard')} | "
                     f"{pm('priority')} | {pg('weak')} | {pg('slow')} |")
        L.append("")
    L += ["**The expected-work rule applied to every online client** (judged by each round's true outcome; "
          "from the invitations of the `rule` case)", "",
          "| predictor | Light / Standard / Priority share | jobs that would succeed | epochs delivered per unit cost |",
          "|---|---|---|---|"]
    for v, r in res["how"]["rule"].items():
        s = r["rule_share"]
        L.append(f"| {v} | {s['light']:.0%} / {s['standard']:.0%} / {s['priority']:.0%} | {r['rule_ok']:.0%} | "
                 f"{r['rule_epochs_per_cost']:.2f} |")
    L += ["", "**Ways of choosing the job, applied to every online client** (predictor v2; judged by each "
          "round's true outcome; the oracle skips clients that cannot finish any job)", "",
          "| rule | Light / Standard / Priority share | jobs that succeed | epochs delivered per unit cost | "
          "successful jobs per unit cost |", "|---|---|---|---|---|"]
    for name, r in res["how"]["rule"]["v2"]["rules"].items():
        s = r["share"]
        L.append(f"| {name} | {s['light']:.0%} / {s['standard']:.0%} / {s['priority']:.0%} | {r['ok']:.0%} | "
                 f"{r['epochs_per_cost']:.2f} | {r['jobs_per_cost']:.2f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="femnist", choices=("femnist", "cifar"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[403, 404, 405])
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. system.drop_model=exposure")
    ap.add_argument("--out", help="folder for report.md and result.json")
    args = ap.parse_args()
    import seeds as seedreg
    seedreg.check(args.seeds, "dev")
    res = check(args.dataset, args.seeds, args.set, args.rounds)
    text = render(res)
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.md"), "w") as fh:
            fh.write(text)
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
