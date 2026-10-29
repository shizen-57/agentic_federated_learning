#!/usr/bin/env python3
"""Is the success predictor calibrated, and does the safety layer recover
from pessimism? (fourth external review, finding 5; docs/VALIDATION.md)

    python scripts/predictor_stress.py --out results/predictor_stress --jobs 3
    python scripts/predictor_stress.py --dataset cifar --out runs/predictor_stress_cifar --set fl.device=auto

The tuned method (configs/rl2_tuned.json) on real FEMNIST or CIFAR-10 (s2
settings, persistent split, development pool, 200 rounds), seeds 290-292, in two
worlds: `stable`, and `shock` -- at round 80 half the devices (fixed per
seed) lose 98% of their bandwidth and 95% of their battery capacity; at
round 140 they are restored. Every executed job's predicted success (for the
mode that ran) and outcome are recorded, with the round's abstentions,
Light overrides and gate fills. Reported: calibration (ECE, 10 bins; Brier;
mean predicted vs observed) overall, by mode, weak vs strong clients, a
client's first 3 invitations vs later ones, and before / during / after the
shock for hit and untouched clients; the exceptions per period; and recovery
-- how soon restored clients are invited again and how they then fare, and
how many clients are never invited again (avoided for good). For those, the
simulator's ground truth (env.oracle_eligible, before any job runs, same
random numbers) says how often they could have succeeded: avoided rightly or
wrongly.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

SEEDS = (290, 291, 292)
ROUNDS, SHOCK, RESTORE = 200, 80, 140
PERIODS = (("before", 1, SHOCK), ("shock", SHOCK, RESTORE), ("after", RESTORE, ROUNDS + 1))


GROUPS = {"femnist": "s2_femnist", "cifar": "s2_cifar"}


def config(seed: int, group: str = "s2_femnist", sets: tuple = ()) -> dict:
    import rl2lib
    from sweep import build_jobs
    with open(os.path.join(ROOT, "configs", "rl2_tuned.json")) as fh:
        tuned = json.load(fh)
    src = rl2lib.source_group("configs/sweep_advanced.yaml", group)
    e = rl2lib.group_entry(src, {"data.split": "persistent"}, ROUNDS, [tuned["base"]], [seed])
    e["override"] = rl2lib.resolve(e, tuned["overrides"])
    return build_jobs({"seeds": [seed], "sets": {}, "groups": {"g": e}}, None, None, "/tmp/afl_stress",
                      list(sets))[0]["raw"]


def episode(args) -> dict:
    world, seed, group, sets = args
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from afl.config import Cfg
    from afl.engine import Experiment
    exp = Experiment(Cfg.from_dict(config(seed, group, sets)))
    env, pol = exp.env, exp.policy
    hit = np.random.default_rng([seed, 99]).permutation(env.N)[: env.N // 2]
    hit_set = set(int(i) for i in hit)
    bw0, cap0 = env.bandwidth.copy(), env.energy_cap.copy()
    jobs, rounds, now, oracle, could = [], [], [0], [False], [{}]
    count: dict[int, int] = {}
    real_round, real_out, real_oracle, real_run = env.new_round, env.outcome, env.oracle_eligible, exp.run_round

    def new_round(t):
        if world == "shock" and t == SHOCK:
            env.bandwidth[hit] *= 0.02
            env.energy_cap[hit] *= 0.05
            env.energy[hit] = np.minimum(env.energy[hit], env.energy_cap[hit])
        if world == "shock" and t == RESTORE:
            env.bandwidth[:], env.energy_cap[:] = bw0, cap0
        real_round(t)
        # ground truth before any job runs (pure; same random numbers): could
        # this client have succeeded in some mode, and in Light, this round?
        oracle[0] = True
        try:
            could[0] = {int(i): (int(real_oracle(int(i))), int(real_out(int(i), "light")["success"]))
                        for i in np.where(env._draw["available"])[0]}
        finally:
            oracle[0] = False

    def oracle_eligible(i):
        oracle[0] = True
        try:
            return real_oracle(i)
        finally:
            oracle[0] = False

    def outcome(i, mode, frac=None):
        out = real_out(i, mode, frac)
        if not oracle[0]:
            count[i] = count.get(i, 0) + 1
            jobs.append({"t": now[0], "i": int(i), "mode": mode, "p": float(pol._p_round[i][mode]),
                         "ok": int(out["success"]), "weak": bool(env.weak[i]), "k": count[i],
                         "hit": int(i) in hit_set})
        return out

    def run_round(t):
        now[0] = t
        row = real_run(t)
        allowed = pol.allowed_modes or {}
        online = np.where(env._draw["available"])[0]
        rounds.append({"t": t, "abstain": row["n_abstain"], "override": row["n_light_override"],
                       "fill": row["gate_filled"], "selected": row["n_selected"],
                       "online": [int(i) for i in online],
                       "ruled_out": [int(i) for i in online if not allowed.get(int(i))],
                       "could": {i: c[0] for i, c in could[0].items()},
                       "light_ok": {i: c[1] for i, c in could[0].items()}})
        return row
    env.new_round, env.outcome, env.oracle_eligible, exp.run_round = new_round, outcome, oracle_eligible, run_round
    exp.run(verbose=False)
    return {"world": world, "seed": seed, "jobs": jobs, "rounds": rounds, "hit": sorted(hit_set), "N": env.N,
            "weak": [int(i) for i in np.where(env.weak)[0]]}


def calib(jobs: list[dict]) -> dict:
    if not jobs:
        return {"n": 0}
    import numpy as np
    p = np.array([j["p"] for j in jobs])
    y = np.array([j["ok"] for j in jobs])
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10)
              if (bins == b).any()) / len(p)
    return {"n": len(p), "mean_p": float(p.mean()), "rate": float(y.mean()), "ece": float(ece),
            "brier": float(((p - y) ** 2).mean())}


def report(res: list[dict]) -> tuple[str, dict]:
    out, L = {}, ["# Predictor and safety-layer stress test", "",
                  "Tuned method, real FEMNIST, seeds " + ", ".join(map(str, SEEDS)) + f", {ROUNDS} rounds. "
                  f"Shock world: half the devices lose 98% bandwidth and 95% battery at round {SHOCK}, "
                  f"restored at {RESTORE}. Calibration of the executed job's predicted success.", ""]
    for world in ("stable", "shock"):
        js = [j for r in res if r["world"] == world for j in r["jobs"]]
        groups = {"all": js}
        groups.update({f"mode {m}": [j for j in js if j["mode"] == m] for m in ("light", "standard", "priority")})
        groups.update({"weak clients": [j for j in js if j["weak"]], "strong clients": [j for j in js if not j["weak"]],
                       "first 3 invitations": [j for j in js if j["k"] <= 3],
                       "later invitations": [j for j in js if j["k"] > 3]})
        for name, a, b in PERIODS:
            for h in ((True, "hit"), (False, "untouched")) if world == "shock" else ((None, "all"),):
                sel = [j for j in js if a <= j["t"] < b and (h[0] is None or j["hit"] == h[0])]
                groups[f"{name} ({h[1]})"] = sel
        out[world] = {k: calib(v) for k, v in groups.items()}
        L += [f"## {world}", "", "| group | jobs | mean predicted | observed | ECE | Brier |",
              "|---|---|---|---|---|---|"]
        for k, c in out[world].items():
            L.append(f"| {k} | {c['n']} | " + (f"{c['mean_p']:.3f} | {c['rate']:.3f} | {c['ece']:.3f} | "
                                                f"{c['brier']:.3f} |" if c["n"] else "- | - | - | - |"))
        L += ["", "Exceptions per round (mean over seeds and rounds):", ""]
        for name, a, b in PERIODS:
            rr = [x for r in res if r["world"] == world for x in r["rounds"] if a <= x["t"] < b]
            L.append(f"- {name}: abstain {st.mean(x['abstain'] for x in rr):.2f}, Light override "
                     f"{st.mean(x['override'] for x in rr):.2f}, gate fill {st.mean(x['fill'] for x in rr):.2f}, "
                     f"invited {st.mean(x['selected'] for x in rr):.2f}")
        never = [r["N"] - len({j["i"] for j in r["jobs"] if j["t"] >= ROUNDS - 60}) for r in res if r["world"] == world]
        L.append(f"- clients not invited at all in the last 60 rounds: {never} of {res[0]['N']} per seed")
        # Why? For each client not invited in the last 60 rounds: how often it
        # was online there, and of those rounds how often the safety layer had
        # ruled it out (no allowed job) -- "avoided for good" vs "not chosen".
        av = []
        for r in (r for r in res if r["world"] == world):
            late = [x for x in r["rounds"] if x["t"] > ROUNDS - 60]
            invited = {j["i"] for j in r["jobs"] if j["t"] > ROUNDS - 60}
            for i in range(r["N"]):
                if i in invited:
                    continue
                on = sum(i in x["online"] for x in late)
                out_ = sum(i in x["ruled_out"] for x in late)
                ok = sum(x["could"].get(i, 0) for x in late)
                av.append({"weak": i in r["weak"], "online": on, "ruled_out_share": out_ / on if on else None,
                           "could_share": ok / on if on else None})
        if av:
            ro = [a["ruled_out_share"] for a in av if a["ruled_out_share"] is not None]
            stuck = [a for a in av if a["ruled_out_share"] is not None and a["ruled_out_share"] >= 0.9]
            cs = [a["could_share"] for a in stuck]
            out[world + "_avoided"] = {"n": len(av), "weak_share": sum(a["weak"] for a in av) / len(av),
                                       "median_online_rounds": st.median(a["online"] for a in av),
                                       "ruled_out_90pct_or_more": len(stuck),
                                       "median_ruled_out_share": st.median(ro) if ro else None,
                                       "stuck_median_could_share": st.median(cs) if cs else None,
                                       "stuck_could_half_or_more": sum(1 for x in cs if x >= 0.5)}
            a = out[world + "_avoided"]
            L.append(f"- of those {a['n']} clients (all seeds): {a['weak_share']:.0%} weak; median {a['median_online_rounds']} "
                     f"of 60 rounds online; ruled out by the safety layer in a median {a['median_ruled_out_share']:.0%} "
                     f"of their online rounds; {a['ruled_out_90pct_or_more']} ruled out in >= 90% (avoided for good)")
            if cs:
                L.append(f"- were they right to be avoided? Of the {len(stuck)} avoided for good, some mode would have "
                         f"succeeded in a median {a['stuck_median_could_share']:.0%} of their online rounds "
                         f"(simulator ground truth); {a['stuck_could_half_or_more']} could have succeeded in at least "
                         f"half of them (avoided wrongly)")
        # every late online client-round the safety layer ruled out: how often
        # was a job actually possible (false abstention)?
        ro_r = [(x["could"].get(i, 0), x["light_ok"].get(i, 0)) for r in res if r["world"] == world
                for x in r["rounds"] if x["t"] > ROUNDS - 60 for i in x["ruled_out"]]
        ok_r = [x["could"].get(i, 0) for r in res if r["world"] == world for x in r["rounds"]
                if x["t"] > ROUNDS - 60 for i in x["online"] if i not in x["ruled_out"]]
        if ro_r:
            out[world + "_ruled_out_rounds"] = {"n": len(ro_r), "could": st.mean(c for c, _ in ro_r),
                                                "light_ok": st.mean(lt for _, lt in ro_r),
                                                "allowed_could": st.mean(ok_r) if ok_r else None}
            b = out[world + "_ruled_out_rounds"]
            L.append(f"- last 60 rounds, every online client-round the safety layer ruled out ({b['n']}): a job was "
                     f"possible in {b['could']:.0%} (Light alone in {b['light_ok']:.0%}); for allowed client-rounds "
                     f"{b['allowed_could']:.0%}")
        if world == "shock":
            lags, back, rate_b, rate_a = [], [], [], []
            for r in (r for r in res if r["world"] == "shock"):
                for i in r["hit"]:
                    after = [j["t"] for j in r["jobs"] if j["i"] == i and j["t"] >= RESTORE]
                    back.append(bool(after))
                    if after:
                        lags.append(after[0] - RESTORE)
                rate_b += [j["ok"] for j in r["jobs"] if j["hit"] and j["t"] < SHOCK]
                rate_a += [j["ok"] for j in r["jobs"] if j["hit"] and j["t"] >= RESTORE + 20]
            out["recovery"] = {"hit_invited_again": sum(back) / len(back), "median_lag": st.median(lags) if lags else None,
                               "rate_before": st.mean(rate_b), "rate_after": st.mean(rate_a) if rate_a else None}
            rc = out["recovery"]
            L += ["", f"Recovery of the hit clients after round {RESTORE}: {rc['hit_invited_again']:.0%} invited "
                  f"again; median wait {rc['median_lag']} rounds; success rate {rc['rate_before']:.3f} before the "
                  f"shock vs {rc['rate_after']:.3f} from round {RESTORE + 20} on."]
        L.append("")
    return "\n".join(L), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--dataset", choices=sorted(GROUPS), default="femnist")
    ap.add_argument("--set", action="append", default=[], help="operational override, e.g. fl.device=auto")
    args = ap.parse_args()
    import rl2lib
    if rl2lib.non_operational(args.set):
        raise SystemExit("--set may only be operational")
    import multiprocessing as mp
    tasks = [(w, s, GROUPS[args.dataset], tuple(args.set)) for w in ("stable", "shock") for s in SEEDS]
    with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn")) as pool:
        res = list(pool.map(episode, tasks))
    text, summary = report(res)
    text = text.replace("real FEMNIST", f"real {'CIFAR-10' if args.dataset == 'cifar' else 'FEMNIST'}", 1)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
