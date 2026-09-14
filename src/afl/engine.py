"""The round loop (Sec. 3.4).

One round is: observe -> predict success per (client, mode) -> gate eligibility
-> policy acts -> local training -> acceptance -> aggregation -> metrics ->
reward -> policy learns. Every quantity the experimental design asks for
(Sec. 3.23) is written to a per-round CSV so the plots in Chapter 5 are produced
from the log rather than recomputed.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M
from .aggregate import aggregate
from .data import build as build_data
from .local import accepted, local_train
from .models import build_model, get_flat, set_flat
from .policies import build_policy
from .predictor import SuccessPredictor
from .sim import ClientEnv

LOG_FIELDS = [
    "round", "accuracy", "test_loss", "reward", "cost", "latency",
    "n_selected", "n_success", "fail_ratio", "arl_online", "arl_oracle",
    "cv", "cv_selection", "jain", "ece", "mode_light", "mode_standard",
    "mode_priority", "wall_s",
]


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.d = build_data(cfg)
        self.N = cfg.data.num_clients
        self.env = ClientEnv(cfg, self.d["n_i"], cfg.seed)
        self.predictor = SuccessPredictor(cfg)
        self.policy = build_policy(cfg)

        self.model = build_model(cfg, self.d["dim"], self.d["num_classes"])
        self.global_flat = get_flat(self.model)

        self.x_succ = np.zeros(self.N)
        self.x_sel = np.zeros(self.N)
        self.since_success = np.zeros(self.N)
        self.issued_round = np.zeros(self.N, dtype=int)
        self.local_loss = np.ones(self.N)     # refreshed lazily, see _refresh_losses
        self.gen = torch.Generator().manual_seed(cfg.seed + 3)
        self.last_update: dict[int, torch.Tensor] = {}  # MIFA/MimiC substitution store

        self.norm = {k: M.Normalizer() for k in ("acc", "cost", "lat", "arl", "faildmg")}
        self.cal_probs: list[float] = []
        self.cal_out: list[int] = []
        self.rows: list[dict] = []

    # ------------------------------------------------------------ helpers

    def _client_xy(self, i: int):
        idx = self.d["parts"][i]
        return self.d["x_train"][idx], self.d["y_train"][idx]

    def _refresh_losses(self, clients: list[int]) -> None:
        """Update the cached local loss for the clients we are about to touch.

        Power-of-Choice and Oort both need a local loss signal. Evaluating every
        client every round would be the dominant cost of the whole experiment,
        so only selected clients are refreshed -- which is also what a real
        server could observe.
        """
        set_flat(self.model, self.global_flat)
        self.model.eval()
        with torch.no_grad():
            for i in clients:
                x, y = self._client_xy(i)
                if len(y) == 0:
                    continue
                out = self.model(torch.from_numpy(x[:128]))
                self.local_loss[i] = float(F.cross_entropy(out, torch.from_numpy(y[:128])))

    def _evaluate(self) -> tuple[float, float]:
        set_flat(self.model, self.global_flat)
        self.model.eval()
        x = torch.from_numpy(self.d["x_test"])
        y = torch.from_numpy(self.d["y_test"])
        with torch.no_grad():
            out = self.model(x)
            loss = float(F.cross_entropy(out, y))
            acc = float((out.argmax(1) == y).float().mean())
        return acc, loss

    def _state(self, t: int) -> dict:
        obs = {i: self.env.observe(i) for i in range(self.N)}
        hist = {
            i: {
                "success_rate": float(self.x_succ[i] / max(self.x_sel[i], 1.0)),
                "rounds_since_success": float(self.since_success[i]),
                "selected_frac": float(self.x_sel[i] / max(t, 1)),
            }
            for i in range(self.N)
        }
        p_succ = {
            i: {m: self.predictor.predict(obs[i], hist[i], m) for m in self.cfg.system.modes}
            for i in range(self.N)
        }
        # e^elig_{i,t} (Eq. 3.26): at least one mode clears the threshold.
        eligible = {
            i: int(max(p_succ[i].values()) >= self.cfg.policy.elig_threshold)
            for i in range(self.N)
        }
        t_expected = {
            i: {m: sum(self.env.workload(i, m)[:2]) for m in self.cfg.system.modes}
            for i in range(self.N)
        }
        return {
            "obs": obs, "hist": hist, "p_succ": p_succ, "eligible": eligible,
            "t_expected": t_expected, "n_i": self.d["n_i"],
            "local_loss": self.local_loss,
            "repr_value": M.standalone_value(self.d["sketch"]),
        }

    # --------------------------------------------------------------- round

    def run_round(self, t: int) -> dict:
        t0 = time.time()
        self.env.new_round(t)
        state = self._state(t)

        selected, assign = self.policy.act(state, t)
        self._refresh_losses(selected)

        # V^repr_{i,t}(S_t) for every client (Eq. 3.26), not only the selected
        # ones: Eq. (3.27) sums ARL' over the full population C, so an
        # eligible, high-value client who was never invited must still show up
        # here with Y^succ = 0 via the success dict's default below.
        rep_vals = M.per_client_representation_value(self.d["sketch"], selected)
        values: dict[int, float] = {i: float(rep_vals[i]) for i in range(self.N)}

        updates, weights, steps, succ_p = [], [], [], []
        success: dict[int, int] = {}
        pending_spends: list[tuple[int, str]] = []
        round_cost = 0.0
        latencies = []
        mode_counts = {m: 0 for m in self.cfg.system.modes}

        for i in selected:
            mode = assign[i]
            mode_counts[mode] += 1
            self.x_sel[i] += 1
            # The server broadcasts the current global model with the invitation,
            # so the update is computed against round t. Staleness only bites in
            # the asynchronous variant, where this is set when the model is sent.
            self.issued_round[i] = t
            round_cost += self.env.cost(i, mode)
            out = self.env.outcome(i, mode)
            latencies.append(out["latency"])
            # Energy spend is deferred until after the oracle diagnostic below
            # is computed (Sec. 3.11, Eq. 3.29): applying it immediately would
            # make a client's own real energy draw down before its own
            # counterfactual is evaluated, so a client that genuinely could
            # have succeeded under the pre-decision energy state gets
            # incorrectly reported as oracle-ineligible.
            pending_spends.append((i, mode))

            ok = out["success"] == 1
            if ok:
                x, y = self._client_xy(i)
                if len(y) == 0:
                    ok = False
                else:
                    upd, n_steps = local_train(
                        self.cfg, self.global_flat, x, y,
                        self.cfg.system.local_epochs[mode],
                        self.d["dim"], self.d["num_classes"], self.gen)
                    if accepted(upd, self.issued_round[i], t, self.cfg):
                        updates.append(upd)
                        weights.append(self.d["n_i"][i])
                        steps.append(n_steps)
                        succ_p.append(state["p_succ"][i][mode])
                    else:
                        ok = False

            success[i] = int(ok)
            if ok:
                self.x_succ[i] += 1
                self.since_success[i] = 0
                self.last_update[i] = updates[-1].detach().clone()
            else:
                self.since_success[i] += 1
                # MIFA/MimiC-style post-hoc correction (Table 3.5): the client's
                # own outcome is still recorded as a failure above -- this does
                # not prevent anything or change fairness/ARL bookkeeping, it
                # only patches the aggregation afterward with the client's last
                # known-good update, so Table 3.5's prevention-vs-imputation
                # question can be asked as a clean ablation against the same
                # selection policy with imputation switched off.
                if self.cfg.fl.impute_missing and i in self.last_update:
                    updates.append(self.last_update[i])
                    weights.append(self.d["n_i"][i])
                    steps.append(1)

            # The predictor learns from every issued invitation, success or not.
            self.predictor.update(state["obs"][i], state["hist"][i], mode, int(ok))
            self.cal_probs.append(state["p_succ"][i][mode])
            self.cal_out.append(int(ok))

        delta = aggregate(self.cfg.fl.aggregator, updates, weights, steps)
        if delta is not None:
            self.global_flat = self.global_flat + delta

        acc, test_loss = self._evaluate()
        prev_acc = self.rows[-1]["accuracy"] if self.rows else acc
        d_acc = acc - prev_acc

        # Latency is measured over the full selected set, not only the clients
        # that succeeded (Sec. 3.22 (v)): a straggler that blew the deadline
        # still cost the round its time.
        latency = max(latencies) if latencies else 0.0
        arl_on = M.arl_online(values, state["eligible"], success)
        arl_or = M.arl_oracle(values, self.env, success)
        cv = M.cv_fairness(self.x_succ)          # successful-contribution CV (core metric, Eq. 3.32)
        cv_sel = M.cv_fairness(self.x_sel)        # raw-selection-count CV (Task 2 ablation comparator)
        n_succ = int(sum(success.values()))
        fail_ratio = 1.0 - n_succ / max(len(selected), 1)

        # Now safe to apply this round's real energy spend -- after the oracle
        # diagnostic above has queried every client's pre-spend state.
        for i, mode in pending_spends:
            self.env.spend(i, mode)

        # Task 1/2 ablations: swap the value-weighted counterfactual damage term
        # for a naive raw failure count, and/or the capability-conditioned
        # successful-contribution fairness for plain selection-count fairness,
        # to test whether either of the thesis's two specific claims actually
        # changes the learned policy's behaviour versus the naive alternative.
        r = self.cfg.reward
        if r.damage_metric == "raw_fail":
            damage_term = self.norm["faildmg"](fail_ratio)
        else:
            damage_term = self.norm["arl"](arl_on)
        fairness_term = cv_sel if r.fairness_metric == "cv_selection" else cv

        reward = (r.alpha * self.norm["acc"](d_acc)
                  - r.beta * self.norm["cost"](round_cost)
                  - r.delta * self.norm["lat"](latency)
                  - r.lambda_arl * damage_term
                  - r.lambda_fair * fairness_term)
        self.policy.observe_outcome(state, (selected, assign), reward,
                                    {"success": success})

        row = {
            "round": t, "accuracy": acc, "test_loss": test_loss, "reward": reward,
            "cost": round_cost, "latency": latency,
            "n_selected": len(selected), "n_success": n_succ,
            "fail_ratio": fail_ratio,
            "arl_online": arl_on, "arl_oracle": arl_or,
            "cv": cv, "cv_selection": cv_sel, "jain": M.jain(self.x_succ),
            "ece": M.calibration_error(self.cal_probs[-200:], self.cal_out[-200:]),
            "mode_light": mode_counts["light"],
            "mode_standard": mode_counts["standard"],
            "mode_priority": mode_counts["priority"],
            "wall_s": time.time() - t0,
        }
        self.rows.append(row)
        return row

    # ----------------------------------------------------------------- run

    def run(self, verbose: bool = True) -> list[dict]:
        for t in range(1, self.cfg.fl.rounds + 1):
            row = self.run_round(t)
            if verbose and (t % max(1, self.cfg.fl.rounds // 20) == 0 or t == 1):
                print(f"[{self.cfg.tag}] r{t:3d} acc={row['accuracy']:.4f} "
                      f"succ={row['n_success']}/{row['n_selected']} "
                      f"ARL'={row['arl_online']:.3f} ARL*={row['arl_oracle']:.3f} "
                      f"CV={row['cv']:.3f} R={row['reward']:.3f}", flush=True)
        return self.rows

    def save(self) -> str:
        os.makedirs(self.cfg.out, exist_ok=True)
        path = os.path.join(self.cfg.out, f"{self.cfg.tag}.csv")
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            wr.writeheader()
            wr.writerows(self.rows)
        self.cfg.dump(os.path.join(self.cfg.out, f"{self.cfg.tag}.config.json"))
        return path
