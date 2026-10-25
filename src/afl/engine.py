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
    "round", "accuracy", "acc_worst10", "test_loss", "reward", "cost", "latency",
    "n_selected", "n_success", "fail_ratio", "arl_online", "arl_selected", "arl_oracle",
    "cv", "cv_selection", "jain", "ece", "mode_light", "mode_standard",
    "mode_priority", "weak_selected", "weak_success", "late_delivered",
    "budget_dropped", "debt_mean", "arl_v2", "departed", "wall_s",
]

LIGHTER = {"priority": "standard", "standard": "light", "light": None}
SKETCH_DIM = 1024


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        if cfg.fl.deterministic:
            # Must be set before the first CUDA call to take effect.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.d = build_data(cfg)
        self.N = cfg.data.num_clients
        self.env = ClientEnv(cfg, self.d["n_i"], cfg.seed, sketch=self.d["sketch"])
        self.predictor = SuccessPredictor(cfg)
        self.policy = build_policy(cfg)

        # Data lives on the training device once, so a GPU run does not copy a
        # client's samples across the bus every time it is selected.
        self.dev = _device(cfg.fl.device)
        self.x_train = torch.from_numpy(self.d["x_train"]).to(self.dev)
        self.y_train = torch.from_numpy(self.d["y_train"]).to(self.dev)
        self.part_idx = [torch.from_numpy(p).to(self.dev) for p in self.d["parts"]]
        self.x_test = torch.from_numpy(self.d["x_test"]).to(self.dev)
        self.y_test = torch.from_numpy(self.d["y_test"]).to(self.dev)
        n_test = len(self.d["y_test"])
        self.eval_idx = None
        if 0 < cfg.fl.eval_subset < n_test:
            pick = np.random.default_rng(cfg.seed + 5).choice(n_test, cfg.fl.eval_subset,
                                                             replace=False)
            self.eval_idx = torch.from_numpy(np.sort(pick)).to(self.dev)

        self.model = build_model(cfg, self.d["dim"], self.d["num_classes"]).to(self.dev)
        self.global_flat = get_flat(self.model)

        self.x_succ = np.zeros(self.N)
        self.x_sel = np.zeros(self.N)
        self.since_success = np.zeros(self.N)
        self.issued_round = np.zeros(self.N, dtype=int)
        self.local_loss = np.ones(self.N)     # refreshed lazily, see _refresh_losses
        self.gen = torch.Generator().manual_seed(cfg.seed + 3)
        self.last_update: dict[int, torch.Tensor] = {}  # MIFA/MimiC substitution store

        self.norm = {k: M.Normalizer() for k in ("acc", "cost", "lat", "arl", "faildmg", "debt", "v2c")}
        self.cov_ema = np.zeros(self.d["num_classes"])   # ARL-v2 class-coverage history
        if cfg.policy.value_feature == "marginal":
            self.value_input = M.expected_marginal_value(
                self.d["sketch"], cfg.fl.clients_per_round, seed=cfg.seed + 41)
        elif cfg.policy.value_feature == "standalone":
            self.value_input = M.standalone_value(self.d["sketch"])
        else:
            raise ValueError(f"unknown policy.value_feature {cfg.policy.value_feature}")

        # State for the optional pieces; all empty/unused in the original runs.
        self.late: list[tuple] = []               # semi-async updates due next round
        self.debt = np.zeros(self.N)              # fairness debt (Sec. 3.13)
        self.mem: dict[int, torch.Tensor] = {}    # MIFA: latest update of every client
        self.mimic_c: dict[int, torch.Tensor] = {}  # MimiC: correction variables
        self.ef_err: dict[int, torch.Tensor] = {}   # FedCG: top-k error feedback
        self.hist_sketch = None                   # FedCG: projected update history
        if cfg.policy.name == "fedcg":
            P = self.global_flat.numel()
            g = torch.Generator().manual_seed(cfg.seed + 31)
            self.sk_idx = torch.randint(0, SKETCH_DIM, (P,), generator=g).to(self.dev)
            self.sk_sign = (torch.randint(0, 2, (P,), generator=g) * 2 - 1).float().to(self.dev)
            self.hist_sketch = np.zeros((self.N, SKETCH_DIM))
            self.has_hist = np.zeros(self.N, dtype=bool)
        self.cal_probs: list[float] = []
        self.cal_out: list[int] = []
        self.rows: list[dict] = []
        self._prev_acc_r: float | None = None

    # ------------------------------------------------------------ helpers

    def _client_xy(self, i: int):
        idx = self.part_idx[i]
        return self.x_train[idx], self.y_train[idx]

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
                out = self.model(x[:128])
                self.local_loss[i] = float(F.cross_entropy(out, y[:128]))

    def _evaluate(self, full: bool = True) -> tuple[float, float, float]:
        """Accuracy, loss, and mean accuracy over the worst 10% of classes.

        The worst-classes number is what representation loss should move: if a
        policy keeps losing the clients that hold rare classes, overall accuracy
        can stay high while those classes are never learned. For 10 classes it
        is the single worst class.
        """
        set_flat(self.model, self.global_flat)
        self.model.eval()
        x, y = self.x_test, self.y_test
        if not full and self.eval_idx is not None:
            x, y = x[self.eval_idx], y[self.eval_idx]
        C = self.d["num_classes"]
        loss_sum, correct = 0.0, 0
        per_class_ok = torch.zeros(C, device=self.dev)
        per_class_n = torch.zeros(C, device=self.dev)
        with torch.no_grad():
            # Batched so a CIFAR-sized test set fits in memory on CPU.
            for s in range(0, len(y), 2048):
                out = self.model(x[s:s + 2048])
                yb = y[s:s + 2048]
                loss_sum += float(F.cross_entropy(out, yb, reduction="sum"))
                ok = (out.argmax(1) == yb).float()
                correct += int(ok.sum())
                per_class_ok.index_add_(0, yb, ok)
                per_class_n.index_add_(0, yb, torch.ones_like(ok))
        n = max(len(y), 1)
        seen = per_class_n > 0
        cls_acc = (per_class_ok[seen] / per_class_n[seen]).cpu().numpy()
        k = max(1, int(round(0.1 * len(cls_acc))))
        worst = float(np.sort(cls_acc)[:k].mean()) if len(cls_acc) else 0.0
        return correct / n, loss_sum / n, worst

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
        state = {
            "obs": obs, "hist": hist, "p_succ": p_succ, "eligible": eligible,
            "t_expected": t_expected, "n_i": self.d["n_i"],
            "local_loss": self.local_loss,
            "repr_value": self.value_input,
            "sketch": self.d["sketch"],
            "n_work": self.env.n_work,
            "debt": self.debt / (1.0 + self.debt.max()),
        }
        if self.hist_sketch is not None:
            self._probe_missing_history()
            state["update_sketch"] = self.hist_sketch
        return state

    # ------------------------------------------------- optional components

    def _sketch(self, v: torch.Tensor) -> np.ndarray:
        """Count-sketch projection of a flat update to SKETCH_DIM dimensions."""
        out = torch.zeros(SKETCH_DIM, device=self.dev).index_add_(0, self.sk_idx, v * self.sk_sign)
        return out.cpu().numpy()

    def _probe_missing_history(self) -> None:
        """FedCG needs every client's historical update; for clients that have
        never reported, use one epoch of local training from the current model
        (the stand-in for the initial gradients DivFL-style selection assumes)."""
        for i in np.where(~self.has_hist)[0]:
            x, y = self._client_xy(int(i))
            if len(y) == 0:
                continue
            upd, _ = local_train(self.cfg, self.global_flat, x, y, 1,
                                 self.d["dim"], self.d["num_classes"], self.gen)
            self.hist_sketch[i] = self._sketch(upd)
            self.has_hist[i] = True

    def _compress(self, i: int, upd: torch.Tensor, frac: float) -> torch.Tensor:
        """FedCG upload: top-k sparsification at ratio frac with error feedback."""
        v = upd + self.ef_err[i] if i in self.ef_err else upd
        k = max(1, int(round(frac * v.numel())))
        if k >= v.numel():
            self.ef_err[i] = torch.zeros_like(v)
            return v
        idx = torch.topk(v.abs(), k, sorted=False).indices
        sent = torch.zeros_like(v)
        sent[idx] = v[idx]
        self.ef_err[i] = v - sent
        return sent

    def _est_time(self, i: int, mode: str, state: dict, frac: float | None) -> float:
        """Server-side estimate of a client's round time from observed descriptors."""
        sysc = self.cfg.system
        obs = state["obs"][i]
        frac = sysc.upload_frac[mode] if frac is None else frac
        return (sysc.local_epochs[mode] * (self.env.n_work[i] / 1200.0) / max(obs["speed_hat"], 1e-3)
                + 0.25 * frac / max(obs["bw_hat"], 1e-3))

    def _enforce_budget(self, selected, assign, state, fracs):
        """Eq. (3.64)-(3.65): per-round cost budget C_max and time limit L_max.

        Applied to every policy's chosen action, so constrained comparisons are
        fair. A client whose estimated time exceeds L_max is moved to lighter
        modes and dropped if even Light does not fit. While the round's cost
        exceeds C_max, the lowest-priority client (last in the policy's order)
        is moved to a lighter mode, or dropped once it is already on Light.
        Only observable quantities are used. (For the RL policy this is an
        action projection: the gradient still uses the sampled action.)
        """
        sysc = self.cfg.system
        if sysc.cost_max >= 1e9 and sysc.latency_max >= 1e9:
            return selected, assign, 0
        assign = dict(assign)
        keep, dropped = [], 0
        for i in selected:
            m = assign[i]
            while self._est_time(i, m, state, fracs.get(i)) > sysc.latency_max and LIGHTER[m]:
                m = LIGHTER[m]
            if self._est_time(i, m, state, fracs.get(i)) > sysc.latency_max:
                dropped += 1
                continue
            assign[i] = m
            keep.append(i)
        while keep and sum(self.env.cost(i, assign[i], fracs.get(i)) for i in keep) > sysc.cost_max:
            for i in reversed(keep):
                if LIGHTER[assign[i]]:
                    assign[i] = LIGHTER[assign[i]]
                    break
            else:
                keep.pop()
                dropped += 1
        return keep, {i: assign[i] for i in keep}, dropped

    # --------------------------------------------------------------- round

    def run_round(self, t: int) -> dict:
        t0 = time.time()
        self.env.new_round(t)
        state = self._state(t)

        selected, assign = self.policy.act(state, t)
        fracs = dict(getattr(self.policy, "upload_frac", {}) or {})
        # A client still finishing last round's late update cannot take a new job.
        busy = {item[0] for item in self.late}
        if busy:
            selected = [i for i in selected if i not in busy]
        selected, assign, budget_dropped = self._enforce_budget(selected, assign, state, fracs)
        self._refresh_losses(selected)

        # V^repr_{i,t}(S_t) for every client (Eq. 3.26), not only the selected
        # ones: Eq. (3.27) sums ARL' over the full population C, so an
        # eligible, high-value client who was never invited must still show up
        # here with Y^succ = 0 via the success dict's default below.
        rep_vals = M.per_client_representation_value(self.d["sketch"], selected)
        values: dict[int, float] = {i: float(rep_vals[i]) for i in range(self.N)}

        updates, weights, steps, succ_p, recv_ids = [], [], [], [], []
        success: dict[int, int] = {}
        reasons: dict[int, str] = {}

        # Semi-asynchronous deliveries from last round, now one round stale:
        # Eq. (3.18)'s staleness check decides whether they still count.
        late_delivered = 0
        for i, upd, n_steps, issued, p in self.late:
            if accepted(upd, issued, t, self.cfg):
                updates.append(upd)
                weights.append(self.d["n_i"][i])
                steps.append(n_steps)
                succ_p.append(p)
                recv_ids.append(i)
                self.x_succ[i] += 1
                self.since_success[i] = 0
                self.last_update[i] = upd.detach().clone()
                late_delivered += 1
        self.late = []
        pending_spends: list[tuple[int, str]] = []
        round_cost = 0.0
        latencies = []
        mode_counts = {m: 0 for m in self.cfg.system.modes}

        for i in selected:
            mode = assign[i]
            frac = fracs.get(i)
            mode_counts[mode] += 1
            self.x_sel[i] += 1
            # The server broadcasts the current global model with the invitation,
            # so the update is computed against round t. Staleness only bites in
            # the asynchronous variant, where this is set when the model is sent.
            self.issued_round[i] = t
            round_cost += self.env.cost(i, mode, frac)
            out = self.env.outcome(i, mode, frac)
            reasons[i] = out["reason"]
            latencies.append(out["latency"])
            # Energy spend is deferred until after the oracle diagnostic below
            # is computed (Sec. 3.11, Eq. 3.29): applying it immediately would
            # make a client's own real energy draw down before its own
            # counterfactual is evaluated, so a client that genuinely could
            # have succeeded under the pre-decision energy state gets
            # incorrectly reported as oracle-ineligible.
            pending_spends.append((i, mode, frac))

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
                    if self.cfg.fl.aggregator == "fedcg":
                        upd = self._compress(i, upd, frac if frac is not None else 1.0)
                    if accepted(upd, self.issued_round[i], t, self.cfg):
                        updates.append(upd)
                        weights.append(self.d["n_i"][i])
                        steps.append(n_steps)
                        succ_p.append(state["p_succ"][i][mode])
                        recv_ids.append(i)
                    else:
                        ok = False
            elif (self.cfg.fl.async_late and out["reason"] == "deadline"
                  and self.env.late_ok(i, mode, frac)):
                # Too slow for this round, but done within the staleness bound:
                # train against the round-t model now, deliver next round.
                x, y = self._client_xy(i)
                if len(y) > 0:
                    upd, n_steps = local_train(
                        self.cfg, self.global_flat, x, y,
                        self.cfg.system.local_epochs[mode],
                        self.d["dim"], self.d["num_classes"], self.gen)
                    self.late.append((i, upd, n_steps, t, state["p_succ"][i][mode]))

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
                    succ_p.append(1.0)          # not a success: no Eq. (3.54) boost
                    recv_ids.append(i)

            # The predictor learns from every issued invitation, success or not.
            self.predictor.update(state["obs"][i], state["hist"][i], mode, int(ok))
            self.cal_probs.append(state["p_succ"][i][mode])
            self.cal_out.append(int(ok))

        agg = self.cfg.fl.aggregator
        if agg == "mifa":
            # MIFA (Gu et al. 2021, Alg. 1): keep every client's latest update
            # (zero until it first reports) and step by their mean over all N.
            for i, u in zip(recv_ids, updates):
                self.mem[i] = u
            delta = sum(self.mem.values()) / self.N if self.mem else None
        elif agg == "mimic":
            # MimiC (Sun et al. 2024, Alg. 1): correct each received update with
            # the client's correction variable from its last active round,
            # v_i = g_i + c_i; step by the mean v; then c_i = v - g_i.
            if updates:
                corrected = [u + self.mimic_c[i] if i in self.mimic_c else u
                             for i, u in zip(recv_ids, updates)]
                delta = sum(corrected) / len(corrected)
                for i, u in zip(recv_ids, updates):
                    self.mimic_c[i] = delta - u
            else:
                delta = None
        else:
            probs = succ_p if self.cfg.fl.importance_reweight else None
            delta = aggregate(agg, updates, weights, steps, success_probs=probs,
                              p_floor=self.cfg.fl.reweight_floor)
        if delta is not None:
            self.global_flat = self.global_flat + delta
        if self.hist_sketch is not None:
            for i, u in zip(recv_ids, updates):
                self.hist_sketch[i] = self._sketch(u)
                self.has_hist[i] = True

        # The reward always sees the same evaluation set from round to round
        # (the subset, if one is configured) so dAcc never mixes two test sets;
        # only the logged accuracy of the final round switches to the full set.
        acc_r, loss_r, worst_r = self._evaluate(full=False)
        if t == self.cfg.fl.rounds and self.eval_idx is not None:
            acc, test_loss, acc_worst = self._evaluate(full=True)
        else:
            acc, test_loss, acc_worst = acc_r, loss_r, worst_r
        d_acc = acc_r - (self._prev_acc_r if self._prev_acc_r is not None else acc_r)
        self._prev_acc_r = acc_r

        # Latency is measured over the full selected set, not only the clients
        # that succeeded (Sec. 3.22 (v)): a straggler that blew the deadline
        # still cost the round its time.
        latency = max(latencies) if latencies else 0.0
        arl_on = M.arl_online(values, state["eligible"], success)
        # The part of ARL'_t caused by the policy's own invitations failing,
        # without the never-invited clients Eq. (3.27) also sums over.
        arl_sel = M.arl_online({i: values[i] for i in selected}, state["eligible"], success)
        arl_or = M.arl_oracle(values, self.env, success)
        cv = M.cv_fairness(self.x_succ)          # successful-contribution CV (core metric, Eq. 3.32)
        cv_sel = M.cv_fairness(self.x_sel)        # raw-selection-count CV (Task 2 ablation comparator)
        n_succ = int(sum(success.values()))
        fail_ratio = 1.0 - n_succ / max(len(selected), 1)

        # Now safe to apply this round's real energy spend -- after the oracle
        # diagnostic above has queried every client's pre-spend state.
        for i, mode, frac in pending_spends:
            self.env.spend(i, mode, frac)
        for i, why in reasons.items():
            self.env.record(i, why)

        # ARL-v2: the representation loss this round's decisions caused. Only
        # invited clients that failed count; each is weighted by
        #   gap_i   success probability its assigned mode gave up vs. its best mode
        #           (zero if the scheduler already chose the safest mode),
        #   V_i     its representation value this round (Eq. 3.26), and
        #   a_i     1 + its data share in classes that have been under-covered
        #           lately, so a long absence costs more than a short gap.
        sketch = self.d["sketch"]
        deficit = 1.0 - self.cov_ema / max(float(self.cov_ema.max()), 1e-9)
        v2: dict[int, float] = {}
        for i in selected:
            if success.get(i, 0):
                v2[i] = 0.0
                continue
            p = state["p_succ"][i]
            gap = max(p.values()) - p[assign[i]]
            v2[i] = float(gap * values[i] * (1.0 + sketch[i] @ deficit))
        arl_v2 = float(sum(v2.values()))
        ok_ids = [i for i in selected if success.get(i, 0)]
        cov = sketch[ok_ids].max(axis=0) if ok_ids else np.zeros_like(self.cov_ema)
        b = self.cfg.reward.arl_v2_ema
        self.cov_ema = (1 - b) * self.cov_ema + b * cov

        # Fairness debt (Sec. 3.13): each eligible client accrues its fair share
        # K/N of service per round and pays it off with a successful contribution.
        rho = len(selected) / self.N if selected else 0.0
        elig = np.array([state["eligible"][i] for i in range(self.N)], dtype=float)
        served = np.array([success.get(i, 0) for i in range(self.N)], dtype=float)
        self.debt = np.maximum(0.0, self.debt + rho * elig - served)

        # Task 1/2 ablations: swap the value-weighted counterfactual damage term
        # for a naive raw failure count, and/or the capability-conditioned
        # successful-contribution fairness for plain selection-count fairness,
        # to test whether either of the thesis's two specific claims actually
        # changes the learned policy's behaviour versus the naive alternative.
        r = self.cfg.reward
        if r.damage_metric == "raw_fail":
            damage_term = self.norm["faildmg"](fail_ratio)
        elif r.damage_metric == "arl_selected":
            damage_term = self.norm["arl"](arl_sel)
        elif r.damage_metric == "arl":
            damage_term = self.norm["arl"](arl_on)
        elif r.damage_metric == "arl_v2":
            # With per-decision credit the damage leaves the shared reward and
            # is handed to the policy client by client instead.
            damage_term = 0.0 if self.cfg.policy.per_client_credit else self.norm["arl"](arl_v2)
        else:
            raise ValueError(f"unknown reward.damage_metric {r.damage_metric}")
        fairness_term = cv_sel if r.fairness_metric == "cv_selection" else cv

        reward = (r.alpha * self.norm["acc"](d_acc)
                  - r.beta * self.norm["cost"](round_cost)
                  - r.delta * self.norm["lat"](latency)
                  - r.lambda_arl * damage_term
                  - r.lambda_fair * fairness_term)
        if r.lambda_debt > 0:
            reward -= r.lambda_debt * self.norm["debt"](float(self.debt.mean()))
        client_adv = None
        if r.damage_metric == "arl_v2" and self.cfg.policy.per_client_credit:
            client_adv = {i: -r.lambda_arl * self.norm["v2c"](v2[i]) for i in selected}
        self.policy.observe_outcome(state, (selected, assign), reward,
                                    {"success": success, "client_adv": client_adv})

        weak = self.env.weak
        row = {
            "round": t, "accuracy": acc, "acc_worst10": acc_worst,
            "test_loss": test_loss, "reward": reward,
            "cost": round_cost, "latency": latency,
            "n_selected": len(selected), "n_success": n_succ,
            "fail_ratio": fail_ratio,
            "arl_online": arl_on, "arl_selected": arl_sel, "arl_oracle": arl_or,
            "cv": cv, "cv_selection": cv_sel, "jain": M.jain(self.x_succ),
            "ece": M.calibration_error(self.cal_probs[-200:], self.cal_out[-200:]),
            "mode_light": mode_counts["light"],
            "mode_standard": mode_counts["standard"],
            "mode_priority": mode_counts["priority"],
            "weak_selected": int(sum(weak[i] for i in selected)),
            "weak_success": int(sum(weak[i] and success[i] for i in selected)),
            "late_delivered": late_delivered,
            "budget_dropped": budget_dropped,
            "debt_mean": float(self.debt.mean()),
            "arl_v2": arl_v2,
            "departed": int(self.env.departed.sum()),
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
        # Write-then-rename: the sweep runners skip any run whose CSV exists,
        # so a session killed mid-write must not leave a truncated one behind.
        tmp = path + ".part"
        with open(tmp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            wr.writeheader()
            wr.writerows(self.rows)
        os.replace(tmp, path)
        self.cfg.dump(os.path.join(self.cfg.out, f"{self.cfg.tag}.config.json"))
        return path
