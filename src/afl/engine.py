"""The round loop (Sec. 3.4).

One round is: observe -> predict success per (client, mode) -> gate eligibility
-> policy acts -> local training -> acceptance -> aggregation -> metrics ->
reward -> policy learns. Every quantity the experimental design asks for
(Sec. 3.23) is written to a per-round CSV so the plots in Chapter 5 are produced
from the log rather than recomputed.
"""
from __future__ import annotations

import csv
import hashlib
import json
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
from .policies.base import best_work_mode, est_time, mode_by_rule
from .predictor import SuccessPredictor, SuccessPredictorV2
from .provenance import LOADED as CODE_LOADED, code_hash
from .sim import ClientEnv
from .diagnostics import DiagWriter, norm_state

LOG_FIELDS = [
    "round", "accuracy", "accuracy_fixed", "eval_fixed_n", "acc_worst10", "test_loss", "reward", "cost", "latency",
    "n_selected", "n_success", "fail_ratio", "empty_round", "arl_online", "arl_selected", "arl_oracle",
    "cv", "cv_selection", "jain", "ece", "mode_light", "mode_standard",
    "mode_priority", "weak_selected", "weak_success", "late_delivered",
    "budget_dropped", "debt_mean", "arl_v2", "departed",
    "arl_v3_loss", "arl_v3_gain", "arl_v4_loss", "arl_v4_deficit",
    # protocol-fix diagnostics (docs/ADVANCED_RESULTS.md, "Protocol fix")
    "val_accuracy", "acc_client_worst10", "acc_client_weak", "cov_lost", "cov_lost_avoid",
    "gap_max_all", "gap_max_weak", "ece_light", "ece_standard", "ece_priority",
    "ece_weak", "ece_strong", "gate_filled", "acc_labelmix_worst10", "acc_client_measured",
    "wall_s",
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
        if cfg.policy.predictor not in ("v1", "v2"):
            raise ValueError(f"unknown policy.predictor {cfg.policy.predictor}")
        if cfg.policy.gate_mode not in ("top", "thompson"):
            raise ValueError(f"unknown policy.gate_mode {cfg.policy.gate_mode}")
        if cfg.policy.gate_mode == "thompson" and cfg.policy.predictor != "v2":
            raise ValueError("policy.gate_mode=thompson needs policy.predictor=v2")
        self.pv2 = cfg.policy.predictor == "v2"
        self.predictor = SuccessPredictorV2(cfg) if self.pv2 else SuccessPredictor(cfg)
        # Own stream for the Thompson gate, so it never shifts any other draw.
        self.gate_rng = np.random.default_rng(cfg.seed + 7100)
        self.policy = build_policy(cfg)
        ck = getattr(self.policy, "checkpoint_meta", None)
        data_fp = (self.d.get("split") or {}).get("fingerprint")
        if ck is not None and (data_fp is None or ck.get("split_fingerprint") != data_fp):
            # A pre-trained scheduler may only be used on the same persistent
            # split it practised on (practice pool / validation only).
            raise ValueError("checkpoint split fingerprint "
                             f"{ck.get('split_fingerprint')} does not match this run's {data_fp}")
        # Opt-in diagnostic sidecars; reading only, never changes the run.
        self.diag = DiagWriter(cfg.out, cfg.tag) if cfg.diag.enabled else None

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
        # Protocol fix (fl.val_frac): a server validation set, disjoint from the
        # images the reported accuracy is measured on. Everything a policy learns
        # from uses only it.
        self.val_idx = self.test_idx = None
        pool = np.arange(n_test)
        self.split_fp = None
        if self.d.get("split") is not None:
            # data.split = persistent: every example's role is fixed for all
            # seeds (afl.data), so no run's test example is ever another run's
            # validation (reward) example.
            roles = self.d["split"]["val_role"]
            self.val_idx = torch.from_numpy(np.where(roles)[0]).to(self.dev)
            pool = np.where(~roles)[0]
            self.test_idx = torch.from_numpy(pool).to(self.dev)
            self.split_fp = self.d["split"]["fingerprint"]
        elif cfg.fl.val_frac > 0:
            perm = np.random.default_rng(cfg.seed + 6).permutation(n_test)
            n_val = int(round(cfg.fl.val_frac * n_test))
            self.val_idx = torch.from_numpy(np.sort(perm[:n_val])).to(self.dev)
            pool = np.sort(perm[n_val:])
            self.test_idx = torch.from_numpy(pool).to(self.dev)
        if 0 < cfg.fl.eval_subset < len(pool):
            pick = np.random.default_rng(cfg.seed + 5).choice(len(pool), cfg.fl.eval_subset,
                                                             replace=False)
            self.eval_idx = torch.from_numpy(np.sort(pool[pick])).to(self.dev)

        if cfg.fl.clean_rng:
            # Same initial model for every method with this seed, whatever the
            # policy's own construction drew from PyTorch's generator before.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed + 101)
                self.model = build_model(cfg, self.d["dim"], self.d["num_classes"]).to(self.dev)
        else:
            self.model = build_model(cfg, self.d["dim"], self.d["num_classes"]).to(self.dev)
        self.global_flat = get_flat(self.model)
        self.init_checksum = hashlib.sha1(
            self.global_flat.detach().cpu().numpy().tobytes()).hexdigest()[:16]

        self.x_succ = np.zeros(self.N)
        self.x_sel = np.zeros(self.N)
        self.since_success = np.zeros(self.N)
        self.issued_round = np.zeros(self.N, dtype=int)
        self.local_loss = np.ones(self.N)     # refreshed lazily, see _refresh_losses
        self.gen = torch.Generator().manual_seed(cfg.seed + 3)
        self.last_update: dict[int, torch.Tensor] = {}  # MIFA/MimiC substitution store

        nkw = dict(floor=1e-3, clip=3.0, warmup=10) if cfg.reward.robust_norm else {}
        self.norm = {k: M.Normalizer(**nkw) for k in ("acc", "cost", "lat", "arl", "faildmg", "debt",
                                                      "v2c", "v3g", "v3c", "v3cg",
                                                      "v2s", "v3s", "v3sg",
                                                      "v4c", "v4cg", "v4d", "v4g", "prot")}
        self.cov_ema = np.zeros(self.d["num_classes"])   # ARL-v2 class-coverage history
        self.cov_peak = np.zeros(self.d["num_classes"])  # ARL-v4: best coverage so far
        self.cov_dur = np.zeros(self.d["num_classes"])   # ARL-v4: rounds without new coverage
        self.align_ema = np.zeros(self.N)                # ARL-v4: update/validation alignment
        self._prev_worst_r: float | None = None
        self.arl4 = cfg.reward.damage_metric == "arl_v4"
        if cfg.fl.practice and not cfg.fl.val_frac > 0:
            raise ValueError("a practice episode needs fl.val_frac > 0 (its reward uses validation data)")
        if (self.arl4 or cfg.reward.protect_weight > 0) and not cfg.fl.val_frac > 0:
            raise ValueError("ARL-v4 and the protection reward need fl.val_frac > 0 "
                             "(they use the validation set, never the test set)")
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
        # Error of the clients' availability forecasts (REFL). Its own stream,
        # so no other random draw in the run changes.
        self.fc_rng = np.random.default_rng(cfg.seed + 5000)
        self.probed = False                       # FAVOR: initial all-client round done
        self.cal_probs: list[float] = []
        self.cal_out: list[int] = []
        self.cal_mode: list[str] = []
        self.cal_weak: list[bool] = []
        self.gap_cur = np.zeros(self.N)            # rounds since last contribution
        self.gap_max = np.zeros(self.N)
        self._last_cls_acc = np.zeros(self.d["num_classes"])
        self._last_ok = None                       # per-example correctness of the last
        self._last_idx = None                      # reported evaluation, and its indices
        self.test_owner = torch.from_numpy(self.d["test_owner"]).to(self.dev)
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

    def _val_grad(self, n: int = 512) -> torch.Tensor:
        """Gradient of the loss on the first n VALIDATION examples at the current
        global model (ARL-v4's update alignment). Validation data only; reads
        no test example, changes no parameter and draws no random number."""
        set_flat(self.model, self.global_flat)
        self.model.eval()
        idx = self.val_idx[:n]
        self._val_grad_idx = idx
        loss = F.cross_entropy(self.model(self.x_test[idx]), self.y_test[idx])
        grads = torch.autograd.grad(loss, list(self.model.parameters()))
        return torch.cat([g.reshape(-1) for g in grads]).detach()

    def _evaluate(self, full: bool = True, idx=None) -> tuple[float, float, float]:
        """Accuracy, loss, and mean accuracy over the worst 10% of classes.

        The worst-classes number is what representation loss should move: if a
        policy keeps losing the clients that hold rare classes, overall accuracy
        can stay high while those classes are never learned. For 10 classes it
        is the single worst class.
        """
        set_flat(self.model, self.global_flat)
        self.model.eval()
        if idx is None:
            if not full and self.eval_idx is not None:
                idx = self.eval_idx
            elif self.test_idx is not None:
                idx = self.test_idx
        x, y = (self.x_test, self.y_test) if idx is None else (self.x_test[idx], self.y_test[idx])
        C = self.d["num_classes"]
        loss_sum, correct = 0.0, 0
        oks = []
        per_class_ok = torch.zeros(C, device=self.dev)
        per_class_n = torch.zeros(C, device=self.dev)
        with torch.no_grad():
            # Batched so a CIFAR-sized test set fits in memory on CPU.
            for s in range(0, len(y), 2048):
                out = self.model(x[s:s + 2048])
                yb = y[s:s + 2048]
                loss_sum += float(F.cross_entropy(out, yb, reduction="sum"))
                ok = (out.argmax(1) == yb).float()
                oks.append(ok)
                correct += int(ok.sum())
                per_class_ok.index_add_(0, yb, ok)
                per_class_n.index_add_(0, yb, torch.ones_like(ok))
        n = max(len(y), 1)
        seen = per_class_n > 0
        cls_acc = (per_class_ok[seen] / per_class_n[seen]).cpu().numpy()
        full_cls = np.full(C, np.nan)
        full_cls[seen.cpu().numpy()] = cls_acc
        self._last_cls_acc = full_cls
        self._last_ok = torch.cat(oks) if oks else None
        self._last_idx = idx
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
        if self.cfg.policy.predictor_norm == "round" or self.pv2:
            self.predictor.set_reference(np.array(
                [self.predictor.client_features(obs[i], hist[i]) for i in range(self.N)]))
        p_sd = None
        if self.pv2:
            post = {i: {m: self.predictor.predict_client(i, obs[i], hist[i], m)
                        for m in self.cfg.system.modes} for i in range(self.N)}
            p_succ = {i: {m: v[0] for m, v in post[i].items()} for i in range(self.N)}
            p_sd = {i: {m: v[1] for m, v in post[i].items()} for i in range(self.N)}
        else:
            p_succ = {
                i: {m: self.predictor.predict(obs[i], hist[i], m) for m in self.cfg.system.modes}
                for i in range(self.N)
            }
        # e^elig_{i,t} (Eq. 3.26): at least one mode clears the threshold.
        # Gate v2 (thompson): on the optimistic bound p + z*sd, so a client the
        # predictor knows little about is not written off.
        z = self.cfg.policy.gate_z if self.cfg.policy.gate_mode == "thompson" else 0.0
        eligible = {
            i: int(max(p_succ[i][m] + (z * p_sd[i][m] if p_sd else 0.0)
                       for m in self.cfg.system.modes) >= self.cfg.policy.elig_threshold)
            for i in range(self.N)
        }
        # True round time under this round's draws -- what the original Oort
        # baseline used (a leak) -- and the server's estimate from observed
        # speed and bandwidth, which is all a real server has.
        t_expected = {
            i: {m: sum(self.env.workload(i, m)[:2]) for m in self.cfg.system.modes}
            for i in range(self.N)
        }
        state = {"obs": obs}
        t_est = {
            i: {m: self._est_time(i, m, state, None) for m in self.cfg.system.modes}
            for i in range(self.N)
        }
        # Each client's forecast of its own availability next round; right
        # with probability refl_pred_acc, otherwise uninformative (REFL Sec. 5).
        fc = self.env.availability_forecast(t)
        wrong = self.fc_rng.random(self.N) > self.cfg.policy.refl_pred_acc
        noise = self.fc_rng.random(self.N)
        state = {
            "obs": obs, "hist": hist, "p_succ": p_succ, "eligible": eligible,
            "p_sd": p_sd, "gate_rng": self.gate_rng,
            "p_post": ({i: {m: self.predictor.posterior(i, obs[i], hist[i], m)
                            for m in self.cfg.system.modes} for i in range(self.N)}
                       if self.cfg.policy.gate_mode == "thompson" else None),
            "t_expected": t_expected, "t_est": t_est, "n_i": self.d["n_i"],
            "avail_forecast": np.where(wrong, noise, fc),
            "n_late": len(self.late),
            "global_flat": self.global_flat,
            "local_loss": self.local_loss,
            "repr_value": self.value_input,
            "sketch": self.d["sketch"],
            "n_work": self.env.n_work,
            # K x the mean cost of a Light job: the scale of the safety layer's
            # spend target (RL2 C6). Observable: from data sizes and mode costs.
            "cost_ref": float(min(self.cfg.fl.clients_per_round, self.N) * np.mean(
                [self.env.cost(i, "light") for i in range(self.N)])),
            "debt": self.debt / (1.0 + self.debt.max()),
            "cov_deficit": 1.0 - self.cov_ema / max(float(self.cov_ema.max()), 1e-9),
            # Protocol fix (system.checkin_pool): the clients that checked in
            # this round. None = every policy may pick any client (original).
            "candidates": ([int(i) for i in np.where(self.env._draw["available"])[0]]
                           if self.cfg.system.checkin_pool else None),
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

    def _probe_all(self) -> dict[int, torch.Tensor]:
        """One local epoch on every client from the current model.

        FAVOR's Step 2: before the first round every device trains once and
        uploads, so the server has a model of each client to build the state
        from. Not charged to any client and no client is marked as having
        participated -- this is the setup phase of the paper's workflow.
        """
        out = {}
        for i in range(self.N):
            x, y = self._client_xy(i)
            if len(y) == 0:
                continue
            out[i], _ = local_train(self.cfg, self.global_flat, x, y, 1,
                                    self.d["dim"], self.d["num_classes"], self.gen)
        return out

    def _compress(self, i: int, upd: torch.Tensor, frac: float) -> torch.Tensor:
        """Upload at ratio frac: top-k sparsification with error feedback.

        Used by FedCG, and for every policy when fl.compress_upload is on, so a
        mode's upload fraction is what the server actually receives.
        """
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

    def _frac(self, mode: str, frac: float | None) -> float:
        return self.cfg.system.upload_frac[mode] if frac is None else frac

    def _est_time(self, i: int, mode: str, state: dict, frac: float | None) -> float:
        """Server-side estimate of a client's round time from observed descriptors."""
        return est_time(self.cfg, {"obs": state["obs"], "n_work": self.env.n_work}, i, mode, frac)

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

    def _client_metrics(self, success: dict, late_ids: set) -> dict:
        """Who benefits: per-client accuracy, contribution gaps, calibration.

        A client's accuracy is the reported per-class accuracy weighted by its
        own label distribution (its expected accuracy on data like its own).
        Gaps count rounds without a contribution, invited or not. Calibration
        is over every invitation so far, by mode and by weak/other clients.
        """
        sk = self.d["sketch"]
        cls = self._last_cls_acc
        seen = ~np.isnan(cls)
        w = sk[:, seen]
        tot = w.sum(axis=1)
        mix = np.where(tot > 0, (w @ cls[seen]) / np.maximum(tot, 1e-12), np.nan)
        measured = self._measured_client_acc()
        # Measured on each client's own held-out examples where they exist
        # (FEMNIST: the writer's own handwriting). On a Dirichlet split the
        # clients' data are draws from the same class-conditional pool, so the
        # label-mixture value is exactly the expected accuracy on a client's
        # own data, and is used there.
        acc_c = measured if measured is not None else mix

        def worst10(v):
            v = v[~np.isnan(v)]
            if not len(v):
                return 0.0
            return float(np.sort(v)[:max(1, int(round(0.1 * len(v))))].mean())
        weak = self.env.weak
        contributed = np.zeros(self.N, dtype=bool)
        for i, v in success.items():
            contributed[i] = bool(v)
        for i in late_ids:
            contributed[i] = True
        self.gap_cur = np.where(contributed, 0.0, self.gap_cur + 1)
        self.gap_max = np.maximum(self.gap_max, self.gap_cur)
        P, O = np.asarray(self.cal_probs), np.asarray(self.cal_out)
        modes, wk = np.asarray(self.cal_mode), np.asarray(self.cal_weak, dtype=bool)

        def ece(mask):
            return M.calibration_error(list(P[mask]), list(O[mask])) if mask.any() else 0.0
        out = {
            "acc_client_worst10": worst10(acc_c),
            "acc_labelmix_worst10": worst10(mix),
            "acc_client_measured": int(measured is not None),
            "acc_client_weak": float(np.nanmean(acc_c[weak])) if weak.any() else 0.0,
            "gap_max_all": float(self.gap_max.mean()),
            "gap_max_weak": float(self.gap_max[weak].mean()) if weak.any() else 0.0,
            "ece_weak": ece(wk) if len(P) else 0.0,
            "ece_strong": ece(~wk) if len(P) else 0.0,
        }
        for m in self.cfg.system.modes:
            out[f"ece_{m}"] = ece(modes == m) if len(P) else 0.0
        return out

    def _practice_metrics(self) -> dict:
        """Practice episodes measure nothing on the test set: NaN placeholders."""
        keys = ["acc_client_worst10", "acc_labelmix_worst10", "acc_client_measured",
                "acc_client_weak", "gap_max_all", "gap_max_weak", "ece_weak", "ece_strong"]
        keys += [f"ece_{m}" for m in self.cfg.system.modes]
        return {k: float("nan") for k in keys}

    def _measured_client_acc(self):
        """Each client's accuracy on its own held-out test examples, from the
        last reported evaluation; None if the test set has no owners or that
        evaluation did not cover them (a subset round)."""
        if self._last_ok is None or not bool((self.test_owner >= 0).any()):
            return None
        idx = self._last_idx
        own = self.test_owner if idx is None else self.test_owner[idx]
        if not bool((own >= 0).any()):
            return None
        hits = torch.zeros(self.N, device=self.dev).index_add_(0, own.clamp(min=0),
                                                              self._last_ok * (own >= 0))
        cnt = torch.zeros(self.N, device=self.dev).index_add_(0, own.clamp(min=0),
                                                             (own >= 0).float())
        cnt, hits = cnt.cpu().numpy(), hits.cpu().numpy()
        return np.where(cnt > 0, hits / np.maximum(cnt, 1), np.nan)

    def _refl_aggregate(self, updates, weights, stale):
        """REFL's staleness-aware aggregation (Abdelmoniem et al. 2023, Eq. 5).

        Fresh updates get scale 1. A stale update s, tau_s rounds late, gets
        w_s = (1 - beta) / (tau_s + 1) + beta * (1 - exp(-Lambda_s / Lambda_max)),
        where Lambda_s = ||u_F - (u_s + n_F u_F) / (n_F + 1)||^2 / ||u_F||^2 is
        how far it pulls the mean of the n_F fresh updates u_F. The scales are
        multiplied into the usual data-size weights and normalised.
        """
        if not updates:
            return None
        beta = self.cfg.policy.refl_beta
        fresh = [u for u, s in zip(updates, stale) if s == 0]
        scales = [1.0] * len(updates)
        st_idx = [k for k, s in enumerate(stale) if s > 0]
        if st_idx:
            lam = {}
            if fresh:
                u_f = sum(fresh) / len(fresh)
                nf = len(fresh)
                den = max(float(u_f @ u_f), 1e-12)
                for k in st_idx:
                    mixed = (updates[k] + nf * u_f) / (nf + 1)
                    lam[k] = float(((u_f - mixed) ** 2).sum()) / den
            lam_max = max(lam.values()) if lam else 0.0
            for k in st_idx:
                boost = (1 - np.exp(-lam[k] / lam_max)) if lam_max > 0 else 0.0
                scales[k] = (1 - beta) / (stale[k] + 1) + beta * boost
        w = [wi * sc for wi, sc in zip(weights, scales)]
        return aggregate("fedavg", updates, w, [1] * len(updates))

    # --------------------------------------------------------------- round

    def run_round(self, t: int) -> dict:
        t0 = time.time()
        self.env.new_round(t)
        state = self._state(t)
        if getattr(self.policy, "needs_probe", False) and not self.probed:
            self.policy.init_probe(self._probe_all(), self.global_flat)
            self.probed = True

        selected, assign = self.policy.act(state, t)
        dg = None
        if self.diag is not None:
            dg = {"proposed": list(selected), "proposed_modes": dict(assign), "invites": []}
        if self.cfg.policy.mode_rule != "own":
            assign = {i: mode_by_rule(self.cfg, state["p_succ"][i]) for i in selected}
        fracs = dict(getattr(self.policy, "upload_frac", {}) or {})
        # A client still finishing last round's late update cannot take a new job.
        busy = {item[0] for item in self.late}
        if busy:
            selected = [i for i in selected if i not in busy]
        if dg is not None:
            dg["busy_removed"] = [i for i in dg["proposed"] if i in busy]
            dg["pre_budget"] = list(selected)
            dg["rule_modes"] = dict(assign)
        selected, assign, budget_dropped = self._enforce_budget(selected, assign, state, fracs)
        self._refresh_losses(selected)

        # V^repr_{i,t}(S_t) for every client (Eq. 3.26), not only the selected
        # ones: Eq. (3.27) sums ARL' over the full population C, so an
        # eligible, high-value client who was never invited must still show up
        # here with Y^succ = 0 via the success dict's default below.
        rep_vals = M.per_client_representation_value(self.d["sketch"], selected)
        values: dict[int, float] = {i: float(rep_vals[i]) for i in range(self.N)}

        updates, weights, steps, succ_p, recv_ids = [], [], [], [], []
        stale: list[int] = []                 # rounds of delay per received update (REFL)
        grad_var: dict[int, float] = {}       # DELTA: local gradient variance
        want_var = getattr(self.policy, "wants_grad_stats", False)
        compress = self.cfg.fl.compress_upload or self.cfg.fl.aggregator == "fedcg"
        success: dict[int, int] = {}
        reasons: dict[int, str] = {}

        # Semi-asynchronous deliveries from last round, now one round stale:
        # Eq. (3.18)'s staleness check decides whether they still count.
        late_delivered = 0
        late_ids: set[int] = set()
        for i, upd, n_steps, issued, p in self.late:
            if accepted(upd, issued, t, self.cfg):
                late_ids.add(i)
                updates.append(upd)
                weights.append(self.d["n_i"][i])
                steps.append(n_steps)
                succ_p.append(p)
                recv_ids.append(i)
                stale.append(t - issued)
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
            out = self.env.outcome(i, mode, frac)
            # Protocol fix (system.charge_actual): an offline client, or one
            # whose battery cannot cover the job, does no work and spends
            # nothing. Originally every invitation was charged in full.
            ran = not (self.cfg.system.charge_actual and out["reason"] in ("unavailable", "energy"))
            charged = self.env.cost(i, mode, frac) if ran else 0.0
            if ran:
                round_cost += charged
            reasons[i] = out["reason"]
            latencies.append(out["latency"])
            # Energy spend is deferred until after the oracle diagnostic below
            # is computed (Sec. 3.11, Eq. 3.29): applying it immediately would
            # make a client's own real energy draw down before its own
            # counterfactual is evaluated, so a client that genuinely could
            # have succeeded under the pre-decision energy state gets
            # incorrectly reported as oracle-ineligible.
            if ran:
                pending_spends.append((i, mode, frac))

            ok = out["success"] == 1
            if ok:
                x, y = self._client_xy(i)
                if len(y) == 0:
                    ok = False
                    reasons[i] = "rejected"
                else:
                    gs = {} if want_var else None
                    upd, n_steps = local_train(
                        self.cfg, self.global_flat, x, y,
                        self.cfg.system.local_epochs[mode],
                        self.d["dim"], self.d["num_classes"], self.gen, grad_stats=gs)
                    if compress:
                        upd = self._compress(i, upd, self._frac(mode, frac))
                    if accepted(upd, self.issued_round[i], t, self.cfg):
                        updates.append(upd)
                        weights.append(self.d["n_i"][i])
                        steps.append(n_steps)
                        succ_p.append(state["p_succ"][i][mode])
                        recv_ids.append(i)
                        stale.append(0)
                        if gs:
                            grad_var[i] = gs["var"]
                    else:
                        ok = False
                        reasons[i] = "rejected"
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
                    if compress:
                        upd = self._compress(i, upd, self._frac(mode, frac))
                    self.late.append((i, upd, n_steps, t, state["p_succ"][i][mode]))

            if dg is not None:
                dg["invites"].append(self._diag_invite(t, i, mode, frac, out, ok, reasons[i],
                                                       ran, charged, state, dg))
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
            if self.pv2:
                self.predictor.update_client(i, state["obs"][i], state["hist"][i], mode, int(ok))
            else:
                self.predictor.update(state["obs"][i], state["hist"][i], mode, int(ok))
            self.cal_probs.append(state["p_succ"][i][mode])
            self.cal_out.append(int(ok))
            self.cal_mode.append(mode)
            self.cal_weak.append(bool(self.env.weak[i]))

        if self.cfg.fl.absence_counter:
            # A client that was not invited (or whose late update did not
            # arrive) has also gone one more round without contributing.
            for i in range(self.N):
                if i not in success and i not in late_ids:
                    self.since_success[i] += 1

        if self.arl4 and self.cfg.reward.arl4_align and updates:
            # ARL-v4: how well each delivered update points downhill on the
            # VALIDATION loss of the model it was trained from.
            g = self._val_grad()
            gn = float(g.norm()) + 1e-12
            for i, u in zip(recv_ids, updates):
                cos = float(-(u @ g) / (gn * (float(u.norm()) + 1e-12)))
                self.align_ema[i] = 0.7 * self.align_ema[i] + 0.3 * cos

        agg = self.cfg.fl.aggregator
        base_flat = self.global_flat
        agg_w = getattr(self.policy, "agg_weights", None)
        if agg_w is not None:
            # DELTA: the policy's own unbiased (Horvitz-Thompson) weights,
            # applied as they are -- not renormalised over what arrived.
            delta = (sum(u * float(w) for u, w in zip(updates, agg_w(recv_ids)))
                     if updates else None)
        elif agg == "refl":
            delta = self._refl_aggregate(updates, weights, stale)
        elif agg == "mifa":
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

        # Keep a fixed held-out evaluation series for budget/target comparisons.
        # Final headline accuracy and client metrics still use the full test set.
        # Evaluate it last so _last_* describes the reported client population.
        if self.cfg.fl.practice:
            # Pre-training episode (RL2 M6): validation only, no test set at all.
            acc_r, _, worst_r = self._evaluate(idx=self.val_idx)
            acc = test_loss = acc_worst = float("nan")
        elif self.val_idx is not None:
            # Reward and learning baselines see the validation set only; the
            # reported accuracy is computed afterwards and never fed back.
            acc_r, _, worst_r = self._evaluate(idx=self.val_idx)
            acc, test_loss, acc_worst = self._evaluate(full=False)
        else:
            acc_r, loss_r, worst_r = self._evaluate(full=False)
            acc, test_loss, acc_worst = acc_r, loss_r, worst_r
        acc_fixed = acc
        eval_fixed_n = 0 if self.cfg.fl.practice else (
            len(self._last_ok) if self._last_ok is not None else 0)
        if t == self.cfg.fl.rounds and self.eval_idx is not None and not self.cfg.fl.practice:
            acc, test_loss, acc_worst = self._evaluate(full=True)
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
        # Set-level coverage loss: what the delivered set covers vs. the invited
        # set, and vs. what it would have covered had every invited client that
        # could succeed in *some* mode this round been given that mode (same
        # random draws). Unlike a sum of per-client values, this sees two
        # failed holders of the same rare class.
        sketch = self.d["sketch"]
        delivered = [i for i in selected if success.get(i, 0)]
        feasible = [i for i in selected if self.env.oracle_eligible(i)]
        cov_d = M.coverage(sketch[delivered])
        cov_lost = M.coverage(sketch[selected]) - cov_d if selected else 0.0
        cov_lost_avoid = M.coverage(sketch[feasible]) - cov_d if feasible else 0.0
        cv = M.cv_fairness(self.x_succ)          # successful-contribution CV (core metric, Eq. 3.32)
        cv_sel = M.cv_fairness(self.x_sel)        # raw-selection-count CV (Task 2 ablation comparator)
        n_succ = int(sum(success.values()))
        # No invitations means no failed invitations. Log empty rounds so a
        # zero here cannot be mistaken for evidence of successful service.
        fail_ratio = 1.0 - n_succ / len(selected) if selected else 0.0

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

        # ARL-v3 (ARL-v2 fixed, docs/ADVANCED_RESULTS.md). ARL-v2 measured the
        # loss as success probability given up, which Light -- always the
        # safest mode -- never gives up, so it taught the policy to put almost
        # everyone on Light and do a third less training. ARL-v3 measures
        # expected useful work instead,
        #   W(m) = P(success | m) x local_epochs[m]  for modes safe enough to
        #          plan on (P >= elig_threshold, or the safest one if none is),
        #   W(m) = 0                                 for a gamble below that,
        # and gives credit for getting the data in:
        #   loss_i = m_i * (W(best mode) - W(assigned mode)) / epochs_max
        #            (best = base.best_work_mode; zero when the policy chose it)
        #   gain_i = m_i * Y_i
        # with m_i = V_i * (1 + share of i's data in lately under-covered
        # classes), so both count most for rare, recently missing data. The
        # gain is what makes inviting such a client, and keeping it alive,
        # pay off; the loss pushes for as much training as is safe. The loss
        # is an expectation, not an outcome, so it grades the mode choice
        # with low variance.
        ep = self.cfg.system.local_epochs
        ep_max = float(max(ep.values()))
        thr = self.cfg.policy.elig_threshold
        phi = M.coverage_shapley(sketch, selected)
        v3_loss: dict[int, float] = {}
        v3_gain: dict[int, float] = {}
        for i in selected:
            p = state["p_succ"][i]
            # Shapley share of the invited set's coverage, not the leave-one-out
            # value: two invited holders of the same rare class get half each,
            # instead of zero each.
            mult = phi[i] * (1.0 + sketch[i] @ deficit)
            best = best_work_mode(self.cfg, p)
            safe = {m for m in p if p[m] >= thr} or {best}
            work = {m: (p[m] * ep[m] if m in safe else 0.0) for m in p}
            v3_loss[i] = float(mult * (work[best] - work[assign[i]]) / ep_max)
            v3_gain[i] = float(mult * success.get(i, 0))
        arl_v3_loss = float(sum(v3_loss.values()))
        arl_v3_gain = float(sum(v3_gain.values()))
        ok_ids = [i for i in selected if success.get(i, 0)]
        cov = sketch[ok_ids].max(axis=0) if ok_ids else np.zeros_like(self.cov_ema)
        b = self.cfg.reward.arl_v2_ema
        self.cov_ema = (1 - b) * self.cov_ema + b * cov

        # ARL-v4 (docs/RL2_DESIGN.md): expected work the chosen mode gave up
        # against the best confidently feasible, affordable mode; a deficit
        # term that grows while classes go uncovered; per-client difference
        # credit on deficit-weighted delivered coverage.
        rw = self.cfg.reward
        self.cov_peak, self.cov_dur = M.deficit_update(self.cov_ema, self.cov_peak, self.cov_dur, cov)
        arl_v4_deficit = M.deficit_term(self.cov_ema, self.cov_peak, self.cov_dur)
        per_budget = None
        if self.cfg.policy.cost_target_mult > 0 and selected:
            per_budget = self.cfg.policy.cost_target_mult * state["cost_ref"] / max(
                min(self.cfg.fl.clients_per_round, self.N), 1)
        sd = state.get("p_sd")
        v4: dict[int, float] = {}
        for i in selected:
            p = state["p_succ"][i]
            pc = {m: max(0.0, p[m] - (rw.arl4_z * sd[i][m] if sd else 0.0)) for m in p}
            afford = (None if per_budget is None else
                      {m for m in p if self.env.cost(i, m) <= per_budget})
            q = 1.0 + max(0.0, float(self.align_ema[i])) if rw.arl4_align else 1.0
            mult = phi[i] * (1.0 + sketch[i] @ deficit) * q
            v4[i] = float(mult * M.arl4_expected_gap(pc, assign[i], ep, thr, afford))
        arl_v4_loss = float(sum(v4.values()))
        v4_credit = M.difference_credit(sketch, ok_ids, 1.0 + deficit)
        v4_credit = {i: v4_credit.get(i, 0.0) for i in selected}

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
        elif r.damage_metric == "arl_v3":
            if self.cfg.policy.per_client_credit:
                damage_term = 0.0
            else:
                damage_term = (self.norm["arl"](arl_v3_loss)
                               - r.lambda_cov / max(r.lambda_arl, 1e-9) * self.norm["v3g"](arl_v3_gain))
        elif r.damage_metric == "arl_v4":
            if self.cfg.policy.per_client_credit:
                damage_term = r.arl4_mu * self.norm["v4d"](arl_v4_deficit)
            else:
                damage_term = (self.norm["arl"](arl_v4_loss)
                               + r.arl4_mu * self.norm["v4d"](arl_v4_deficit)
                               - r.lambda_cov / max(r.lambda_arl, 1e-9)
                               * self.norm["v4g"](float(sum(v4_credit.values()))))
        else:
            raise ValueError(f"unknown reward.damage_metric {r.damage_metric}")
        fairness_term = cv_sel if r.fairness_metric == "cv_selection" else cv

        # Same calls, same order and same arithmetic as one expression; named
        # only so the diagnostics can record each part.
        z_acc = self.norm["acc"](d_acc)
        z_cost = self.norm["cost"](round_cost)
        z_lat = self.norm["lat"](latency)
        reward = (r.alpha * z_acc
                  - r.beta * z_cost
                  - r.delta * z_lat
                  - r.lambda_arl * damage_term
                  - r.lambda_fair * fairness_term)
        z_debt = None
        if r.lambda_debt > 0:
            z_debt = self.norm["debt"](float(self.debt.mean()))
            reward -= r.lambda_debt * z_debt
        z_prot = None
        if r.protect_weight > 0:
            # Reward v2 protection term: gain in the worst-10%-class accuracy
            # on the validation set (never the test set).
            d_worst = worst_r - (self._prev_worst_r if self._prev_worst_r is not None else worst_r)
            self._prev_worst_r = worst_r
            z_prot = self.norm["prot"](d_worst)
            reward += r.protect_weight * z_prot
        shared_reward = reward
        client_adv = None
        if r.batch_client_norm:
            # Protocol fix: one set of statistics for the whole round, so a
            # client's credit does not depend on where it sits in the order.
            def norm(key, vals):
                return dict(zip(selected, self.norm[key].batch([vals[i] for i in selected])))
        else:
            def norm(key, vals):
                return {i: self.norm[key](vals[i]) for i in selected}
        set_adv = None
        if r.damage_metric == "arl_v2" and self.cfg.policy.per_client_credit:
            z = norm("v2c", v2)
            client_adv = {i: -r.lambda_arl * z[i] for i in selected}
            if self.cfg.policy.credit_split:
                set_adv = -r.lambda_arl * self.norm["v2s"].batch([arl_v2])[0]
        elif r.damage_metric == "arl_v4" and self.cfg.policy.per_client_credit:
            zl, zg = norm("v4c", v4), norm("v4cg", v4_credit)
            client_adv = {i: -r.lambda_arl * zl[i] + r.lambda_cov * zg[i] for i in selected}
        elif r.damage_metric == "arl_v3" and self.cfg.policy.per_client_credit:
            zl, zg = norm("v3c", v3_loss), norm("v3cg", v3_gain)
            client_adv = {i: -r.lambda_arl * zl[i] + r.lambda_cov * zg[i] for i in selected}
            if self.cfg.policy.credit_split:
                # The round's totals, standardised with the statistics from
                # before the round (a baseline that does not depend on this
                # round's action).
                set_adv = (-r.lambda_arl * self.norm["v3s"].batch([arl_v3_loss])[0]
                           + r.lambda_cov * self.norm["v3sg"].batch([arl_v3_gain])[0])
        credits = None if client_adv is None else dict(client_adv)   # diagnostics
        set_credit = set_adv
        if client_adv is not None and self.cfg.policy.credit_total:
            # One round objective, R = shared reward + sum of the clients'
            # standardised ARL credits, for the whole joint action.
            reward = reward + float(sum(client_adv.values()))
            client_adv, set_adv = None, None
        self.policy.observe_outcome(state, (selected, assign), reward, {
            "success": success, "client_adv": client_adv, "set_adv": set_adv, "t": t,
            # For the baselines that learn from what arrived (FAVOR, DELTA, REFL).
            "acc": acc_r, "recv_ids": list(recv_ids), "updates": dict(zip(recv_ids, updates)),
            "stale": dict(zip(recv_ids, stale)), "grad_var": grad_var, "base_flat": base_flat,
            "cost": round_cost,
        })

        weak = self.env.weak
        row = {
            "round": t, "accuracy": acc, "acc_worst10": acc_worst,
            "accuracy_fixed": acc_fixed, "eval_fixed_n": eval_fixed_n,
            "test_loss": test_loss, "reward": reward,
            "cost": round_cost, "latency": latency,
            "n_selected": len(selected), "n_success": n_succ,
            "fail_ratio": fail_ratio, "empty_round": int(not selected),
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
            "arl_v3_loss": arl_v3_loss, "arl_v3_gain": arl_v3_gain,
            "arl_v4_loss": arl_v4_loss, "arl_v4_deficit": arl_v4_deficit,
            "val_accuracy": acc_r, "cov_lost": cov_lost, "cov_lost_avoid": cov_lost_avoid,
            **(self._practice_metrics() if self.cfg.fl.practice
               else self._client_metrics(success, late_ids)),
            "gate_filled": int(getattr(self.policy, "gate_filled", 0)),
            "wall_s": time.time() - t0,
        }
        self.rows.append(row)
        if dg is not None:
            self._diag_round(t, state, selected, assign, success, reasons, recv_ids, late_ids,
                             mode_counts, dg, row, {
                                 "d_acc": d_acc, "acc_for_reward": acc_r, "cost": round_cost,
                                 "latency": latency, "damage_metric": r.damage_metric,
                                 "damage_raw": {"fail_ratio": fail_ratio, "arl_selected": arl_sel,
                                                "arl_online": arl_on, "arl_v2": arl_v2,
                                                "arl_v3_loss": arl_v3_loss, "arl_v3_gain": arl_v3_gain,
                                                "arl_v4_loss": arl_v4_loss, "arl_v4_deficit": arl_v4_deficit},
                                 "fairness_raw": fairness_term, "debt_raw": float(self.debt.mean()),
                                 "z": {"acc": z_acc, "cost": z_cost, "lat": z_lat, "debt": z_debt,
                                       "protect": z_prot,
                                       "damage_term": damage_term},
                                 "norm": {"acc": norm_state(self.norm["acc"], z_acc),
                                          "cost": norm_state(self.norm["cost"], z_cost),
                                          "lat": norm_state(self.norm["lat"], z_lat)},
                                 "weights": {"alpha": r.alpha, "beta": r.beta, "delta": r.delta,
                                             "lambda_arl": r.lambda_arl, "lambda_fair": r.lambda_fair,
                                             "lambda_debt": r.lambda_debt, "lambda_cov": r.lambda_cov},
                                 "shared_reward": shared_reward, "client_credit": credits,
                                 "set_credit": set_credit,
                                 "credit_total": bool(self.cfg.policy.credit_total), "total": reward})
        return row

    # --------------------------------------------------------- diagnostics

    def _diag_invite(self, t, i, mode, frac, out, ok, reason, ran, charged, state, dg) -> dict:
        """One invitation, as the server saw it and as it turned out. Reads
        only; the analysis-only block holds simulator truths no policy sees."""
        dec = getattr(self.policy, "diag_decision", None) or {}
        pick = next((p for p in dec.get("picks", []) if p["client"] == i), {})
        env = self.env
        return {
            "t": t, "client": int(i), "order": dg["proposed"].index(i) if i in dg["proposed"] else None,
            "how": pick.get("how", "fallback" if i in getattr(self.policy, "gate_fill", []) else "chosen"),
            "logp_pick": pick.get("logp_pick"), "mode_probs": pick.get("mode_probs"),
            "mode_entropy": pick.get("mode_entropy"),
            "cov_gain": pick.get("cov_gain"), "cov_gain_weighted": pick.get("cov_gain_weighted"),
            "proposed_mode": dg["proposed_modes"].get(i), "rule_mode": dg["rule_modes"].get(i),
            "mode": mode, "epochs": self.cfg.system.local_epochs[mode],
            "upload_frac": self._frac(mode, frac),
            "p_succ": dict(state["p_succ"][i]), "eligible": int(state["eligible"][i]),
            "obs": dict(state["obs"][i]), "hist": dict(state["hist"][i]),
            "t_est": {m: state["t_est"][i][m] for m in self.cfg.system.modes},
            "env_reason": out["reason"], "reason": reason, "success": int(ok),
            "latency": out["latency"], "t_train": out["t_train"], "t_comm": out["t_comm"],
            "ran": bool(ran), "charged": charged, "late_queued": any(x[0] == i for x in self.late),
            "analysis_only": {
                "weak": bool(env.weak[i]), "tier": int(env.tier[i]),
                "available": bool(env._draw["available"][i]),
                "energy_at_start": float(env._energy_at_round_start[i]),
                "energy_cap": float(env.energy_cap[i]), "reliability": float(env.reliability[i]),
                "drop_roll": float(env._draw["drop"][i]),
                "t_true": {m: state["t_expected"][i][m] for m in self.cfg.system.modes},
            },
        }

    def _diag_round(self, t, state, selected, assign, success, reasons, recv_ids, late_ids,
                    mode_counts, dg, row, reward_parts) -> None:
        modes = self.cfg.system.modes
        p = state["p_succ"]
        cands = state["candidates"] if state["candidates"] is not None else list(range(self.N))
        pmax = [max(p[i].values()) for i in range(self.N)]
        dec = getattr(self.policy, "diag_decision", None)
        reason_counts: dict[str, int] = {}
        for why in reasons.values():
            reason_counts[why] = reason_counts.get(why, 0) + 1
        w = self.predictor.w
        rec = {
            "t": t,
            "online": int(self.env._draw["available"].sum()),
            "departed": int(self.env.departed.sum()),
            "candidates": len(cands),
            "eligible": int(sum(state["eligible"][i] for i in cands)),
            "elig_threshold": self.cfg.policy.elig_threshold,
            "p_max": [round(v, 5) for v in pmax],
            "p_mean_by_mode": {m: float(np.mean([p[i][m] for i in cands])) if cands else None
                               for m in modes},
            "fill": list(getattr(self.policy, "gate_fill", []) or []),
            "gate_filled": int(getattr(self.policy, "gate_filled", 0)),
            "proposed": dg["proposed"], "busy_removed": dg["busy_removed"],
            "budget_removed": [i for i in dg["pre_budget"] if i not in selected],
            "invited": list(selected), "modes": {int(i): assign[i] for i in selected},
            "succeeded": [i for i in selected if success.get(i, 0)],
            "late_delivered": sorted(late_ids), "late_queued": [x[0] for x in self.late],
            "aggregated": len(recv_ids),
            "unique_contributors": int((self.x_succ > 0).sum()),
            "mode_counts": dict(mode_counts), "reason_counts": reason_counts,
            "empty_round": int(not selected),
            "decision": None if dec is None else {k: v for k, v in dec.items() if k != "picks"},
            "predictor": {"seen": int(self.predictor.seen), "w_norm": float(np.linalg.norm(w)),
                          "w": [float(v) for v in w], "b": float(self.predictor.b)},
            "reward": reward_parts,
            "accuracy": row["accuracy"], "accuracy_fixed": row["accuracy_fixed"],
        }
        self.diag.write("round", rec)
        for inv in dg["invites"]:
            self.diag.write("invite", inv)
        upd = getattr(self.policy, "diag_update", None)
        if upd is not None:
            self.diag.write("update", {"t": t, **upd})
        self.diag.flush()

    # ----------------------------------------------------------------- run

    def run(self, verbose: bool = True) -> list[dict]:
        for t in range(1, self.cfg.fl.rounds + 1):
            row = self.run_round(t)
            if verbose and (t % max(1, self.cfg.fl.rounds // 20) == 0 or t == 1):
                print(f"[{self.cfg.tag}] r{t:3d} acc={row['accuracy']:.4f} "
                      f"succ={row['n_success']}/{row['n_selected']} "
                      f"ARL'={row['arl_online']:.3f} ARL*={row['arl_oracle']:.3f} "
                      f"CV={row['cv']:.3f} R={row['reward']:.3f}", flush=True)
        self.policy.finish_run()
        return self.rows

    def save(self) -> str:
        if self.cfg.fl.practice:
            raise RuntimeError("a practice (pre-training) episode is never saved as a result")
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
        if self.diag is not None:
            self.diag.close()
        meta = self.cfg.to_dict()
        meta["init_checksum"] = self.init_checksum   # same for every method under fl.clean_rng
        # Which version of the simulator made this run: the code loaded at
        # start-up. If the files changed while it ran, say so -- the sweep and
        # the confirmatory report then refuse to use the result.
        meta["code_hash"] = CODE_LOADED
        if self.split_fp:
            meta["split_fingerprint"] = self.split_fp
        if getattr(self.policy, "checkpoint_sha", None):
            meta["init_checkpoint_sha256"] = self.policy.checkpoint_sha
        if code_hash() != CODE_LOADED:
            meta["code_changed_during_run"] = True
            print(f"[{self.cfg.tag}] WARNING: simulator source changed during the run; "
                  "result is flagged and will not be reused", flush=True)
        with open(os.path.join(self.cfg.out, f"{self.cfg.tag}.config.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        return path
