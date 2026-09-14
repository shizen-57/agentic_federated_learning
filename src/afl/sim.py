"""Ground-truth client environment (Sec. 3.6, 3.7).

The simulator owns everything the server is not allowed to see: true device
speed, true channel state, true availability. It exposes two interfaces.

``observe``     the noisy per-client descriptors the server may condition on
                (o_t of Sec. 3.8).
``outcome``     the ground-truth participation outcome tilde-Y^succ_{i,t}(E) of a
                client under a configuration.

``outcome`` is driven entirely by per-round random numbers drawn *before* any
action is chosen. That is the common-random-numbers construction the thesis
requires for ARL* (Eq. 3.29): asking "what would client i have done under
Priority instead of Light?" re-uses the same jitter and channel draw, so the
counterfactual differs only by the decision, never by luck.
"""
from __future__ import annotations

import numpy as np


class ClientEnv:
    def __init__(self, cfg, n_i: np.ndarray, seed: int):
        self.cfg = cfg
        self.N = cfg.data.num_clients
        self.n_i = n_i
        rng = np.random.default_rng(seed + 1000)

        # Static device profile. Three tiers, so a valuable-but-slow client is a
        # thing that actually exists in the population rather than an accident.
        tier = rng.choice([0, 1, 2], size=self.N, p=[0.35, 0.4, 0.25])
        self.speed = np.array([0.35, 1.0, 2.2])[tier] * rng.uniform(0.8, 1.2, self.N)
        self.bandwidth = np.array([0.3, 1.0, 2.0])[tier] * rng.uniform(0.7, 1.3, self.N)
        self.reliability = np.clip(0.55 + 0.15 * tier + rng.normal(0, 0.08, self.N), 0.3, 0.98)
        self.energy_cap = np.array([0.4, 1.0, 2.5])[tier] * rng.uniform(0.8, 1.2, self.N)
        self.tier = tier

        self._round_rng: np.random.Generator | None = None
        self._draw: dict | None = None
        self.energy = self.energy_cap.copy()

    # ------------------------------------------------------------------ round

    def new_round(self, t: int) -> None:
        """Draw every random number this round will ever need, up front."""
        rng = np.random.default_rng([self.cfg.seed, t])
        self._round_rng = rng
        self._draw = {
            "available": rng.random(self.N) < self.cfg.system.availability_p,
            "compute_jitter": rng.lognormal(0.0, 0.35, self.N),
            "channel": rng.lognormal(0.0, 0.45, self.N),
            "drop": rng.random(self.N),          # upload / link failure roll
            "accept": rng.random(self.N),        # update-validation roll
        }
        # Energy recharges a little each round; a client drained last round may
        # be back but weaker, which is what makes availability non-stationary.
        self.energy = np.minimum(self.energy_cap, self.energy + 0.25 * self.energy_cap)

    # ------------------------------------------------------------- mechanics

    def workload(self, i: int, mode: str) -> tuple[float, float, float]:
        """Return (train_time, comm_time, energy) for client i under a mode.

        The constants are calibrated so that the three modes sit on different
        sides of the deadline for different device tiers: Light is feasible for
        almost everyone, Standard is feasible for mid and fast tiers, and
        Priority is out of reach for slow devices. Without that separation the
        configuration half of the action would be decorative -- the policy would
        have nothing to trade off.
        """
        sysc = self.cfg.system
        epochs = sysc.local_epochs[mode]
        frac = sysc.upload_frac[mode]
        d = self._draw
        t_train = epochs * (self.n_i[i] / 1200.0) / self.speed[i] * d["compute_jitter"][i]
        t_comm = 0.25 * frac / self.bandwidth[i] * d["channel"][i]
        energy = 0.08 * epochs / self.speed[i] + 0.05 * frac
        return t_train, t_comm, energy

    def cost(self, i: int, mode: str) -> float:
        """C^comp + C^comm of Sec. 3.6, in normalised units."""
        sysc = self.cfg.system
        return (sysc.local_epochs[mode] * self.n_i[i] / 1000.0
                + sysc.upload_frac[mode] * 1.0)

    def outcome(self, i: int, mode: str) -> dict:
        """tilde-Y^succ_{i,t}(E): completion, upload, deadline, acceptance.

        Pure with respect to the round's drawn random numbers, so it can be
        called for configurations that were never assigned (the ARL* oracle).
        """
        d = self._draw
        sysc = self.cfg.system
        if not d["available"][i]:
            return dict(success=0, latency=sysc.deadline, reason="unavailable",
                        t_train=0.0, t_comm=0.0)

        t_train, t_comm, energy = self.workload(i, mode)
        total = t_train + t_comm

        if energy > self.energy[i]:
            return dict(success=0, latency=total, reason="energy",
                        t_train=t_train, t_comm=t_comm)
        if d["drop"][i] > self.reliability[i]:
            return dict(success=0, latency=total, reason="dropped",
                        t_train=t_train, t_comm=t_comm)
        if total > sysc.deadline:
            return dict(success=0, latency=total, reason="deadline",
                        t_train=t_train, t_comm=t_comm)
        # Acceptance (Eq. 3.18) is mostly decided by the norm check inside the
        # trainer; the simulator only injects a small residual rejection rate.
        if d["accept"][i] > 0.98:
            return dict(success=0, latency=total, reason="rejected",
                        t_train=t_train, t_comm=t_comm)
        return dict(success=1, latency=total, reason="ok",
                    t_train=t_train, t_comm=t_comm)

    def spend(self, i: int, mode: str) -> None:
        _, _, energy = self.workload(i, mode)
        self.energy[i] = max(0.0, self.energy[i] - energy)

    # -------------------------------------------------------------- observed

    def observe(self, i: int) -> dict:
        """Noisy descriptors the server is allowed to condition on (Sec. 3.8)."""
        rng = self._round_rng
        return {
            "bw_hat": float(self.bandwidth[i] * rng.lognormal(0, 0.15)),
            "speed_hat": float(self.speed[i] * rng.lognormal(0, 0.15)),
            "energy_hat": float(self.energy[i] / self.energy_cap[i]),
            "avail_hint": float(self._draw["available"][i] * rng.uniform(0.7, 1.0)
                                + (1 - self._draw["available"][i]) * rng.uniform(0.0, 0.3)),
        }

    def oracle_eligible(self, i: int) -> int:
        """e^oracle_{i,t} (Eq. 3.28): does *any* mode succeed for i this round?"""
        return int(any(self.outcome(i, m)["success"] == 1 for m in self.cfg.system.modes))
