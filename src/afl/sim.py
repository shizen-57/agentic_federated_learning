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

Calibration against real FL deployments. This remains a simulation -- it is
not a replay of a published trace file and not a deployment on physical
devices -- but two of its mechanisms are shaped to match specific, cited
measurements from real FL systems rather than picked arbitrarily:

- Availability follows a diurnal (day/night) cycle, not a flat per-round coin
  flip. Real device eligibility (charging + idle + unmetered network) is
  concentrated overnight; Hard et al. 2018 (Google's applied Gboard FL system,
  arXiv:1812.02903) report that round completion is "nearly flat" around noon
  and that the cycle's phase, shape, and amplitude vary per device (by
  timezone/demographics/usage pattern) -- both properties are implemented
  below via a per-client phase offset and amplitude scale on top of a shared
  cosine cycle.
- Device speed/bandwidth/reliability are drawn from a small number of
  heavy-tailed tiers rather than a single homogeneous value. This mirrors the
  qualitative shape of the trace-driven simulation approach used by FedScale
  and Oort (Lai et al., NSDI 2021 / ICML 2022), which calibrate device
  heterogeneity from real measured smartphone benchmarks (AI Benchmark)
  rather than assuming identical clients.
"""
from __future__ import annotations

import numpy as np


def rarity_score(sketch: np.ndarray) -> np.ndarray:
    """How much of a client's data sits in classes that few clients hold.

    score_i = sum_k c_{i,k} * log(1 / q_k), with q_k the mean share of class k
    across clients. A client whose data is mostly in rare classes scores high.
    """
    q = np.clip(sketch.mean(axis=0), 1e-6, None)
    return sketch @ np.log(1.0 / q)


class ClientEnv:
    def __init__(self, cfg, n_i: np.ndarray, seed: int, sketch: np.ndarray | None = None):
        self.cfg = cfg
        self.N = cfg.data.num_clients
        self.n_i = n_i
        sysc = cfg.system
        # Simulated workload is driven by n_work, not n_i, so the deadline
        # separation between modes can be held fixed across datasets.
        self.n_work = (n_i * (sysc.work_ref_samples / max(float(n_i.mean()), 1e-9))
                       if sysc.work_ref_samples > 0 else n_i)
        rng = np.random.default_rng(seed + 1000)

        # Persistently weak clients (off unless system.weak_frac > 0). Drawn from
        # their own stream so the default device profile below is unchanged.
        self.weak = np.zeros(self.N, dtype=bool)
        n_weak = int(round(sysc.weak_frac * self.N))
        rng_w = np.random.default_rng(seed + 2000)
        if n_weak > 0:
            if sysc.weak_select == "rarity" and sketch is not None:
                weak_idx = np.argsort(-rarity_score(sketch), kind="stable")[:n_weak]
            elif sysc.weak_select in ("rarity", "random"):
                weak_idx = rng_w.choice(self.N, size=n_weak, replace=False)
            else:
                raise ValueError(f"unknown system.weak_select {sysc.weak_select}")
            self.weak[weak_idx] = True

        # Static device profile. Three tiers, so a valuable-but-slow client is a
        # thing that actually exists in the population rather than an accident.
        tier = rng.choice([0, 1, 2], size=self.N, p=[0.35, 0.4, 0.25])
        tier[self.weak] = 0                      # weak clients are always low-end
        self.speed = np.array([0.35, 1.0, 2.2])[tier] * rng.uniform(0.8, 1.2, self.N)
        self.bandwidth = np.array([0.3, 1.0, 2.0])[tier] * rng.uniform(0.7, 1.3, self.N)
        self.reliability = np.clip(0.55 + 0.15 * tier + rng.normal(0, 0.08, self.N), 0.3, 0.98)
        self.energy_cap = np.array([0.4, 1.0, 2.5])[tier] * rng.uniform(0.8, 1.2, self.N)
        self.tier = tier
        if n_weak > 0:
            self.reliability[self.weak] = np.clip(
                sysc.weak_reliability + rng_w.normal(0, 0.05, n_weak), 0.1, 0.98)
        self.avail_shift = np.where(self.weak, -sysc.weak_avail_drop, 0.0)
        self._prev_available: np.ndarray | None = None

        # Diurnal availability profile: each client gets its own peak hour and
        # amplitude scale, matching Hard et al. 2018's finding that the cycle's
        # phase/shape/amplitude differ per device (timezone, demographics,
        # usage pattern) rather than every client sharing one global clock.
        # Peak hours cluster around a shared overnight-charging window (~3am)
        # rather than spreading uniformly across the day: real clients in one
        # deployment mostly charge overnight in their own local time, with
        # modest per-client spread (timezone, habits) rather than a totally
        # random phase -- a uniform spread would average out to a flat
        # population-level curve and lose the very midday dip Hard et al.
        # report.
        self.peak_hour = np.mod(rng.normal(3.0, 3.0, self.N), 24.0)
        self.amplitude_scale = np.clip(rng.normal(1.0, 0.25, self.N), 0.3, 1.6)

        self._round_rng: np.random.Generator | None = None
        self._draw: dict | None = None
        self.energy = self.energy_cap.copy()

    def availability_prob(self, t: int) -> np.ndarray:
        """P(A_{i,t}=1) for every client: a per-client diurnal cosine cycle.

        Round t maps to an hour-of-day via rounds_per_day; each client's
        probability oscillates around the configured mean with its own peak
        hour and amplitude, clipped to a sane range so the mean stays
        controllable via system.availability_p.
        """
        sysc = self.cfg.system
        hour = 24.0 * (t % sysc.rounds_per_day) / sysc.rounds_per_day
        phase = 2 * np.pi * (hour - self.peak_hour) / 24.0
        wave = sysc.diurnal_amplitude * self.amplitude_scale * np.cos(phase)
        return np.clip(sysc.availability_p + wave + self.avail_shift, 0.03, 0.99)

    # ------------------------------------------------------------------ round

    def new_round(self, t: int) -> None:
        """Draw every random number this round will ever need, up front."""
        rng = np.random.default_rng([self.cfg.seed, t])
        self._round_rng = rng
        self._draw = {
            "available": rng.random(self.N) < self.availability_prob(t),
            "compute_jitter": rng.lognormal(0.0, 0.35, self.N),
            "channel": rng.lognormal(0.0, 0.45, self.N),
            "drop": rng.random(self.N),          # upload / link failure roll
            "accept": rng.random(self.N),        # update-validation roll
        }
        # Sticky availability: with probability rho a client keeps last round's
        # on/off state instead of re-rolling, so an offline client tends to stay
        # offline for a stretch. The extra draw comes after the others (and only
        # when rho > 0) so runs without persistence are unchanged; it is still
        # drawn before any action, so common random numbers hold.
        rho = self.cfg.system.avail_persistence
        if rho > 0:
            keep = rng.random(self.N) < rho
            if self._prev_available is not None:
                self._draw["available"] = np.where(keep, self._prev_available,
                                                   self._draw["available"])
        self._prev_available = self._draw["available"].copy()
        # Energy recharges a little each round; a client drained last round may
        # be back but weaker, which is what makes availability non-stationary.
        self.energy = np.minimum(self.energy_cap, self.energy + 0.25 * self.energy_cap)
        # Snapshot for outcome()'s feasibility check (see there): every query
        # this round, real or counterfactual, must see the same pre-spend
        # energy regardless of how many spend() calls happen in between, or
        # the ARL* oracle (Eq. 3.29) silently depends on call order.
        self._energy_at_round_start = self.energy.copy()

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
        t_train = epochs * (self.n_work[i] / 1200.0) / self.speed[i] * d["compute_jitter"][i]
        t_comm = 0.25 * frac / self.bandwidth[i] * d["channel"][i]
        energy = 0.08 * epochs / self.speed[i] + 0.05 * frac
        return t_train, t_comm, energy

    def cost(self, i: int, mode: str) -> float:
        """C^comp + C^comm of Sec. 3.6, in normalised units."""
        sysc = self.cfg.system
        return (sysc.local_epochs[mode] * self.n_work[i] / 1000.0
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

        if energy > self._energy_at_round_start[i]:
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
