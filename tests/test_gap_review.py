"""Fixes from the project gap review (docs/GAP_REVIEW.md): matched
comparators (A1), the fairness-queue baseline (C5), the new log columns
(A3, B6, B10), the configurable baseline settings for matched tuning (B1)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]


def _cfg(policy: dict | None = None, rounds: int = 8, **fl):
    from afl.config import Cfg
    return Cfg.from_dict({"data": {"dataset": "synthetic", "num_clients": 16}, "out": "/tmp/afl_gap",
                          "fl": {"rounds": rounds, "clients_per_round": 4, "model": "mlp", "clean_rng": True,
                                 "val_frac": 0.3, **fl},
                          "system": {"charge_actual": True, "checkin_pool": True},
                          "policy": {"strict_gate": True, **(policy or {})}})


def test_every_comparator_has_a_mimic_version_that_differs_only_in_aggregation():
    from afl.conditions import CONDITIONS
    for c in ("arl3_full_pg", "v2_rawfail", "b_oort_ew", "b_favor_ew", "b_delta_ew", "b_cov_ew", "b_fairq_ew"):
        a, b = CONDITIONS[c], CONDITIONS[f"{c}_mimic"]
        assert b["fl.aggregator"] == "mimic" and {k: v for k, v in a.items() if k != "fl.aggregator"} == \
            {k: v for k, v in b.items() if k != "fl.aggregator"}
    assert CONDITIONS["ppo_full_mimic"]["fl.aggregator"] == CONDITIONS["rl_light_mimic"]["fl.aggregator"]


def _state(n, online, t_est=None, p=0.9):
    modes = ("light", "standard", "priority")
    return {"candidates": list(online), "p_succ": [{m: p for m in modes} for _ in range(n)],
            "obs": [{"speed_hat": 1.0 if t_est is None else 1.0 / t_est[i], "bw_hat": 1.0} for i in range(n)],
            "n_work": np.full(n, 500.0)}


def test_fairness_queues_guarantee_every_online_client_its_share():
    """RBCS-F's mechanism: a slow client is skipped while its queue is small,
    then picked once its debt outweighs the efficiency gap -- over time every
    online client reaches about the promised rate beta."""
    from afl.policies.published import FairQueuePolicy
    cfg = _cfg({"name": "fairq", "fairq_beta": 0.2, "fairq_v": 1.0, "mode_rule": "light"})
    pol = FairQueuePolicy(cfg)
    n = 16
    t_est = [0.5] * 8 + [3.0] * 8                     # half fast, half very slow
    picks = np.zeros(n)
    rounds = 400
    for t in range(rounds):
        sel, modes = pol.act(_state(n, range(n), t_est), t)
        assert len(sel) == 4 and set(modes.values()) == {"standard"}   # the engine applies the mode rule
        picks[sel] += 1
    assert (picks / rounds >= 0.2 - 0.02).all()       # the promised rate, slow clients included
    # without the fairness queues (beta = 0) the slow clients are never picked
    pol0 = FairQueuePolicy(_cfg({"name": "fairq", "fairq_beta": 0.0, "fairq_v": 1.0, "mode_rule": "light"}))
    p0 = np.zeros(n)
    for t in range(50):
        p0[pol0.act(_state(n, range(n), t_est), t)[0]] += 1
    assert p0[8:].sum() == 0 and p0[:8].sum() == 200


def test_fairness_queues_only_grow_while_a_client_is_online():
    from afl.policies.published import FairQueuePolicy
    pol = FairQueuePolicy(_cfg({"name": "fairq", "fairq_beta": 0.1, "mode_rule": "light"}))
    for t in range(20):
        pol.act(_state(16, range(12)), t)              # clients 12-15 are never online
    assert (pol.Z[12:] == 0).all()
    assert pol.act(_state(16, []), 0) == ([], {})


def test_fairq_runs_end_to_end_and_new_columns_are_logged(tmp_path):
    from afl.engine import Experiment, LOG_FIELDS
    for col in ("cost_ref", "budget_lighter", "n_never", "n_gap50"):
        assert col in LOG_FIELDS
    cfg = _cfg({"name": "fairq", "mode_rule": "expected_work", "hard_budget_mult": 0.0}, rounds=6,
               aggregator="mimic")
    cfg.out = str(tmp_path)
    rows = Experiment(cfg).run(verbose=False)
    assert len(rows) == 6 and all(r["cost_ref"] > 0 for r in rows)
    assert rows[-1]["n_never"] <= rows[0]["n_never"] <= 16
    assert all(0 <= r["n_gap50"] <= 16 and r["budget_lighter"] == 0 for r in rows)


def test_budget_step_counts_the_jobs_it_makes_lighter(tmp_path):
    """B10: with a tight cap the projection moves jobs to lighter modes, and
    the log says how many."""
    from afl.engine import Experiment
    cfg = _cfg({"name": "random", "mode_rule": "own"}, rounds=5)
    cfg.system.cost_max = 3.0                           # Standard x 4 does not fit; Light might
    cfg.out = str(tmp_path)
    rows = Experiment(cfg).run(verbose=False)
    assert sum(r["budget_lighter"] for r in rows) > 0
    assert all(r["cost"] <= 3.0 + 1e-9 for r in rows)


@pytest.mark.parametrize("key, value, attr", [("oort_eps", 0.5, "eps"), ("oort_alpha", 4.0, "alpha")])
def test_oort_settings_are_configurable_with_unchanged_defaults(key, value, attr):
    from afl.policies.baselines import OortPolicy
    assert OortPolicy(_cfg({"name": "oort"})).eps == 0.2 and OortPolicy(_cfg({"name": "oort"})).alpha == 2.0
    assert getattr(OortPolicy(_cfg({"name": "oort", key: value})), attr) == value


def test_rl2_figures_from_fabricated_runs(tmp_path):
    """C3: the thesis figures draw from paired per-seed results."""
    import plot_rl2 as pl
    m = lambda e, w: {"eq5": e, "w10b": w, "curve": {500: (e - 0.1, 495.0), 1400: (e, 1390.0)}}  # noqa: E731
    seeds = range(300, 310)
    runs = {"femnist": {"tuned_pt": {s: m(0.80 + 0.001 * (s % 3), 0.55) for s in seeds},
                        "ctl": {s: m(0.77, 0.50) for s in seeds}, "best": {s: m(0.78, 0.53) for s in seeds}}}
    tests = [{"family": "P1", "dataset": "femnist", "vs": "ctl", "endpoint": "eq5"},
             {"family": "P2", "dataset": "femnist", "vs": "best", "endpoint": "w10b"}]
    rows = pl.comparisons(runs, "tuned_pt", tests)
    assert rows[1]["mean"] == pytest.approx(0.02) and rows[1]["ci"] == pytest.approx((0.02, 0.02))
    pl.plot_primary(rows, str(tmp_path / "p.png"), {0: True, 1: False})
    pl.plot_seeds(rows, str(tmp_path / "s.png"))
    pts = pl.cost_points(runs["femnist"]["ctl"])
    assert pts == [(495.0, pytest.approx(0.67)), (1390.0, pytest.approx(0.77))]
    pl.plot_cost({"ctl": pts}, str(tmp_path / "c.png"), "t")
    pl.plot_worlds({"femnist churn": {"eq5": 0.01, "w10b": -0.02}}, str(tmp_path / "w.png"))
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("p.png", "s.png", "c.png", "w.png"))
