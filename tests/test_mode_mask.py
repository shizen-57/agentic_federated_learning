"""policy.mode_mask = feasible: the RL policy may only draw modes it predicts
the client can finish, and its update stays an exact policy gradient of the
masked distribution it actually sampled from."""
from __future__ import annotations

import itertools

import pytest
import torch

from test_advanced import _tiny
from test_sixth_review import _forced, _state


def _policy(N, K, mask="feasible", coverage=True):
    from afl.policies.rl import RLPolicy
    cfg = _tiny({"arl3_full": None, "policy.credit_total": True, "policy.prior_baseline": True,
                 "fl.clean_rng": True, "policy.coverage_features": coverage,
                 "policy.entropy_coef": 0.0, "policy.rl_lr": 0.0, "policy.mode_mask": mask})
    cfg.data.num_clients, cfg.fl.clients_per_round = N, K
    return RLPolicy(cfg)


@pytest.mark.parametrize("coverage", [True, False])
def test_infeasible_modes_are_never_drawn(coverage):
    st = _state(4)
    for i in range(4):
        st["p_succ"][i] = {"light": 0.9, "standard": 0.4, "priority": 0.3}
    st["p_succ"][3] = {"light": 0.3, "standard": 0.45, "priority": 0.1}   # none passes 0.5
    pol = _policy(4, 4, coverage=coverage)
    torch.manual_seed(0)
    for _ in range(50):
        _, assign = pol.act(st, 1)
        assert all(assign[i] == "light" for i in range(3))
        assert assign[3] == "standard"                 # only the most likely mode is left
        for lp in pol._pending["logp_mode"].values():
            assert float(lp.detach()) == pytest.approx(0.0, abs=1e-6)   # a single allowed mode


def test_off_draws_from_the_unmasked_distribution():
    st = _state(3)
    for i in range(3):
        st["p_succ"][i] = {"light": 0.9, "standard": 0.1, "priority": 0.1}
    pol = _policy(3, 3, mask="none")
    torch.manual_seed(0)
    drawn = set()
    for _ in range(60):
        drawn |= set(pol.act(st, 1)[1].values())
    assert drawn == {"light", "standard", "priority"}


def test_masked_update_is_an_unbiased_gradient():
    # 3 clients, 2 picked; priority infeasible for everyone, standard for client 2.
    N, K = 3, 2
    pol = _policy(N, K)
    st = _state(N)
    for i in range(N):
        st["p_succ"][i] = {"light": 0.9, "standard": 0.7 if i < 2 else 0.4, "priority": 0.2}
    allowed = {0: (0, 1), 1: (0, 1), 2: (0,)}
    params = list(pol.net.parameters())
    table = {frozenset(s): v for s, v in zip(itertools.combinations(range(N), K), (0.01, 0.03, 0.02))}
    R = lambda a: table[frozenset(a[:K])] + 0.004 * (a[K] - a[K + 1])  # noqa: E731
    actions = [p + m for p in itertools.permutations(range(N), K)
               for m in itertools.product(*(allowed[c] for c in p))]
    exact, total = 0.0, 0.0
    for a in actions:
        _forced(pol, a)
        pol.act(st, 1)
        exact = exact + torch.exp(pol._pending["logp"]) * R(a)
        total += float(torch.exp(pol._pending["logp"].detach()))
    assert total == pytest.approx(1.0, abs=1e-5)            # the masked actions are all of them
    grads = torch.autograd.grad(exact, params, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    expected = [torch.zeros_like(p) for p in params]
    for a in actions:
        _forced(pol, a)
        pol.act(st, 1)
        prob = float(torch.exp(pol._pending["logp"].detach()))
        pol.baseline, pol.baseline_n = 0.3, 5
        pol.observe_outcome(None, None, R(a), {"client_adv": None, "set_adv": None})
        for e, p in zip(expected, params):
            e += prob * p.grad
    for e, g in zip(expected, grads):
        assert torch.allclose(e, -g, atol=1e-7)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_unknown_mask_is_rejected():
    with pytest.raises(ValueError):
        _policy(3, 2, mask="bogus")
