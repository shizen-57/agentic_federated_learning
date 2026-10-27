"""RL2 M2: the SAFE-PPO learner -- GAE, exact log-probabilities of the sampled
distribution, an unbiased policy gradient, the critic, the entropy schedule."""
from __future__ import annotations

import itertools

import pytest
import torch

from test_advanced import _tiny
from test_sixth_review import _state


def _ppo(N, K, **over):
    from afl.policies.safe_rl import SafePPOPolicy
    o = {"policy.name": "safe_ppo", "fl.clean_rng": True, "policy.coverage_features": True,
         "policy.ppo_ent_start": 0.0, "policy.ppo_ent_end": 0.0, "policy.ppo_norm_adv": False}
    o.update(over)
    cfg = _tiny({"arl3_full": None, **o})
    cfg.data.num_clients, cfg.fl.clients_per_round = N, K
    return SafePPOPolicy(cfg)


def test_gae_matches_a_hand_calculation():
    from afl.policies.safe_rl import gae
    r, v = [1.0, 0.0, 2.0], [0.5, 0.2, 1.0]
    g, lam, last = 0.9, 0.8, 0.3
    d2 = 2.0 + g * last - 1.0
    d1 = 0.0 + g * 1.0 - 0.2
    d0 = 1.0 + g * 0.2 - 0.5
    a2, a1 = d2, d1 + g * lam * d2
    a0 = d0 + g * lam * a1
    adv, ret = gae(r, v, last, g, lam)
    assert adv == pytest.approx([a0, a1, a2])
    assert ret == pytest.approx([a0 + 0.5, a1 + 0.2, a2 + 1.0])


NETS = [("mlp", False), ("mlp", True), ("attention", False), ("attention", True)]


@pytest.mark.parametrize("mask", ["none", "feasible"])
@pytest.mark.parametrize("enc,mem", NETS)
def test_stored_actions_are_re_evaluated_exactly(mask, enc, mem):
    pol = _ppo(5, 3, **{"policy.mode_mask": mask, "policy.encoder": enc, "policy.memory": mem})
    st = _state(5)
    st["p_succ"][4] = {"light": 0.9, "standard": 0.3, "priority": 0.2}
    torch.manual_seed(0)
    for t in range(1, 6):
        sel, assign = pol.act(st, t)
        last = pol._last
        again = pol._pass(last["inp"], {"picks": last["picks"], "modes": last["modes"]})
        assert float(again["logp"].detach()) == pytest.approx(last["logp"], abs=1e-6)
        assert float(again["value"].detach()) == pytest.approx(last["value"], abs=1e-6)
        if mask == "feasible" and 4 in assign:
            assert assign[4] == "light"
        pol.observe_outcome(st, (sel, assign), 0.1, {"success": {i: 1 for i in sel[:1]}})
        pol.buf = []


@pytest.mark.parametrize("enc,mem", NETS)
def test_policy_gradient_is_unbiased(enc, mem):
    # One-round batch, gamma irrelevant: A(a) = R(a) - V(s). Summing the clipped
    # surrogate's gradient (ratio = 1) over every action, weighted by its
    # probability, gives exactly -grad E[R].
    N, K = 3, 2
    pol = _ppo(N, K, **{"policy.encoder": enc, "policy.memory": mem})
    st = _state(N)
    inp = pol._inputs(st, 1)
    params = [p for n, p in pol.net.named_parameters() if not n.startswith("critic")]
    table = {frozenset(s): v for s, v in zip(itertools.combinations(range(N), K), (0.01, 0.03, 0.02))}
    R = lambda a: table[frozenset(a[0])] + 0.004 * (a[1][0] - a[1][1])  # noqa: E731
    acts = [(list(p), list(m)) for p in itertools.permutations(range(N), K)
            for m in itertools.product(range(3), repeat=K)]
    v = float(pol._pass(inp, {"picks": acts[0][0], "modes": acts[0][1]})["value"].detach())
    exact = sum(torch.exp(pol._pass(inp, {"picks": p, "modes": m})["logp"]) * R((p, m)) for p, m in acts)
    grads = [torch.zeros_like(p) if g is None else g
             for p, g in zip(params, torch.autograd.grad(exact, params, allow_unused=True))]
    expected = [torch.zeros_like(p) for p in params]
    for p_, m in acts:
        out = pol._pass(inp, {"picks": p_, "modes": m})
        prob = float(torch.exp(out["logp"]).detach())
        old = out["logp"].detach()
        ratio = torch.exp(out["logp"] - old)
        loss = -(ratio * (R((p_, m)) - v))
        g = torch.autograd.grad(loss, params, allow_unused=True)
        for e, gi in zip(expected, g):
            if gi is not None:
                e += prob * gi
    for e, g in zip(expected, grads):
        assert torch.allclose(e, -g, atol=1e-6)


def test_critic_learns_a_constant_return():
    pol = _ppo(4, 2, **{"policy.ppo_batch": 4, "policy.ppo_epochs": 8, "policy.ppo_lr": 0.01,
                        "policy.ppo_gamma": 0.0})
    st = _state(4)
    torch.manual_seed(0)
    for t in range(1, 60):
        pol.act(st, t)
        pol.observe_outcome(st, None, 1.0, {})
    assert pol._value_only(pol._inputs(st, 60)) == pytest.approx(1.0, abs=0.15)
    assert pol.n_updates >= 10


def test_entropy_bonus_decays():
    pol = _ppo(4, 2, **{"policy.ppo_ent_start": 0.01, "policy.ppo_ent_end": 0.001})
    first = pol._ent_coef()
    pol.n_updates = 10 ** 6
    assert first == pytest.approx(0.01) and pol._ent_coef() == pytest.approx(0.001)


def test_attention_is_permutation_equivariant():
    N = 6
    pol = _ppo(N, 3, **{"policy.encoder": "attention", "policy.memory": True})
    st = _state(N)
    inp = pol._inputs(st, 1)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    pinp = dict(inp)
    pinp["x"], pinp["allowed"], pinp["pm"], pinp["mem"] = (inp["x"][perm], inp["allowed"][perm],
                                                           inp["pm"][perm], inp["mem"][perm])
    pinp["cov"] = (inp["cov"][0][perm], inp["cov"][1])
    with torch.no_grad():
        a, b = pol.net.start(inp), pol.net.start(pinp)
        la = pol.net.select_logits(inp, [], a)
        lb = pol.net.select_logits(pinp, [], b)
        assert torch.allclose(la[perm], lb, atol=1e-5)
        assert torch.allclose(pol.net.value(inp, a), pol.net.value(pinp, b), atol=1e-5)
        inv = torch.argsort(perm)
        ma = pol.net.mode_logits(inp, [0, 2], a)
        mb = pol.net.mode_logits(pinp, [int(inv[0]), int(inv[2])], b)
        assert torch.allclose(ma[perm], mb, atol=1e-5)


def test_memory_records_observed_outcomes():
    pol = _ppo(4, 2, **{"policy.memory": True, "policy.mem_ema": 0.5})
    st = _state(4)
    sel, assign = pol.act(st, 1)
    i = sel[0]
    k = pol.modes.index(assign[i])
    pol.observe_outcome(st, (sel, assign), 0.0, {"success": {i: 1}})
    assert pol.m_ema[i, k] == pytest.approx(0.75) and pol.m_last[i, -1] == 1.0
    assert pol.m_since[i] == 0 and pol.m_count[i] == 1
    assert pol.m_win[-1, 0] == pytest.approx(1 / len(sel))
    with pytest.raises(ValueError):
        _ppo(4, 2, **{"policy.encoder": "lstm"})
