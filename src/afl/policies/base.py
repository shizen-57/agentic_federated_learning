from __future__ import annotations


def best_work_mode(cfg, p_succ: dict) -> str:
    """The mode with the most expected training among the safe-enough ones.

    Expected training of mode m = P(success | m) x local_epochs[m]. Only modes
    whose success probability clears the eligibility threshold count; if none
    does, the safest mode is used. This is the "expected_work" mode rule
    (policy.mode_rule) and the reference point of ARL-v3's lost-work term.
    """
    modes = list(cfg.system.modes)
    ok = [m for m in modes if p_succ[m] >= cfg.policy.elig_threshold]
    if not ok:
        return max(modes, key=lambda m: p_succ[m])
    return max(ok, key=lambda m: p_succ[m] * cfg.system.local_epochs[m])


def mode_by_rule(cfg, p_succ: dict) -> str:
    rule = cfg.policy.mode_rule
    if rule == "expected_work":
        return best_work_mode(cfg, p_succ)
    if rule == "light":
        return "light"
    raise ValueError(f"unknown policy.mode_rule {rule}")


class Policy:
    """Interface every orchestration policy implements.

    `act` receives the observed state for one round and returns
    (selected clients, {client: mode}) -- the joint action a_t = (S_t, E_t).
    Baselines that only select clients still have to return a mode, so the
    comparison in Table 3.4 holds the configuration space fixed and varies only
    the decision rule.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def act(self, state: dict, t: int):
        raise NotImplementedError

    def observe_outcome(self, state: dict, action, reward: float, info: dict) -> None:
        """Hook for learning policies; a no-op for the fixed baselines."""

    def finish_round(self) -> dict:
        return {}
