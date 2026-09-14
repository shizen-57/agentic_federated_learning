from __future__ import annotations


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
