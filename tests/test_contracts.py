"""Theorem 6.2 empirical test (per-interface): if Phi_{ij}(t) holds and
a_ij in A^safe_{ij}(s_t), then Phi_{ij}(t+1) holds.
"""

from __future__ import annotations

import numpy as np

from certified_marl.env.arrivals import UniformPoisson
from certified_marl.env.csgrag import CSGRAGEnv
from certified_marl.env.geometry import Rect
from certified_marl.shield.contracts import phi_ij
from certified_marl.shield.feasibility_kernel import safe_action_set


def _env(seed: int = 0) -> CSGRAGEnv:
    regions = [Rect(0.0, 0.0, 5.0, 10.0), Rect(5.0, 0.0, 10.0, 10.0)]
    kernels = [Rect(0.5, 0.5, 4.0, 9.5), Rect(6.0, 0.5, 9.5, 9.5)]
    rng = np.random.default_rng(seed)
    arr = UniformPoisson(bounds=(0.0, 0.0, 10.0, 10.0), rate=0.5, rng=rng)
    return CSGRAGEnv(
        regions=regions, kernels=kernels, U_bar=[20.0, 20.0],
        arrivals=arr, obstacles=[], dt=1.0, speed=1.0,
        delta_star_default=1.0, delta_step=0.1, rng=rng, horizon=100,
    )


def test_phi_preserved_under_safe_action():
    env = _env(seed=0)
    state = env.reset(seed=0)
    rng = np.random.default_rng(42)
    for _ in range(100):
        # Pre-step: predicate holds at t
        for ifs in state.interfaces.values():
            ok, reason = phi_ij(state, ifs)
            assert ok, f"Phi failed pre-step with reason={reason}"
        actions = {}
        for key, ifs in state.interfaces.items():
            safe = safe_action_set(state, ifs, env.delta_step, env.obstacles)
            actions[key] = int(rng.choice(safe))
        state, info = env.step(actions)
        # Theorem 6.2: predicate must hold at t+1
        for ifs in state.interfaces.values():
            ok, reason = phi_ij(state, ifs)
            assert ok, f"Phi failed post-step with reason={reason}"
        if info["done"]:
            state = env.reset()
