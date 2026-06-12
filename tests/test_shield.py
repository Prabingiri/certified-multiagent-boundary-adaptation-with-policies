"""Non-emptiness of A^safe_{ij}(s_t) (the paper, Section 3.4): the safe set
is never empty and always contains 0 (hold). An empty safe set would make
the masked softmax undefined.
"""

from __future__ import annotations

import numpy as np

from certified_marl.env.arrivals import UniformPoisson
from certified_marl.env.csgrag import CSGRAGEnv
from certified_marl.env.geometry import Obstacle, Rect
from certified_marl.shield.feasibility_kernel import safe_action_set


def _make_env(seed: int = 0) -> CSGRAGEnv:
    """2-region horizontal split on a 10x10 workspace."""
    regions = [
        Rect(0.0, 0.0, 5.0, 10.0),
        Rect(5.0, 0.0, 10.0, 10.0),
    ]
    kernels = [
        Rect(0.5, 0.5, 4.0, 9.5),
        Rect(6.0, 0.5, 9.5, 9.5),
    ]
    U_bar = [20.0, 20.0]
    rng = np.random.default_rng(seed)
    arrivals = UniformPoisson(bounds=(0.0, 0.0, 10.0, 10.0), rate=0.5, rng=rng)
    return CSGRAGEnv(
        regions=regions, kernels=kernels, U_bar=U_bar,
        arrivals=arrivals, obstacles=[], dt=1.0, speed=1.0,
        delta_star_default=1.0, delta_step=0.1, rng=rng, horizon=50,
    )


def test_safe_set_is_never_empty():
    env = _make_env(seed=0)
    state = env.reset(seed=0)
    assert len(state.interfaces) == 1
    ifs = next(iter(state.interfaces.values()))
    safe = safe_action_set(state, ifs, env.delta_step, env.obstacles)
    assert len(safe) >= 1
    assert 0 in safe  # hold is always feasible (the paper, Section 3.4)


def test_safe_set_hold_under_stress():
    """Randomly perturb delta values toward caps and verify hold is still safe."""
    env = _make_env(seed=1)
    state = env.reset(seed=1)
    ifs = next(iter(state.interfaces.values()))
    # Push both directed slacks to the cap
    ifs.band_ij.delta = ifs.delta_star
    ifs.band_ji.delta = ifs.delta_star
    safe = safe_action_set(state, ifs, env.delta_step, env.obstacles)
    assert 0 in safe
    # At the cap, expand direction (whichever decreases one of the deltas further
    # toward 0 via Eq. 3) can be feasible; the other is clipped but still falls
    # in [0, delta_star] because of the clip - so both may be feasible. We only
    # assert HOLD is feasible, which is the theorem.


def test_safe_set_runtime_over_rollout():
    """Over 200 random epochs, safe set is non-empty at every active interface."""
    env = _make_env(seed=7)
    state = env.reset(seed=7)
    rng = np.random.default_rng(11)
    for _ in range(200):
        # Sample random safe actions per interface.
        actions = {}
        for key, ifs in state.interfaces.items():
            safe = safe_action_set(state, ifs, env.delta_step, env.obstacles)
            assert 0 in safe
            assert len(safe) >= 1
            actions[key] = int(rng.choice(safe))
        state, info = env.step(actions)
        if info["done"]:
            state = env.reset()



