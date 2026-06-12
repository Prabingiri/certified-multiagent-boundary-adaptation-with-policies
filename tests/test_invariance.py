"""Theorem 6.2 empirical test: under shielded execution, the certified
execution state (Definition 6.1) is preserved for all t.
"""

from __future__ import annotations

import numpy as np

from certified_marl.env.arrivals import BoundaryHotspot
from certified_marl.env.csgrag import CSGRAGEnv
from certified_marl.env.geometry import Rect
from certified_marl.metrics.safety import safety_from_state
from certified_marl.shield.feasibility_kernel import safe_action_set


def _4region_line(seed: int = 0) -> CSGRAGEnv:
    regions = [
        Rect(0.0, 0.0, 10.0, 10.0),
        Rect(10.0, 0.0, 20.0, 10.0),
        Rect(20.0, 0.0, 30.0, 10.0),
        Rect(30.0, 0.0, 40.0, 10.0),
    ]
    kernels = [
        Rect(0.5, 0.5, 9.0, 9.5),
        Rect(10.5, 0.5, 19.0, 9.5),
        Rect(20.5, 0.5, 29.0, 9.5),
        Rect(30.5, 0.5, 39.0, 9.5),
    ]
    rng = np.random.default_rng(seed)
    # Boundary hotspot at interface Gamma_23 (x=20).
    arr = BoundaryHotspot(center=(20.0, 5.0), sigma=1.5, rate=1.5,
                          bounds=(0.0, 0.0, 40.0, 10.0), rng=rng)
    return CSGRAGEnv(
        regions=regions, kernels=kernels, U_bar=[30.0, 30.0, 30.0, 30.0],
        arrivals=arr, obstacles=[], dt=1.0, speed=1.0,
        delta_star_default=1.5, delta_step=0.15, rng=rng, horizon=300,
    )


def test_four_region_line_zero_violations_under_shield():
    env = _4region_line(seed=0)
    state = env.reset(seed=0)
    rng = np.random.default_rng(123)
    for _ in range(300):
        actions = {}
        for key, ifs in state.interfaces.items():
            safe = safe_action_set(state, ifs, env.delta_step, env.obstacles)
            actions[key] = int(rng.choice(safe))
        state, info = env.step(actions)
        if info["done"]:
            break
    metrics = safety_from_state(state)
    # Theorem 6.2: ALL violation types must be zero.
    assert metrics.geom_violations == 0, f"geom violations: {metrics.geom_violations}"
    assert metrics.ker_violations == 0, f"ker violations: {metrics.ker_violations}"
    assert metrics.cert_violations == 0, f"cert violations: {metrics.cert_violations}"
    # srv violations would mean CS-LSTF admitted an event it could not serve
    # within budget - impossible under the Eq. 9 admission test. Assert zero.
    assert metrics.srv_violations == 0, f"srv violations: {metrics.srv_violations}"
