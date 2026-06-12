"""Handshake sign regression tests.

The trainer samples an action in the CONTROLLER-CENTERED frame and then
canonicalizes to the (min(i,j), max(i,j)) frame expected by env.step:

    canonical_action = sampled_action       if c == i
                     = -sampled_action      if c == j

A sign error here is silent: training proceeds numerically but the
policy gradient points in the wrong direction.

Invariants tested
-----------------
1. Canonicalization-identity: given a controller endpoint c, the
   physical boundary motion caused by executing the canonical action
   must equal the motion that the semantic action (in the controller's
   frame) describes.

2. Frame-independence: swapping the roles of i and j as controller (by
   changing load values so controller_of returns the other endpoint)
   must NOT change the physical outcome when the controller-frame
   action has the same semantic meaning.

3. Antisymmetry of the Eq. 3 update: under any signed action a_ij, the
   post-step deltas satisfy
       (new_delta_ij - old_delta_ij) + (new_delta_ji - old_delta_ji) = 0
   i.e., the total band width on the interface is preserved
   (conservation of eligibility region).
"""
from __future__ import annotations

import numpy as np
import pytest

from certified_marl.env.arrivals import make_arrival
from certified_marl.env.csgrag import CSGRAGEnv
from certified_marl.env.geometry import Rect
from certified_marl.shield.matching import controller_of
from certified_marl.trainers.masked_ppo import actor_state_vec


def _two_region_env(seed: int = 0) -> CSGRAGEnv:
    regions = [Rect(0.0, 0.0, 4.0, 4.0), Rect(4.0, 0.0, 8.0, 4.0)]
    kernels = [Rect(0.5, 0.5, 3.5, 3.5), Rect(4.5, 0.5, 7.5, 3.5)]
    rng = np.random.default_rng(seed)
    arrivals = make_arrival(
        dict(kind="boundary_hotspot", center=[4.0, 2.0], sigma=1.0,
             rate=0.0, service_time=0.2),   # rate=0: isolate boundary dynamics
        bounds=(0.0, 0.0, 8.0, 4.0), rng=rng,
    )
    return CSGRAGEnv(
        regions=regions, kernels=kernels, U_bar=[8.0, 8.0],
        arrivals=arrivals, obstacles=[], bounds=(0.0, 0.0, 8.0, 4.0),
        dt=1.0, speed=1.0, delta_star_default=1.0, delta_step=0.1,
        rng=rng, horizon=50,
    )


def _canonicalize(action_signed: int, c: int, i: int) -> int:
    """Replicate the trainer's canonicalization rule."""
    return action_signed if c == i else -action_signed


def test_total_band_width_conserved():
    """Eq. 3 update must preserve delta_ij + delta_ji (antisymmetric update)."""
    env = _two_region_env(seed=0)
    s = env.reset()
    ifs = s.interfaces[(0, 1)]
    # Set both deltas to interior values so the a_ij update isn't clipped.
    ifs.band_ij.delta = 0.5
    ifs.band_ji.delta = 0.5
    total_before = ifs.band_ij.delta + ifs.band_ji.delta

    s2, _ = env.step({(0, 1): +1})
    ifs2 = s2.interfaces[(0, 1)]
    total_after = ifs2.band_ij.delta + ifs2.band_ji.delta

    assert abs(total_after - total_before) < 1e-9, (
        f"band total changed: {total_before} -> {total_after}"
    )


def test_canonicalization_controller_is_i():
    """When c == i, canonical action equals controller-frame action (direct)."""
    env = _two_region_env(seed=1)
    s = env.reset()
    # Set loads so controller_of(0, 1, loads) == 0.
    s.agents[0].load_pressure_ewma = 0.1
    s.agents[1].load_pressure_ewma = 0.8
    loads = [a.load_pressure_ewma for a in s.agents]
    c = controller_of(0, 1, loads)
    assert c == 0

    for controller_action in (-1, 0, +1):
        canonical = _canonicalize(controller_action, c, i=0)
        assert canonical == controller_action, (
            f"c=i: expected canonical == controller action ({controller_action}), "
            f"got {canonical}"
        )


def test_canonicalization_controller_is_j():
    """When c == j, canonical action is negated controller-frame action."""
    env = _two_region_env(seed=2)
    s = env.reset()
    s.agents[0].load_pressure_ewma = 0.8
    s.agents[1].load_pressure_ewma = 0.1
    loads = [a.load_pressure_ewma for a in s.agents]
    c = controller_of(0, 1, loads)
    assert c == 1

    for controller_action in (-1, 0, +1):
        canonical = _canonicalize(controller_action, c, i=0)
        assert canonical == -controller_action, (
            f"c=j: expected canonical == -controller action ({-controller_action}), "
            f"got {canonical}"
        )


def test_frame_independence_of_physical_outcome():
    """Swapping controller (by swapping loads) and re-canonicalizing the
    SAME SEMANTIC action must produce the same delta changes.

    Semantic action: 'cbar expands into c' (action_signed = +1 in controller
    frame, per masked_ppo.py module docstring). Physical outcome: delta on
    cbar's side decreases by delta_step; delta on c's side increases.
    """
    env_a = _two_region_env(seed=3)
    s_a = env_a.reset()
    s_a.agents[0].load_pressure_ewma = 0.1   # c = 0
    s_a.agents[1].load_pressure_ewma = 0.8   # cbar = 1
    ifs_a = s_a.interfaces[(0, 1)]
    ifs_a.band_ij.delta = 0.5
    ifs_a.band_ji.delta = 0.5
    # Controller (=0) samples +1 ("cbar expands into c").
    canonical_a = _canonicalize(+1, c=0, i=0)   # expect +1
    s_a, _ = env_a.step({(0, 1): canonical_a})
    # Record post-step deltas.
    post_a_ij = s_a.interfaces[(0, 1)].band_ij.delta   # c-side
    post_a_ji = s_a.interfaces[(0, 1)].band_ji.delta   # cbar-side

    env_b = _two_region_env(seed=3)
    s_b = env_b.reset()
    # Swap loads so now c = 1, cbar = 0.
    s_b.agents[0].load_pressure_ewma = 0.8   # cbar = 0
    s_b.agents[1].load_pressure_ewma = 0.1   # c = 1
    ifs_b = s_b.interfaces[(0, 1)]
    ifs_b.band_ij.delta = 0.5
    ifs_b.band_ji.delta = 0.5
    # Controller (=1) samples +1 ("cbar expands into c").
    canonical_b = _canonicalize(+1, c=1, i=0)   # expect -1
    s_b, _ = env_b.step({(0, 1): canonical_b})
    post_b_ij = s_b.interfaces[(0, 1)].band_ij.delta   # cbar-side in env_b
    post_b_ji = s_b.interfaces[(0, 1)].band_ji.delta   # c-side in env_b

    c_delta_a = post_a_ij
    c_delta_b = post_b_ji
    cbar_delta_a = post_a_ji
    cbar_delta_b = post_b_ij

    assert abs(c_delta_a - c_delta_b) < 1e-9, (
        f"c-side delta mismatch across frames: {c_delta_a} vs {c_delta_b}"
    )
    assert abs(cbar_delta_a - cbar_delta_b) < 1e-9, (
        f"cbar-side delta mismatch across frames: {cbar_delta_a} vs {cbar_delta_b}"
    )


def test_actor_state_is_controller_centered():
    """actor_state_vec's first coordinate must be the CONTROLLER's load,
    not a fixed ordering by index. This is what anchors the learned
    policy to the 'lighter endpoint has slack' canonical orientation."""
    env = _two_region_env(seed=4)
    s = env.reset()
    s.agents[0].load_pressure_ewma = 0.1
    s.agents[1].load_pressure_ewma = 0.8
    # q_i is a property = len(queue). Push placeholder events to set it.
    s.agents[0].queue = [None] * 2
    s.agents[1].queue = [None] * 7
    vec = actor_state_vec(s, 0, 1)
    # First component is controller's load. Controller = 0, load = 0.1.
    assert abs(vec[0] - 0.1) < 1e-6, (
        f"actor_state first coord should be controller load 0.1, got {vec[0]}"
    )
    assert abs(vec[1] - 2.0) < 1e-6, (
        f"actor_state second coord should be controller q_i = 2, got {vec[1]}"
    )
    # Swap loads -> controller becomes 1. First coord should now be 0.1
    # (controller's load) but indexed FROM agent 1, whose q_i is 7.
    s.agents[0].load_pressure_ewma = 0.8
    s.agents[1].load_pressure_ewma = 0.1
    vec2 = actor_state_vec(s, 0, 1)
    assert abs(vec2[0] - 0.1) < 1e-6
    assert abs(vec2[1] - 7.0) < 1e-6, (
        f"after controller flip, q_i should be agent-1's (7), got {vec2[1]}"
    )
