"""Endpoint-disjoint active-interface matching (the paper, Section 6, used in the Theorem 6.2 proof).

Property: let E^act(t) be the active-interface set selected at t by greedy
weighted matching. Then no two interfaces in E^act(t) share an endpoint.

This test runs `active_interfaces(loads, neighbors)` over a wide range
of synthetic load configurations -- including the degenerate cases
(all-tie, star-topology, bipartite) -- and asserts the returned set is a
matching. The check covers tie-breaking and all-zero weights, where the
greedy construction must still return a valid matching.
"""
from __future__ import annotations

import itertools
import random

import pytest

from certified_marl.shield.matching import (
    active_interfaces,
    greedy_weighted_matching,
    propose_dominant,
)


def _assert_matching(edges: list[tuple[int, int]]) -> None:
    """Assert the edge list is a valid matching (no shared endpoints)."""
    used: set[int] = set()
    for (i, j) in edges:
        assert i not in used, f"endpoint {i} appears twice in {edges}"
        assert j not in used, f"endpoint {j} appears twice in {edges}"
        used.add(i)
        used.add(j)


def test_active_interfaces_line_topology():
    """4-region line: each region's neighbors are (i-1, i+1)."""
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    for _ in range(50):
        loads = [random.random() for _ in range(4)]
        e = active_interfaces(loads, neighbors)
        _assert_matching(e)


def test_active_interfaces_star_topology():
    """Star: one central node connected to all others. At most 1 edge
    can be active (central is the choke point)."""
    neighbors = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}
    for _ in range(50):
        loads = [random.random() for _ in range(5)]
        e = active_interfaces(loads, neighbors)
        _assert_matching(e)
        # Star must produce at most 1 active edge.
        assert len(e) <= 1


def test_active_interfaces_grid_topology():
    """3x3 grid: 9 nodes, each with 2-4 neighbors."""
    neighbors: dict[int, list[int]] = {i: [] for i in range(9)}
    for r in range(3):
        for c in range(3):
            i = r * 3 + c
            if c + 1 < 3:
                j = r * 3 + (c + 1)
                neighbors[i].append(j); neighbors[j].append(i)
            if r + 1 < 3:
                j = (r + 1) * 3 + c
                neighbors[i].append(j); neighbors[j].append(i)
    random.seed(0)
    for _ in range(100):
        loads = [random.random() for _ in range(9)]
        e = active_interfaces(loads, neighbors)
        _assert_matching(e)


def test_active_interfaces_all_equal_loads():
    """Adversarial: all loads equal -> weights all zero. Greedy must
    still produce a valid matching (possibly empty)."""
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    loads = [0.5, 0.5, 0.5, 0.5]
    e = active_interfaces(loads, neighbors)
    _assert_matching(e)


def test_greedy_matching_pathological_ties():
    """Edge set with all-tied weights. Deterministic tie-break on tuple
    order must still produce a matching."""
    edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    weights = {e: 1.0 for e in edges}
    m = greedy_weighted_matching(edges, weights)
    _assert_matching(m)


def test_endpoint_disjointness_under_env_rollout():
    """End-to-end: run a real env for 200 steps under hold, assert that
    on every step the active_interfaces set is a matching."""
    import numpy as np
    from certified_marl.env.arrivals import make_arrival
    from certified_marl.env.csgrag import CSGRAGEnv
    from certified_marl.env.geometry import Rect

    rng = np.random.default_rng(0)
    env = CSGRAGEnv(
        regions=[Rect(0, 0, 4, 4), Rect(4, 0, 8, 4),
                 Rect(8, 0, 12, 4), Rect(12, 0, 16, 4)],
        kernels=[Rect(0.5, 0.5, 3.5, 3.5), Rect(4.5, 0.5, 7.5, 3.5),
                 Rect(8.5, 0.5, 11.5, 3.5), Rect(12.5, 0.5, 15.5, 3.5)],
        U_bar=[8.0, 8.0, 8.0, 8.0],
        arrivals=make_arrival(
            dict(kind="boundary_hotspot", center=[8.0, 2.0], sigma=1.0,
                 rate=3.0, service_time=0.2),
            bounds=(0, 0, 16, 4), rng=rng,
        ),
        obstacles=[], bounds=(0, 0, 16, 4),
        dt=1.0, speed=1.0, delta_star_default=1.0, delta_step=0.1,
        rng=rng, horizon=200,
    )
    s = env.reset()
    for _ in range(200):
        nbrs = {i: [] for i in range(env.n)}
        for (i, j) in s.interfaces.keys():
            nbrs[i].append(j); nbrs[j].append(i)
        loads = [a.load_pressure_ewma for a in s.agents]
        e = active_interfaces(loads, nbrs)
        _assert_matching(e)
        s, info = env.step({})
        if info["done"]:
            break
