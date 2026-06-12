"""Seed-determinism regression tests.

Given a seed, a full rollout must produce bit-identical state
trajectories. Python dict insertion order (guaranteed since 3.7) and
NumPy default_rng both preserve ordering, but set iteration, implicit
hashing, and parallel reductions can introduce nondeterminism. These
tests codify the expected invariant.

Invariants tested
-----------------
1. Two fresh env constructions with the same seed produce identical
   arrival event streams.

2. Two fresh env constructions with the same seed, stepped through the
   same action sequence, produce identical state trajectories (deltas,
   agent loads, violation counts).

3. Different seeds produce different trajectories, confirming the seed
   actually drives the stochasticity.
"""
from __future__ import annotations

import numpy as np
import pytest

from certified_marl.env.arrivals import make_arrival
from certified_marl.env.csgrag import CSGRAGEnv
from certified_marl.env.geometry import Rect


def _fresh_env(seed: int) -> CSGRAGEnv:
    regions = [Rect(0.0, 0.0, 4.0, 4.0), Rect(4.0, 0.0, 8.0, 4.0)]
    kernels = [Rect(0.5, 0.5, 3.5, 3.5), Rect(4.5, 0.5, 7.5, 3.5)]
    rng = np.random.default_rng(seed)
    arrivals = make_arrival(
        dict(kind="boundary_hotspot", center=[4.0, 2.0], sigma=1.0,
             rate=3.0, service_time=0.2),
        bounds=(0.0, 0.0, 8.0, 4.0), rng=rng,
    )
    return CSGRAGEnv(
        regions=regions, kernels=kernels, U_bar=[8.0, 8.0],
        arrivals=arrivals, obstacles=[], bounds=(0.0, 0.0, 8.0, 4.0),
        dt=1.0, speed=1.0, delta_star_default=1.0, delta_step=0.1,
        rng=rng, horizon=100,
    )


def _run_holdpolicy(env: CSGRAGEnv, n_steps: int = 100) -> dict:
    """Run a pure-hold rollout and collect state signatures."""
    s = env.reset()
    trace = []
    for _ in range(n_steps):
        s, info = env.step({})   # hold = no boundary actions
        trace.append(dict(
            t=s.t,
            delta_ij=s.interfaces[(0, 1)].band_ij.delta,
            delta_ji=s.interfaces[(0, 1)].band_ji.delta,
            load_0=s.agents[0].load_pressure_ewma,
            load_1=s.agents[1].load_pressure_ewma,
            q0=len(s.agents[0].queue),
            q1=len(s.agents[1].queue),
            viol=sum(s.violations.values()),
            admitted=info["admitted"],
            rejected=info["rejected"],
            completed=info["completed"],
        ))
        if info["done"]:
            break
    return dict(trace=trace, final_viol=sum(s.violations.values()))


def test_same_seed_same_trajectory():
    """Two independent constructions with seed=7 produce identical traces."""
    env_a = _fresh_env(seed=7)
    env_b = _fresh_env(seed=7)
    trace_a = _run_holdpolicy(env_a)
    trace_b = _run_holdpolicy(env_b)
    assert trace_a["trace"] == trace_b["trace"], (
        "same-seed trajectory drift detected -- nondeterminism in env"
    )
    assert trace_a["final_viol"] == trace_b["final_viol"]


def test_different_seeds_different_trajectory():
    """Seed 0 and seed 1 must produce different rollouts."""
    t0 = _run_holdpolicy(_fresh_env(seed=0))
    t1 = _run_holdpolicy(_fresh_env(seed=1))
    assert t0["trace"] != t1["trace"], (
        "different seeds produced identical rollouts -- seed has no effect"
    )


def test_arrivals_are_seed_deterministic():
    """Arrival event streams are deterministic given seed."""
    env_a = _fresh_env(seed=42)
    env_b = _fresh_env(seed=42)
    s_a = env_a.reset()
    s_b = env_b.reset()
    # 10 time steps, collect new_events via step()
    adms_a, adms_b = [], []
    for _ in range(10):
        _, info_a = env_a.step({})
        _, info_b = env_b.step({})
        adms_a.append(info_a["admitted"])
        adms_b.append(info_b["admitted"])
    assert adms_a == adms_b, (
        f"arrival streams differ under same seed: {adms_a} vs {adms_b}"
    )
