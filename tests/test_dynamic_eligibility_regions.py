from __future__ import annotations

import numpy as np
from unittest.mock import patch

from certified_marl.env.arrivals import Event, UniformPoisson
from certified_marl.env.csgrag import CSGRAGEnv, build_agent_eligibility_region
from certified_marl.env.geometry import EligibilityRegion, Rect
from certified_marl.shield.feasibility_kernel import check_edge_action_safe


def _two_region_env(*, dynamic_service_regions: bool) -> CSGRAGEnv:
    regions = [
        Rect(0.0, 0.0, 10.0, 10.0),
        Rect(10.0, 0.0, 20.0, 10.0),
    ]
    kernels = [
        Rect(1.0, 1.0, 9.0, 9.0),
        Rect(11.0, 1.0, 19.0, 9.0),
    ]
    rng = np.random.default_rng(0)
    arrivals = UniformPoisson(bounds=(0.0, 0.0, 20.0, 10.0), rate=0.0, rng=rng)
    return CSGRAGEnv(
        regions=regions,
        kernels=kernels,
        U_bar=[30.0, 30.0],
        arrivals=arrivals,
        obstacles=[],
        dt=1.0,
        speed=1.0,
        delta_star_default=1.0,
        delta_step=1.0,
        rng=rng,
        horizon=10,
        initial_delta_fraction=1.0,
        dynamic_service_regions=dynamic_service_regions,
    )


def test_dynamic_service_region_changes_certificate_surface():
    env = _two_region_env(dynamic_service_regions=True)
    state = env.reset(seed=0)
    region0 = build_agent_eligibility_region(state, 0, dynamic_service_regions=True)
    assert isinstance(region0, EligibilityRegion)
    assert state.agents[0].U_current > state.agents[0].region.diag
    assert region0.contains((10.5, 5.0))
    assert not state.agents[0].region.contains((10.5, 5.0))


def test_dynamic_service_region_blocks_stranding_borrowed_event():
    env = _two_region_env(dynamic_service_regions=True)
    state = env.reset(seed=0)
    ifs = next(iter(state.interfaces.values()))
    borrowed_event = Event(x=10.8, y=5.0, occurrence_time=0.0, service_time=0.0)
    state.agents[0].queue.append(borrowed_event)
    # a_01 = -1 shrinks delta_10 from 1.0 to 0.0, which strands the queued
    # borrowed-strip event outside E_0(t+1).
    ok_hold, reason_hold = check_edge_action_safe(
        state, ifs, 0, env.delta_step, env.obstacles, speed=env.speed
    )
    ok_contract, reason_contract = check_edge_action_safe(
        state, ifs, -1, env.delta_step, env.obstacles, speed=env.speed
    )
    assert ok_hold, reason_hold
    assert not ok_contract
    assert reason_contract == "cert_queue"


def test_legacy_mode_keeps_old_owner_rectangle_semantics():
    env = _two_region_env(dynamic_service_regions=False)
    state = env.reset(seed=0)
    ifs = next(iter(state.interfaces.values()))
    borrowed_event = Event(x=10.8, y=5.0, occurrence_time=0.0, service_time=0.0)
    state.agents[0].queue.append(borrowed_event)
    ok_contract, reason_contract = check_edge_action_safe(
        state, ifs, -1, env.delta_step, env.obstacles, speed=env.speed
    )
    assert ok_contract, reason_contract


def test_apply_time_cert_reject_counted():
    env = _two_region_env(dynamic_service_regions=True)
    state = env.reset(seed=0)
    ifs = next(iter(state.interfaces.values()))

    with patch(
        "certified_marl.shield.feasibility_kernel.check_edge_action_safe",
        return_value=(False, "cert_queue"),
    ):
        state, _ = env.step({ifs.key: -1})

    # A cert-clause rejection is counted in the aggregate `cert` counter.
    assert state.violations["cert"] == 1
