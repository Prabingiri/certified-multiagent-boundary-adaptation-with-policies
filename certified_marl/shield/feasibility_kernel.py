r"""Executable feasibility mask for boundary actions."""

from __future__ import annotations

import numpy as np

from certified_marl.env.csgrag import (
    CSGRAGState,
    InterfaceState,
    build_agent_eligibility_region,
)
from certified_marl.env.geometry import (
    EligibilityRegion,
    GEOM_EPS,
    Obstacle,
    Rect,
    ulsp_bound_region,
)


# ---------------------------------------------------------------------------
# Simulation of a post-update slack pair (without mutating the env state)
# ---------------------------------------------------------------------------


def _simulate_post_update(
    interface: InterfaceState,
    a_ij: int,
    delta_step: float,
) -> tuple[float, float]:
    """Return post-update directed slacks without mutating state."""
    new_delta_ij = float(np.clip(
        interface.band_ij.delta - a_ij * delta_step,
        0.0, interface.delta_star,
    ))
    new_delta_ji = float(np.clip(
        interface.band_ji.delta - (-a_ij) * delta_step,
        0.0, interface.delta_star,
    ))
    return new_delta_ij, new_delta_ji


def _rects_overlap(a: Rect, b: Rect) -> bool:
    """Positive-area intersection between two rectangles."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    return ix1 > ix0 + GEOM_EPS and iy1 > iy0 + GEOM_EPS


def _intersects(o: Obstacle, r: Rect) -> bool:
    """Obstacle/region positive-area intersection."""
    ix0 = max(o.rect.x0, r.x0)
    iy0 = max(o.rect.y0, r.y0)
    ix1 = min(o.rect.x1, r.x1)
    iy1 = min(o.rect.y1, r.y1)
    return ix1 > ix0 and iy1 > iy0


def _intersects_region(o: Obstacle, region: Rect | EligibilityRegion) -> bool:
    if isinstance(region, Rect):
        return _intersects(o, region)
    return any(_intersects(o, r) for r in region.member_rects)


def _queue_membership_ok(
    agent,
    region: Rect | EligibilityRegion,
 ) -> bool:
    """Every queued event must remain inside the proposed service region."""
    for e in agent.queue:
        if not region.contains((e.x, e.y)):
            return False
    return True


def _queue_envelope_bound(
    agent,
    region: Rect | EligibilityRegion,
    *,
    speed: float,
) -> float:
    """Conservative queue envelope for already-committed events."""
    diameter = region.diag / max(speed, 1e-9)
    return sum(diameter + e.service_time for e in agent.queue)


# ---------------------------------------------------------------------------
# Per-edge feasibility test and executable action set
# ---------------------------------------------------------------------------


def check_edge_action_safe(
    s: CSGRAGState,
    interface: InterfaceState,
    a_ij: int,
    delta_step: float,
    obstacles: list[Obstacle],
    tol: float = GEOM_EPS,
    speed: float = 1.0,
    dynamic_service_regions: bool | None = None,
) -> tuple[bool, str]:
    """Return whether a candidate action passes the local feasibility kernel."""
    if dynamic_service_regions is None:
        dynamic_service_regions = bool(getattr(s, "dynamic_service_regions", False))

    if not isinstance(a_ij, (int, np.integer)) or bool(isinstance(a_ij, bool)):
        return False, "action"
    if a_ij not in (-1, 0, 1):
        return False, "action"

    # Clause (1): slack admissibility.
    new_dij, new_dji = _simulate_post_update(interface, a_ij, delta_step)
    if not (0.0 - tol <= new_dij <= interface.delta_star + tol):
        return False, "slack"
    if not (0.0 - tol <= new_dji <= interface.delta_star + tol):
        return False, "slack"

    # Clause (2): inverse-action consistency is enforced by simulating
    # the opposite directed slack with -a_ij.
    delta_overrides = {interface.key: (new_dij, new_dji)}

    a_i = s.agents[interface.i]
    a_j = s.agents[interface.j]
    region_i = build_agent_eligibility_region(
        s,
        interface.i,
        dynamic_service_regions=dynamic_service_regions,
        delta_overrides=delta_overrides,
    )
    region_j = build_agent_eligibility_region(
        s,
        interface.j,
        dynamic_service_regions=dynamic_service_regions,
        delta_overrides=delta_overrides,
    )
    # Clause (3): each endpoint kernel remains inside its service region.
    if not region_i.contains_region(a_i.kernel):
        return False, "ker"
    if not region_j.contains_region(a_j.kernel):
        return False, "ker"

    U_i_new = ulsp_bound_region(
        region_i,
        [o for o in obstacles if _intersects_region(o, region_i)],
    ) / max(speed, 1e-9)
    U_j_new = ulsp_bound_region(
        region_j,
        [o for o in obstacles if _intersects_region(o, region_j)],
    ) / max(speed, 1e-9)
    # Clause (5): response bounds remain within the certified ceilings.
    if U_i_new > a_i.U_bar + tol:
        return False, "cert_budget"
    if U_j_new > a_j.U_bar + tol:
        return False, "cert_budget"

    if dynamic_service_regions:
        current_region_i = build_agent_eligibility_region(
            s,
            interface.i,
            dynamic_service_regions=dynamic_service_regions,
        )
        current_region_j = build_agent_eligibility_region(
            s,
            interface.j,
            dynamic_service_regions=dynamic_service_regions,
        )
        # Clause (6): already committed events remain eligible.
        if not _queue_membership_ok(a_i, region_i):
            return False, "cert_queue"
        if not _queue_membership_ok(a_j, region_j):
            return False, "cert_queue"
        # Clause (7): committed workload remains covered by the new bound.
        current_W_i = _queue_envelope_bound(a_i, current_region_i, speed=speed)
        current_W_j = _queue_envelope_bound(a_j, current_region_j, speed=speed)
        post_W_i = _queue_envelope_bound(a_i, region_i, speed=speed)
        post_W_j = _queue_envelope_bound(a_j, region_j, speed=speed)
        if current_W_i <= a_i.U_current + tol and post_W_i > U_i_new + tol:
            return False, "cert_workload"
        if current_W_j <= a_j.U_current + tol and post_W_j > U_j_new + tol:
            return False, "cert_workload"

    return True, "ok"


def safe_action_set(
    s: CSGRAGState,
    interface: InterfaceState,
    delta_step: float,
    obstacles: list[Obstacle],
    speed: float = 1.0,
    dynamic_service_regions: bool | None = None,
) -> list[int]:
    """Return the feasible subset of {-1, 0, +1}."""
    out: list[int] = []
    for a in (-1, 0, +1):
        ok, _ = check_edge_action_safe(
            s,
            interface,
            a,
            delta_step,
            obstacles,
            speed=speed,
            dynamic_service_regions=dynamic_service_regions,
        )
        if ok:
            out.append(a)
    assert 0 in out, "Hold action must remain feasible."
    return out
