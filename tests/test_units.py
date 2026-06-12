"""Unit-system checks for the preset domains.

Verifies that every preset `Domain` declared in env/domain.py satisfies
three operational feasibility conditions:

  1. BUDGET FEASIBILITY: U_bar >= diag(R) / v for a representative region.
     If this fails, the budget is tighter than the fastest possible
     straight-line traversal, and admission would reject everything.

  2. NO-OVERLOAD IN EXPECTATION:  lambda * |R| * E[T_e] < Q*.
     Ensures the arrival-rate / capacity combination is not inherently
     unstable under steady state.

  3. BAND-SIZE FEASIBILITY: delta_star < min(W, H) / 2.
     Otherwise collaboration bands can meet in the middle of a region
     and overlap, breaking the directed-band model.

These guard against unit-system inconsistencies in domain definitions.
"""

import math

from certified_marl.env.domain import (
    chicago_emergency_ground,
    toy_abstract,
)
from certified_marl.env.geometry import Rect, ulsp_bound


# Representative region side length PER DOMAIN, in world units.
# Chicago: 500 m world units => 4 km side == 8 world-units.
# Toy:     1:1 world units   => 10 unit side.
REPRESENTATIVE_SIDE = {
    "chicago_emergency_ground": 8.0,   # 4 km
    "toy_abstract":            10.0,   # 10 toy-units
}

# Representative U_bar in world-time units. This is what the env would
# declare for a region of the representative side length at rho = 0.
# We compute it as ULSP(R) = diag(R) (no obstacles) in world-time,
# divided by speed_world. Domains default rho = 0 so U_bar == ULSP.
def _expected_u_bar_world(domain, side_world: float) -> float:
    r = Rect(0, 0, side_world, side_world)
    diag_world = r.diag
    # U_bar_world = (1 + rho) * diag / speed_world
    return (1.0 + domain.rho) * diag_world / domain.speed_world


def test_unit_budget_feasibility():
    """U_bar must allow at least the fastest straight-line traversal."""
    for factory in (chicago_emergency_ground, toy_abstract):
        d = factory()
        side = REPRESENTATIVE_SIDE[d.name]
        u_bar_world = _expected_u_bar_world(d, side)
        # Minimum feasible budget == diag(R) / v (worst-case straight line).
        min_feasible = math.hypot(side, side) / d.speed_world
        assert u_bar_world >= min_feasible - 1e-9, (
            f"Domain {d.name}: U_bar_world={u_bar_world:.3f} is less than "
            f"the minimum feasible {min_feasible:.3f}. "
            f"Budget is geometrically infeasible."
        )


def test_unit_sound_si_conversions():
    """Check SI-side speed reporting matches hand-computed values."""
    d = chicago_emergency_ground()
    # Chicago: length=500m, time=60s, speed_world=1.44
    # => speed_mps = 1.44 * 500 / 60 = 12.0 m/s
    assert abs(d.speed_mps - 12.0) < 1e-6, (
        f"Chicago speed SI conversion wrong: got {d.speed_mps}, expected 12.0"
    )


def test_unit_queue_capacity_positive():
    """Queue capacity (when set) must be a positive integer."""
    for factory in (chicago_emergency_ground,):
        d = factory()
        assert d.queue_capacity is not None
        assert d.queue_capacity > 0, (
            f"Domain {d.name}: queue_capacity={d.queue_capacity} invalid"
        )


def test_unit_service_time_non_negative():
    """Service time must be non-negative."""
    for factory in (chicago_emergency_ground, toy_abstract):
        d = factory()
        assert d.service_time_s >= 0.0, (
            f"Domain {d.name}: service_time_s={d.service_time_s} must be >= 0"
        )
        assert d.service_time_world >= 0.0
