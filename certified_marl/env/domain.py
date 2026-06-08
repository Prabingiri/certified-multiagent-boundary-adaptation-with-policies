r"""Operational domain specification.

Decouples the dimensionless "world units" the env/shield/trainer run in from
physical meaning. A `Domain` object loaded at config time pins units via two
primitives -- length_unit_m (meters per world-length) and time_unit_s (seconds
per world-time) -- from which every physical quantity is derived. The code
stays dimensionless; configs pin units through `Domain`.

A Chicago ground-response preset anchors speed/rate/budget to published
operational data; a toy preset gives a 1:1 unit mapping for the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Domain:
    r"""Operational domain specifying the unit system and reference scales.

    Attributes
    ----------
    name : str
        Human-readable domain identifier (used in logs, YAML configs).
    length_unit_m : float
        Conversion factor: 1 world-length == length_unit_m meters.
    time_unit_s : float
        Conversion factor: 1 world-time == time_unit_s seconds.
    speed_world : float
        Agent traversal speed in world-units per world-time.
    rho : float
        Certificate slack ratio: U_bar = (1 + rho) * ULSP(R). Default 0.0
        corresponds to the "geometric certificate" interpretation -- the
        budget IS the partition's worst-case bound, no external slack.
    arrival_rate_per_km2_per_hr : float
        Operational aggregate rate, in SI units (for reporting and
        literature comparison). Not used internally; env uses world-units.
    service_time_s : float
        Mean on-scene / on-site service duration in seconds. Typically set
        to a small value (1-10 s) for response-time-focused studies;
        non-zero so the code path is exercised. Default 1.0 s.
    queue_capacity : int
        Per-agent admitted-queue capacity Q*_i. Used by the capacity-aware
        admission predicate (Eq. 11); None means unbounded.

    Invariants
    ----------
    length_unit_m > 0, time_unit_s > 0, speed_world > 0, rho >= 0.
    speed_mps = speed_world * length_unit_m / time_unit_s should match
    the literature-reported value for the operational setting.
    """

    name: str
    length_unit_m: float
    time_unit_s: float
    speed_world: float
    rho: float = 0.0
    arrival_rate_per_km2_per_hr: float = 0.0  # SI-side reporting only
    service_time_s: float = 1.0
    queue_capacity: int | None = 16

    def __post_init__(self) -> None:
        assert self.length_unit_m > 0, "length_unit_m must be positive."
        assert self.time_unit_s > 0, "time_unit_s must be positive."
        assert self.speed_world > 0, "speed_world must be positive."
        assert self.rho >= 0, "rho must be non-negative."
        if self.queue_capacity is not None:
            assert self.queue_capacity > 0, "queue_capacity must be positive."

    # ------------------------------------------------------------------
    # SI-side properties (for reporting, sanity checks, and paper values)
    # ------------------------------------------------------------------

    @property
    def speed_mps(self) -> float:
        """Agent speed in meters per second."""
        return self.speed_world * self.length_unit_m / self.time_unit_s

    @property
    def service_time_world(self) -> float:
        """Service time s(e) expressed in world-time units."""
        return self.service_time_s / self.time_unit_s

    # ------------------------------------------------------------------
    # Conversions: SI <-> world units
    # ------------------------------------------------------------------

    def to_world_length(self, meters: float) -> float:
        """Convert meters to world-length units."""
        return meters / self.length_unit_m

    def to_world_time(self, seconds: float) -> float:
        """Convert seconds to world-time units."""
        return seconds / self.time_unit_s

    def from_world_length(self, units: float) -> float:
        """Convert world-length units to meters."""
        return units * self.length_unit_m

    def from_world_time(self, units: float) -> float:
        """Convert world-time units to seconds."""
        return units * self.time_unit_s

    def arrival_rate_world(self, area_world_units2: float) -> float:
        """Convert the SI-side aggregate rate to a per-area rate in
        world-time units. Used by `env.arrivals.make_arrival` factory when
        a Domain is provided.

        rate_world = lambda * (length_unit_m / 1000)^2 * area_world_units2
                     * (time_unit_s / 3600)
        """
        area_km2 = area_world_units2 * (self.length_unit_m / 1000.0) ** 2
        per_world_time = self.arrival_rate_per_km2_per_hr * (self.time_unit_s / 3600.0)
        return area_km2 * per_world_time


# ---------------------------------------------------------------------------
# Preset domains anchored to published operational data
# ---------------------------------------------------------------------------


def chicago_emergency_ground() -> Domain:
    r"""Chicago ground-response SI anchor.

    Anchors the unit system to Chicago urban ground response (~12 m/s,
    ~0.24 incidents/km^2/hr, 606 km^2 city area). World units: 1 length =
    500 m, 1 time = 60 s, so speed_world = 1.44 -> 1.44*500/60 = 12 m/s.
    """
    return Domain(
        name="chicago_emergency_ground",
        length_unit_m=500.0,
        time_unit_s=60.0,
        speed_world=1.44,  # -> 12 m/s
        rho=0.0,
        arrival_rate_per_km2_per_hr=0.24,
        service_time_s=10.0,  # ~10 s nominal on-site stabilization
        queue_capacity=16,
    )


def toy_abstract() -> Domain:
    r"""Abstract / toy domain for unit tests and the 4-region line sanity.

    Declares an explicit 1:1 unit mapping so the abstract numerical
    values used by the unit tests remain valid.
    """
    return Domain(
        name="toy_abstract",
        length_unit_m=1.0,
        time_unit_s=1.0,
        speed_world=1.0,
        rho=0.0,
        arrival_rate_per_km2_per_hr=0.0,
        service_time_s=0.0,
        queue_capacity=None,  # unbounded
    )


# Registry for YAML-configurable domain selection.
DOMAIN_REGISTRY: dict[str, callable] = {
    "chicago_emergency_ground": chicago_emergency_ground,
    "toy_abstract": toy_abstract,
}


def get_domain(name: str) -> Domain:
    """Look up a preset Domain by name."""
    if name not in DOMAIN_REGISTRY:
        raise KeyError(
            f"Unknown domain {name!r}. Available: {sorted(DOMAIN_REGISTRY)}"
        )
    return DOMAIN_REGISTRY[name]()


__all__ = [
    "Domain",
    "chicago_emergency_ground",
    "toy_abstract",
    "DOMAIN_REGISTRY",
    "get_domain",
]
