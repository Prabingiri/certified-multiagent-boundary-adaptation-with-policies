r"""Geometry primitives for the CSG-RAG environment.

Regions are nodes in the time-varying graph G(t); the codebase needs only
a few operations on them -- point/region containment, shared-interface
detection, the diagonal, and the OA-BAR ULSP bound (Giri & Trajcevski,
MDM 2025, Theorem 1). Any shape supporting these works; this module
provides the axis-aligned `Rect`, which matches the OA-BAR partition output.

Numerical convention: geometric predicates use a GEOM_EPS tolerance.
Zero-measure (degenerate) rectangles are valid -- they represent delta=0
bands and 1-D interfaces -- and `contains` returns False on them.

References
----------
 - Giri & Trajcevski 2025. Obstacles-Aware Partitioning for Bounding
   Worst-Case Response Time of Mobile Surveillance Fleet. MDM 2025.
 - Duncan, Goodrich, Kobourov 2001. Balanced Aspect Ratio trees. J. Algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python 3.7 compatibility for the requested base env
    from typing_extensions import Protocol, runtime_checkable

import numpy as np

# Tolerance for all floating-point geometric predicate checks. Guards
# against floating-point residue in np.clip.
GEOM_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Region protocol (abstract interface)
# ---------------------------------------------------------------------------


@runtime_checkable
class Region(Protocol):
    """Abstract geometric region in the CSG-RAG partition.

    A concrete region must implement:
        contains(pt) -> bool                  point membership
        contains_region(other) -> bool        region containment (e.g. K_i in R_i)
        shares_interface_with(other) -> str|None  shared 1-D interface tag
        diag -> float                         upper bound on intra-region distance
        area -> float                         measure (diagnostics)

    `Rect` (below) is the only implementation here; other shapes can be
    added without changes to `shield`, `env.cs_lstf`, `trainers`, `metrics`.
    """

    def contains(self, pt: tuple[float, float]) -> bool: ...
    def contains_region(self, other: "Region") -> bool: ...
    def shares_interface_with(self, other: "Region") -> Optional[str]: ...
    @property
    def diag(self) -> float: ...
    @property
    def area(self) -> float: ...


# ---------------------------------------------------------------------------
# Concrete: axis-aligned rectangle (matches OA-BAR output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    r"""Axis-aligned rectangle [x0, x1] x [y0, y1] in R^2 (a Region).

    Matches the OA-BAR partition output (Giri & Trajcevski, MDM 2025) and
    gives closed-form Region methods. Degenerate (zero-measure) rectangles
    are valid -- they represent delta=0 bands and 1-D interfaces. The
    invariant x1 >= x0, y1 >= y0 is enforced in __post_init__; `contains`
    returns False on degenerate instances.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (self.x1 >= self.x0 and self.y1 >= self.y0):
            raise ValueError(
                f"Invalid rect {self}: must have x1 >= x0 and y1 >= y0."
            )

    # ------------- Region protocol implementations -------------

    def contains(self, pt: tuple[float, float]) -> bool:
        """Closed-interval membership test; False on degenerate rectangles."""
        if self.is_degenerate:
            return False
        x, y = pt
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def contains_region(self, other: "Region") -> bool:
        """Rect-in-Rect containment (the K_i subset R_i invariant,
        Definition 6.1 condition 2)."""
        if isinstance(other, Rect):
            return (
                self.x0 <= other.x0 + GEOM_EPS
                and self.y0 <= other.y0 + GEOM_EPS
                and self.x1 + GEOM_EPS >= other.x1
                and self.y1 + GEOM_EPS >= other.y1
            )
        raise NotImplementedError(
            f"Rect.contains_region only supports Rect operands; got {type(other).__name__}."
        )

    # Alias for callers that pass a Rect.
    def contains_rect(self, other: "Rect") -> bool:
        """Alias for `contains_region` when both operands are Rect."""
        return self.contains_region(other)

    def shares_interface_with(self, other: "Region", tol: float = GEOM_EPS) -> Optional[str]:
        """Return a tag identifying the shared 1-D interface, or None.

        For Rect-Rect, the tag gives the axis/orientation of the shared edge
        (used to orient the interface band):
            'x_right'  : self.x1 == other.x0   (other to the right of self)
            'x_left'   : self.x0 == other.x1   (other to the left)
            'y_top'    : self.y1 == other.y0   (other above)
            'y_bottom' : self.y0 == other.y1   (other below)
        """
        if not isinstance(other, Rect):
            raise NotImplementedError(
                f"Rect.shares_interface_with only supports Rect operands; got {type(other).__name__}."
            )
        # Vertical interfaces: self.x1 == other.x0 (other to the right), etc.
        if abs(self.x1 - other.x0) < tol and max(self.y0, other.y0) < min(self.y1, other.y1) - tol:
            return "x_right"
        if abs(self.x0 - other.x1) < tol and max(self.y0, other.y0) < min(self.y1, other.y1) - tol:
            return "x_left"
        # Horizontal interfaces:
        if abs(self.y1 - other.y0) < tol and max(self.x0, other.x0) < min(self.x1, other.x1) - tol:
            return "y_top"
        if abs(self.y0 - other.y1) < tol and max(self.x0, other.x0) < min(self.x1, other.x1) - tol:
            return "y_bottom"
        return None

    # ------------- Geometric quantities -------------

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def diag(self) -> float:
        r"""Euclidean diagonal d = sqrt(W^2 + H^2).

        This is the *d^j* term in OA-BAR's ULSP bound (Giri & Trajcevski,
        MDM 2025, Theorem 1): ULSP(R^j) <= d^j + (1/2) * sum_q P(P_{j,q}).
        """
        return float(np.hypot(self.width, self.height))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_degenerate(self) -> bool:
        """True if zero-measure (zero width or height): a delta=0 band or a
        1-D interface. `contains()` returns False on such instances."""
        return self.width <= GEOM_EPS or self.height <= GEOM_EPS


# Re-export the canonical name used by the rest of the codebase.
# `RectRegion` is the more explicit name aligned with the Region protocol;
# `Rect` is the short alias used throughout the env and shield code.
RectRegion = Rect


@dataclass(frozen=True)
class EligibilityRegion:
    r"""Executable service-eligibility region E_i(t) for one agent.

    The owner rectangle `owner_rect` remains the fixed accountability region
    R_i^0. Borrowed service strips are neighbor-side interface bands that are
    currently eligible for agent i to serve. This keeps ownership static while
    making the certified service surface dynamic.

    In the current rectangular interface model, borrowed strips are pairwise
    interior-disjoint up to measure-zero boundaries, so simple summation of
    their areas/perimeter contributions is conservative and typically exact.
    """

    owner_rect: Rect
    borrowed_strips: Sequence[Rect] = ()

    @property
    def member_rects(self) -> tuple[Rect, ...]:
        return (self.owner_rect, *tuple(self.borrowed_strips))

    def contains(self, pt: tuple[float, float]) -> bool:
        return any(r.contains(pt) for r in self.member_rects)

    def contains_region(self, other: "Region") -> bool:
        """Containment for kernel checks.

        Kernels remain fixed inside the owner rectangle, so owner-side
        containment is the relevant invariant. For non-owner-contained regions
        we conservatively return False rather than attempting a generic union
        containment proof.
        """
        if isinstance(other, Rect):
            return self.owner_rect.contains_region(other)
        raise NotImplementedError(
            "EligibilityRegion.contains_region currently supports Rect only."
        )

    def shares_interface_with(self, other: "Region") -> Optional[str]:
        raise NotImplementedError(
            "EligibilityRegion is a service surface, not a partition primitive."
        )

    @property
    def bounding_box(self) -> Rect:
        rects = self.member_rects
        return Rect(
            min(r.x0 for r in rects),
            min(r.y0 for r in rects),
            max(r.x1 for r in rects),
            max(r.y1 for r in rects),
        )

    @property
    def diag(self) -> float:
        return self.bounding_box.diag

    @property
    def area(self) -> float:
        return float(sum(r.area for r in self.member_rects))


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------


@dataclass
class Obstacle:
    r"""Axis-aligned rectangular obstacle (placeholder; extends to polygons later).

    Obstacles enter the ULSP bound via their *interior-restricted perimeter*:

        ULSP(R) = diag(R) + (1/2) * sum_q P(P_q restricted to interior of R).

    Boundary-aligned portions of an obstacle (edges that lie on the region
    boundary del R) are EXCLUDED because they do not induce detours within
    R -- a mobile unit can skirt them by traveling along the region boundary.
    See `interior_perimeter_in()` below.

    References
    ----------
    Giri & Trajcevski, MDM 2025, Theorem 1 (proof of the ULSP bound).
    """

    rect: Rect

    def interior_perimeter_in(self, region: Rect) -> float:
        r"""Compute P(P_q) -- the obstacle perimeter restricted to the
        INTERIOR of `region`.

        For an axis-aligned rectangular obstacle, we clip the obstacle to
        the region and sum the edges of the clipped rectangle that lie
        strictly inside the region (i.e., not coincident with the region
        boundary).
        """
        ix0 = max(self.rect.x0, region.x0)
        iy0 = max(self.rect.y0, region.y0)
        ix1 = min(self.rect.x1, region.x1)
        iy1 = min(self.rect.y1, region.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        w, h = ix1 - ix0, iy1 - iy0
        per = 0.0
        # Left edge of clipped obstacle at x = ix0 contributes h if interior.
        if ix0 > region.x0 + GEOM_EPS:
            per += h
        # Right edge at x = ix1.
        if ix1 < region.x1 - GEOM_EPS:
            per += h
        # Bottom edge at y = iy0.
        if iy0 > region.y0 + GEOM_EPS:
            per += w
        # Top edge at y = iy1.
        if iy1 < region.y1 - GEOM_EPS:
            per += w
        return per


def ulsp_bound(region: Rect, obstacles: list[Obstacle]) -> float:
    r"""OA-BAR upper bound on worst-case geodesic distance (Giri &
    Trajcevski, MDM 2025, Theorem 1):

        LSP(R) <= ULSP(R) := diag(R) + (1/2) * sum_q P(P_q interior of R),

    the Euclidean diagonal plus half the interior-restricted obstacle
    perimeter. Drives the design-time certificate U_bar_i = (1+rho)*U0_i,
    U0_i := ULSP(R_i(0)), and the runtime bound U_i(t) (the paper, Section 2).
    """
    obst_term = 0.5 * sum(o.interior_perimeter_in(region) for o in obstacles)
    return region.diag + obst_term


def ulsp_bound_region(region: Rect | EligibilityRegion, obstacles: list[Obstacle]) -> float:
    r"""ULSP-style upper bound for either a Rect or an EligibilityRegion.

    For EligibilityRegion, we use the diagonal of the union bounding box plus
    the sum of obstacle interior-perimeter contributions on each member
    rectangle. Because the borrowed strips are disjoint up to shared
    boundaries, this is conservative and typically exact under the current
    rectangular service-surface construction.
    """
    if isinstance(region, Rect):
        return ulsp_bound(region, obstacles)
    if isinstance(region, EligibilityRegion):
        obst_term = 0.5 * sum(
            o.interior_perimeter_in(r)
            for o in obstacles
            for r in region.member_rects
        )
        return region.diag + obst_term
    raise NotImplementedError(
        f"ulsp_bound_region does not support {type(region).__name__}."
    )


# ---------------------------------------------------------------------------
# Interface band (edge-local collaboration geometry)
# ---------------------------------------------------------------------------


@dataclass
class InterfaceBand:
    r"""Directed interface band B^(i)_{ij}(t) (the paper, Section 3.2).

    Given a shared interface Gamma_ij = del R_i intersect del R_j between
    two neighboring regions, the directed band on the i-side is the set

        B^(i)_ij(t) = { x in R_i(t) : dist(x, Gamma_ij(t)) <= delta_ij(t) },

    where delta_ij(t) in [0, delta_star_ij] is the directed slack.
    The corresponding two-sided shared buffer band is

        B_ij(t) = B^(i)_ij(t) union B^(j)_ij(t).

    An event with location x_e in B_ij(t) is buffer-eligible for the
    neighboring pair {i, j} (the paper, Section 3.4).

    For axis-aligned rectangular regions, the shared interface Gamma_ij is
    a line segment (vertical or horizontal) and the band is itself an
    axis-aligned rectangle -- which is what `band_rect()` returns.

    Attributes
    ----------
    owner, neighbor : int
        Region indices i (owner of this directed side) and j (opposite).
    axis : {'vertical', 'horizontal'}
        Orientation of the shared interface Gamma_ij.
    interface_coord : float
        x (if vertical) or y (if horizontal) of the interface.
    lo, hi : float
        Extent of the interface segment along the perpendicular axis.
    delta : float
        Current band width on the i-side, in [0, delta_star].
    delta_star : float
        Interface-local certified cap delta^star_ij > 0; any
        delta_ij in [0, delta_star] preserves endpoint certificates.
    direction : int
        +1/-1: which side of the interface R_i lies on.
    """

    owner: int
    neighbor: int
    axis: str
    interface_coord: float
    lo: float
    hi: float
    delta: float
    delta_star: float
    direction: int

    def band_rect(self, delta: float | None = None) -> Rect:
        """Return the band as a (possibly degenerate) axis-aligned rectangle.

        When delta == 0, the returned rectangle is degenerate (zero width);
        `Rect.contains()` correctly returns False on it, so no special case
        is needed at the call site.
        """
        use_delta = self.delta if delta is None else float(delta)
        if self.axis == "vertical":
            if self.direction == +1:
                return Rect(
                    self.interface_coord,
                    self.lo,
                    self.interface_coord + use_delta,
                    self.hi,
                )
            return Rect(
                self.interface_coord - use_delta,
                self.lo,
                self.interface_coord,
                self.hi,
            )
        # horizontal
        if self.direction == +1:
            return Rect(
                self.lo,
                self.interface_coord,
                self.hi,
                self.interface_coord + use_delta,
            )
        return Rect(
            self.lo,
            self.interface_coord - use_delta,
            self.hi,
            self.interface_coord,
        )

    def contains(self, pt: tuple[float, float], eps: float = GEOM_EPS) -> bool:
        """True iff `pt` lies in the directed band B^(owner)_{owner, neighbor}(t).

        The `delta <= eps` short-circuit handles both the semantic case
        delta == 0 ("no collaboration on this side") and the numerical case
        delta ~ 1e-17 (floating-point residue from repeated np.clip).
        """
        if self.delta <= eps:
            return False
        return self.band_rect().contains(pt)
