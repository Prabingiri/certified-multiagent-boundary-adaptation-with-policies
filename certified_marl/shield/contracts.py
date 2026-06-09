r"""Assume-guarantee local contracts Phi_{ij}.

For each interface (i, j) in the region-adjacency graph G(t) = (V, E(t)),
the *local safety predicate* is

    Phi_ij(t) := delta_ij(t) in [0, delta_star_ij]
              /\ K_i subset R_i(t) /\ K_j subset R_j(t)
              /\ U_i(t) <= U_bar_i /\ U_j(t) <= U_bar_j

This is the part of the certified execution state (the paper, Definition
6.1) that is LOCAL to the endpoints (i, j). The active interfaces are
endpoint-disjoint by construction (the paper, Section 6, under Eq. 17),
so conjoining Phi_ij over them gives the whole-state guarantee of
Theorem 6.2 (feasibility preservation under shielded boundary updates).

This is compositional shielding: a global specification (Definition 6.1)
reduced to local obligations (Phi_ij) discharged by per-interface shields
(A^safe_ij in `feasibility_kernel.py`).
"""

from __future__ import annotations

from certified_marl.env.csgrag import CSGRAGState, InterfaceState
from certified_marl.env.geometry import GEOM_EPS


def phi_ij(
    s: CSGRAGState,
    interface: InterfaceState,
    tol: float = GEOM_EPS,
) -> tuple[bool, str]:
    r"""Evaluate the local predicate Phi_{ij} at the current state.

    Parameters
    ----------
    s : CSGRAGState
        Current global state (need the endpoint agents' U_current, region,
        and kernel).
    interface : InterfaceState
        The edge whose local predicate is being evaluated; both directed
        slacks (band_ij, band_ji) and the certified cap delta_star are
        checked.
    tol : float
        Numerical tolerance for <= comparisons. Defaults to GEOM_EPS (1e-12).

    Returns
    -------
    (ok, reason) : tuple[bool, str]
        `ok = True` iff all five conjuncts of Phi_ij hold.
    """
    i, j = interface.i, interface.j
    a_i, a_j = s.agents[i], s.agents[j]

    # Clause 1: slack admissibility on both directed sides.
    if not (0.0 - tol <= interface.band_ij.delta <= interface.delta_star + tol):
        return False, "slack"
    if not (0.0 - tol <= interface.band_ji.delta <= interface.delta_star + tol):
        return False, "slack"


    if not a_i.region.contains_region(a_i.kernel):
        return False, "ker"
    if not a_j.region.contains_region(a_j.kernel):
        return False, "ker"

    # Clause 3: certificate budget preservation U_i(t) <= U_bar_i.
    if a_i.U_current > a_i.U_bar + tol:
        return False, "cert"
    if a_j.U_current > a_j.U_bar + tol:
        return False, "cert"

    return True, "ok"
