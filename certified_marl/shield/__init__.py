r"""Compositional edge-local feasibility kernel (the shield).

This subpackage realizes the bottom layer of the three-layer safe
hierarchical architecture (package root docstring, section "Three-layer
safe hierarchical architecture"):

    Layer L (safety):  feasibility kernel = A^safe_{ij}(s_t).

The shield is constructed so that Theorem 6.2 of the paper (feasibility
preservation under shielded boundary updates) follows from the local
per-interface check plus the endpoint-disjoint matching (the paper,
Section 6, under Eq. 17).

Exports
-------
    phi_ij                     : the local predicate Phi_{ij}(t).
    check_edge_action_safe     : the per-edge feasibility test.
    safe_action_set            : A^safe_{ij}(s_t) = {a in {-1,0,+1} : feasible}.
    greedy_weighted_matching   : endpoint-disjoint active-interface selection.
    propose_dominant           : dominant-neighbor proposal step.
    active_interfaces          : propose -> weight -> match pipeline.
    controller_of              : handshake rule for antisymmetric execution.

See the paper, Section 3.4 (feasibility kernel) and Section 6 (preservation).
"""

from certified_marl.shield.contracts import phi_ij
from certified_marl.shield.feasibility_kernel import (
    check_edge_action_safe,
    safe_action_set,
)
from certified_marl.shield.matching import (
    active_interfaces,
    controller_of,
    greedy_weighted_matching,
    propose_dominant,
)

__all__ = [
    "phi_ij",
    "check_edge_action_safe",
    "safe_action_set",
    "greedy_weighted_matching",
    "active_interfaces",
    "controller_of",
    "propose_dominant",
]
