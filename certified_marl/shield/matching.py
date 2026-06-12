r"""Endpoint-disjoint active-interface selection.

"""

from __future__ import annotations

from typing import Iterable


def propose_dominant(
    loads: list[float],
    neighbors: dict[int, list[int]],
) -> list[tuple[int, int]]:
    r"""Each agent proposes one dominant-imbalance neighbor.

    Implements the paper's Sec. 4.3 proposal: j^*(t) = argmax_{j in N(i)}
    |l_i(t) - l_j(t)|. Ties are broken deterministically by the smallest
    neighbor index to ensure reproducibility across runs.

    Parameters
    ----------
    loads : list[float]
        l_i(t) for i = 0, ..., n-1 (the reduced Markov load from CS-LSTF,
        the paper Eq. 16).
    neighbors : dict[int, list[int]]
        One-hop neighbor map: neighbors[i] contains j iff region i shares an
        interface with region j.

    Returns
    -------
    list of UNORDERED pairs (i, j) with i < j. Duplicates are merged
    (if both i proposes j and j proposes i, the pair appears once).
    """
    props: set[tuple[int, int]] = set()
    for i, ns in neighbors.items():
        if not ns:
            continue
        best_j, best_val = None, -1.0
        for j in ns:
            val = abs(loads[i] - loads[j])
            # argmax with deterministic tie-break on smaller j.
            if val > best_val or (val == best_val and (best_j is None or j < best_j)):
                best_val = val
                best_j = j
        if best_j is not None:
            # Store as a canonical (min, max) unordered pair.
            props.add((min(i, best_j), max(i, best_j)))
    return list(props)


def greedy_weighted_matching(
    proposed_edges: Iterable[tuple[int, int]],
    weights: dict[tuple[int, int], float],
) -> list[tuple[int, int]]:
    r"""Greedy weighted matching on a simple graph.

    Algorithm. Sort edges by weight descending; accept an edge iff neither
    endpoint is already used by a previously accepted edge. Deterministic
    tie-break on the tuple (i, j) (smaller tuple first).

    The structural guarantee uses endpoint-disjointness, not optimality of
    the matching objective. Greedy matching gives the needed disjoint active
    interfaces with deterministic tie-breaking.

    Returns
    -------
    list of accepted edges (a matching on the input graph).
    """
    edges = sorted(proposed_edges, key=lambda e: (-weights[e], e))
    used: set[int] = set()
    match: list[tuple[int, int]] = []
    for (i, j) in edges:
        if i in used or j in used:
            continue
        match.append((i, j))
        used.add(i)
        used.add(j)
    return match


def active_interfaces(
    loads: list[float],
    neighbors: dict[int, list[int]],
) -> list[tuple[int, int]]:
    r"""Full pipeline: dominant-neighbor proposal -> weighted matching.

    Convenience wrapper combining `propose_dominant` and
    `greedy_weighted_matching`. Returns the active-interface set.
    """
    proposed = propose_dominant(loads, neighbors)
    w = {e: abs(loads[e[0]] - loads[e[1]]) for e in proposed}
    return greedy_weighted_matching(proposed, w)


def controller_of(i: int, j: int, loads: list[float]) -> int:
    r"""Controller selection rule.

    The controller endpoint c(i, j; t) is the lighter-loaded endpoint:

        c(i, j; t) = i if l_i(t) <= l_j(t)
                   = j otherwise.

    Only the controller samples a stochastic action; the opposite endpoint
    executes the inverse motion enforcing antisymmetry a_ji = -a_ij
    by construction. Tie-break favors the numerically smaller index i.

    Rationale. Placing the controller at the lighter side anchors the
    policy's "canonical orientation" so that the state vector s_cc_bar(t)
    is always described from the perspective of the endpoint that
    has slack to absorb load, which simplifies the learning signal.
    """
    if loads[i] <= loads[j]:
        return i
    return j
