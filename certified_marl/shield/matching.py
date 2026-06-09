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

    This is not an OPTIMAL matching -- that would be MWM in O(V^3) via
    Gabow's blossom algorithm (or O(V*E) via Edmonds). Greedy is a
    2-approximation for maximum weight and is MORE than sufficient here
    because the load imbalance weights are only a heuristic for which
    interfaces benefit most from collaboration, not a quantity we commit
    to maximize exactly (the paper, Section 5.1). Optimality of the
    matching is NOT required for Theorem 6.2; only endpoint-disjointness
    is required, and that is guaranteed by any matching.

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
    `greedy_weighted_matching`. Returns E_act(t) of the paper.
    """
    proposed = propose_dominant(loads, neighbors)
    w = {e: abs(loads[e[0]] - loads[e[1]]) for e in proposed}
    return greedy_weighted_matching(proposed, w)


def active_interfaces_band_aware(
    loads: list[float],
    neighbors: dict[int, list[int]],
    gen_by_agent_neighbor: dict,
    band_threshold: int = 2,
    band_weight: float = 0.1,
) -> list[tuple[int, int]]:
    r"""Band-aware matching primitive (SG_v2 symmetry).

    Identical structure to ``active_interfaces`` (proposal then greedy
    matching), but the proposal step is band-aware. Each region:
      (a) proposes its dominant-gradient neighbor (same as SG_v1), AND
      (b) additionally proposes any neighbor with per-epoch band activity
          >= ``band_threshold``, even when |l_i - l_j| <= 0.
    The matching weight combines load-gradient and band-activity:
        weight(i, j) = |l_i - l_j| + band_weight * (gen_ij + gen_ji).

    The greedy matching still returns an endpoint-disjoint matching, so the
    endpoint-disjointness property (and thus the feasibility preservation of
    Theorem 6.2) holds unchanged.
    """
    props: set[tuple[int, int]] = set()
    for i, ns in neighbors.items():
        if not ns:
            continue
        # (a) Standard dominant-gradient proposal.
        best_j, best_val = None, -1.0
        for j in ns:
            val = abs(loads[i] - loads[j])
            if val > best_val or (val == best_val and (best_j is None or j < best_j)):
                best_val = val
                best_j = j
        if best_j is not None:
            props.add((min(i, best_j), max(i, best_j)))
        # (b) Band-aware additional proposals.
        band_i = gen_by_agent_neighbor.get(i, {})
        for j in ns:
            bij = band_i.get(j, 0) + gen_by_agent_neighbor.get(j, {}).get(i, 0)
            if bij >= band_threshold:
                props.add((min(i, j), max(i, j)))
    # Combined weight: load gradient + band activity.
    w = {}
    for (i, j) in props:
        load_diff = abs(loads[i] - loads[j])
        band_act = float(
            gen_by_agent_neighbor.get(i, {}).get(j, 0)
            + gen_by_agent_neighbor.get(j, {}).get(i, 0)
        )
        w[(i, j)] = load_diff + band_weight * band_act
    return greedy_weighted_matching(list(props), w)


def controller_of(i: int, j: int, loads: list[float]) -> int:
    r"""Handshake rule (the paper, Eq. 19).

    The controller endpoint c(i, j; t) is the LIGHTER-load endpoint:

        c(i, j; t) = i if l_i(t) <= l_j(t)
                   = j otherwise.

    Only the controller samples a stochastic action; the opposite endpoint
    executes the inverse motion (Eq. 4), enforcing antisymmetry a_ji = -a_ij
    by construction. Tie-break favors the numerically smaller index i.

    Rationale. Placing the controller at the lighter side anchors the
    policy's "canonical orientation" so that the state vector s_cc_bar(t)
    (Eq. 20) is always described from the perspective of the endpoint that
    has slack to absorb load, which simplifies the learning signal.
    """
    if loads[i] <= loads[j]:
        return i
    return j
