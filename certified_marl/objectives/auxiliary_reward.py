r"""Auxiliary CPAC reward terms used by the CPAC trainer.

The auxiliary reward adds three interface-local terms to the base
potential-difference reward:

    - omega_zeta * max{zeta_i, zeta_j}
    - omega_Delta * |zeta_i - zeta_j|
    + omega_c * C_ij^cross

This module returns those terms for one active interface. The capacity-overflow
chi term is an optional extension, disabled by default. The caller adds the
base reward and motion cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass
class AuxiliaryRewardWeights:
    """Coefficients for auxiliary reward terms; defaults are no-op."""

    # max rejection share: -omega_zeta * max{zeta_i, zeta_j}
    omega_zeta: float = 0.0

    # rejection imbalance: -omega_Delta * |zeta_i - zeta_j|
    omega_Delta: float = 0.0

    # owner-cross admission credit: +omega_c * C_ij^cross
    omega_c: float = 0.0

    # optional capacity-overflow extension: -omega_chi * max(0, chi_i-1, chi_j-1)
    omega_chi: float = 0.0

    # Reference workload-intensity scale: dt of the env (used to convert
    # per-epoch generated counts into a continuous-time intensity lambda_i).
    # Default 1.0 (matches env.dt default).
    intensity_dt: float = 1.0

    # chi clip: an instantaneous chi computed from a single-epoch generated
    # count can be very large (e.g. 3 events on a sub-second epoch implies
    # chi=20+). The overflow penalty then dominates training. Cap chi at
    # chi_max so the penalty stays in a reasonable scale. Default 3.0:
    # "infeasibility at 3x or more is the same signal" - we don't need to
    # punish 20x differently from 3x.
    chi_max: float = 3.0

    # Optional edge-local rejection shares. When True, max-rejection and rejection-imbalance terms read edge-local shares instead of per-agent shares. Default False uses the per-agent share.
    use_edge_local_zeta: bool = False


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def _zeta_pair(state: Any, i: int, j: int,
               use_edge_local: bool = False) -> tuple[float, float]:
    """Return (zeta_{i|ij}, zeta_{j|ij}) for an active interface (i, j).

    When `use_edge_local` is True, read from
    `state.edge_local_rejection_share[(i, j)]` / `[(j, i)]` (per-edge,
    when available). Otherwise read `state.rejection_share_this_epoch` (per-agent share).
    """
    if use_edge_local:
        els = getattr(state, "edge_local_rejection_share", None) or {}
        zi = float(els.get((i, j), 0.0))
        zj = float(els.get((j, i), 0.0))
        return zi, zj
    rs = getattr(state, "rejection_share_this_epoch", None) or {}
    return float(rs.get(i, 0.0)), float(rs.get(j, 0.0))


def _chi_pair(state: Any, i: int, j: int,
              intensity_dt: float) -> tuple[float, float]:
    """Return (chi_i, chi_j) = (lambda_i / lambda_i^cert).

    lambda_i is taken as generated_this_epoch_by_agent[i] / intensity_dt
    (offered intensity proxy under inhomogeneous Poisson). When
    lambda_i^cert is missing or non-positive, returns 0 so the
    overflow penalty cannot fire on a malformed state.
    """
    gen = getattr(state, "generated_this_epoch_by_agent", None) or {}
    lcert = getattr(state, "lambda_cert_per_agent", None) or {}
    dt = max(float(intensity_dt), 1.0e-9)
    lam_i = float(gen.get(i, 0)) / dt
    lam_j = float(gen.get(j, 0)) / dt
    lcert_i = float(lcert.get(i, 0.0))
    lcert_j = float(lcert.get(j, 0.0))
    chi_i = lam_i / lcert_i if lcert_i > 0.0 else 0.0
    chi_j = lam_j / lcert_j if lcert_j > 0.0 else 0.0
    return chi_i, chi_j


def _chi_clipped(chi: float, chi_max: float) -> float:
    """Cap chi at chi_max to keep overflow penalty in a reasonable scale."""
    return min(max(chi, 0.0), max(float(chi_max), 0.0))


def _cross_count(state: Any, i: int, j: int) -> float:
    """Return C_ij^true-cross(t) for canonical pair (min, max)."""
    cm = getattr(state, "cross_admitted_this_epoch_by_pair", None) or {}
    edge = (min(i, j), max(i, j))
    return float(cm.get(edge, 0))


# ---------------------------------------------------------------------------
# Per-term reward components
# ---------------------------------------------------------------------------


def max_rejection_rate_penalty(state: Any, i: int, j: int,
                                weights: AuxiliaryRewardWeights) -> float:
    """(c) -omega_zeta * max{zeta_i, zeta_j}."""
    if weights.omega_zeta <= 0.0:
        return 0.0
    zi, zj = _zeta_pair(state, i, j,
                        use_edge_local=weights.use_edge_local_zeta)
    return -weights.omega_zeta * max(zi, zj)


def rejection_imbalance_penalty(state: Any, i: int, j: int,
                                 weights: AuxiliaryRewardWeights) -> float:
    """(d) -omega_Delta * |zeta_i - zeta_j|."""
    if weights.omega_Delta <= 0.0:
        return 0.0
    zi, zj = _zeta_pair(state, i, j,
                        use_edge_local=weights.use_edge_local_zeta)
    return -weights.omega_Delta * abs(zi - zj)


def true_cross_reward(state: Any, i: int, j: int,
                       weights: AuxiliaryRewardWeights) -> float:
    """(e) +omega_c * C_ij^true-cross(t)."""
    if weights.omega_c <= 0.0:
        return 0.0
    c = _cross_count(state, i, j)
    return weights.omega_c * c


def capacity_overflow_penalty(state: Any, i: int, j: int,
                               weights: AuxiliaryRewardWeights) -> float:
    """(f) -omega_chi * max(0, chi_i-1, chi_j-1)."""
    if weights.omega_chi <= 0.0:
        return 0.0
    chi_i, chi_j = _chi_pair(state, i, j, weights.intensity_dt)
    chi_i = _chi_clipped(chi_i, weights.chi_max)
    chi_j = _chi_clipped(chi_j, weights.chi_max)
    excess = max(0.0, chi_i - 1.0, chi_j - 1.0)
    return -weights.omega_chi * excess


# ---------------------------------------------------------------------------
# Aggregator: single call site for the trainer
# ---------------------------------------------------------------------------


def auxiliary_reward_terms(state: Any, i: int, j: int,
                                weights: AuxiliaryRewardWeights) -> float:
    """Return auxiliary reward terms for one active interface.

    Returns 0.0 when every weight is 0.0, so this function is safe to call
    unconditionally in the trainer (no overhead beyond a few dict reads
    when disabled).
    """
    if (weights.omega_zeta <= 0.0
            and weights.omega_Delta <= 0.0
            and weights.omega_c <= 0.0
            and weights.omega_chi <= 0.0):
        return 0.0
    r_c = max_rejection_rate_penalty(state, i, j, weights)
    r_d = rejection_imbalance_penalty(state, i, j, weights)
    r_e = true_cross_reward(state, i, j, weights)
    r_f = capacity_overflow_penalty(state, i, j, weights)
    return float(r_c + r_d + r_e + r_f)


__all__ = [
    "AuxiliaryRewardWeights",
    "max_rejection_rate_penalty",
    "rejection_imbalance_penalty",
    "true_cross_reward",
    "capacity_overflow_penalty",
    "auxiliary_reward_terms",
]
