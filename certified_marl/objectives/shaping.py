r"""Pairwise CPAC reward helpers.

The CPAC objective uses the pair-local stress potential

    Psi_ij(t) = omega_l * |l_i(t) - l_j(t)|
              + omega_b * (bar_b_i(t) + bar_b_j(t))
              + kappa * (l_i(t) + l_j(t))

and subtracts eta for non-hold boundary actions.
"""
from __future__ import annotations

from dataclasses import dataclass


from certified_marl.env.csgrag import CSGRAGState


@dataclass
class ShapingWeights:
    r"""Coefficients for the CPAC stress potential and action penalty."""

    beta: float = 1.0       # omega_l in the paper
    omega_b: float = 0.5    # latent-overload weight
    kappa: float = 0.1      # total pair-load regularizer
    eta: float = 0.01     # action-oscillation indicator penalty (eta_m)


def psi_ij(
    state: CSGRAGState,
    i: int,
    j: int,
    w: ShapingWeights | None = None,
    speed: float = 1.0,
) -> float:
    r"""Evaluate Psi_{ij}(t) at the current state.

    Base form (Eq. 25):
        Psi_ij = beta * |l_i - l_j|              [pairwise load mismatch]
               + omega_b * (bar_b_i + bar_b_j)    [endpoint latent overload]
               + kappa * (l_i + l_j)             [absolute local load]


    Parameters
    ----------
    state : CSGRAGState
    i, j : int  (endpoint indices, order-symmetric)
    w : ShapingWeights
    speed : float
        Agent travel speed. Pass env.speed.

    Returns
    -------
    float >= 0. Lower is better.
    """
    w = w or ShapingWeights()
    a_i = state.agents[i]
    a_j = state.agents[j]
    l_i = a_i.load_pressure_ewma
    l_j = a_j.load_pressure_ewma
    b_i = a_i.latent_overload_ewma
    b_j = a_j.latent_overload_ewma
    val = (
        w.beta * abs(l_i - l_j)
        + w.omega_b * (b_i + b_j)
        + w.kappa * (l_i + l_j)
    )
    return val


def pairwise_reward(
    psi_before: float,
    psi_after: float,
    action: int,
    w: ShapingWeights | None = None,
) -> float:
    r"""Compute the one-step shaping reward r_ij(t) (the paper, Eq. 26).

        r_ij(t) = Psi_ij(t) - Psi_ij(t+1) - eta * 1[a_ij(t) != 0].

    Parameters
    ----------
    psi_before : float
        Psi_ij(t), evaluated BEFORE the edge action and arrival/service
        transition.
    psi_after : float
        Psi_ij(t+1), evaluated AFTER the transition.
    action : int
        The executed signed action a_ij(t) in {-1, 0, +1}. Only its zero/
        non-zero status matters here.
    w : ShapingWeights

    Returns
    -------
    float -- may be positive (progress), negative (regression), or zero.

    Note
    ----
    A positive return means Psi DECREASED across the transition, which is
    the intended direction. The sign convention makes the reward align
    naturally with standard RL "higher is better" semantics.
    """
    w = w or ShapingWeights()
    oscillation_cost = w.eta if action != 0 else 0.0
    return (psi_before - psi_after) - oscillation_cost


def pairwise_reward_nstep(
    psi_window: list[float],
    action: int,
    n: int,
    w: ShapingWeights | None = None,
) -> float:
    r"""n-step potential-difference shaping reward.

        r_ij(t) = (1/n) * (Psi(s_t) - Psi(s_{t+n})) - eta * 1[a_ij != 0]

    This is a sum-of-one-step-potential-differences telescoped over n
    steps and divided by n; equivalent to averaged TD(n) shaping. By
    Wiewiora (JAIR 2003), composition of potential-based shaping terms is
    potential-based, so the optimal policy of the underlying CMDP is
    preserved for any n >= 1.

    Reduces single-step variance and credits multi-epoch boundary-motion
    effects to their causal action, so the policy learns coherent
    multi-step shifts rather than uncoordinated single-step actions.

    Parameters
    ----------
    psi_window : list[float]
        Psi values at successive states [Psi(s_t), Psi(s_{t+1}), ...,
        Psi(s_{t+m})] for some m. Must have length >= 2. The function
        uses Psi(s_t) and Psi(s_{t+min(n, m-1)}); shorter windows fall
        back to a smaller effective n at episode boundaries.
    action : int
        Executed signed action a_ij(t) at step t.
    n : int
        Target n-step horizon. Effective n = min(n, len(psi_window) - 1).
    w : ShapingWeights

    Returns
    -------
    float. n=1 recovers the one-step reward exactly.
    """
    if len(psi_window) < 2:
        # Fallback: cannot compute even a 1-step difference.
        return -(w.eta if (w and action != 0) else
                 (ShapingWeights().eta if action != 0 else 0.0))
    w = w or ShapingWeights()
    n_eff = max(1, min(n, len(psi_window) - 1))
    psi_t = psi_window[0]
    psi_tn = psi_window[n_eff]
    progressive = (psi_t - psi_tn) / float(n_eff)
    oscillation_cost = w.eta if action != 0 else 0.0
    return progressive - oscillation_cost

