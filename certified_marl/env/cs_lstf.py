r"""CS-LSTF: certificate-sensitive admission and dispatch.

Service layer of the boundary-collaboration framework. Each admission is
screened against the fixed per-region response budget using a workload
envelope; admitted events are dispatched by least slack-time. Boundary
adaptation is handled separately by the controllers and shield.

The screen is conservative: it may reject events a finer routing analysis
could serve. Travel time is Euclidean distance / speed, exact in obstacle-free
domains. See the paper (Section 4) for the admission and slack definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from certified_marl.env.arrivals import Event
from certified_marl.env.geometry import Rect


# ---------------------------------------------------------------------------
# Agent state (the "node view" in the CSG-RAG formulation)
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    r"""Per-agent state exported to the shield and controller.

    Carries three categories of information:

    1. STATIC geometric quantities (set at construction, never mutated):
        region   : R_i(0), the ownership region (a Rect).
        kernel   : K_i, the invariant certified kernel, K_i subset R_i.
        U_bar    : fixed response budget.

    2. DYNAMIC service state (mutated by CS-LSTF admission/dispatch):
        position : p_i(t), the agent's current location.
        U_current: U_i(t), the current regional response bound (recomputed
                   after any certified boundary adaptation).
        queue    : Q_i(t), admitted but unfinished events committed to i.

    3. SMOOTHED CONTROL SIGNALS (exported to CPAC for interface decisions):
        latent_overload_ewma : bar_b_i(t), EWMA of rejected-pressure count
                               (the paper, Eq. 14).
        load_pressure_ewma   : l_i(t), reduced load signal (Eq. 16).

    The smoothed fields are EWMAs updated at the end of each epoch by
    `update_load_signals`, so CPAC observes post-service load.
    """

    idx: int
    position: tuple[float, float]
    region: Rect
    kernel: Rect
    U_bar: float
    U_current: float
    eligibility_diag: Optional[float] = None
    queue: list[Event] = field(default_factory=list)
    latent_overload_ewma: float = 0.0
    load_pressure_ewma: float = 0.0

    @property
    def q_i(self) -> int:
        """Admitted backlog size q_i(t) = |Q_i(t)| (the paper, Section 4.2)."""
        return len(self.queue)


# ---------------------------------------------------------------------------
# Core CS-LSTF primitives (pure functions on AgentState + Event)
# ---------------------------------------------------------------------------


def travel_time(
    p: tuple[float, float],
    x: tuple[float, float],
    speed: float = 1.0,
) -> float:
    r"""Travel time = Euclidean distance / speed.

    Exact in obstacle-free domains; a lower bound when obstacles constrain
    motion (the paper's d_G in Eq. 8 is obstacle-aware).
    """
    dx, dy = x[0] - p[0], x[1] - p[1]
    return float(np.hypot(dx, dy)) / speed


def lst_margin(
    agent: AgentState,
    event: Event,
    t: float,
    speed: float = 1.0,
) -> float:
    r"""Certificate-aware least-slack-time margin (the paper, Eq. 9).

        LST_i(e, t) = U_bar_i - (t - r_e) - tau_i(e, t) - s(e).

    Negative iff even an immediate start misses the budget (admission then
    rejects). Uses the fixed budget U_bar_i, not U_i(t): U_bar_i is the
    contractual bound and U_i(t) <= U_bar_i is held by the shield (Sec. 3.3).
    """
    tau = travel_time(agent.position, (event.x, event.y), speed)
    return agent.U_bar - (t - event.occurrence_time) - tau - event.service_time


def workload_envelope(
    agent: AgentState,
    t: float,
    speed: float = 1.0,
) -> float:
    r"""Conservative workload envelope W_i(t) (the paper, Eq. 10).

        W_i(t) = sum_{e' in Q_i} (tau_hat_i(e', t) + s(e')),

    with tau_hat_i a per-event remaining-travel bound. We use region
    diagonal / speed (exact obstacle-free; the paper's tau_hat uses ULSP).
    """
    diag = agent.eligibility_diag if agent.eligibility_diag is not None else agent.region.diag
    diameter = float(diag) / speed
    return sum(diameter + e.service_time for e in agent.queue)


def least_slack_margin(
    agent: AgentState,
    t: float,
    speed: float = 1.0,
) -> float:
    r"""Tightest LST across the admitted backlog (m_i(t), the paper Sec. 4.2):

        m_i(t) = min_{e in Q_i(t)} LST_i(e, t);  U_bar_i if the queue is empty.

    Supplies m_i(t) for the load signal and the actor's 7-D state. Returns
    U_bar_i (not +inf) on an empty queue so it stays finite as an NN
    feature; the load-signal g_i uses +inf for the empty case.
    """
    if not agent.queue:
        return float(agent.U_bar)
    return min(lst_margin(agent, e, t, speed) for e in agent.queue)


def admit(
    agent: AgentState,
    event: Event,
    t: float,
    speed: float = 1.0,
) -> bool:
    r"""Certificate-consistent admission predicate (the paper, Eq. 11).

        ADMIT_i(Q_i, e) = 1   iff   LST_i(e, t) >= 0
                                AND W_i(t) + tau_i(e, t) + s(e) <= U_bar_i.

    Both clauses are needed: LST>=0 protects e, and the W_i envelope keeps
    already-committed events from being pushed past budget.
    """
    if lst_margin(agent, event, t, speed) < 0.0:
        return False
    W = workload_envelope(agent, t, speed)
    tau = travel_time(agent.position, (event.x, event.y), speed)
    return (W + tau + event.service_time) <= agent.U_bar


def dispatch_local(
    agent: AgentState,
    t: float,
    speed: float = 1.0,
) -> Optional[Event]:
    r"""Local service priority (the paper, Eq. 12).

        e*_i(t) in argmin_{e in Q_i} LST_i(e, t), tie-broken by min tau_i.

    Least-slack-time-first on the certificate-aware slack (Eq. 9).
    """
    if not agent.queue:
        return None
    best, best_key = None, (float("inf"), float("inf"))
    for e in agent.queue:
        lst = lst_margin(agent, e, t, speed)
        tau = travel_time(agent.position, (e.x, e.y), speed)
        key = (lst, tau)
        if key < best_key:
            best_key = key
            best = e
    return best


def interface_dispatch(
    agent_i: AgentState,
    agent_j: AgentState,
    event: Event,
    t: float,
    speed: float = 1.0,
) -> Optional[AgentState]:
    r"""Dispatch a buffer-eligible event to the better endpoint (the paper, Eq. 13).

    For an event in the shared band B_ij(t), assignable to either endpoint:

        assign(e) in argmin_{u in {i,j}} LST_u(e, t)  s.t.  ADMIT_u(Q_u, e) = 1.

    Returns the chosen AgentState, or None if neither endpoint admits.
    """
    ok_i = admit(agent_i, event, t, speed)
    ok_j = admit(agent_j, event, t, speed)
    if ok_i and ok_j:
        lst_i = lst_margin(agent_i, event, t, speed)
        lst_j = lst_margin(agent_j, event, t, speed)
        # argmin LST (Eq. 13): assign to the endpoint where the event has
        # the tighter deadline. This preserves the worst-case response tail.
        return agent_i if lst_i <= lst_j else agent_j
    if ok_i:
        return agent_i
    if ok_j:
        return agent_j
    return None


# ---------------------------------------------------------------------------
# Load signals: latent overload, pressure, and reduced load signal (Eqs. 14-16)
# ---------------------------------------------------------------------------


def update_load_signals(
    agent: AgentState,
    latent_count_this_epoch: int,
    t: float,
    gamma_b: float = 1.0,
    gamma_m: float = 1.0,
    gamma_w: float = 0.0,   # spatial-utilization weight in load
    alpha_b: float = 0.9,
    alpha: float = 0.9,
    speed: float = 1.0,
) -> None:
    r"""Mutate the agent's smoothed control signals in place.

    Computes, in order (the paper Eqs. 14-16; the code's p_i is the
    paper's g_i):

        bar_b_i(t) = alpha_b * bar_b_i(t-1) + (1 - alpha_b) * b_i^-(t)     [Eq. 14]
        m_i(t)     = min_{e in Q_i} LST_i(e, t)  or  +inf if empty
        p_i(t)     = q_i(t) + gamma_b * bar_b_i(t) + gamma_m * [-m_i(t)]_+   [Eq. 15]
        l_i(t)     = alpha * l_i(t-1) + (1 - alpha) * p_i(t)           [Eq. 16]

    where b_i^-(t) is the latent overload count (events eligible under
    geometric ownership that fail the admission predicate). l_i is the
    single local-load scalar CPAC observes; the EWMAs filter transients.
    """
    # EWMA of latent overload.
    agent.latent_overload_ewma = (
        alpha_b * agent.latent_overload_ewma + (1 - alpha_b) * float(latent_count_this_epoch)
    )

    # Tightest committed slack m_i(t).
    if agent.queue:
        m_i = min(lst_margin(agent, e, t, speed) for e in agent.queue)
    else:
        m_i = float("inf")

    # Instantaneous load pressure p_i(t) = g_i(t) (Eq. 15). The optional
    # gamma_w term adds the certificate-utilization fraction W_i/U_bar_i
    # (default gamma_w=0 recovers the paper's g_i exactly).
    util_frac = 0.0
    if gamma_w > 0.0 and agent.U_bar > 0.0:
        W = workload_envelope(agent, t, speed)
        util_frac = float(W) / float(agent.U_bar)
    p_i = (
        agent.q_i
        + gamma_b * agent.latent_overload_ewma
        + gamma_m * max(0.0, -m_i)
        + gamma_w * util_frac
    )

    # Reduced Markov load l_i(t) via EWMA (Eq. 16).
    agent.load_pressure_ewma = alpha * agent.load_pressure_ewma + (1 - alpha) * p_i
