r"""CSG-RAG Environment: Constrained Stochastic Game on a Region-Adjacency Graph.

The environment binds geometry (regions + kernels + interface bands),
arrivals (the stochastic event stream), and CS-LSTF (certified admission /
dispatch) into a single discrete-epoch `step(actions)` API used by the
trainers.

State (at epoch t)
------------------
For the formal CSG-RAG tuple see the package root docstring. At runtime
the env exposes:

    node features per region i:
        x_i(t) = (q_i, bar_b_i, m_i, U_i, U_bar_i, p_i_xy, kernel_rect)
    edge features per interface (i, j) in E(t):
        e_ij(t) = (delta_ij, delta_star_ij, Delta_ell_ij, eta_ij, h_ij)
    joint:
        s_t = ({x_i}, {e_ij})

Actions
-------
Edge-local a_ij(t) in {-1, 0, +1} on ACTIVE interfaces (selected by the
matching step in `shield.matching.active_interfaces`), with antisymmetry
a_ji = -a_ij (the paper, Eq. 4).

Step order (each epoch)
-----------------------
A single call to `step(actions)` performs, in order:

    1. ARRIVALS    : sample new events from the arrival process on [t, t+dt].
    2. ADMISSION   : for each event, run CS-LSTF admission either at the
                     geometric owner (if the event is in private interior)
                     or via interface_dispatch (if the event is in a buffer
                     band B_ij). Rejected events become latent overload.
    3. SERVICE     : for each agent, dispatch the tightest-slack committed
                     event (if any). Compute realized response time; record
                     any fixed-budget exceedance.
    4. EDGE ACTION : apply the learned edge actions (one per active
                     interface), double-checking safety via the shield.
                     Residues below GEOM_EPS are snapped to exact 0 /
                     delta_star to avoid floating-point residue.
    5. CERTIFICATE : recompute U_i(t+1) on updated geometry. Check the
                     certified-state conditions (Definition 6.1: kernel
                     containment, U_i <= U_bar).
    6. LOAD SIGNAL : update bar_b_i and l_i EWMAs (Eqs. 14-16 of the paper).
    7. CLOCK       : advance t := t + 1; set info["done"] if t >= horizon.

Safety counters
---------------
The env maintains per-category violation counters (geom, ker, cert, srv,
team) that are zero under the shield (Theorem 6.2). They are read at eval
time by `metrics.safety.safety_from_state` (Block A of the results table).
A non-zero counter indicates a bug in the shield or CS-LSTF, not a policy
failure -- the shield guarantees these by construction.

See the paper, Section 3 (system model + dynamics) and Section 4 (CS-LSTF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from certified_marl.env.arrivals import ArrivalProcess, Event
from certified_marl.env.cs_lstf import (
    AgentState,
    admit,
    dispatch_local,
    interface_dispatch,
    travel_time,
    update_load_signals,
)
from certified_marl.env.domain import Domain
from certified_marl.env.geometry import (
    EligibilityRegion,
    GEOM_EPS,
    InterfaceBand,
    Obstacle,
    Rect,
    ulsp_bound,
    ulsp_bound_region,
)


# ---------------------------------------------------------------------------
# Edge and joint state containers
# ---------------------------------------------------------------------------


@dataclass
class InterfaceState:
    r"""Bookkeeping for an unordered interface {i, j}.

    Stores the directed slack on each side (band_ij owned by i, band_ji
    owned by j) and the interface-local certified cap delta_star.
    """

    i: int
    j: int
    band_ij: InterfaceBand
    band_ji: InterfaceBand
    delta_star: float

    @property
    def key(self) -> tuple[int, int]:
        """Canonical unordered-pair key (min, max)."""
        return (min(self.i, self.j), max(self.i, self.j))


@dataclass
class CSGRAGState:
    r"""Full joint environment state s_t (numpy-native).

    The trainer wraps this into tensors at rollout time. Keeping the state
    numpy-native lets `env/`, `shield/`, and `metrics/` run without torch
    (lazy-import discipline).

    Attributes
    ----------
    t : int
        Epoch counter; increments by 1 per `step` call.
    agents : list[AgentState]
        Per-region node state.
    interfaces : dict[(int, int), InterfaceState]
        Edge state keyed by canonical (min, max) pair.
    obstacles : list[Obstacle]
        Workspace obstacles.
    bounds : tuple of four floats
        Workspace bounding box (x0, y0, x1, y1).
    violations : dict[str, int]
        Per-category counters (geom, ker, cert, srv, team).
    response_times : list[float]
        Realized T_e = C_i(e) - r_e for every completed event. Used by
        `metrics.tail.tail_metrics` for empirical evaluation.
    rejections : int
        Count of events that failed admission at every eligible endpoint.
    """

    t: int
    agents: list[AgentState]
    interfaces: dict[tuple[int, int], InterfaceState]
    obstacles: list[Obstacle]
    bounds: tuple[float, float, float, float]
    violations: dict[str, int] = field(default_factory=lambda: dict(
        geom=0, ker=0, cert=0, srv=0, team=0,
    ))
    response_times: list[float] = field(default_factory=list)
    rejections: int = 0
    dynamic_service_regions: bool = False

    # Per-epoch state for the rejection-aware reward r^max (the paper, Eq. 27
    # and CPAC auxiliary reward path), populated by step() and read by the reward extensions.
    # Instantaneous (per-epoch), distinct from the cumulative
    # rejected_by_agent; inert unless the reward weights are non-zero.
    generated_this_epoch_by_agent: dict = field(default_factory=dict)   # G_i(t)
    rejected_this_epoch_by_agent: dict = field(default_factory=dict)    # R_i(t)
    cross_admitted_this_epoch_by_pair: dict = field(default_factory=dict)  # owner-cross dispatches
    # zeta_i(t) = R_i / (G_i + eps): per-agent rejection share.
    rejection_share_this_epoch: dict = field(default_factory=dict)
    # lambda_i^cert(t) = nu / (U_i + tau_s): certificate-induced intensity.
    lambda_cert_per_agent: dict = field(default_factory=dict)
    # Edge-local rejection-rate components: per-agent, per-neighbor
    # generated/rejected counts -> zeta_{i|ij}.
    gen_by_agent_neighbor: dict = field(default_factory=dict)
    rej_by_agent_neighbor: dict = field(default_factory=dict)
    edge_local_rejection_share: dict = field(default_factory=dict)  # zeta_{i|ij}

    def neighbors(self, i: int) -> list[int]:
        """Return the neighbor-set N(i) in the current region-adjacency graph."""
        out: list[int] = []
        for (a, b) in self.interfaces.keys():
            if a == i:
                out.append(b)
            elif b == i:
                out.append(a)
        return out


def build_agent_eligibility_region(
    s: CSGRAGState,
    i: int,
    *,
    dynamic_service_regions: bool | None = None,
    delta_overrides: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> Rect | EligibilityRegion:
    r"""Construct the executable service region E_i(t) from state.

    Ownership remains fixed at `AgentState.region == R_i^0`. When dynamic
    service regions are enabled, agent i may additionally serve the
    neighbor-side directed strips adjacent to every active interface touching
    i. Those borrowed strips define the extra certified service surface.
    """
    if dynamic_service_regions is None:
        dynamic_service_regions = bool(getattr(s, "dynamic_service_regions", False))
    owner_rect = s.agents[i].region
    if not dynamic_service_regions:
        return owner_rect

    borrowed: list[Rect] = []
    for key, ifs in s.interfaces.items():
        a, b = key
        if i not in key:
            continue
        override = delta_overrides.get(key) if delta_overrides is not None else None
        if override is None:
            rect_ij = ifs.band_ij.band_rect()
            rect_ji = ifs.band_ji.band_rect()
        else:
            rect_ij = ifs.band_ij.band_rect(delta=override[0])
            rect_ji = ifs.band_ji.band_rect(delta=override[1])
        if i == a:
            if not rect_ji.is_degenerate:
                borrowed.append(rect_ji)
        else:
            if not rect_ij.is_degenerate:
                borrowed.append(rect_ij)
    return EligibilityRegion(owner_rect=owner_rect, borrowed_strips=tuple(borrowed))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CSGRAGEnv:
    r"""Discrete-epoch CSG-RAG environment.

    Parameters
    ----------
    regions : list[Rect]
        Initial ownership partition {R_i(0)}. Rectangular to match OA-BAR
        output; the env code is geometry-agnostic once `geometry.Region`
        protocol is in place.
    kernels : list[Rect]
        Invariant certified kernels K_i; must satisfy K_i subset R_i.
    U_bar : list[float]
        Per-agent allowable response budget. Typically (1 + rho) * ULSP(R_i(0))
        with rho around 0.3-0.5.
    arrivals : ArrivalProcess
        Stochastic spatial-temporal event generator. See `env.arrivals`.
    obstacles : list[Obstacle] | None
        Workspace obstacles (axis-aligned rectangles).
    bounds : tuple[float, float, float, float] | None
        Workspace bounding box (x0, y0, x1, y1). If None, computed as the
        bounding box of the regions.
    dt : float
        Continuous-time duration of one epoch. Default 1.0 (unit-time
        epochs; arrival rates are specified per unit time).
    speed : float
        Mobile-unit traversal speed (for travel_time). Default 1.0.
    delta_star_default : float
        Default certified cap delta^star_{ij} used for every interface.
        Can be set per-interface from OA-BAR clearance where available.
    delta_step : float
        Delta_delta, the per-step slack increment in the directed-slack
        update (Eq. 3) of the paper.
    rng : np.random.Generator | None
        Master rng for env-side stochasticity (not arrival-side).
    horizon : int
        Maximum epoch index; `info["done"] = True` when t reaches it.
    """

    def __init__(
        self,
        regions: list[Rect],
        kernels: list[Rect],
        U_bar: list[float],
        arrivals: ArrivalProcess,
        obstacles: list[Obstacle] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        dt: float = 1.0,
        speed: float = 1.0,
        delta_star_default: float = 1.0,
        delta_step: float = 0.1,
        rng: np.random.Generator | None = None,
        horizon: int = 200,
        # Optional SI-unit anchor for reporting.
        domain: "Domain | None" = None,
        # Equal head-start for all policies: delta_ij(0) = beta * delta*_ij.
        # Default 0.0 (zero initial slack); the shield keeps beta in [0, 1].
        initial_delta_fraction: float = 0.0,
        # Spatial-load weight on W_i/U_bar; 0.0 = the paper's load exactly.
        gamma_w_load: float = 0.0,
        # Service-surface mode: False = static owner rectangle; True = the
        # dynamic eligibility region E_i(t) = R_i^0 union borrowed strips.
        dynamic_service_regions: bool = False,
        # Rejection-aware reward r^max params (the paper, Eq. 27 and
        # CPAC auxiliary reward path); inert unless the reward weights are non-zero.
        nu_load_factor: float = 0.85,            # nu in lambda_i^cert = nu/(U_i+tau_s)
        tau_s_nominal: float = 1.0,              # tau_s nominal service time
        aux_epsilon: float = 1.0e-6,       # eps in zeta_i = R_i/(G_i+eps)
    ) -> None:
        assert len(regions) == len(kernels) == len(U_bar), \
            "regions, kernels, U_bar must have the same length (one entry per agent)."
        assert 0.0 <= initial_delta_fraction <= 1.0, \
            "initial_delta_fraction must lie in [0, 1] (fraction of delta*_ij)."
        self.n = len(regions)
        self.regions0 = list(regions)
        self.kernels = list(kernels)
        self.U_bar = list(U_bar)
        self.arrivals = arrivals
        self.obstacles = list(obstacles or [])
        if bounds is None:
            bounds = (
                min(r.x0 for r in regions),
                min(r.y0 for r in regions),
                max(r.x1 for r in regions),
                max(r.y1 for r in regions),
            )
        self.bounds = bounds
        self.dt = dt
        self.speed = speed
        self.delta_star_default = delta_star_default
        self.delta_step = delta_step
        self.rng = rng or np.random.default_rng(0)
        self.horizon = horizon
        self._state: Optional[CSGRAGState] = None
        self.domain = domain  # SI-anchor for reporting; optional.
        self.initial_delta_fraction = float(initial_delta_fraction)
        self.gamma_w_load = float(gamma_w_load)
        self.dynamic_service_regions = bool(dynamic_service_regions)
        # Auxiliary-reward parameters (load and rejection signals).
        self.nu_load_factor = float(nu_load_factor)
        self.tau_s_nominal = float(tau_s_nominal)
        self.aux_epsilon = float(aux_epsilon)
        # Regime A's adversarial arrivals query env state via set_env;
        # other arrival classes do not implement it.
        if hasattr(self.arrivals, "set_env"):
            self.arrivals.set_env(self)

    # ---------- construction ----------

    def _build_interfaces(
        self, regions: list[Rect],
    ) -> dict[tuple[int, int], InterfaceState]:
        r"""Discover shared interfaces Gamma_ij and instantiate their bands.

        For every unordered pair (i, j) with i < j, check whether R_i and
        R_j share a 1-D boundary segment (via `Rect.shares_interface_with`).
        If so, build two directed `InterfaceBand` objects (one owned by
        each endpoint) with initial delta = 0.

        The `direction` attribute of each band encodes which side of the
        interface the owner region sits on, so band expansion grows INTO
        the owner's territory away from the neighbor.
        """
        ifs: dict[tuple[int, int], InterfaceState] = {}
        n = len(regions)
        for i in range(n):
            for j in range(i + 1, n):
                axis_tag = regions[i].shares_interface_with(regions[j])
                if axis_tag is None:
                    continue
                if axis_tag in ("x_right", "x_left"):
                    # Vertical interface at x = x_iface
                    if axis_tag == "x_right":
                        x_iface = regions[i].x1
                        dir_i, dir_j = -1, +1
                    else:  # x_left
                        x_iface = regions[i].x0
                        dir_i, dir_j = +1, -1
                    lo = max(regions[i].y0, regions[j].y0)
                    hi = min(regions[i].y1, regions[j].y1)
                    band_ij = InterfaceBand(
                        owner=i, neighbor=j, axis="vertical",
                        interface_coord=x_iface, lo=lo, hi=hi,
                        delta=0.0, delta_star=self.delta_star_default,
                        direction=dir_i,
                    )
                    band_ji = InterfaceBand(
                        owner=j, neighbor=i, axis="vertical",
                        interface_coord=x_iface, lo=lo, hi=hi,
                        delta=0.0, delta_star=self.delta_star_default,
                        direction=dir_j,
                    )
                else:
                    # Horizontal interface at y = y_iface
                    if axis_tag == "y_top":
                        y_iface = regions[i].y1
                        dir_i, dir_j = -1, +1
                    else:  # y_bottom
                        y_iface = regions[i].y0
                        dir_i, dir_j = +1, -1
                    lo = max(regions[i].x0, regions[j].x0)
                    hi = min(regions[i].x1, regions[j].x1)
                    band_ij = InterfaceBand(
                        owner=i, neighbor=j, axis="horizontal",
                        interface_coord=y_iface, lo=lo, hi=hi,
                        delta=0.0, delta_star=self.delta_star_default,
                        direction=dir_i,
                    )
                    band_ji = InterfaceBand(
                        owner=j, neighbor=i, axis="horizontal",
                        interface_coord=y_iface, lo=lo, hi=hi,
                        delta=0.0, delta_star=self.delta_star_default,
                        direction=dir_j,
                    )
                ifs[(i, j)] = InterfaceState(
                    i=i, j=j, band_ij=band_ij, band_ji=band_ji,
                    delta_star=self.delta_star_default,
                )
        return ifs

    def reset(self, seed: int | None = None) -> CSGRAGState:
        r"""Reset to epoch 0 with zero slacks and empty queues.

        Re-seeds both the env rng and the arrival rng for reproducibility.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.arrivals.rng = np.random.default_rng(seed + 1)
        # Deterministic arrival sources (e.g.
        # historical_replay) hold a monotone cursor over a fixed event
        # list. Without rewinding the cursor, post-reset rollouts on
        # the same env see zero arrivals and training metrics collapse.
        if hasattr(self.arrivals, "reset"):
            try:
                self.arrivals.reset()
            except Exception:
                pass
        agents: list[AgentState] = []
        for i in range(self.n):
            r = self.regions0[i]
            pos = ((r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0)  # centroid
            # ULSP is a length; divide by speed to compare against the
            # time-budget U_bar.
            U0 = ulsp_bound(r, self._obstacles_in(r, t=0.0)) / max(self.speed, 1e-9)
            agents.append(AgentState(
                idx=i, position=pos, region=r, kernel=self.kernels[i],
                U_bar=self.U_bar[i], U_current=U0, eligibility_diag=r.diag, queue=[],
                latent_overload_ewma=0.0, load_pressure_ewma=0.0,
            ))
        interfaces = self._build_interfaces([a.region for a in agents])

        if self.initial_delta_fraction > 0.0:
            init_delta = self.initial_delta_fraction * self.delta_star_default
            for ifs in interfaces.values():
                ifs.band_ij.delta = init_delta
                ifs.band_ji.delta = init_delta
        self._state = CSGRAGState(
            t=0, agents=agents, interfaces=interfaces,
            obstacles=list(self.obstacles), bounds=self.bounds,
            dynamic_service_regions=self.dynamic_service_regions,
        )
        self._refresh_certificates(t_cont=0.0)

        _init_viols = {
            k: v for k, v in self._state.violations.items()
            if v > 0 and k in ("geom", "ker", "cert")
        }
        if _init_viols:
            raise RuntimeError(
                f"reset() produced an infeasible initial state at t=0; "
                f"violations: {_init_viols}. Likely cause: "
                f"initial_delta_fraction={self.initial_delta_fraction} "
                f"is too large for delta_star_default="
                f"{self.delta_star_default} or U_bar={self.U_bar} is "
                f"insufficient for the initial geometry."
            )
        return self._state

    @property
    def state(self) -> CSGRAGState:
        assert self._state is not None, "Call env.reset() first."
        return self._state

    # ---------- helpers ----------

    def _current_obstacles(self, t: float) -> list[Obstacle]:
        """The (static) workspace obstacle set."""
        return list(self.obstacles)

    def _obstacles_in(self, region: Rect, t: float | None = None) -> list[Obstacle]:
        """Subset of obstacles that intersect `region` (for ULSP scoping)."""
        candidates = list(self.obstacles)
        out = []
        for o in candidates:
            ix0 = max(o.rect.x0, region.x0)
            iy0 = max(o.rect.y0, region.y0)
            ix1 = min(o.rect.x1, region.x1)
            iy1 = min(o.rect.y1, region.y1)
            if ix1 > ix0 and iy1 > iy0:
                out.append(o)
        return out

    def _obstacles_in_region(
        self,
        region: Rect | EligibilityRegion,
        t: float | None = None,
    ) -> list[Obstacle]:
        """Obstacle subset for either a Rect or an EligibilityRegion."""
        if isinstance(region, Rect):
            return self._obstacles_in(region, t=t)
        candidates = list(self.obstacles)
        out: list[Obstacle] = []
        for o in candidates:
            for r in region.member_rects:
                ix0 = max(o.rect.x0, r.x0)
                iy0 = max(o.rect.y0, r.y0)
                ix1 = min(o.rect.x1, r.x1)
                iy1 = min(o.rect.y1, r.y1)
                if ix1 > ix0 and iy1 > iy0:
                    out.append(o)
                    break
        return out

    def _eligibility_region(
        self,
        i: int,
        delta_overrides: dict[tuple[int, int], tuple[float, float]] | None = None,
    ) -> Rect | EligibilityRegion:
        return build_agent_eligibility_region(
            self.state,
            i,
            dynamic_service_regions=self.dynamic_service_regions,
            delta_overrides=delta_overrides,
        )

    def _refresh_certificates(self, t_cont: float) -> None:
        """Recompute U_current and dynamic workload diameter on the live service surface."""
        s = self.state
        for a in s.agents:
            region = self._eligibility_region(a.idx)
            obs_here = self._obstacles_in_region(region, t=t_cont)
            a.U_current = ulsp_bound_region(region, obs_here) / max(self.speed, 1e-9)
            a.eligibility_diag = region.diag
            if a.U_current > a.U_bar + GEOM_EPS:
                s.violations["cert"] += 1
            if not a.region.contains_region(a.kernel):
                s.violations["ker"] += 1

    def _owner_of(self, x: float, y: float) -> int | None:
        """Which agent's region geometrically contains (x, y)? None if outside."""
        for a in self.state.agents:
            if a.region.contains((x, y)):
                return a.idx
        return None

    def _buffer_pair(self, x: float, y: float) -> tuple[int, int] | None:
        """If (x, y) lies in any interface band B_ij(t), return the pair (i, j).

        Checks both directed sides of each interface. Returns None if the
        point is not in any buffer band (either private interior or
        outside the workspace).
        """
        for key, ifs in self.state.interfaces.items():
            if ifs.band_ij.contains((x, y)) or ifs.band_ji.contains((x, y)):
                return key
        return None

    # ---------- core step ----------

    def step(
        self,
        edge_actions: dict[tuple[int, int], int],
    ) -> tuple[CSGRAGState, dict]:
        r"""Advance the environment by one epoch.

        Parameters
        ----------
        edge_actions : dict
            Mapping from canonical unordered key (i, j) with i < j to
            signed action in {-1, 0, +1}. The convention (the paper, Sec. 3.3
            and Eq. 3, delta_ij(t+1) = Pi[delta_ij - a_ij * Delta_delta]):
            a_ij = +1 expands agent i into R_j ("i expands into j"; delta_ij
            DECREASES while delta_ji INCREASES), a_ij = -1 expands j into R_i,
            a_ij = 0 holds. The antisymmetric inverse action a_ji = -a_ij
            (Eq. 4) on (j, i) is applied automatically by the update rule.

        Returns
        -------
        (state, info) : tuple[CSGRAGState, dict]
            `state` is the same CSGRAGState object, mutated in place.
            `info` contains per-step diagnostics:
                admitted  : int, events admitted this epoch.
                rejected  : int, events that failed admission at all endpoints.
                completed : int, events served this epoch.
                done      : bool, True iff t has reached horizon.

        Diagnostics
        -----------
        After `step`, the violation counters record rejected or exceeded
        conditions and response_times records empirical completed-event
        response times.
        """
        s = self.state
        info: dict = dict(admitted=0, rejected=0, completed=0)
        # Reset per-epoch auxiliary-reward fields.

        s.generated_this_epoch_by_agent = {k: 0 for k in range(self.n)}
        s.rejected_this_epoch_by_agent = {k: 0 for k in range(self.n)}
        s.cross_admitted_this_epoch_by_pair = {}
        # Edge-local rejection: per-agent, per-neighbor.
        s.gen_by_agent_neighbor = {k: {} for k in range(self.n)}
        s.rej_by_agent_neighbor = {k: {} for k in range(self.n)}
        # Per-edge collaboration attribution. cross_admitted_by_edge counts
        # only true owner-cross service: a buffer-eligible event
        # geometrically owned by one endpoint and committed to the other.
        info["cross_admitted_by_edge"] = {}
        info["owner_cross_admitted_by_edge"] = info["cross_admitted_by_edge"]
        info["buffer_admitted_by_edge"] = {}
        info["same_owner_buffer_admitted_by_edge"] = {}
        info["buffer_rejected_by_edge"] = {}

        def _inc_info_dict(name: str, key: tuple[int, int], amount: int = 1) -> None:
            d = info.setdefault(name, {})
            d[key] = d.get(key, 0) + amount

        t_cont = float(s.t) * self.dt
        new_events = self.arrivals.step(t_cont, self.dt)
        latent_counts = [0 for _ in range(self.n)]

        # --- 1. Arrivals + admission ---------------------------------------
        for ev in new_events:
            pair = self._buffer_pair(ev.x, ev.y)
            if pair is not None:
                # Buffer-eligible: dispatch to better-admitting endpoint.
                i, j = pair
                edge = (min(i, j), max(i, j))
                owner_of_ev = self._owner_of(ev.x, ev.y)
                # Auxiliary-reward stats: event incident to both pair endpoints.
                s.generated_this_epoch_by_agent[i] = s.generated_this_epoch_by_agent.get(i, 0) + 1
                s.generated_this_epoch_by_agent[j] = s.generated_this_epoch_by_agent.get(j, 0) + 1
                # Edge-local: buffer event at (i,j) counts for the (i,j)
                # bucket at BOTH endpoints.
                s.gen_by_agent_neighbor.setdefault(i, {})
                s.gen_by_agent_neighbor.setdefault(j, {})
                s.gen_by_agent_neighbor[i][j] = s.gen_by_agent_neighbor[i].get(j, 0) + 1
                s.gen_by_agent_neighbor[j][i] = s.gen_by_agent_neighbor[j].get(i, 0) + 1
                chosen = interface_dispatch(
                    s.agents[i], s.agents[j], ev, t_cont, self.speed,
                )
                if chosen is None:
                    s.rejections += 1
                    info["rejected"] += 1
                    _inc_info_dict("buffer_rejected_by_edge", edge)
                    latent_counts[i] += 1
                    latent_counts[j] += 1
                    # Per-region rejection tracking: attribute the
                    # rejection to the GEOMETRIC OWNER at event location.
                    # For buffer-eligible events, geometric owner is the
                    # endpoint whose region physically contains x_e.
                    if not hasattr(s, "rejected_by_agent"):
                        s.rejected_by_agent = {k: 0 for k in range(len(s.agents))}
                    if owner_of_ev is not None:
                        s.rejected_by_agent[owner_of_ev] = s.rejected_by_agent.get(owner_of_ev, 0) + 1
                        # Auxiliary-reward stats: per-epoch rejection attribution.
                        s.rejected_this_epoch_by_agent[owner_of_ev] = (
                            s.rejected_this_epoch_by_agent.get(owner_of_ev, 0) + 1)
                        # Edge-local: buffer event rejected at (i,j) counts
                        # for the (i,j) bucket at BOTH endpoints i and j.
                        s.rej_by_agent_neighbor.setdefault(i, {})
                        s.rej_by_agent_neighbor.setdefault(j, {})
                        s.rej_by_agent_neighbor[i][j] = s.rej_by_agent_neighbor[i].get(j, 0) + 1
                        s.rej_by_agent_neighbor[j][i] = s.rej_by_agent_neighbor[j].get(i, 0) + 1
                else:
                    chosen.queue.append(ev)
                    info["admitted"] += 1
                    _inc_info_dict("buffer_admitted_by_edge", edge)
                    if owner_of_ev is not None and chosen.idx != owner_of_ev:
                        _inc_info_dict("cross_admitted_by_edge", edge)
                        # Auxiliary-reward stats: true-cross per-epoch tracking.
                        s.cross_admitted_this_epoch_by_pair[edge] = (
                            s.cross_admitted_this_epoch_by_pair.get(edge, 0) + 1)
                    else:
                        _inc_info_dict("same_owner_buffer_admitted_by_edge", edge)
                continue
            # Private / kernel: only the geometric owner is eligible.
            owner = self._owner_of(ev.x, ev.y)
            if owner is None:
                # Event outside workspace partition; should not happen if
                # the arrival bounds match the region union. Reject.
                s.rejections += 1
                info["rejected"] += 1
                continue
            a = s.agents[owner]
            # Auxiliary-reward stats: per-epoch generated count for the geometric owner.
            s.generated_this_epoch_by_agent[owner] = s.generated_this_epoch_by_agent.get(owner, 0) + 1
            # Edge-local: private event at owner counts for EVERY edge
            # incident to owner (the event is affected by all of owner's
            # boundary actions via the capacity envelope).
            s.gen_by_agent_neighbor.setdefault(owner, {})
            for _nbr in s.neighbors(owner):
                s.gen_by_agent_neighbor[owner][_nbr] = (
                    s.gen_by_agent_neighbor[owner].get(_nbr, 0) + 1)
            # Certificate-aware admission (the paper, Eq. 11).
            if admit(a, ev, t_cont, self.speed):
                a.queue.append(ev)
                info["admitted"] += 1
            else:
                s.rejections += 1
                info["rejected"] += 1
                latent_counts[owner] += 1
                if not hasattr(s, "rejected_by_agent"):
                    s.rejected_by_agent = {k: 0 for k in range(len(s.agents))}
                s.rejected_by_agent[owner] = s.rejected_by_agent.get(owner, 0) + 1
                # Auxiliary-reward stats: per-epoch rejection attribution.
                s.rejected_this_epoch_by_agent[owner] = (
                    s.rejected_this_epoch_by_agent.get(owner, 0) + 1)
                # Edge-local admit path: private rejection.
                s.rej_by_agent_neighbor.setdefault(owner, {})
                for _nbr in s.neighbors(owner):
                    s.rej_by_agent_neighbor[owner][_nbr] = (
                        s.rej_by_agent_neighbor[owner].get(_nbr, 0) + 1)

        # --- 2. Service (one simulated step per agent) ---------------------
        for a in s.agents:
            e = dispatch_local(a, t_cont, self.speed)
            if e is None:
                continue
            tau = travel_time(a.position, (e.x, e.y), self.speed)
            # Service model: an admitted event completes within the epoch.
            # Accumulate per-agent Euclidean travel (operational-burden metric).
            if not hasattr(s, "travel_distance_by_agent"):
                s.travel_distance_by_agent = {k: 0.0 for k in range(len(s.agents))}
            prev_pos = a.position
            this_travel = ((prev_pos[0] - e.x) ** 2 + (prev_pos[1] - e.y) ** 2) ** 0.5
            s.travel_distance_by_agent[a.idx] = (
                s.travel_distance_by_agent.get(a.idx, 0.0) + this_travel
            )

            a.position = (e.x, e.y)
            completion_time = t_cont + tau + e.service_time
            realized_T = completion_time - e.occurrence_time
            s.response_times.append(realized_T)
            # Per-region response times (keyed by serving agent) for the
            # worst-region and std metrics.
            if not hasattr(s, "response_times_by_agent"):
                s.response_times_by_agent = {k: [] for k in range(len(s.agents))}
            s.response_times_by_agent.setdefault(a.idx, []).append(realized_T)
            try:
                a.queue.remove(e)
            except ValueError:
                pass
            info["completed"] += 1
            # Empirical fixed-budget exceedance diagnostic.
            if realized_T > a.U_bar + GEOM_EPS:
                s.violations["srv"] += 1

        # --- 3. Apply edge actions (shield-double-checked) -----------------
        # The env re-verifies every action against the shield, so a buggy
        # policy cannot corrupt the certified-state invariants.
        from certified_marl.shield.feasibility_kernel import check_edge_action_safe

        # Apply-time shield uses the same obstacle set as the downstream
        # certificate recomputation, so the two checks agree.
        _t_now = float(s.t) * self.dt
        _active_obstacles = self._current_obstacles(_t_now)
        for key, a_ij in edge_actions.items():
            if a_ij == 0:
                continue
            ifs = s.interfaces[key]
            ok, reason = check_edge_action_safe(
                s=s, interface=ifs, a_ij=a_ij,
                delta_step=self.delta_step, obstacles=_active_obstacles,
                speed=self.speed,
            )
            if not ok:
                # Should never fire under correct shield masking; a hit means
                # a bug in the masked softmax or action canonicalization. Map
                # any cert-clause reason to the aggregate `cert` counter.
                if reason == "cert" or reason.startswith("cert_"):
                    s.violations["cert"] = s.violations.get("cert", 0) + 1
                else:
                    counter_key = reason if reason in s.violations else "geom"
                    s.violations[counter_key] = s.violations.get(counter_key, 0) + 1
                continue
            # Apply Eq. 3 update with snap-to-exact on tiny residues
            # (floating-point residue guard).
            new_delta_ij = float(np.clip(
                ifs.band_ij.delta - a_ij * self.delta_step,
                0.0, ifs.delta_star,
            ))
            new_delta_ji = float(np.clip(
                ifs.band_ji.delta - (-a_ij) * self.delta_step,
                0.0, ifs.delta_star,
            ))
            if abs(new_delta_ij) < GEOM_EPS:
                new_delta_ij = 0.0
            elif abs(new_delta_ij - ifs.delta_star) < GEOM_EPS:
                new_delta_ij = ifs.delta_star
            if abs(new_delta_ji) < GEOM_EPS:
                new_delta_ji = 0.0
            elif abs(new_delta_ji - ifs.delta_star) < GEOM_EPS:
                new_delta_ji = ifs.delta_star
            ifs.band_ij.delta = new_delta_ij
            ifs.band_ji.delta = new_delta_ji

        # --- 4. Recompute certificates U_i(t+1) ----------------------------
        # U_i(t+1) is recomputed on the post-update geometry (the boundary
        # bands may have changed) against the static obstacle set.
        #
        # UNITS: ulsp_bound returns a LENGTH (geometric diameter + obstacle
        # perimeter contributions). U_bar is a TIME budget (world-time
        # units). The certificate comparison must be apples-to-apples, so
        # we divide the raw ULSP length by self.speed (world-length per
        # world-time) to obtain the time-to-traverse certificate that
        # U_bar dominates. This division matters whenever speed != 1.0
        # (e.g. the Chicago config); at speed == 1.0 the two coincide.
        t_next = float(s.t + 1) * self.dt
        self._refresh_certificates(t_cont=t_next)

        # --- 5. Update load signals (Eqs. 14-16 of the paper) -------------
        # gamma_w (default 0) matches the paper's load exactly; gamma_w > 0
        # adds the W_i/U_bar utilization fraction to the load pressure.
        gamma_w = float(getattr(self, "gamma_w_load", 0.0))
        for a in s.agents:
            update_load_signals(a, latent_counts[a.idx], t_cont,
                                gamma_w=gamma_w, speed=self.speed)

        # --- 5b. Auxiliary-reward statistics ------------
        # Compute zeta_i(t) = R_i(t) / (G_i(t) + epsilon) and
        # lambda_i^cert(t) = nu / (U_i + tau_s). Both use the post-step certificates
        # already refreshed in step 4. Defaults are inert: the reward
        # consumer must opt in via positive weights to alter training.
        eps_aux = float(self.aux_epsilon)
        nu_aux = float(self.nu_load_factor)
        tau_s_aux = float(self.tau_s_nominal)
        s.rejection_share_this_epoch = {
            k: (float(s.rejected_this_epoch_by_agent.get(k, 0)) /
                (float(s.generated_this_epoch_by_agent.get(k, 0)) + eps_aux))
            for k in range(self.n)
        }
        s.lambda_cert_per_agent = {
            k: nu_aux / (max(float(s.agents[k].U_current), eps_aux) + tau_s_aux)
            for k in range(self.n)
        }
        # Edge-local zeta finalization:
        # zeta_{i|ij} = R_{i|ij} / (G_{i|ij} + eps) per ordered (i, j).
        s.edge_local_rejection_share = {}
        for i_agent in range(self.n):
            gen_by_n = s.gen_by_agent_neighbor.get(i_agent, {})
            rej_by_n = s.rej_by_agent_neighbor.get(i_agent, {})
            for j_neighbor in gen_by_n.keys() | rej_by_n.keys():
                _g = float(gen_by_n.get(j_neighbor, 0))
                _r = float(rej_by_n.get(j_neighbor, 0))
                s.edge_local_rejection_share[(i_agent, j_neighbor)] = (
                    _r / (_g + eps_aux)
                )

        # --- 6. Advance clock ----------------------------------------------
        s.t += 1
        info["done"] = s.t >= self.horizon
        return s, info
