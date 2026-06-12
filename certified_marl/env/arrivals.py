r"""Event arrival processes for the CSG-RAG environment.

Provides the spatial arrival generators used by the paper's regimes
(Section 7):

    uniform              UniformPoisson            -- U  (uniform Poisson)
    boundary_hotspot     BoundaryHotspot           -- S1 (single Gaussian hotspot)
    multi_hotspot        MultiHotspot              -- SB (corner hotspots)
    shifting_hotspot     ShiftingHotspot           -- N  (translating hotspot)
    boundary_adversarial BoundaryAdversarialArrival-- A  (adaptive stress)

Regime R (real Chicago trace) uses `HistoricalReplay` in
`env.historical_arrivals`, dispatched via `make_arrival(kind="historical_replay")`.

Each generator emits Event(x, y, occurrence_time, service_time). Each
ArrivalProcess owns its own numpy Generator, so the workload stream is
seeded independently of the env's tie-break/decision RNG.

References
----------
 - The paper, Section 7.
 - Bertsimas & van Ryzin 1991. Stochastic and dynamic vehicle routing.
   Operations Research.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Event:
    """A single service request.

    Attributes
    ----------
    x, y           : spatial location (R^2).
    occurrence_time : r_e, the continuous-time stamp at which the event occurs.
    service_time   : s(e) >= 0, on-site processing time.
    """

    x: float
    y: float
    occurrence_time: float
    service_time: float = 0.0


# ---------------------------------------------------------------------------
# Base class and concrete arrival regimes
# ---------------------------------------------------------------------------


class ArrivalProcess:
    """Abstract arrival process; subclasses implement `step(t, dt)`.

    The `step` method advances the process from time t to t + dt and
    returns the list of Events born in that interval. Time is continuous
    but realized on the discrete-epoch clock of the env.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def step(self, t: float, dt: float) -> list[Event]:
        raise NotImplementedError


class UniformPoisson(ArrivalProcess):
    r"""Spatially-uniform homogeneous Poisson process.

    N(t + dt) - N(t) ~ Poisson(lambda * dt). Each event's location is
    drawn independently and uniformly over the bounding rectangle of the
    workspace.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        rate: float,
        service_time: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng or np.random.default_rng(0))
        self.bounds = bounds
        self.rate = rate
        self.service_time = service_time

    def step(self, t: float, dt: float) -> list[Event]:
        k = self.rng.poisson(self.rate * dt)
        x0, y0, x1, y1 = self.bounds
        out: list[Event] = []
        for _ in range(k):
            x = self.rng.uniform(x0, x1)
            y = self.rng.uniform(y0, y1)
            out.append(Event(
                x, y,
                occurrence_time=t + self.rng.uniform(0, dt),
                service_time=self.service_time,
            ))
        return out


class BoundaryHotspot(ArrivalProcess):
    r"""Poisson arrivals with locations from a Gaussian blob at `center`.

    Intended to model a single boundary hotspot that straddles the shared
    interface of two neighboring regions. Events outside the workspace are
    clipped to the bounding box.
    """

    def __init__(
        self,
        center: tuple[float, float],
        sigma: float,
        rate: float,
        bounds: tuple[float, float, float, float],
        service_time: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng or np.random.default_rng(0))
        self.center = center
        self.sigma = sigma
        self.rate = rate
        self.bounds = bounds
        self.service_time = service_time

    def step(self, t: float, dt: float) -> list[Event]:
        k = self.rng.poisson(self.rate * dt)
        cx, cy = self.center
        x0, y0, x1, y1 = self.bounds
        out: list[Event] = []
        for _ in range(k):
            x = np.clip(self.rng.normal(cx, self.sigma), x0, x1)
            y = np.clip(self.rng.normal(cy, self.sigma), y0, y1)
            out.append(Event(
                float(x), float(y),
                occurrence_time=t + self.rng.uniform(0, dt),
                service_time=self.service_time,
            ))
        return out


class ShiftingHotspot(ArrivalProcess):
    r"""Gaussian blob whose center translates along a line segment over time.

    Center trajectory: triangle wave between `start` and `end` with period
    `period`. Explicitly:

        alpha(t) = 2 * phase(t)        if phase < 0.5
                 = 2 * (1 - phase(t))  otherwise
        center(t) = start + alpha(t) * (end - start)

    where phase(t) = (t / period) mod 1. The hotspot is non-stationary,
    so a fixed band width cannot track it (regime N).
    """

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        period: float,
        sigma: float,
        rate: float,
        bounds: tuple[float, float, float, float],
        service_time: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng or np.random.default_rng(0))
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.period = period
        self.sigma = sigma
        self.rate = rate
        self.bounds = bounds
        self.service_time = service_time

    def _center_at(self, t: float) -> tuple[float, float]:
        phase = (t / self.period) % 1.0
        alpha = 2 * phase if phase < 0.5 else 2 * (1 - phase)
        c = self.start + alpha * (self.end - self.start)
        return float(c[0]), float(c[1])

    def step(self, t: float, dt: float) -> list[Event]:
        k = self.rng.poisson(self.rate * dt)
        cx, cy = self._center_at(t)
        x0, y0, x1, y1 = self.bounds
        out: list[Event] = []
        for _ in range(k):
            x = np.clip(self.rng.normal(cx, self.sigma), x0, x1)
            y = np.clip(self.rng.normal(cy, self.sigma), y0, y1)
            out.append(Event(
                float(x), float(y),
                occurrence_time=t + self.rng.uniform(0, dt),
                service_time=self.service_time,
            ))
        return out


class MultiHotspot(ArrivalProcess):
    r"""Multi-hotspot arrival process.

    Formal semantics:

        lambda_hot    = q * Lambda / k + (1 - q) * Lambda / N
        lambda_normal =                   (1 - q) * Lambda / N

    Implementation: at every step, draw the total event count
    k_total ~ Poisson(rate * dt), then assign each event to
    a region by the mass distribution (hot vs normal). Within
    a region, the event location is a Gaussian blob at the region
    center with given sigma.

    Parameters
    ----------
    rate : float
        Total Lambda (events/epoch) across the whole workspace.
    hot_centers : list of (x, y)
        Centroids of the k_hot hot regions. k = len(hot_centers).
    q : float in [0, 1]
        Concentration of mass on hot regions (q).
        q=1.0 means all mass on hot regions; q=0.0 reduces to uniform.
    sigma : float
        Standard deviation of Gaussian blob per region.
    normal_centers : list of (x, y)
        Centroids of the (N - k_hot) normal regions.
    service_time : float
    bounds : 4-tuple
    rng : Generator
    """

    def __init__(
        self,
        rate: float,
        hot_centers: list,
        q: float,
        sigma: float,
        normal_centers: list,
        bounds: tuple[float, float, float, float],
        service_time: float = 0.0,
        rng: "np.random.Generator | None" = None,
    ):
        super().__init__(rng or np.random.default_rng(0))
        assert 0.0 <= q <= 1.0
        self.rate = rate
        self.hot_centers = list(hot_centers)
        self.q = float(q)
        self.sigma = float(sigma)
        self.normal_centers = list(normal_centers)
        self.bounds = bounds
        self.service_time = service_time

    def step(self, t: float, dt: float) -> list[Event]:
        k_total = self.rng.poisson(self.rate * dt)
        out: list[Event] = []
        k_hot = len(self.hot_centers)
        N = k_hot + len(self.normal_centers)
        # Per-region mass fractions:
        # hot mass fraction = q + (1-q) * (k_hot / N)
        # normal mass fraction = (1-q) * (N - k_hot) / N
        hot_mass = self.q + (1.0 - self.q) * (k_hot / max(N, 1))
        x0, y0, x1, y1 = self.bounds
        for _ in range(k_total):
            # Decide: hot region or normal region
            if self.rng.uniform() < hot_mass and k_hot > 0:
                idx = self.rng.integers(0, k_hot)
                cx, cy = self.hot_centers[idx]
            else:
                if not self.normal_centers:
                    # All regions are hot; fall through to first hot region
                    cx, cy = self.hot_centers[0]
                else:
                    idx = self.rng.integers(0, len(self.normal_centers))
                    cx, cy = self.normal_centers[idx]
            x = float(np.clip(self.rng.normal(cx, self.sigma), x0, x1))
            y = float(np.clip(self.rng.normal(cy, self.sigma), y0, y1))
            out.append(Event(
                x, y,
                occurrence_time=t + float(self.rng.uniform(0, dt)),
                service_time=self.service_time,
            ))
        return out


class BoundaryAdversarialArrival(ArrivalProcess):
    r"""Adversarial boundary-stress arrival process.

    At each decision epoch, identifies the active interface (i, j) in the
    region-adjacency graph with the largest absolute reduced-Markov-load
    mismatch
    \[
        (i^*, j^*) \in \arg\max_{(i,j) \in \mathcal{E}} |\ell_i(t) - \ell_j(t)|,
    \]
    and concentrates arrivals as a Gaussian blob near the centroid of the
    shared interface $\Gamma_{i^*j^*}$. This produces a moving overload
    pattern that targets the weakest current boundary at every epoch.

    The strategy queries the env state for the current $\ell_i(t)$ values
    via a back-reference set at construction time (see
    ``set_env``). When the env reference is unset (e.g., during arrival
    pre-instantiation in tests), the process falls back to spatially
    uniform arrivals at the same total rate.

    Parameters
    ----------
    rate : float
        Total arrival intensity per epoch.
    sigma : float
        Standard deviation of the Gaussian blob centred on the targeted
        interface.
    bounds : 4-tuple
        Workspace bounds for clipping.
    service_time : float
    rng : Generator
    """

    def __init__(
        self,
        rate: float,
        sigma: float,
        bounds: tuple[float, float, float, float],
        service_time: float = 0.0,
        rng: "np.random.Generator | None" = None,
    ):
        super().__init__(rng or np.random.default_rng(0))
        self.rate = float(rate)
        self.sigma = float(sigma)
        self.bounds = bounds
        self.service_time = float(service_time)
        self._env = None  # set later via set_env(env)

    def set_env(self, env) -> None:
        """Attach env reference for state queries."""
        self._env = env

    def _target_interface_center(self):
        """Return the (x, y) centroid of the interface with maximum
        $|\\ell_i - \\ell_j|$ in the current env state. Returns None if
        the env reference or its state has not been initialised."""
        env = self._env
        if env is None:
            return None
        state = getattr(env, "_state", None) or getattr(env, "state", None)
        if state is None or not hasattr(state, "agents") or not hasattr(state, "interfaces"):
            return None
        loads = [a.load_pressure_ewma for a in state.agents]
        best_key = None
        best_score = -1.0
        for key in state.interfaces.keys():
            i, j = key
            score = abs(loads[i] - loads[j])
            if score > best_score:
                best_score = score
                best_key = key
        if best_key is None:
            return None
        ifs = state.interfaces[best_key]
        # interface center: midpoint of its endpoints
        if hasattr(ifs, "p0") and hasattr(ifs, "p1"):
            cx = (float(ifs.p0[0]) + float(ifs.p1[0])) / 2
            cy = (float(ifs.p0[1]) + float(ifs.p1[1])) / 2
            return (cx, cy)
        # fallback: midpoint of the two regions' centers
        i, j = best_key
        ri = state.agents[i].region
        rj = state.agents[j].region
        cx_i = (ri.x0 + ri.x1) / 2
        cy_i = (ri.y0 + ri.y1) / 2
        cx_j = (rj.x0 + rj.x1) / 2
        cy_j = (rj.y0 + rj.y1) / 2
        return ((cx_i + cx_j) / 2, (cy_i + cy_j) / 2)

    def step(self, t: float, dt: float) -> list[Event]:
        target = self._target_interface_center()
        x0, y0, x1, y1 = self.bounds
        k_total = self.rng.poisson(self.rate * dt)
        out: list[Event] = []
        if target is None:
            # uniform fallback when state is not yet available
            for _ in range(k_total):
                x = float(self.rng.uniform(x0, x1))
                y = float(self.rng.uniform(y0, y1))
                out.append(Event(
                    x, y,
                    occurrence_time=t + float(self.rng.uniform(0, dt)),
                    service_time=self.service_time,
                ))
            return out
        cx, cy = target
        for _ in range(k_total):
            x = float(np.clip(self.rng.normal(cx, self.sigma), x0, x1))
            y = float(np.clip(self.rng.normal(cy, self.sigma), y0, y1))
            out.append(Event(
                x, y,
                occurrence_time=t + float(self.rng.uniform(0, dt)),
                service_time=self.service_time,
            ))
        return out


# ---------------------------------------------------------------------------
# Factory: instantiate an ArrivalProcess from a dict spec
# ---------------------------------------------------------------------------


def make_arrival(
    spec: dict,
    bounds: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> ArrivalProcess:
    r"""Build an ArrivalProcess from a YAML-compatible dict.

    Dispatches on `spec["kind"]`: "uniform", "boundary_hotspot",
    "shifting_hotspot", "multi_hotspot", "boundary_adversarial", or
    "historical_replay".

    Raises
    ------
    ValueError
        If `spec["kind"]` is not one of the supported kinds.
    """
    kind = spec["kind"]
    if kind == "uniform":
        return UniformPoisson(
            bounds=bounds,
            rate=spec["rate"],
            service_time=spec.get("service_time", 0.0),
            rng=rng,
        )
    if kind == "boundary_hotspot":
        return BoundaryHotspot(
            center=tuple(spec["center"]),
            sigma=spec["sigma"],
            rate=spec["rate"],
            bounds=bounds,
            service_time=spec.get("service_time", 0.0),
            rng=rng,
        )
    if kind == "shifting_hotspot":
        return ShiftingHotspot(
            start=tuple(spec["start"]),
            end=tuple(spec["end"]),
            period=spec["period"],
            sigma=spec["sigma"],
            rate=spec["rate"],
            bounds=bounds,
            service_time=spec.get("service_time", 0.0),
            rng=rng,
        )
    if kind == "multi_hotspot":
        return MultiHotspot(
            rate=spec["rate"],
            hot_centers=[tuple(c) for c in spec["hot_centers"]],
            q=spec["q"],
            sigma=spec["sigma"],
            normal_centers=[tuple(c) for c in spec["normal_centers"]],
            bounds=bounds,
            service_time=spec.get("service_time", 0.0),
            rng=rng,
        )
    if kind == "historical_replay":
        from certified_marl.env.historical_arrivals import HistoricalReplay
        return HistoricalReplay(
            json_path=spec["json_path"],
            bounds=bounds,
            day_offset_hours=spec.get("day_offset_hours", 0),
            duration_hours=spec.get("duration_hours", 24),
            service_time=spec.get("service_time", 0.0),
            urgent_only=spec.get("urgent_only", True),
            seed=spec.get("seed", 0),
            compress_to_epochs=spec.get("compress_to_epochs", None),
        )
    if kind == "boundary_adversarial":
        return BoundaryAdversarialArrival(
            rate=spec["rate"],
            sigma=spec["sigma"],
            bounds=bounds,
            service_time=spec.get("service_time", 0.0),
            rng=rng,
        )
    raise ValueError(f"Unknown arrival kind: {kind!r}")
