r"""Historical-data arrival process.

Replays a real event stream through the CSG-RAG environment. The reference
trace is the City of Chicago "Crimes - 2001 to Present" open dataset
(Chicago Data Portal). Events are loaded from
a JSON file with (date, latitude, longitude, primary_type) fields,
restricted to a set of response-time-sensitive incident types (URGENT_TYPES
below; our own filter, not an official CPD priority code), and
coordinate-transformed from (lat, lon) to the env's world-unit system via an
affine mapping. Each retained incident is treated as one service-request
event. (The data/chicago_911_week_*.json filenames are a misnomer: the
contents are crime-incident records, not 911-dispatch records.)

This is the arrival generator for Regime R, dispatched via
`make_arrival(kind="historical_replay")`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from certified_marl.env.arrivals import ArrivalProcess, Event


# Urgency filter on the Chicago dataset's `primary_type` field.
#
# Criterion: violent / imminent-harm incident types -- crimes against
# persons or active threats -- where dispatch latency materially affects
# victim safety or in-progress apprehension, so worst-case response time
# is the governing metric (the paper's WCRT setting). This is our own
# operational choice, NOT an official CPD priority code.
#
# Included (all valid `primary_type` values, present in the 2023 trace):
#   BATTERY, ASSAULT, ROBBERY, WEAPONS VIOLATION, HOMICIDE,
#   CRIMINAL SEXUAL ASSAULT, KIDNAPPING, ARSON, INTIMIDATION,
#   MOTOR VEHICLE THEFT (in-progress / time-critical interception).
# Excluded: property / financial / administrative offenses where response
#   time is not outcome-critical (e.g. THEFT, CRIMINAL DAMAGE, BURGLARY,
#   DECEPTIVE PRACTICE, NARCOTICS).
# Note: "CRIMINAL SEXUAL ASSAULT" is the post-2017 label (pre-2017 data
#   uses "CRIM SEXUAL ASSAULT"); the 2023 replay uses the current spelling.
URGENT_TYPES = {
    "BATTERY", "ASSAULT", "ROBBERY", "WEAPONS VIOLATION",
    "MOTOR VEHICLE THEFT", "HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT", "KIDNAPPING",
    "ARSON", "INTIMIDATION",
}


class HistoricalReplay(ArrivalProcess):
    r"""Replays a real event stream through the env.

    Parameters
    ----------
    json_path : str | Path
        Path to JSON file (Chicago Data Portal format expected):
        a list of dicts with keys 'date', 'latitude', 'longitude',
        'primary_type'.
    bounds : (x0, y0, x1, y1)
        Env world-unit bounds. The (latitude, longitude) bounding box
        of the input data is affinely mapped to this rectangle.
    day_offset_hours : int
        Hour of the replay's start. Events before this are skipped;
        events are indexed so their first arrival is at env_time = 0.
    duration_hours : int
        How many hours of replay to retain. Events after this are
        truncated.
    service_time : float
        Per-event service time in env time-units.
    urgent_only : bool
        If True, filter to the response-time-sensitive primary_types in
        URGENT_TYPES (our own urgency filter, not an official CPD code).
    seed : int
        Not used for randomness (replay is deterministic); kept for
        API compatibility.
    """

    def __init__(self, json_path, bounds,
                 day_offset_hours: int = 0,
                 duration_hours: int = 24,
                 service_time: float = 0.17,
                 urgent_only: bool = True,
                 seed: int = 0,
                 lat_bbox: tuple | None = None,
                 lon_bbox: tuple | None = None,
                 compress_to_epochs: int | None = None):
        self.bounds = bounds
        self.service_time = service_time
        self.rng = np.random.default_rng(seed)
        with open(json_path) as f:
            raw = json.load(f)
        # Parse and filter.
        records = []
        for d in raw:
            if not d.get("latitude") or not d.get("longitude"): continue
            if urgent_only and d.get("primary_type") not in URGENT_TYPES:
                continue
            try:
                la = float(d["latitude"]); lo = float(d["longitude"])
                if lat_bbox is not None and not (lat_bbox[0] <= la <= lat_bbox[1]):
                    continue
                if lon_bbox is not None and not (lon_bbox[0] <= lo <= lon_bbox[1]):
                    continue
                dt = datetime.fromisoformat(d["date"].replace("Z", ""))
                records.append((dt, la, lo))
            except Exception:
                continue
        records.sort(key=lambda r: r[0])
        if not records:
            raise ValueError(f"No valid records in {json_path}")
        # Truncate to the requested hour window.
        t0 = records[0][0]
        cutoff_lo = (t0.replace(hour=0, minute=0, second=0, microsecond=0)
                     + (records[0][0] - records[0][0]))
        # Filter by offset + duration
        import datetime as _dt
        start_dt = t0.replace(minute=0, second=0, microsecond=0) \
                     + _dt.timedelta(hours=day_offset_hours)
        end_dt = start_dt + _dt.timedelta(hours=duration_hours)
        records = [(t, la, lo) for (t, la, lo) in records
                   if start_dt <= t < end_dt]
        if not records:
            raise ValueError(
                f"No records in hour window [{start_dt}, {end_dt})"
            )
        # Compute bbox of SOURCE data (from the filtered slice).
        lats = [la for _, la, _ in records]
        lons = [lo for _, _, lo in records]
        self.src_lat_lo, self.src_lat_hi = min(lats), max(lats)
        self.src_lon_lo, self.src_lon_hi = min(lons), max(lons)
        x0, y0, x1, y1 = bounds
        # Affine: longitude -> x, latitude -> y.
        # Chicago: longitude increases eastward, latitude increases northward.
        # World-unit x increases right, y increases up.
        def xform(la, lo):
            fx = (lo - self.src_lon_lo) / max(1e-9,
                        self.src_lon_hi - self.src_lon_lo)
            fy = (la - self.src_lat_lo) / max(1e-9,
                        self.src_lat_hi - self.src_lat_lo)
            # Clamp to bounds to avoid edge bleed.
            eps = 1e-6
            fx = max(eps, min(1 - eps, fx))
            fy = max(eps, min(1 - eps, fy))
            return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        # Convert each record to an Event scheduled at the env epoch.
        start_ts = start_dt.timestamp()
        # Optional time compression: replay duration_hours of source events
        # into compress_to_epochs minutes of env time (used by the Sec. 7.4
        # load-sensitivity sweep). None = real-time.
        src_duration_min = duration_hours * 60.0
        time_scale = 1.0
        if compress_to_epochs is not None and src_duration_min > 0:
            time_scale = float(compress_to_epochs) / src_duration_min
        self.time_scale = time_scale
        self._events = []
        for dt, la, lo in records:
            # Env time: minutes since start_dt, optionally compressed.
            t_env = ((dt.timestamp() - start_ts) / 60.0) * time_scale
            x, y = xform(la, lo)
            self._events.append(Event(x=float(x), y=float(y),
                                       occurrence_time=float(t_env),
                                       service_time=float(service_time)))
        # Sort by occurrence_time.
        self._events.sort(key=lambda e: e.occurrence_time)
        self._cursor = 0
        self._n_total = len(self._events)
        self._start_dt = start_dt
        self._end_dt = end_dt

    def step(self, t: float, dt: float) -> list[Event]:
        """Return events with occurrence_time in [t, t + dt).

        Rewinds the cursor if env-time `t` regressed below it (e.g. after
        env.reset()), so post-reset rollouts replay from the start.
        """
        # Cursor rewind on time regression (i.e. after env.reset()).
        if (self._cursor > 0
                and (self._cursor >= self._n_total
                     or self._events[self._cursor].occurrence_time > t + dt)):
            # Only rewind if we've fallen out of the [t, t+dt) window
            # entirely - preserves forward-progress within an episode.
            if t < self._events[max(0, self._cursor - 1)].occurrence_time:
                self._cursor = 0
        out = []
        while (self._cursor < self._n_total
               and self._events[self._cursor].occurrence_time < t + dt):
            if self._events[self._cursor].occurrence_time >= t:
                out.append(self._events[self._cursor])
            self._cursor += 1
        return out

    def reset(self) -> None:
        """Reset the replay cursor to the first event (explicit control;
        `step()` also auto-rewinds after env.reset())."""
        self._cursor = 0

    def __repr__(self):
        return (f"HistoricalReplay(n={self._n_total}, "
                f"window={self._start_dt} to {self._end_dt})")
