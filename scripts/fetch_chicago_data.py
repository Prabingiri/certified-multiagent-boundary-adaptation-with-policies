r"""Fetch additional Chicago crime data from the Chicago Data Portal.

The portal exposes the "Crimes - 2001 to Present" dataset via Socrata SODA
API at https://data.cityofchicago.org/resource/ijzp-q8t2.json.

This script fetches one or more 7-day windows in 2023, applies the same
URGENT_TYPES filter as historical_arrivals.py, and saves to the same
schema as the existing data/chicago_911_week_2023-01.json.

Usage:
    # Fetch a single week (Jan 8-14, 2023):
    python scripts/fetch_chicago_data.py --start 2023-01-08 --days 7 \
        --out data/chicago_911_week_2023-02.json

    # Fetch 12 weeks of 2023 (sequentially, jumping forward 4 weeks each):
    for w in 02 06 10 14 18 22 26 30 34 38 42 46; do
      python scripts/fetch_chicago_data.py --start 2023-{w}-01 \
          --days 7 --out data/chicago_911_week_2023-{w}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

URGENT_TYPES = {
    "BATTERY", "ASSAULT", "ROBBERY", "WEAPONS VIOLATION",
    "MOTOR VEHICLE THEFT", "HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT", "KIDNAPPING",
    "ARSON", "INTIMIDATION",
}

API_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"


def fetch_window(start_date: str, days: int, urgent_only: bool = False,
                 app_token: str | None = None,
                 timeout_sec: int = 60) -> list[dict]:
    """Fetch crimes between [start_date, start_date+days) UTC.

    Returns a list of dicts in the same schema as the existing
    chicago_911_week_2023-01.json file.
    """
    start_dt = datetime.fromisoformat(start_date)
    end_dt = start_dt + timedelta(days=days)
    where = (f"date between '{start_dt.isoformat()}' "
             f"and '{end_dt.isoformat()}'")
    params = {
        "$where": where,
        "$select": "date,latitude,longitude,primary_type,district",
        "$limit": "50000",
        "$order": "date ASC",
    }
    qs = urllib.parse.urlencode(params)
    url = f"{API_URL}?{qs}"
    req = urllib.request.Request(url)
    if app_token:
        req.add_header("X-App-Token", app_token)
    print(f"[fetch] GET {API_URL}", file=sys.stderr)
    print(f"[fetch] $where = {where}", file=sys.stderr)
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read()
    events = json.loads(raw)
    print(f"[fetch] {len(events)} events fetched (pre-filter)", file=sys.stderr)
    # Drop events missing geocoordinates (some incidents are unlocated).
    before = len(events)
    events = [e for e in events
              if e.get("latitude") and e.get("longitude")
              and e.get("date") and e.get("primary_type")]
    if len(events) < before:
        print(f"[fetch] dropped {before - len(events)} events with "
              f"missing geo/date/type", file=sys.stderr)
    if urgent_only:
        before = len(events)
        events = [e for e in events
                  if e.get("primary_type") in URGENT_TYPES]
        print(f"[fetch] urgent_only filter: {before} -> {len(events)} "
              f"events", file=sys.stderr)
    return events


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2023-01-01",
                   help="ISO date for window start (UTC). Default 2023-01-01, "
                        "the week Regime R replays.")
    p.add_argument("--days", type=int, default=7,
                   help="window length in days (default 7)")
    p.add_argument("--out", default="data/chicago_911_week_2023-01.json",
                   help="output JSON file path. Default is the file Regime "
                        "R's config expects.")
    p.add_argument("--urgent-only", action="store_true",
                   help="filter to URGENT_TYPES at fetch time (otherwise "
                        "the env's HistoricalReplay applies the filter)")
    p.add_argument("--app-token", default=None,
                   help="optional Socrata app token (raises rate limit "
                        "from 1k/hr to 1M/hr); see "
                        "https://dev.socrata.com/foundry/data.cityofchicago.org/ijzp-q8t2")
    p.add_argument("--timeout-sec", type=int, default=60)
    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events = fetch_window(args.start, args.days,
                          urgent_only=args.urgent_only,
                          app_token=args.app_token,
                          timeout_sec=args.timeout_sec)
    with open(out_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"[fetch] wrote {len(events)} events -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
