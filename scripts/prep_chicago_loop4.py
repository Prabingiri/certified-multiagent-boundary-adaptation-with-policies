r"""Pre-process weekly Chicago crime JSON into the 4-district Loop4 subset.

Filters to URGENT_TYPES and to CPD districts {1, 2, 12, 18}, then re-assigns
synthetic (latitude, longitude) coordinates so each district's events fall
cleanly into a designated quadrant of the env coordinate system. Time stamps
and per-district event counts are preserved exactly.

Region layout (env coordinate convention: x = longitude proxy increasing east,
y = latitude proxy increasing north). The synthetic lat-lon space is
[0.0, 1.0] x [0.0, 1.0] so the env's affine transform maps it cleanly to
[bounds.x0, bounds.x1] x [bounds.y0, bounds.y1]:

    +------------------ NORTH (y=1.0) -----------------+
    |                       |                          |
    | Region 0: Dist 12     | Region 1: Dist 18        |
    | Near West (HOT-A)     | Near North (COLD-A)      |
    | lon[0.00, 0.50]       | lon[0.50, 1.00]          |
    | lat[0.50, 1.00]       | lat[0.50, 1.00]          |
    |                       |                          |
    |---------------------- + -------------------------|
    |                       |                          |
    | Region 2: Dist 2      | Region 3: Dist 1         |
    | Wentworth (HOT-B)     | Central/Loop (COLD-B)    |
    | lon[0.00, 0.50]       | lon[0.50, 1.00]          |
    | lat[0.00, 0.50]       | lat[0.00, 0.50]          |
    |                       |                          |
    +------------------ SOUTH (y=0.0) -----------------+

This puts the HOT pair (12, 2) on the left column and the COLD pair (18, 1)
on the right column. The vertical interior boundary (x=0.5) is the primary
hot-to-cold lending boundary; horizontal boundaries within each column are
secondary.

Note: the geographic-accurate Chicago map is for the dataset figure only and
does not feed into the env. The env uses this synthetic quadrant layout for
clean region/boundary geometry. Per-district event rates and arrival timing
are preserved unchanged.

Usage:
    python scripts/prep_chicago_loop4.py --in data/chicago_911_week_2023-01.json \
        --out data/chicago_loop4_week_2023-01.json --seed 1
    # Or batch all 12 weekly files (seed = week index, reproducing the
    # data files used in the reported experiments):
    python scripts/prep_chicago_loop4.py --batch
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

URGENT_TYPES = {
    "BATTERY", "ASSAULT", "ROBBERY", "WEAPONS VIOLATION",
    "MOTOR VEHICLE THEFT", "HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT", "KIDNAPPING",
    "ARSON", "INTIMIDATION",
}

# CPD districts -> env quadrant (lon_lo, lon_hi, lat_lo, lat_hi) in [0,1]^2.
# Margin keeps events away from region boundaries to avoid affine edge clipping.
MARGIN = 0.02
QUADRANT = {
    "012": (0.00 + MARGIN, 0.50 - MARGIN, 0.50 + MARGIN, 1.00 - MARGIN),  # NW: Hot-A
    "018": (0.50 + MARGIN, 1.00 - MARGIN, 0.50 + MARGIN, 1.00 - MARGIN),  # NE: Cold-A
    "002": (0.00 + MARGIN, 0.50 - MARGIN, 0.00 + MARGIN, 0.50 - MARGIN),  # SW: Hot-B
    "001": (0.50 + MARGIN, 1.00 - MARGIN, 0.00 + MARGIN, 0.50 - MARGIN),  # SE: Cold-B
}
LOOP4_DISTRICTS = set(QUADRANT.keys())


def remap(in_path: Path, out_path: Path, seed: int = 0) -> dict:
    rng = random.Random(seed)
    with open(in_path) as f:
        raw = json.load(f)
    kept = []
    per_dist = {d: 0 for d in LOOP4_DISTRICTS}
    for d in raw:
        ptype = d.get("primary_type")
        dist = d.get("district")
        if ptype not in URGENT_TYPES:
            continue
        if dist not in LOOP4_DISTRICTS:
            continue
        lon_lo, lon_hi, lat_lo, lat_hi = QUADRANT[dist]
        new_lon = rng.uniform(lon_lo, lon_hi)
        new_lat = rng.uniform(lat_lo, lat_hi)
        kept.append({
            "date": d["date"],
            "primary_type": ptype,
            "district": dist,
            "latitude": f"{new_lat:.6f}",
            "longitude": f"{new_lon:.6f}",
        })
        per_dist[dist] += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(kept, f)
    return {"in": str(in_path), "out": str(out_path),
            "n_kept": len(kept), "per_district": per_dist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, default=None)
    ap.add_argument("--out", dest="out_path", type=str, default=None)
    ap.add_argument("--batch", action="store_true",
                    help="Process all 12 weekly files.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Coordinate-assignment seed. In --batch mode the "
                         "per-week seed is seed + week index; the default "
                         "reproduces the released experiment data.")
    ap.add_argument("--data-dir", type=str, default="data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.batch:
        results = []
        for wk in range(1, 13):
            wk_str = f"{wk:02d}"
            in_p = data_dir / f"chicago_911_week_2023-{wk_str}.json"
            out_p = data_dir / f"chicago_loop4_week_2023-{wk_str}.json"
            if not in_p.exists():
                print(f"SKIP wk{wk_str}: {in_p} not found")
                continue
            r = remap(in_p, out_p, seed=args.seed + wk)
            results.append(r)
            print(f"wk{wk_str}: kept={r['n_kept']:4d}  per-dist={r['per_district']}")
        total = sum(r["n_kept"] for r in results)
        print(f"\nTOTAL across {len(results)} weeks: {total} events")
    else:
        if not args.in_path or not args.out_path:
            ap.error("--in and --out required when --batch is not set")
        r = remap(Path(args.in_path), Path(args.out_path), seed=args.seed)
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
