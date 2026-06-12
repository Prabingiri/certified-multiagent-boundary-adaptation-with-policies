r"""Scale Chicago Loop4 weekly events while preserving temporal structure.

For each real CPD priority-1 event in a week, generate `factor` replicas:
  - Same district, primary_type (real attributes preserved)
  - Same synthetic env-coord (lat, lon) as assigned in data prep
  - Timestamp jittered uniformly in [-30 min, +30 min] from the source event

Net effect: factor-x event volume; the daily/hourly profile is preserved
(low overnight, afternoon peak - matching real CPD dispatch temporal
structure) and the spatial distribution is unchanged.

Per-week jitter seeds: week index for the 5x files and factor * week index
for the 4x and 10x files. These seeds reproduce the scaled data files used
in the reported load-sensitivity experiments byte-for-byte.

Usage:
    python scripts/scale_chicago_loop4.py            # 4x, 5x, 10x; all weeks
    python scripts/scale_chicago_loop4.py --factors 4

Inputs:  data/chicago_loop4_week_2023-{01..12}.json
Outputs: data/chicago_loop4_{4x,5x,10x}_week_2023-{01..12}.json
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

JITTER_MIN = 30  # minutes


def scale_week(in_path: Path, out_path: Path, factor: int,
               seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    data = json.loads(in_path.read_text())
    out = []
    for ev in data:
        ts_str = ev["date"]
        try:
            t = datetime.fromisoformat(ts_str.replace("Z", ""))
        except Exception:
            continue
        # Original event, then factor - 1 replicas
        out.append(ev)
        for _ in range(factor - 1):
            jitter = rng.uniform(-JITTER_MIN, JITTER_MIN)
            new_t = t + timedelta(minutes=jitter)
            new_ev = dict(ev)
            new_ev["date"] = new_t.isoformat()
            out.append(new_ev)
    # Sort by timestamp for replay determinism.
    def _ts(e):
        try:
            return datetime.fromisoformat(e["date"].replace("Z", ""))
        except Exception:
            return datetime.min
    out.sort(key=_ts)
    out_path.write_text(json.dumps(out))
    return len(data), len(out)


def week_seed(factor: int, week: int) -> int:
    return week if factor == 5 else factor * week


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=int, nargs="+", default=[4, 5, 10],
                    help="event-volume multipliers to generate")
    ap.add_argument("--data-dir", type=str, default="data")
    args = ap.parse_args()

    base = Path(args.data_dir)
    for factor in args.factors:
        tag = f"{factor}x"
        for wk in range(1, 13):
            wk_str = f"{wk:02d}"
            in_p = base / f"chicago_loop4_week_2023-{wk_str}.json"
            out_p = base / f"chicago_loop4_{tag}_week_2023-{wk_str}.json"
            if not in_p.exists():
                print(f"SKIP {tag} wk{wk_str}: {in_p} not found")
                continue
            n_in, n_out = scale_week(in_p, out_p, factor,
                                     seed=week_seed(factor, wk))
            print(f"{tag} wk{wk_str}: {n_in} -> {n_out} events")


if __name__ == "__main__":
    main()
