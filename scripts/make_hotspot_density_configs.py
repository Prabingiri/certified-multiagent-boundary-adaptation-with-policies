"""Hotspot-density sweep: fix the grid (g=8, 64 regions), vary the NUMBER of
hotspots at fixed total load, spread evenly across the workspace.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

CFG_DIR = Path(__file__).resolve().parent.parent / "configs" / "experiment"
G = 8                # 8x8 = 64 regions
C = 16.0             # cells per region side
INSET = 4.0
U_BAR = 28.353
LAMBDA = 0.5 * G * G  # fixed total load = 32

# h -> (cols, rows) even factorization on the 8x8 grid (exact rectangles).
FACTOR = {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (4, 2), 16: (4, 4), 32: (8, 4)}


def even_indices(p: int, size: int = G) -> list[int]:
    """p indices spread evenly over 0..size-1."""
    return [int((i + 0.5) * size / p) for i in range(p)]


def region_center(col: int, row: int) -> tuple[float, float]:
    return (C * col + C / 2.0, C * row + C / 2.0)


def placement(h: int) -> list[list[float]]:
    """Even sub-grid placement of h hotspots on the GxG grid.

    Exact rectangles for the doubling set (FACTOR); for arbitrary h (e.g. the
    linear 5/10/15/20 sweep), use a near-square cols x rows even sub-grid and
    take the first h cells -- still evenly spread with idle neighbours.
    """
    if h in FACTOR:
        cols_n, rows_n = FACTOR[h]
    else:
        cols_n = math.ceil(math.sqrt(h))
        rows_n = math.ceil(h / cols_n)
    cols, rows = even_indices(cols_n), even_indices(rows_n)
    pts = [list(region_center(c, r)) for r in rows for c in cols]
    return pts[:h]


def make(h: int) -> dict:
    hot = placement(h)
    assert len(hot) == h, f"h={h}: built {len(hot)} hotspots"

    bound = C * G
    regions, kernels, centers = [], [], []
    for r in range(G):
        for c in range(G):
            x0, y0 = c * C, r * C
            regions.append([x0, y0, x0 + C, y0 + C])
            kernels.append([x0 + INSET, y0 + INSET, x0 + C - INSET, y0 + C - INSET])
            centers.append([x0 + C / 2.0, y0 + C / 2.0])
    hot_set = {tuple(p) for p in hot}
    normal = [p for p in centers if tuple(p) not in hot_set]

    return {
        "experiment": {
            "name": f"sb_density_h{h}", "seed": 0, "num_updates": 0,
            "log_path": f"runs/sb_density_h{h}/log.jsonl", "save_every": 0,
            "trainer": "none", "domain": "canonical_cells",
        },
        "env": {
            "regions": regions, "kernels": kernels,
            "U_bar": [U_BAR] * (G * G), "bounds": [0.0, 0.0, bound, bound],
            "dt": 1.0, "speed": 1.0, "delta_star_default": 4.0, "delta_step": 1.0,
            "horizon": 500, "initial_delta_fraction": 0.0,
        },
        "arrivals": {
            "kind": "multi_hotspot", "rate": LAMBDA, "q": 1.0, "sigma": 3.0,
            "hot_centers": hot, "normal_centers": normal, "service_time": 1.0,
        },
        "model": {"state_dim": 6, "critic_dim": 9, "hidden": 64},
        "trainer": {
            "lr_actor": 0.0003, "lr_critic": 0.001, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_eps": 0.2, "entropy_coef": 0.01,
            "value_coef": 0.5, "epochs_per_update": 4, "minibatch_size": 128,
            "max_grad_norm": 0.5, "rollout_length": 256,
            "admission_reward_coef": 0.0, "epsilon_start": 0.0, "epsilon_final": 0.0,
        },
        "shaping": {"beta": 1.0, "omega_b": 0.5, "kappa": 0.1, "eta": 0.01},
    }


if __name__ == "__main__":
    hs = [int(a) for a in sys.argv[1:]] or [1, 5, 10, 15, 20, 32]
    for h in hs:
        cfg = make(h)
        out = CFG_DIR / f"sb_density_h{h}.yaml"
        with open(out, "w") as f:
            yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
        cov = 100.0 * h / (G * G)
        print(f"wrote {out}: h={h} ({cov:.1f}% coverage), load/hotspot={LAMBDA/h:.2f}, "
              f"hot={cfg['arrivals']['hot_centers'] if h <= 4 else '...'}")
