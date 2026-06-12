"""Generate four-corner SB configs at g = 6, 8, 10, 12 for the scaling sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG_DIR = HERE.parent / "configs" / "experiment"
C = 16.0
INSET = 4.0
U_BAR = 28.353


def make(g: int) -> dict:
    n = g * g
    bound = C * g
    regions, kernels, centers = [], [], []
    for r in range(g):
        for c in range(g):
            x0, y0 = c * C, r * C
            regions.append([x0, y0, x0 + C, y0 + C])
            kernels.append([x0 + INSET, y0 + INSET, x0 + C - INSET, y0 + C - INSET])
            centers.append((x0 + C / 2.0, y0 + C / 2.0))

    lo, hi = C / 2.0, bound - C / 2.0
    hot = [[lo, lo], [hi, lo], [lo, hi], [hi, hi]]  # four corner-region centers
    hot_set = {tuple(h) for h in hot}
    normal = [[cx, cy] for (cx, cy) in centers if (cx, cy) not in hot_set]

    return {
        "experiment": {
            "name": f"a5_4corner_g{g}", "seed": 0, "num_updates": 0,
            "log_path": f"runs/a5_4corner_g{g}/log.jsonl", "save_every": 0,
            "trainer": "none", "domain": "canonical_cells",
        },
        "env": {
            "regions": regions, "kernels": kernels,
            "U_bar": [U_BAR] * n, "bounds": [0.0, 0.0, bound, bound],
            "dt": 1.0, "speed": 1.0,
            "delta_star_default": 4.0, "delta_step": 1.0,
            "horizon": 500, "initial_delta_fraction": 0.0,
        },
        "arrivals": {
            "kind": "multi_hotspot", "rate": 0.5 * n, "q": 1.0, "sigma": 3.0,
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


def write(g: int) -> None:
    cfg = make(g)
    out = CFG_DIR / f"a5_4corner_g{g}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    n = g * g
    print(f"wrote {out}: n={n}, bounds={16*g}x{16*g}, rate={0.5*n}, "
          f"hot={cfg['arrivals']['hot_centers']}, normals={len(cfg['arrivals']['normal_centers'])}")


def validate_against_a50() -> bool:
    """g=4 generator output must match the committed a5_4corner_a50.yaml."""
    ref = CFG_DIR / "a5_4corner_a50.yaml"
    with open(ref) as f:
        committed = yaml.safe_load(f)
    gen = make(4)
    ok = True
    for section in ("env", "arrivals"):
        for k, v in gen[section].items():
            if committed[section].get(k) != v:
                print(f"  MISMATCH {section}.{k}: gen={v!r} vs committed={committed[section].get(k)!r}")
                ok = False
    print(f"  g=4 vs a5_4corner_a50: {'MATCH' if ok else 'DIFF'}")
    return ok


if __name__ == "__main__":
    print("validating generator against committed a5_4corner_a50 (g=4):")
    if not validate_against_a50():
        print("generator does not reproduce the template; aborting")
        sys.exit(1)
    print("generating new scales:")
    for g in (6, 8, 10, 12):
        write(g)
