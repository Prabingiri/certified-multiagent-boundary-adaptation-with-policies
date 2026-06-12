r"""4-pane regime taxonomy figure.

Panels:

  S1 - single hotspot
  SB - corner hotspots
  N  - shifting hotspot (with trajectory arrow)
  R  - real Chicago Loop4

U (uniform) and A (adaptive stress) are not panel-illustrated: U is uniform
Poisson;
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from sigspatial_style import apply as _apply_style


from certified_marl.env.arrivals import (
    BoundaryHotspot, MultiHotspot, ShiftingHotspot,
)


def _simulate_arrivals(arrival_proc, horizon: int, dt: float = 1.0):
    """Run an ArrivalProcess for `horizon` epochs and return the
    (x, y) coordinates of every generated event. Identical to what
    the env feeds CS-LSTF during a rollout."""
    xs, ys = [], []
    for t in range(horizon):
        events = arrival_proc.step(t * dt, dt)
        for ev in events:
            xs.append(ev.x); ys.append(ev.y)
    return np.array(xs), np.array(ys)

HOT_FILL = "#fde2cf"
COLD_FILL = "#cfe1f0"
NEUTRAL_FILL = "#f0f0f0"
GRID_COLOR = "#1a1a1a"
HOT_X = "#b00020"


def _shade_region(ax, rect, color, alpha=0.55):
    x0, y0, x1, y1 = rect
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     facecolor=color, edgecolor="none",
                                     alpha=alpha, zorder=1))


def _draw_grid(ax, regions, *, color=GRID_COLOR, lw=1.3, alpha=0.95):
    xs = sorted({float(r[0]) for r in regions} | {float(r[2]) for r in regions})
    ys = sorted({float(r[1]) for r in regions} | {float(r[3]) for r in regions})
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    for x in xs:
        ax.plot([x, x], [y0, y1], color=color, lw=lw, alpha=alpha, zorder=5)
    for y in ys:
        ax.plot([x0, x1], [y, y], color=color, lw=lw, alpha=alpha, zorder=5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")


def _hotspots(ax, points, *, size=160):
    for x, y in points:
        ax.scatter([x], [y], s=size, marker="X", color=HOT_X,
                   edgecolors="white", linewidths=1.4, zorder=11)


def _box(ax, txt, *, loc="ll"):
    pos = {"ll": (0.04, 0.06, "left", "bottom")}
    x, y, ha, va = pos[loc]
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=8,
            bbox=dict(facecolor="white", edgecolor="#888", alpha=0.92, pad=2),
            zorder=20)


def _set_xy(ax, c=16):
    ticks = [0, 2*c, 4*c]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")


def _plot_events(ax, xs, ys, *, alpha=0.55, size=3, subsample=None):

    if subsample is not None and subsample < 1.0:
        rng = np.random.default_rng(0)
        keep = rng.random(len(xs)) < subsample
        xs, ys = xs[keep], ys[keep]
    ax.scatter(xs, ys, s=size, c="#444", alpha=alpha, linewidths=0, zorder=6)


def pane_s1(ax):
    regions = [[i*16, j*16, (i+1)*16, (j+1)*16] for j in range(4) for i in range(4)]
    cx, cy = 24.0, 24.0
    hot_idx = next(i for i, r in enumerate(regions) if r[0] <= cx < r[2] and r[1] <= cy < r[3])
    for i, r in enumerate(regions):
        _shade_region(ax, r, HOT_FILL if i == hot_idx else NEUTRAL_FILL, alpha=0.45)
    _draw_grid(ax, regions)

    rng_sim = np.random.default_rng(101)
    arr = BoundaryHotspot(center=(cx, cy), sigma=3.0, rate=6.0,
                           bounds=(0.0, 0.0, 64.0, 64.0), rng=rng_sim)
    xs, ys = _simulate_arrivals(arr, horizon=500)
    _plot_events(ax, xs, ys, subsample=0.10)
    _hotspots(ax, [(cx, cy)])
    sigma_circ = mpatches.Circle((cx, cy), 6.0, fill=False, edgecolor=HOT_X,
                                  linestyle="--", linewidth=0.9, alpha=0.6, zorder=8)
    ax.add_patch(sigma_circ)
    ax.set_title("S1 - Single hotspot", loc="left", fontsize=12)
    _set_xy(ax)


def pane_sb(ax):
    regions = [[i*16, j*16, (i+1)*16, (j+1)*16] for j in range(4) for i in range(4)]
    hot_centers = [(8, 8), (56, 8), (8, 56), (56, 56)]
    hot_idxs = set()
    for hx, hy in hot_centers:
        for i, r in enumerate(regions):
            if r[0] <= hx < r[2] and r[1] <= hy < r[3]:
                hot_idxs.add(i)
    for i, r in enumerate(regions):
        _shade_region(ax, r, HOT_FILL if i in hot_idxs else NEUTRAL_FILL, alpha=0.45)
    _draw_grid(ax, regions)
    # Run the actual MultiHotspot arrival generator (q=1.0 -> 100% hot).
    # rate=8 x horizon=500 -> ~4000 events split across 4 corner Gaussians.
    rng_sim = np.random.default_rng(102)
    arr = MultiHotspot(rate=8.0, hot_centers=hot_centers, q=1.0, sigma=3.0,
                        normal_centers=[(24.0, 24.0)],  # unused (q=1.0)
                        bounds=(0.0, 0.0, 64.0, 64.0), rng=rng_sim)
    xs, ys = _simulate_arrivals(arr, horizon=500)
    _plot_events(ax, xs, ys, subsample=0.10)
    _hotspots(ax, hot_centers)
    ax.set_title("SB - Corner hotspots", loc="left",  fontsize=12)
    _set_xy(ax)


def pane_n(ax):
    """N: shifting hotspot - Gaussian center translates (12,12) -> (52,52) at
    period 1000. Trajectory shown as fading X markers + arrow + sampled events
    drawn from Gaussians at each trajectory point."""
    regions = [[i*16, j*16, (i+1)*16, (j+1)*16] for j in range(4) for i in range(4)]
    for r in regions:
        _shade_region(ax, r, NEUTRAL_FILL, alpha=0.4)
    _draw_grid(ax, regions)
    start = (12.0, 12.0); end = (52.0, 52.0)
    pts = [(start[0] + t*(end[0]-start[0]), start[1] + t*(end[1]-start[1]))
           for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    # Run the actual ShiftingHotspot arrival generator: rate=6 x horizon=500
    # -> ~3000 events distributed along the moving trajectory (triangle wave
    # over period=1000).
    rng_sim = np.random.default_rng(103)
    arr = ShiftingHotspot(start=start, end=end, period=1000.0,
                           sigma=3.0, rate=6.0,
                           bounds=(0.0, 0.0, 64.0, 64.0), rng=rng_sim)
    xs, ys = _simulate_arrivals(arr, horizon=500)
    _plot_events(ax, xs, ys, subsample=0.10)
    for i, (x, y) in enumerate(pts):
        alpha = 0.30 + 0.70 * (i / (len(pts)-1))
        ax.scatter([x], [y], s=120, marker="X", color=HOT_X,
                   edgecolors="white", linewidths=1.1, zorder=11, alpha=alpha)
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=HOT_X, lw=1.6, alpha=0.65))
    ax.set_title("N - Shifting hotspot", loc="left", fontsize=12)
    _set_xy(ax)


def _load_chicago_events_in_env_coords():
    raw = json.loads((REPO_ROOT / "data/chicago_loop4_week_2023-01.json").read_text())
    QUAD = {"012": (0.0, 4.0, 4.0, 8.0), "018": (4.0, 4.0, 8.0, 8.0),
            "002": (0.0, 0.0, 4.0, 4.0), "001": (4.0, 0.0, 8.0, 4.0)}
    pts = []
    for d in raw:
        dist = d.get("district")
        if dist not in QUAD: continue
        try:
            la = float(d["latitude"]); lo = float(d["longitude"])
        except Exception:
            continue
        x = lo * 8.0; y = la * 8.0
        pts.append((x, y))
    return pts


def pane_r(ax):
    regions = [[0, 4, 4, 8], [4, 4, 8, 8], [0, 0, 4, 4], [4, 0, 8, 4]]
    role = {0: "HOT", 1: "COLD", 2: "HOT", 3: "COLD"}
    for i, r in enumerate(regions):
        _shade_region(ax, r, HOT_FILL if role[i] == "HOT" else COLD_FILL, alpha=0.4)
    obs_path = REPO_ROOT / "data/chicago_loop4_obstacles_top10.json"
    if obs_path.exists():
        obs = json.loads(obs_path.read_text())
        colors = {"water": "#467ba0", "highway": "#404040", "park": "#3d7a4a"}
        for dist_id, lst in obs.items():
            for o in lst:
                x0, y0, x1, y1 = o["rect_env"]
                ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                 facecolor=colors[o["kind"]], edgecolor="none",
                                                 alpha=0.78, zorder=4))
    pts = _load_chicago_events_in_env_coords()
    if pts:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=3, c="#222222", alpha=0.45, linewidths=0, zorder=6)
    _draw_grid(ax, regions)

    # Two lines: district ID + arrival rate. Full Loop-4 names are given in
    # the figure caption.
    label_pos = {
        ("Dist 12",  "0.86 e/h"): (2.0, 6.05),
        ("Dist 18",  "0.58 e/h"): (6.0, 6.05),
        ("Dist  2",  "0.89 e/h"): (2.0, 2.05),
        ("Dist  1",  "0.55 e/h"): (6.0, 2.05),
    }
    for (line1, line2), (x, y) in label_pos.items():
        ax.text(x, y, f"{line1}\n{line2}", ha="center", va="center", fontsize=9,
                color="#111",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5),
                zorder=12)
    ax.set_title("R - Chicago (real)", loc="left", fontsize=12)
    ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 4, 8])
    ax.set_xlabel("x (cells)", fontsize=14); ax.set_ylabel("y (cells)", fontsize=14)
    ax.tick_params(labelsize=12)


def main():
    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.8), constrained_layout=True)
    pane_s1(axes[0, 0])
    pane_sb(axes[0, 1])
    pane_n(axes[1, 0])
    pane_r(axes[1, 1])

    from matplotlib.lines import Line2D
    # Single combined entry for events: synthetic in S1/SB/N, real Chicago
    # crime incidents in R. Same gray dot styling across all four panes.
    legend_handles = [
        Line2D([0], [0], marker="X", color="w", markerfacecolor=HOT_X,
               markeredgecolor="white", markersize=9, markeredgewidth=1.0,
               label="hotspot"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333",
               markersize=4, label="arrival"),
        mpatches.Patch(facecolor="#467ba0", alpha=0.78, label="water"),
        mpatches.Patch(facecolor="#404040", alpha=0.78, label="roads"),
        mpatches.Patch(facecolor="#3d7a4a", alpha=0.78, label="parks"),
    ]

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.07), ncol=5, fontsize=13,
               frameon=True, framealpha=0.95,
               columnspacing=1.0, handletextpad=0.5, borderpad=0.5,
               handlelength=1.1)

    out_pdf = REPO_ROOT / "figures/fig_regime_taxonomy_4pane_v5.pdf"
    out_png = REPO_ROOT / "figures/fig_regime_taxonomy_4pane_v5.png"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
