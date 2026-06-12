"""Certified band-width trade-off figure 

Output: figures/fig_part1_trilemma_clean.pdf
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sigspatial_style as style

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures", "fig_part1_trilemma_clean.pdf")

# Non-method palette for geometric quantities.
CURVE_COLOR = "#222222"        # near-black for the math curve
ACCENT_COLOR = "#9c1a3a"       # crimson - not a method color
KERNEL_FILL = "#e2e8da"        # pale neutral green-gray for kernel area
KERNEL_EDGE = "#5a6c4a"        # dark olive - not the Wong CPAC green
BAND_FILL = "#f0d8d8"          # pale neutral pink for band area
BAND_LABEL = "#7a3030"         # muted brick


def kernel_fraction(c: int, dstar: int) -> float:
    if 2 * dstar >= c:
        return 0.0
    return (1.0 - 2.0 * dstar / c) ** 2


def main() -> None:
    style.apply()  # locked fonts only; no per-script overrides

    # Math sanity check.
    assert abs(kernel_fraction(16, 4) - 0.25) < 1e-9

    fig, axes = plt.subplots(
        1, 2, figsize=(3.4, 2.2),
        gridspec_kw={"wspace": 0.45,
                     "left": 0.10,
                     "right": 0.99,
                     "top": 0.96,
                     "bottom": 0.18},
    )

    # ----- Left: kernel fraction curve --------------------------------
    ax = axes[0]
    ratios = np.linspace(0.0, 0.5, 400)
    kf = (1.0 - 2.0 * ratios) ** 2
    ax.plot(ratios, kf, color=CURVE_COLOR, linewidth=1.8, zorder=3)
    # Default operating point at delta*/c = 0.25.
    ax.axvline(0.25, color=ACCENT_COLOR, linestyle="--",
               linewidth=0.9, zorder=2)
    ax.axhline(0.25, color=ACCENT_COLOR, linestyle="--",
               linewidth=0.9, zorder=2)
    ax.scatter([0.25], [0.25], s=44, color=ACCENT_COLOR,
               edgecolor="black", linewidth=0.5, zorder=4)
    ax.annotate("default\n" + r"$\delta^*/c = 0.25$",
                xy=(0.25, 0.25), xytext=(0.30, 0.55),
                fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))
    ax.set_xlabel(r"$\delta^*/c$")
    ax.set_ylabel(r"Kernel fraction $|K_i|/|R_i^0|$")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, width=0.6,
                   direction="out")

    # ----- Right: default region geometry -----------------------------
    ax = axes[1]
    c, d = 16, 4
    # Outer outline.
    ax.add_patch(Rectangle((0, 0), c, c, fill=False,
                           edgecolor="black", linewidth=1.2))
    # Band (4 strips).
    for x0, y0, w, h in [(0, 0, d, c), (c - d, 0, d, c),
                         (d, 0, c - 2 * d, d), (d, c - d, c - 2 * d, d)]:
        ax.add_patch(Rectangle((x0, y0), w, h, fill=True,
                               facecolor=BAND_FILL, edgecolor="none"))
    # Kernel.
    ax.add_patch(Rectangle((d, d), c - 2 * d, c - 2 * d,
                           fill=True, facecolor=KERNEL_FILL,
                           edgecolor=KERNEL_EDGE, linewidth=1.0))
    ax.text(c / 2, c / 2, r"$K_i$", ha="center", va="center",
            fontsize=12)
    ax.text(d / 2, c / 2, "band", ha="center", va="center",
            fontsize=8, color=BAND_LABEL)
    ax.set_xlim(-0.5, c + 0.5)
    ax.set_ylim(-0.5, c + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("cells")
    ax.set_ylabel("cells")
    ax.tick_params(axis="both", which="both", length=3, width=0.6,
                   direction="out")

    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print(f"\nVerified: kernel_fraction(16, 4) = {kernel_fraction(16, 4):.4f}")


if __name__ == "__main__":
    main()
