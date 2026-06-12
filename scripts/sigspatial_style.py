"""Shared figure style for the paper - single source of truth.

Apply by importing and calling `apply()` at the top of any figure script.
All figures share this style block so the full set is
visually consistent.

Wong colorblind-safe palette per Wong, B. (2011). "Points of view: Color
blindness." Nature Methods 8, 441.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


# Wong colorblind-safe categorical palette
PALETTE = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "sky_blue":  "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermillion": "#D55E00",
    "purple":    "#CC79A7",
}

# Method-name -> color contract (used everywhere).
# Palette rationale: orange for the no-collaboration baseline (warm but
# gentle), vermillion for the intermediate fixed-band variant (bolder
# warm), blue for the primary shielded method (cool/professional focus),
# green for the learned CPAC policy. All four are Wong (2011)
# colorblind-safe.
METHOD_COLORS = {
    "hold":         PALETTE["orange"],     # warm baseline
    "fixed_band":   PALETTE["vermillion"], # bolder intermediate
    "safe_greedy":  PALETTE["blue"],       # Wong dark blue (primary method)
    "masked_ppo":  PALETTE["green"],
    "masked_ppo":   PALETTE["green"],
}


LABEL_COLORS = {
    r"$\pi_{\mathrm{hold}}$":  PALETTE["orange"],
    r"$\pi_{\mathrm{FB}}$":    PALETTE["vermillion"],
    r"$\pi_{\mathrm{SG}}$":    PALETTE["blue"],
    r"$\pi^{\mathrm{CPAC}}$":  PALETTE["green"],
}

# Regime-tier colors (for scorecard heatmap F4)
TIER_COLORS = {
    "tier1_designed_for":  "#A6D9A6",  # light green
    "tier3_boundary":      "#F4B5B0",  # soft red
    "neutral_grey":        "#E8E8E8",
}

SIZE_SINGLE = (3.3, 2.4)   # one column
SIZE_DOUBLE = (6.8, 2.6)   # full width, narrow
SIZE_DOUBLE_TALL = (6.8, 3.4)  # full width, taller for 4-panel grids


def apply():
    """Apply the locked rcParams. Call once at script start."""
    mpl.rcParams.update({

        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Math text
        "mathtext.fontset": "cm",
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 1.0,
        # Grid (off by default; enable per-figure if useful)
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Save
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # editable text in vector PDF
        "ps.fonttype": 42,
    })


def annotate_caption_only():
    return lambda *args, **kwargs: None
