"""Region-count scaling figure

Data: runs/_master_phase1.csv, four-corner SB regimes A5_4corner (g=4, 16 reg),
A5_4corner_g6 (36), A5_4corner_g8 (64), A5_4corner_g10 (100), A5_4corner_g12
(144). Per-seed means, 10 seeds.

Output: figures/fig_region_scaling.pdf (+ .png)
"""
from __future__ import annotations

import math
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import sigspatial_style as style

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "runs", "_master_phase1.csv")
OUT = os.path.join(ROOT, "figures", "fig_region_scaling.pdf")


SCALE = [(16, "A5_4corner", 4), (36, "A5_4corner_g6", 6), (64, "A5_4corner_g8", 8),
         (100, "A5_4corner_g10", 10), (144, "A5_4corner_g12", 12)]

POLICIES = [
    ("hold", r"$\pi_{\mathrm{hold}}$", style.METHOD_COLORS["hold"]),
    ("fixed_band", r"$\pi_{\mathrm{FB}}$", style.METHOD_COLORS["fixed_band"]),
    ("safe_greedy", r"$\pi_{\mathrm{SG}}$", style.METHOD_COLORS["safe_greedy"]),
]

# Shared with the rho-tolerance figure so y-labels align across the pair.
YLABEL_X = -0.15


def stat(df, regime, policy, col, scale=1.0):
    """Per-seed mean and analytic 95% CI (1.96 * SEM), matching the rho figure."""
    s = df[(df["regime"] == regime) & (df["policy"] == policy)]
    vals = s.groupby("seed")[col].mean().values * scale
    m = float(vals.mean())
    if len(vals) < 2:
        return m, 0.0
    ci = 1.96 * math.sqrt(vals.var(ddof=1) / len(vals))
    return m, ci


def draw_panel(ax, df, col, ylabel, scale=1.0):
    handles, labels = [], []
    x = list(range(len(SCALE)))
    for policy, label, color in POLICIES:
        means, cis = [], []
        for _, reg, _ in SCALE:
            m, ci = stat(df, reg, policy, col, scale)
            means.append(m)
            cis.append(ci)
        handle = ax.errorbar(
            x, means, yerr=cis, marker="o", markersize=4.2, linewidth=1.35,
            capsize=2.5, capthick=0.8, elinewidth=0.8, color=color, label=label,
        )
        handles.append(handle)
        labels.append(label)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_coords(YLABEL_X, 0.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    return handles, labels


def main():
    style.apply()
    df = pd.read_csv(CSV)

    fig, axes = plt.subplots(
        2, 1, figsize=(3.35, 3.0), sharex=True,
        gridspec_kw={"hspace": 0.18, "left": 0.19, "right": 0.98,
                     "top": 0.88, "bottom": 0.19},
    )
    handles, labels = draw_panel(
        axes[0], df, "admitted", "Admitted (k)", scale=1.0 / 1000.0)
    draw_panel(axes[1], df, "cross_serve_rate", "Cross-service")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[1].set_xticks(list(range(len(SCALE))))
    axes[1].set_xticklabels([f"{n}\n(g={g})" for n, _, g in SCALE])
    axes[1].set_xlabel("Region count")

    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=3, frameon=True, framealpha=0.95, handlelength=1.2,
        columnspacing=0.8, borderpad=0.25,
    )

    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=300)
    print(f"wrote {OUT}")
    print(f"{'regions':>8}{'hold_k':>9}{'FB_k':>9}{'SG_k':>9}{'SG_cross':>10}")
    for n, reg, _ in SCALE:
        h = stat(df, reg, "hold", "admitted", 1 / 1000.)[0]
        fb = stat(df, reg, "fixed_band", "admitted", 1 / 1000.)[0]
        sg = stat(df, reg, "safe_greedy", "admitted", 1 / 1000.)[0]
        cr = stat(df, reg, "safe_greedy", "cross_serve_rate")[0]
        print(f"{n:>8}{h:>9.3f}{fb:>9.3f}{sg:>9.3f}{cr:>10.3f}")


if __name__ == "__main__":
    main()
