"""Hotspot-density sweep figure: fixed grid (g=8), fixed total load, vary the
number of evenly-spread hotspots (1 -> 32, i.e. 1.5% -> 50% coverage).
"""
from __future__ import annotations

import json
import math
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import sigspatial_style as style

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures", "fig_hotspot_density.pdf")
G2 = 64  # 8x8 regions

# Shared y-label x-coordinate so the two panels align (same strategy as the
# region-scaling and rho-tolerance figures). Less negative = closer to the plot.
YLABEL_X = -0.13

# Linear sweep plus the 50%-coverage endpoint (h=32), so the right arm of the
# inverted-U clearly falls below the single-hotspot value. Plotted vs coverage %
# so the 31->50 spacing stays honest. Only h with a runs/ dir are used.
HS = [1, 5, 10, 15, 20, 32]


def load(h):
    d = os.path.join(ROOT, "runs", "p1_hotspot_density", f"sb_density_h{h}")
    rows = []
    for n in os.listdir(d):
        if n.endswith(".jsonl"):
            for line in open(os.path.join(d, n), encoding="utf-8"):
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def per_seed(rows, policy, getter):
    out = {}
    for r in rows:
        if r.get("policy") != policy:
            continue
        out[r["seed"]] = getter(r["results"])
    return [out[s] for s in sorted(out)]


def ci(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def main():
    style.apply()
    adm = lambda res: res["throughput"]["admitted"]
    crs = lambda res: (res["throughput"].get("buffer_admitted_total", 0)
                       / max(res["throughput"]["admitted"], 1))

    avail = [h for h in HS
             if os.path.isdir(os.path.join(ROOT, "runs", "p1_hotspot_density",
                                            f"sb_density_h{h}"))]

    # All three policies; gain is over pi_hold (so pi_hold is flat at 0).
    pols = [("hold", r"$\pi_{\mathrm{hold}}$"),
            ("fixed_band", r"$\pi_{\mathrm{FB}}$"),
            ("safe_greedy", r"$\pi_{\mathrm{SG}}$")]
    gain = {p: ([], []) for p, _ in pols}
    cross = {p: ([], []) for p, _ in pols}
    for h in avail:
        rows = load(h)
        hold = per_seed(rows, "hold", adm)
        for p, _ in pols:
            a = per_seed(rows, p, adm)
            k = min(len(hold), len(a))
            m, e = ci([(a[i] - hold[i]) / 1000.0 for i in range(k)])  # thousands
            gain[p][0].append(m); gain[p][1].append(e)
            m, e = ci(per_seed(rows, p, crs))
            cross[p][0].append(m); cross[p][1].append(e)

    x = list(range(len(avail)))   # even categorical spacing (matches Fig. region scaling)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.35, 3.0), sharex=True,
        gridspec_kw={"hspace": 0.18, "left": 0.19, "right": 0.97,
                     "top": 0.88, "bottom": 0.20},
    )

    def line(ax, y, color, label):
        ax.errorbar(x, y[0], yerr=y[1], marker="o", ms=4.2, lw=1.35, color=color,
                    capsize=2.5, capthick=0.8, elinewidth=0.8, label=label)

    for p, lab in pols:
        line(ax1, gain[p], style.METHOD_COLORS[p], lab)
    ax1.set_ylabel("Adm. gain (k)")
    ax1.yaxis.set_label_coords(YLABEL_X, 0.5)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.grid(axis="y", alpha=0.25, lw=0.5); ax1.set_axisbelow(True)
    gmax = max(gain["safe_greedy"][0])
    ax1.set_ylim(-0.05 * gmax, gmax * 1.12)

    for p, lab in pols:
        line(ax2, cross[p], style.METHOD_COLORS[p], lab)
    ax2.set_ylabel("Cross-service")
    ax2.yaxis.set_label_coords(YLABEL_X, 0.5)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax2.grid(axis="y", alpha=0.25, lw=0.5); ax2.set_axisbelow(True)
    cmax = max(cross["safe_greedy"][0])
    ax2.set_ylim(-0.05 * cmax, cmax * 1.12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(h) for h in avail])
    ax2.set_xlabel("Hotspot count")

    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3,
               frameon=True, framealpha=0.95, handlelength=1.2,
               columnspacing=0.8, borderpad=0.25)

    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=300)
    print(f"wrote {OUT}")
    print(f"{'h':>4}{'cov%':>6}{'SG_gain_k':>10}{'FB_gain_k':>10}{'SG_cross':>9}")
    for i, h in enumerate(avail):
        print(f"{h:>4}{100*h/G2:>6.1f}{gain['safe_greedy'][0][i]:>10.2f}"
              f"{gain['fixed_band'][0][i]:>10.2f}{cross['safe_greedy'][0][i]:>9.3f}")


if __name__ == "__main__":
    main()
