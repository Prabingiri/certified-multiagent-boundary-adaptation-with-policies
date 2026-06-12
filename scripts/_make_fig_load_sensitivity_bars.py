r"""3-panel grouped bar chart: Chicago Loop4 load sensitivity.

Sign flipped: positive bars = improvement over pi_hold (goes UP).
pi_hold shown as orange reference line at 0.

Output: figures/fig_load_sensitivity_bars.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import sigspatial_style as style
from sigspatial_style import METHOD_COLORS

REGIMES = [
    ("$1\\times$",   "runs/chicago_loop4_v3_full"),
    ("$4\\times$",   "runs/chicago_loop4_v54x_full"),
    ("$5\\times$",   "runs/chicago_loop4_v5_full"),
    ("$10\\times$",  "runs/chicago_loop4_v510x_full"),
]

COMPARE_POLICIES = ["fixed_band", "safe_greedy"]
POLICY_LABEL = {
    "hold":        "$\\pi_{\\mathrm{hold}}$",
    "fixed_band":  "$\\pi_{\\mathrm{FB}}$",
    "safe_greedy": "$\\pi_{\\mathrm{SG}}$",
}


METRICS = [
    (
        "Resp. drop (%)",
        lambda r: r["results"]["tail"]["mean"],
        "pct",
    ),
    (
        "Rej. drop (pp)",
        lambda r: (
            r["results"]["throughput"]["rejected"]
            / (
                r["results"]["throughput"]["admitted"]
                + r["results"]["throughput"]["rejected"]
            )
        ),
        "pp",
    ),
]


def collect(d):
    rows = []
    for f in (REPO / d).glob("*.jsonl") if (REPO / d).exists() else []:
        for line in f.read_text().splitlines():
            try: rows.append(json.loads(line))
            except: pass
    return rows


def policy_mean(rows, pol, key_fn):
    rs = [r for r in rows if r["policy"] == pol]
    vals = [key_fn(r) for r in rs]
    return mean(vals) if vals else float("nan")


def paired_improvements(rows, pol, key_fn, mode):
    by_pol_seed = {}
    for r in rows:
        by_pol_seed.setdefault((r["policy"], r["seed"]), []).append(key_fn(r))
    seeds = sorted({s for p, s in by_pol_seed if p == "hold"}
                   & {s for p, s in by_pol_seed if p == pol})
    vals = []
    for seed in seeds:
        v_hold = mean(by_pol_seed[("hold", seed)])
        v_pol = mean(by_pol_seed[(pol, seed)])
        if mode == "pct":
            vals.append((v_hold - v_pol) / v_hold * 100)
        else:
            vals.append((v_hold - v_pol) * 100)
    return np.asarray(vals, dtype=float)


def boot_mean_ci(vals, B=1000, alpha=0.05):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float("nan"), 0.0, 0.0
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    bs = vals[idx].mean(axis=1)
    mu = float(vals.mean())
    lo = float(np.percentile(bs, 100 * alpha / 2))
    hi = float(np.percentile(bs, 100 * (1 - alpha / 2)))
    return mu, mu - lo, hi - mu


def main():
    style.apply()  
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.8, 2.55), constrained_layout=False)
    if len(METRICS) == 1:
        axes = [axes]

    data = {label: collect(d) for label, d in REGIMES}
    n_K = len(REGIMES)
    bar_w = 0.34
    x_base = np.arange(n_K)

    for ax_idx, (ylabel, key_fn, mode) in enumerate(METRICS):
        ax = axes[ax_idx]
        for p_idx, pol in enumerate(COMPARE_POLICIES):
            improvements = []
            for label, _ in REGIMES:
                rows = data[label]
                v_hold = policy_mean(rows, "hold", key_fn)
                v_pol = policy_mean(rows, pol, key_fn)
                if v_hold == v_hold and v_pol == v_pol:
                    if mode == "pct":
                        improvements.append((v_hold - v_pol) / v_hold * 100)
                    else:
                        improvements.append((v_hold - v_pol) * 100)
                else:
                    improvements.append(0.0)
            offset = (p_idx - 0.5) * bar_w
            ax.bar(x_base + offset, improvements, width=bar_w,
                   color=METHOD_COLORS[pol], edgecolor="black",
                   linewidth=0.5)

        # pi_hold reference: thick orange line at 0
        ax.axhline(0, color=METHOD_COLORS["hold"], linewidth=2.5,
                   alpha=0.95, zorder=4)

        ax.set_xticks(x_base)
        ax.set_xticklabels([r[0] for r in REGIMES])
        ax.set_xlabel("Event scaling")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(y=0.16)

    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS["hold"], linewidth=2.5,
               label=POLICY_LABEL["hold"]),
        mpatches.Patch(facecolor=METHOD_COLORS["fixed_band"], edgecolor="black",
                       linewidth=0.5, label=POLICY_LABEL["fixed_band"]),
        mpatches.Patch(facecolor=METHOD_COLORS["safe_greedy"], edgecolor="black",
                       linewidth=0.5, label=POLICY_LABEL["safe_greedy"]),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=3,
               frameon=True, framealpha=0.95, handlelength=1.5)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.22, top=0.82, wspace=0.35)

    out_pdf = REPO / "figures/fig_load_sensitivity_bars_chicago.pdf"
    out_png = REPO / "figures/fig_load_sensitivity_bars_chicago.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=240)
    print(f"wrote {out_pdf} ({out_pdf.stat().st_size} bytes)")
    print(f"wrote {out_png} ({out_png.stat().st_size} bytes)")

    # the ordered Chicago stress axis.
    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(3.35, 2.55), sharex=True,
        constrained_layout=False,
    )
    if len(METRICS) == 1:
        axes = [axes]
    x_vals = np.arange(len(REGIMES), dtype=float)
    pol = "safe_greedy"

    for ax, (ylabel, key_fn, mode) in zip(axes, METRICS):
        means, elos, ehis = [], [], []
        for label, _ in REGIMES:
            vals = paired_improvements(data[label], pol, key_fn, mode)
            mu, lo, hi = boot_mean_ci(vals)
            means.append(mu); elos.append(lo); ehis.append(hi)
        ax.errorbar(
            x_vals, means, yerr=[elos, ehis],
            color=METHOD_COLORS[pol], marker="o", linestyle="-",
            linewidth=1.3, markersize=4.4,
            markeredgecolor="black", markeredgewidth=0.45,
            capsize=2.6, capthick=0.75, elinewidth=0.75,
        )
        ax.axhline(0, color=METHOD_COLORS["hold"], linewidth=1.2,
                   alpha=0.95, zorder=1)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([r[0] for r in REGIMES])
        ax.text(0.02, 0.92, ylabel, transform=ax.transAxes,
                ha="left", va="top")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(y=0.22)

    axes[-1].set_xlabel("Event scaling")
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS["safe_greedy"], marker="o",
               linewidth=1.3, markersize=4.4, markeredgecolor="black",
               markeredgewidth=0.45, label=POLICY_LABEL["safe_greedy"]),
        Line2D([0], [0], color=METHOD_COLORS["hold"], linewidth=1.2,
               label="$\\pi_{\\mathrm{hold}}$"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.23), ncol=2,
                   frameon=True, framealpha=0.95, handlelength=1.4,
                   borderpad=0.25, columnspacing=0.9)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.16, top=0.90, hspace=0.20)
    out_pdf = REPO / "figures/fig_load_sensitivity_sg_chicago.pdf"
    out_png = REPO / "figures/fig_load_sensitivity_sg_chicago.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=240)
    print(f"wrote {out_pdf} ({out_pdf.stat().st_size} bytes)")
    print(f"wrote {out_png} ({out_png.stat().st_size} bytes)")


    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(3.35, 2.65), sharex=True,
        constrained_layout=False,
    )
    if len(METRICS) == 1:
        axes = [axes]
    x_vals = np.arange(len(REGIMES), dtype=float)
    markers = {"fixed_band": "s", "safe_greedy": "o"}
    linestyles = {"fixed_band": "--", "safe_greedy": "-"}

    for ax, (ylabel, key_fn, mode) in zip(axes, METRICS):
        for pol in COMPARE_POLICIES:
            means, elos, ehis = [], [], []
            for label, _ in REGIMES:
                vals = paired_improvements(data[label], pol, key_fn, mode)
                mu, lo, hi = boot_mean_ci(vals)
                means.append(mu); elos.append(lo); ehis.append(hi)
            ax.errorbar(
                x_vals, means, yerr=[elos, ehis],
                color=METHOD_COLORS[pol], marker=markers[pol],
                linestyle=linestyles[pol], linewidth=1.25,
                markersize=4.2, markeredgecolor="black",
                markeredgewidth=0.45, capsize=2.5,
                capthick=0.75, elinewidth=0.75,
                label=POLICY_LABEL[pol],
            )
        ax.axhline(0, color=METHOD_COLORS["hold"], linewidth=1.15,
                   alpha=0.95, zorder=1)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([r[0] for r in REGIMES])
        ax.set_ylabel(ylabel)
        ax.yaxis.set_label_coords(-0.16, 0.5)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(y=0.22)

    axes[-1].set_xlabel("Event scaling")
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS["fixed_band"], marker="s",
               linestyle="--", linewidth=1.25, markersize=4.2,
               markeredgecolor="black", markeredgewidth=0.45,
               label=POLICY_LABEL["fixed_band"]),
        Line2D([0], [0], color=METHOD_COLORS["safe_greedy"], marker="o",
               linestyle="-", linewidth=1.25, markersize=4.2,
               markeredgecolor="black", markeredgewidth=0.45,
               label=POLICY_LABEL["safe_greedy"]),
        Line2D([0], [0], color=METHOD_COLORS["hold"], linewidth=1.15,
               label=POLICY_LABEL["hold"]),
    ]
    axes[0].legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.25), ncol=3,
                   frameon=True, framealpha=0.95, handlelength=1.25,
                   borderpad=0.25, columnspacing=0.65)
    fig.subplots_adjust(left=0.23, right=0.985, bottom=0.16, top=0.885, hspace=0.20)
    out_pdf = REPO / "figures/fig_load_sensitivity_policy_chicago.pdf"
    out_png = REPO / "figures/fig_load_sensitivity_policy_chicago.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=240)
    print(f"wrote {out_pdf} ({out_pdf.stat().st_size} bytes)")
    print(f"wrote {out_png} ({out_png.stat().st_size} bytes)")


    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(3.35, 2.75), sharex=True,
        constrained_layout=False,
    )
    if len(METRICS) == 1:
        axes = [axes]
    x_vals = np.arange(len(REGIMES), dtype=float)
    markers = {"fixed_band": "s", "safe_greedy": "o"}
    linestyles = {"fixed_band": "--", "safe_greedy": "-"}

    for ax, (ylabel, key_fn, mode) in zip(axes, METRICS):
        for pol in COMPARE_POLICIES:
            means, elos, ehis = [], [], []
            for label, _ in REGIMES:
                vals = paired_improvements(data[label], pol, key_fn, mode)
                mu, lo, hi = boot_mean_ci(vals)
                means.append(mu); elos.append(lo); ehis.append(hi)
            ax.errorbar(
                x_vals, means, yerr=[elos, ehis],
                color=METHOD_COLORS[pol], marker=markers[pol],
                linestyle=linestyles[pol], linewidth=1.5,
                markersize=4.6, capsize=2.6, capthick=0.8,
                elinewidth=0.8, label=POLICY_LABEL[pol],
            )
        ax.axhline(0, color=METHOD_COLORS["hold"], linewidth=1.5,
                   alpha=0.95, zorder=1)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([r[0] for r in REGIMES])
        ax.set_xlabel("Event scaling")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].legend(loc="lower right", frameon=True, framealpha=0.95,
                   handlelength=1.5)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.23, top=0.96, wspace=0.34)
    out_pdf = REPO / "figures/fig_load_sensitivity_lines_chicago.pdf"
    out_png = REPO / "figures/fig_load_sensitivity_lines_chicago.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=240)
    print(f"wrote {out_pdf} ({out_pdf.stat().st_size} bytes)")
    print(f"wrote {out_png} ({out_png.stat().st_size} bytes)")

    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(3.35, 2.75), sharex=True,
        constrained_layout=False,
    )
    if len(METRICS) == 1:
        axes = [axes]
    x_vals = np.arange(len(REGIMES), dtype=float)
    dodge = {"fixed_band": -0.08, "safe_greedy": 0.08}
    markers = {"fixed_band": "s", "safe_greedy": "o"}

    for ax, (ylabel, key_fn, mode) in zip(axes, METRICS):
        for pol in COMPARE_POLICIES:
            means, elos, ehis = [], [], []
            for label, _ in REGIMES:
                vals = paired_improvements(data[label], pol, key_fn, mode)
                mu, lo, hi = boot_mean_ci(vals)
                means.append(mu); elos.append(lo); ehis.append(hi)
            ax.errorbar(
                x_vals + dodge[pol], means, yerr=[elos, ehis],
                color=METHOD_COLORS[pol], marker=markers[pol],
                linestyle="none", markersize=5.2,
                markeredgecolor="black", markeredgewidth=0.55,
                capsize=3.0, capthick=0.85, elinewidth=0.85,
                label=POLICY_LABEL[pol],
            )
        ax.axhline(0, color=METHOD_COLORS["hold"], linewidth=1.4,
                   alpha=0.95, zorder=1)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([r[0] for r in REGIMES])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Event scaling")
    axes[0].legend(loc="lower right", frameon=True, framealpha=0.95,
                   handlelength=1.1, borderpad=0.25, labelspacing=0.25)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.16, top=0.97, hspace=0.18)
    out_pdf = REPO / "figures/fig_load_sensitivity_points_chicago.pdf"
    out_png = REPO / "figures/fig_load_sensitivity_points_chicago.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=240)
    print(f"wrote {out_pdf} ({out_pdf.stat().st_size} bytes)")
    print(f"wrote {out_png} ({out_png.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
