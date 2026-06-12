"""Section 7.3 figure: regime-wise generalization scorecard.

  Fig 7a  ->  figures/fig_general_regime_scorecard_7a.pdf
              two panels: (1) rejection-rate reduction (pp), higher is better;
              (2) cross-service rate, the mechanism.
  Fig 7b  ->  figures/fig_general_regime_scorecard_7b.pdf
              one-panel LoadCV support figure.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sigspatial_style as style

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "runs", "_master_phase1.csv")
OUT_7A = os.path.join(ROOT, "figures", "fig_general_regime_scorecard_7a.pdf")
OUT_7B = os.path.join(ROOT, "figures", "fig_general_regime_scorecard_7b.pdf")
OUT_7C = os.path.join(ROOT, "figures", "fig_general_regime_mechanism_scatter.pdf")
OUT_7D = os.path.join(ROOT, "figures", "fig_general_regime_policy_compare.pdf")
RNG_SEED = 0
B = 1000


REGIME_MAP = [
    ("U",  ["A1_uniform"]),
    ("S1", ["A2_single_hotspot"]),
    ("SB", ["A5_2corner", "A5_2corner_heavy", "A5_2edge", "A5_4corner"]),
    ("N",  ["A4_shifting_fast", "A4_shifting_medium", "A4_shifting_slow"]),
    ("A",  ["A_adv_boundary"]),
    ("R",  ["B1_chicago_911"]),
]


def boot_ci(x, B=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0, 0.0
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = x[idx].mean(axis=1)
    return (float(np.percentile(bs, 100 * alpha / 2)),
            float(np.percentile(bs, 100 * (1 - alpha / 2))))


def aggregate_per_seed(df_sub, policy, value_col):
    """Average value_col across sub-variants within each matched seed."""
    p = df_sub[df_sub["policy"] == policy]
    if p.empty:
        return np.array([])
    return p.groupby("seed")[value_col].mean().sort_index().values


def compute():
    rng = np.random.default_rng(RNG_SEED)
    df = pd.read_csv(CSV)

    labels = []
    rej = {"mean": [], "lo": [], "hi": []}     # Delta rejection rate (pp)
    cross = {"mean": [], "lo": [], "hi": []}    # cross-service rate
    load = {"mean": [], "lo": [], "hi": []}     # Delta Load CV (%)

    for paper, csv_regs in REGIME_MAP:
        sub = df[df["regime"].isin(csv_regs)]
        if sub.empty:
            continue
        rej_hold = aggregate_per_seed(sub, "hold", "rejection_rate")
        rej_sg = aggregate_per_seed(sub, "safe_greedy", "rejection_rate")
        cross_sg = aggregate_per_seed(sub, "safe_greedy", "cross_serve_rate")
        load_hold = aggregate_per_seed(sub, "hold", "coef_var")
        load_sg = aggregate_per_seed(sub, "safe_greedy", "coef_var")

        n = min(len(rej_hold), len(rej_sg))
        if n == 0:
            continue
        # Rejection REDUCTION (pp): positive = fewer rejections than hold,
        # so bars point up (improvement), matching the load-study convention.
        d_rej = (rej_hold[:n] - rej_sg[:n]) * 100.0
        d_mean = float(d_rej.mean())
        d_lo, d_hi = boot_ci(d_rej, B=B, rng=rng)

        c_mean = float(cross_sg.mean())
        c_lo, c_hi = boot_ci(cross_sg, B=B, rng=rng)

        n2 = min(len(load_hold), len(load_sg))
        if n2 > 0 and load_hold[:n2].mean() > 0:
            d_load = (load_sg[:n2] - load_hold[:n2]) / load_hold[:n2].mean() * 100.0
            dl_mean = float(d_load.mean())
            dl_lo, dl_hi = boot_ci(d_load, B=B, rng=rng)
        else:
            dl_mean, dl_lo, dl_hi = 0.0, 0.0, 0.0

        labels.append(paper)
        rej["mean"].append(d_mean); rej["lo"].append(d_mean - d_lo); rej["hi"].append(d_hi - d_mean)
        cross["mean"].append(c_mean); cross["lo"].append(c_mean - c_lo); cross["hi"].append(c_hi - c_mean)
        load["mean"].append(dl_mean); load["lo"].append(dl_mean - dl_lo); load["hi"].append(dl_hi - dl_mean)

    return labels, rej, cross, load


def compute_policy_compare():
    rng = np.random.default_rng(RNG_SEED)
    df = pd.read_csv(CSV)
    policies = [("FB", "fixed_band"), ("SG", "safe_greedy")]

    labels = []
    out = {p: {"rej": {"mean": [], "lo": [], "hi": []},
               "cross": {"mean": [], "lo": [], "hi": []}}
           for p, _ in policies}

    for paper, csv_regs in REGIME_MAP:
        sub = df[df["regime"].isin(csv_regs)]
        if sub.empty:
            continue
        rej_hold = aggregate_per_seed(sub, "hold", "rejection_rate")
        if len(rej_hold) == 0:
            continue
        labels.append(paper)
        for p_label, p_slug in policies:
            rej_p = aggregate_per_seed(sub, p_slug, "rejection_rate")
            cross_p = aggregate_per_seed(sub, p_slug, "cross_serve_rate")

            n = min(len(rej_hold), len(rej_p))
            d_rej = (rej_hold[:n] - rej_p[:n]) * 100.0
            d_mean = float(d_rej.mean())
            d_lo, d_hi = boot_ci(d_rej, B=B, rng=rng)

            c_mean = float(cross_p.mean())
            c_lo, c_hi = boot_ci(cross_p, B=B, rng=rng)

            out[p_label]["rej"]["mean"].append(d_mean)
            out[p_label]["rej"]["lo"].append(d_mean - d_lo)
            out[p_label]["rej"]["hi"].append(d_hi - d_mean)
            out[p_label]["cross"]["mean"].append(c_mean)
            out[p_label]["cross"]["lo"].append(c_mean - c_lo)
            out[p_label]["cross"]["hi"].append(c_hi - c_mean)

    return labels, out


def _barv(ax, x, d, *, color, axhline_zero):
    ax.bar(x, d["mean"], width=0.62, yerr=[d["lo"], d["hi"]],
           color=color, edgecolor="black", linewidth=0.6,
           error_kw=dict(ecolor="black", capsize=3.0, elinewidth=0.9,
                         capthick=0.9), zorder=2)
    if axhline_zero:
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.margins(x=0.04)


def main():
    style.apply()

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    labels, rej, cross, load = compute()
    sg = style.METHOD_COLORS["safe_greedy"]
    x = list(range(len(labels)))  # left = U, right = R


    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.35, 2.85), sharex=True,
        gridspec_kw={"hspace": 0.16, "left": 0.15, "right": 0.97,
                     "top": 0.975, "bottom": 0.11},
    )
    _barv(ax1, x, rej, color=sg, axhline_zero=True)
    ax1.set_ylabel(r"Rej. red. (pp)", labelpad=4)

    _barv(ax2, x, cross, color=sg, axhline_zero=False)
    ax2.set_ylabel(r"Cross-service", labelpad=4)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)

    fig.savefig(OUT_7A)
    fig.savefig(OUT_7A.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_7A}")


    fig, ax = plt.subplots(
        1, 1, figsize=(3.35, 2.45),
        gridspec_kw={"left": 0.16, "right": 0.98,
                     "top": 0.96, "bottom": 0.17},
    )
    rej_mean = np.asarray(rej["mean"])
    cross_mean = np.asarray(cross["mean"])
    xerr = np.vstack([rej["lo"], rej["hi"]])
    yerr = np.vstack([cross["lo"], cross["hi"]])
    ax.errorbar(
        rej_mean, cross_mean, xerr=xerr, yerr=yerr,
        fmt="o", color=sg, ecolor="black", elinewidth=0.75,
        capsize=2.5, markersize=4.8, markeredgecolor="black",
        markeredgewidth=0.6, zorder=3,
    )
    offsets = {
        "U": (0.35, 0.018),
        "S1": (0.35, -0.020),
        "SB": (0.35, 0.014),
        "N": (0.35, -0.022),
        "A": (0.35, 0.018),
        "R": (0.35, 0.020),
    }
    for lab, xv, yv in zip(labels, rej_mean, cross_mean):
        dx, dy = offsets.get(lab, (0.35, 0.015))
        ax.text(xv + dx, yv + dy, lab, fontsize=8,
                ha="left", va="center")
    ax.axvline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_xlabel(r"Rej. reduction (pp)")
    ax.set_ylabel(r"Cross-service rate")
    ax.set_xlim(-1.2, max(rej_mean) + 3.3)
    ax.set_ylim(-0.035, max(cross_mean) + 0.075)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(OUT_7C)
    fig.savefig(OUT_7C.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_7C}")

    labels2, comp = compute_policy_compare()
    x2 = np.arange(len(labels2))
    width = 0.34
    fb = style.METHOD_COLORS["fixed_band"]
    offsets = {"FB": -width / 2, "SG": width / 2}
    colors = {"FB": fb, "SG": sg}
    names = {"FB": r"$\pi_{\mathrm{FB}}$", "SG": r"$\pi_{\mathrm{SG}}$"}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.35, 2.55), sharex=True,
        gridspec_kw={"hspace": 0.13, "left": 0.18, "right": 0.98,
                     "top": 0.90, "bottom": 0.13},
    )
    for key in ["FB", "SG"]:
        d = comp[key]["rej"]
        ax1.bar(
            x2 + offsets[key], d["mean"], width=width,
            yerr=[d["lo"], d["hi"]], color=colors[key],
            edgecolor="black", linewidth=0.55,
            error_kw=dict(ecolor="black", capsize=2.2, elinewidth=0.75,
                          capthick=0.75),
            label=names[key], zorder=2,
        )
    ax1.axhline(0, color="black", linewidth=0.75, zorder=1)
    ax1.set_ylabel(r"Rej. drop (pp)")
    ax1.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", frameon=True, ncol=2, fontsize=8,
               handlelength=1.1, columnspacing=0.8, borderpad=0.25)

    for key in ["FB", "SG"]:
        d = comp[key]["cross"]
        ax2.bar(
            x2 + offsets[key], d["mean"], width=width,
            yerr=[d["lo"], d["hi"]], color=colors[key],
            edgecolor="black", linewidth=0.55,
            error_kw=dict(ecolor="black", capsize=2.2, elinewidth=0.75,
                          capthick=0.75),
            zorder=2,
        )
    ax2.set_ylabel(r"Cross-service")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2)
    ax2.tick_params(axis="x", labelsize=7.2)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    fig.savefig(OUT_7D)
    fig.savefig(OUT_7D.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_7D}")

    print("\n--- Caption-ready values ---")
    print(f"{'Regime':<6}{'dRej_pp':>10}{'CrossServe':>12}{'dLoadCV%':>11}")
    for r, dr, cs, dl in zip(labels, rej["mean"], cross["mean"], load["mean"]):
        print(f"{r:<6}{dr:>+10.2f}{cs:>12.4f}{dl:>+11.2f}")


if __name__ == "__main__":
    main()
