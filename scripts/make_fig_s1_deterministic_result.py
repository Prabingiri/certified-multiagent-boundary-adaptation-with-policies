"""Generate the S1 deterministic-policy comparison figure.

Three horizontal bar panels: rejected-event count, admitted-event count,
conditional p_{95} response on completed events. One bar per policy
with bootstrap 95% CI whiskers.

Counts (not rates) are reported in the figure because the rollout
protocol is fixed (matched seeds, identical arrival streams ~3000
events per rollout) and operational magnitude is more interpretable
than a rate in [0,1] under that protocol. Rates remain the reported
quantity in the metrics paragraph and cross-regime scorecard,
where arrival counts vary by regime.

Conventions:
  - No embedded title in the figure (caption carries the narrative).
  - No significance asterisks on bars (CI bars and caption text carry it).
  - x-axis labels are plain operational quantities.
  - Method colors via sigspatial_style.METHOD_COLORS (Wong palette).

Reads runs/_master_phase1.csv directly (authoritative data source).
Recomputes bootstrap CIs and paired Wilcoxon, prints caption-ready
stats to stdout.

Output: figures/fig_s1_deterministic_result.pdf
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import sigspatial_style as style

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "runs", "_master_phase1.csv")
OUT = os.path.join(ROOT, "figures", "fig_s1_deterministic_result.pdf")
REGIME = "A2_single_hotspot"   # internal CSV label; paper label is S1
RNG_SEED = 0
B = 1000


def boot_ci(x: np.ndarray, B: int = 1000, alpha: float = 0.05,
            rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Bootstrap percentile 95% CI on the mean."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = x[idx].mean(axis=1)
    lo = float(np.percentile(bs, 100 * alpha / 2))
    hi = float(np.percentile(bs, 100 * (1 - alpha / 2)))
    return lo, hi


def main() -> None:
    style.apply()
    rng = np.random.default_rng(RNG_SEED)

    df = pd.read_csv(CSV)
    sub = df[df["regime"] == REGIME].copy()
    if len(sub) == 0:
        raise SystemExit(f"No rows for regime {REGIME!r} in {CSV}")

    # Methods in plotting order (top -> bottom on the bar charts).
    methods = [
        ("hold",        r"$\pi_{\mathrm{hold}}$"),
        ("fixed_band",  r"$\pi_{\mathrm{FB}}$"),
        ("safe_greedy", r"$\pi_{\mathrm{SG}}$"),
    ]
    # Metric columns and their published axis labels.
    # Counts (rejected, admitted) for operational magnitude; epochs for p_{95}.
    # `rejected` is the base-env canonical field for events refused by
    # CS-LSTF; it equals the `rejection` count to within rounding
    # (corr 1.0) and matches the paper's rejection-rate definition (sigma,
    # Section 7.1). We plot/label it as "Rejected" so the figure vocabulary
    # is consistent with the text and equations (no undefined "rejection").
    metrics = [
        ("rejected",   r"Rejected events"),
        ("admitted",   r"Admitted events"),
        ("p95",        r"Conditional $p_{95}$ (epochs)"),
    ]

    # Verify expected sample size per policy.
    n_per = sub.groupby("policy").size().to_dict()
    for code, _ in methods:
        if n_per.get(code, 0) != 10:
            print(f"  WARN: policy {code!r} has n={n_per.get(code,0)} (expected 10)")

    # Compute means, bootstrap CIs, and paired Wilcoxon vs hold.
    by_policy = {code: sub[sub["policy"] == code].sort_values("seed").reset_index(drop=True)
                 for code, _ in methods}
    hold_df = by_policy["hold"]

    stats_table: dict[tuple[str, str], dict[str, float]] = {}
    for code, _ in methods:
        for col, _ in metrics:
            vals = by_policy[code][col].values
            mu = float(np.mean(vals))
            lo, hi = boot_ci(vals, B=B, rng=rng)
            entry = {"mean": mu, "lo": lo, "hi": hi}
            if code != "hold":
                w = stats.wilcoxon(by_policy[code][col].values,
                                   hold_df[col].values,
                                   zero_method="wilcox")
                entry["p"] = float(w.pvalue)
            stats_table[(code, col)] = entry

    # Three-panel full-width figure (include via figure* / \textwidth).
    # method labels appear only on the leftmost panel; the others share
    # the same row order with empty y-ticks. Locked sigspatial_style fonts
    # (no per-script overrides) keep the whole figure set consistent.
    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.0, 2.6),
        gridspec_kw={"wspace": 0.20,
                     "left": 0.12,
                     "right": 0.99,
                     "top": 0.97,
                     "bottom": 0.20},
    )

    for k, (ax, (col, xlabel)) in enumerate(zip(axes, metrics)):
        labels, means, errs_lo, errs_hi, colors = [], [], [], [], []
        for code, lbl in methods:
            entry = stats_table[(code, col)]
            mu = entry["mean"]
            labels.append(lbl)
            means.append(mu)
            errs_lo.append(mu - entry["lo"])
            errs_hi.append(entry["hi"] - mu)
            colors.append(style.METHOD_COLORS[code])

        # y positions: top = hold, bottom = SG (visual order matches `methods`).
        y = list(range(len(labels)))[::-1]
        ax.barh(y, means, height=0.62,
                color=colors, edgecolor="black", linewidth=0.6, zorder=2)
        ax.errorbar(means, y, xerr=[errs_lo, errs_hi],
                    fmt="none", ecolor="black", capsize=3.5,
                    elinewidth=0.9, capthick=0.9, zorder=3)
        ax.set_yticks(y)
        # Show method labels on the leftmost panel only; others share
        # the same row order silently.
        if k == 0:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([""] * len(labels))
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", which="both", length=0)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        # x-limits: counts auto-scaled with a 5% headroom on the right
        # so error bars and tick labels do not collide with the spine.
        xmax = max(m + e for m, e in zip(means, errs_hi))
        ax.set_xlim(0.0, xmax * 1.05)

    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=300)
    print(f"wrote {OUT}")

    # Caption-ready stats.
    print("\n--- Caption-ready statistics (regime S1, n=10 matched seeds) ---")
    for code, lbl in methods:
        for col, _ in metrics:
            e = stats_table[(code, col)]
            line = (f"  {lbl:30s} {col:12s} "
                    f"mean={e['mean']:.4f}  "
                    f"95% CI=[{e['lo']:.4f}, {e['hi']:.4f}]")
            if "p" in e:
                line += f"  paired Wilcoxon p={e['p']:.4f}"
            print(line)


if __name__ == "__main__":
    main()
