"""S1 response-budget-tolerance sensitivity plot.

"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import sigspatial_style as style


ROOT = Path(__file__).resolve().parents[1]

OUT_PAPER = ROOT / "figures" / "fig_rho_tolerance_sweep_v3.pdf"

RUN_DIRS = [
    (0.10, "tu18_sensitivity/tu18_rho10"),
    (0.15, "tu18_sensitivity/tu18_rho15"),
    (0.20, "tu18_sensitivity/tu18_rho20"),
    (0.25, "tu18_sensitivity/tu18_rho25"),
    (0.30, "tu18_sensitivity/tu18_rho30"),
]

POLICIES = [
    ("hold", r"$\pi_{\mathrm{hold}}$", style.METHOD_COLORS["hold"]),
    ("fixed_band", r"$\pi_{\mathrm{FB}}$", style.METHOD_COLORS["fixed_band"]),
    ("safe_greedy", r"$\pi_{\mathrm{SG}}$", style.METHOD_COLORS["safe_greedy"]),
]


def load_rows(subdir: str) -> list[dict]:
    rows: list[dict] = []
    root = ROOT / "runs" / subdir
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def metric(
    rows: list[dict],
    policy: str,
    path: tuple[str, ...],
    scale: float = 1.0,
) -> tuple[float, float]:
    values = []
    for row in rows:
        if row.get("policy") != policy:
            continue
        value = row
        for key in path:
            value = value[key]
        values.append(float(value) * scale)
    if not values:
        raise ValueError(f"missing rows for policy={policy}")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(variance / len(values))


def validate_loaded(loaded: list[tuple[float, list[dict]]]) -> None:
    hashes = {
        row["git_hash"]
        for _, rows in loaded
        for row in rows
    }
    if len(hashes) != 1:
        raise ValueError(f"rho subset is not single-snapshot: {sorted(hashes)}")

    generated_by_seed: dict[int, set[float]] = {}
    for _, rows in loaded:
        for row in rows:
            seed = int(row["seed"])
            generated = round(
                row["results"]["throughput"]["admitted"]
                + row["results"]["throughput"]["rejected"],
                6,
            )
            generated_by_seed.setdefault(seed, set()).add(generated)
    mismatched = {
        seed: sorted(values)
        for seed, values in generated_by_seed.items()
        if len(values) != 1
    }
    if mismatched:
        raise ValueError(f"rho subset is not matched by seed: {mismatched}")


def draw_panel(
    ax,
    loaded,
    path: tuple[str, ...],
    ylabel: str,
    scale: float = 1.0,
    floating_labels: bool = False,
):
    handles, labels = [], []
    x_values = [rho for rho, _ in loaded]
    for policy, label, color in POLICIES:
        means, ci95 = [], []
        for _, rows in loaded:
            mean, interval = metric(rows, policy, path, scale)
            means.append(mean)
            ci95.append(interval)
        handle = ax.errorbar(
            x_values,
            means,
            yerr=ci95,
            marker="o",
            markersize=4.2,
            linewidth=1.35,
            capsize=2.5,
            capthick=0.8,
            elinewidth=0.8,
            color=color,
            label=label,
        )
        handles.append(handle)
        labels.append(label)
        if floating_labels:
            offset = {
                "hold": 6,
                "fixed_band": -8,
                "safe_greedy": 0,
            }[policy]
            ax.annotate(
                label,
                xy=(x_values[-1], means[-1]),
                xytext=(8, offset),
                textcoords="offset points",
                color=color,
                fontsize=7.6,
                va="center",
                ha="left",
                clip_on=False,
            )
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_coords(-0.17, 0.5)  # shared with region-scaling figure
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(x_values)
    return handles, labels


def main() -> None:
    style.apply() 

    loaded = [(rho, load_rows(subdir)) for rho, subdir in RUN_DIRS]
    validate_loaded(loaded)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, 3.0),
        sharex=True,
        gridspec_kw={
            "hspace": 0.18,
            "left": 0.19,
            "right": 0.98,
            "top": 0.88,
            "bottom": 0.15,
        },
    )
    handles, labels = draw_panel(
        axes[0],
        loaded,
        ("results", "throughput", "admitted"),
        "Admitted (k)",
        scale=1.0 / 1000.0,
    )
    draw_panel(
        axes[1],
        loaded,
        ("results", "tail", "p95"),
        "$p_{95}$ (epochs)",
    )
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[1].set_xlabel(r"Response-budget tolerance $\rho$")
    axes[0].set_xlim(0.085, 0.315)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        handlelength=1.2,
        columnspacing=0.8,
        borderpad=0.25,
    )

    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_PAPER)
    fig.savefig(OUT_PAPER.with_suffix(".png"), dpi=300)
    print(f"wrote {OUT}")
    print(f"wrote {OUT_PAPER}")


if __name__ == "__main__":
    main()
