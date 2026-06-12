"""dispersion figure for the hotspot-density sweep.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib.pyplot as plt

import sigspatial_style as style

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures", "fig_hotspot_dispersion.pdf")
G2 = 64
HS = [1, 5, 10, 15, 20, 32]   # include the 50%-coverage endpoint

POLICIES = [
    ("hold", r"$\pi_{\mathrm{hold}}$", style.METHOD_COLORS["hold"]),
    ("fixed_band", r"$\pi_{\mathrm{FB}}$", style.METHOD_COLORS["fixed_band"]),
    ("safe_greedy", r"$\pi_{\mathrm{SG}}$", style.METHOD_COLORS["safe_greedy"]),
]


def load(h):
    d = os.path.join(ROOT, "runs", "p1_hotspot_density", f"sb_density_h{h}")
    rows = []
    for n in os.listdir(d):
        if n.endswith(".jsonl"):
            for line in open(os.path.join(d, n), encoding="utf-8"):
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def cv_per_seed(rows, policy):
    out = {}
    for r in rows:
        if r.get("policy") == policy:
            out[r["seed"]] = r["results"]["imbalance"]["mean_coef_var"]
    return [out[s] for s in sorted(out)]


def ci(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def main():
    style.apply()
    avail = [h for h in HS
             if os.path.isdir(os.path.join(ROOT, "runs", "p1_hotspot_density",
                                            f"sb_density_h{h}"))]
    data = {h: load(h) for h in avail}
    x = list(range(len(avail)))   # even categorical spacing (matches density fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.4),
                           gridspec_kw={"left": 0.16, "right": 0.97,
                                        "top": 0.86, "bottom": 0.21})
    series = {}
    for policy, label, color in POLICIES:
        m = [ci(cv_per_seed(data[h], policy)) for h in avail]
        series[policy] = [v[0] for v in m]
        ax.errorbar(x, [v[0] for v in m], yerr=[v[1] for v in m], marker="o",
                    ms=4.2, lw=1.35, color=color, capsize=2.5, capthick=0.8,
                    elinewidth=0.8, label=label)
    ax.set_ylabel("Load-signal CV")
    ax.set_xlabel("Hotspot count")
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in avail])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
              frameon=True, framealpha=0.95, handlelength=1.2,
              columnspacing=0.8, borderpad=0.25)

    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=300)
    print(f"wrote {OUT}")
    print(f"{'h':>4}{'cov%':>6}{'hold_CV':>9}{'SG_CV':>8}{'reduce%':>9}")
    for i, h in enumerate(avail):
        hd, sg = series["hold"][i], series["safe_greedy"][i]
        print(f"{h:>4}{100*h/G2:>6.1f}{hd:>9.3f}{sg:>8.3f}{100*(hd-sg)/hd:>+8.1f}%")


if __name__ == "__main__":
    main()
