r"""CPAC figures.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
_sys.path.insert(0, str(HERE))
import sigspatial_style as style
from sigspatial_style import METHOD_COLORS, apply as _apply_style

_apply_style()
plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLOR_HOLD = METHOD_COLORS["hold"]
COLOR_FB   = METHOD_COLORS["fixed_band"]
COLOR_SG   = METHOD_COLORS["safe_greedy"]
COLOR_CPAC = METHOD_COLORS["masked_ppo"]

PALETTE = {
    "hold": COLOR_HOLD,
    "fixed_band": COLOR_FB,
    "safe_greedy": COLOR_SG,
    "CPAC": COLOR_CPAC,
}


def load_baseline(cfg: str):
    """Deterministic-policy rollouts for one config: out[policy][seed] = row."""
    out = defaultdict(dict)
    for fp in glob.glob(str(REPO_ROOT / "runs" / "p1" / cfg / "*.jsonl")):
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                if d.get("status") != "OK":
                    continue
                pol = d.get("policy")
                seed = d.get("seed")
                if pol in ("hold", "fixed_band", "safe_greedy") and isinstance(seed, int):
                    out[pol][seed] = d
    return out


def _cpac_eval_files(cfg: str, eval_seed_glob: str):
    return sorted(glob.glob(str(
        REPO_ROOT / "runs" / "cpac" / cfg / "eval_trainseed*" /
        f"run_seed{eval_seed_glob}_*.jsonl"
    )))


def load_cpac(cfg: str):
    """CPAC evaluation rollouts grouped by evaluation seed, across all
    available training seeds."""
    by_eval = defaultdict(list)
    by_eval_p95 = defaultdict(list)
    by_eval_rej = defaultdict(list)
    cert_rt = 0
    n_rollouts = 0
    for es in range(10):
        for fp in _cpac_eval_files(cfg, str(es)):
            with open(fp) as f:
                d = json.loads(f.readline())
            r = d["results"]
            by_eval[es].append(r["throughput"]["admitted"])
            by_eval_p95[es].append(r["tail"]["p95"])
            by_eval_rej[es].append(r["throughput"]["rejected"])
            cert_rt += r["safety"].get("cert", 0)
            n_rollouts += 1
    return by_eval, by_eval_p95, by_eval_rej, cert_rt, n_rollouts


def load_baseline_wcrt_and_rate(cfg: str, pol: str):
    """Returns (wcrt_list, admit_rate_list) for a deterministic policy."""
    wcrt, rate = [], []
    for fp in glob.glob(str(REPO_ROOT / "runs" / "p1" / cfg / "*.jsonl")):
        for line in open(fp):
            d = json.loads(line)
            if d.get("status") != "OK" or d.get("policy") != pol:
                continue
            r = d["results"]["throughput"]
            adm, rej = r["admitted"], r["rejected"]
            wcrt.append(d["results"]["tail"]["max"])
            rate.append(adm / (adm + rej) if (adm + rej) > 0 else 0.0)
    return wcrt, rate


def load_cpac_wcrt_and_rate(cfg: str):
    """Returns (wcrt_list, admit_rate_list) across CPAC rollouts."""
    wcrt, rate = [], []
    for fp in _cpac_eval_files(cfg, "*"):
        d = json.loads(open(fp).readline())
        r = d["results"]["throughput"]
        adm, rej = r["admitted"], r["rejected"]
        wcrt.append(d["results"]["tail"]["max"])
        rate.append(adm / (adm + rej) if (adm + rej) > 0 else 0.0)
    return wcrt, rate


def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boots = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)
    ])
    return np.percentile(boots, 100*alpha/2), np.percentile(boots, 100*(1-alpha/2))



# =============================================================
def figure_flagship_v2():
    cfg = "a4_shifting_slow"
    baselines = load_baseline(cfg)
    cpac_by_eval, cpac_p95_by_eval, cpac_rej_by_eval, cert_rt, n_rollouts = load_cpac(cfg)

    methods = [
        ("hold",        r"$\pi_{\mathrm{hold}}$"),
        ("fixed_band",  r"$\pi_{\mathrm{FB}}$"),
        ("safe_greedy", r"$\pi_{\mathrm{SG}}$"),
        ("CPAC",        r"$\pi^{\mathrm{CPAC}}_\theta$"),
    ]
    metrics = [
        ("rej", "Rejected events"),
        ("adm", "Admitted events"),
        ("p95", r"$p_{95}$ response (epochs)"),
    ]

    def policy_values(pol, which):
        if pol == "CPAC":
            src = {"rej": cpac_rej_by_eval, "adm": cpac_by_eval, "p95": cpac_p95_by_eval}[which]
            return [a for v in src.values() for a in v]
        recs = baselines.get(pol, {})
        key = {"rej": ("throughput", "rejected"),
               "adm": ("throughput", "admitted"),
               "p95": ("tail", "p95")}[which]
        return [r["results"][key[0]][key[1]] for r in recs.values()]


    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.0, 2.6),
        gridspec_kw={"wspace": 0.20, "left": 0.12, "right": 0.99,
                     "top": 0.97, "bottom": 0.20},
    )

    y = list(range(len(methods)))[::-1]   # top = hold, bottom = CPAC

    for k, (ax, (which, xlabel)) in enumerate(zip(axes, metrics)):
        labels, means, elo, ehi, colors = [], [], [], [], []
        for code, lbl in methods:
            vals = policy_values(code, which)
            mu = float(np.mean(vals))
            lo, hi = bootstrap_ci(vals)
            labels.append(lbl); means.append(mu)
            elo.append(mu - lo); ehi.append(hi - mu)
            colors.append(PALETTE[code])
        ax.barh(y, means, height=0.62, color=colors,
                edgecolor="black", linewidth=0.6, zorder=2)
        ax.errorbar(means, y, xerr=[elo, ehi], fmt="none", ecolor="black",
                    capsize=3.5, elinewidth=0.9, capthick=0.9, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if k == 0 else [""] * len(labels))
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", which="both", length=0)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        xmax = max(m + e for m, e in zip(means, ehi))
        ax.set_xlim(0.0, xmax * 1.05)

    out = FIG_DIR / "fig_phase2_flagship_v2.pdf"
    selected = FIG_DIR / "fig_phase2_CPAC_v3.pdf"
    for path in (out, selected):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"[write] {out}  (horizontal 4-policy, N regime, n_rollouts={n_rollouts})")
    print(f"[write] {selected}")
    plt.close(fig)


# =============================================================
# Observed max response per regime and policy, with the design-time
# response reference U_bar as a dashed line.
# =============================================================
def figure_certification_clean():
    UBAR_REF = 28.35   # c=16 reference ceiling used in the experiments

    REGIMES = [
        # (label, config)
        ("S1", "f4i_g4_c16_beta0_true_baselines"),
        ("SB", "a5_4corner_a50"),
        ("N",  "a4_shifting_slow"),
        ("A",  "adv_b_boundary_stress"),
        ("R",  "chicago_real_replay"),
    ]

    METHODS = [
        ("hold",        COLOR_HOLD, r"$\pi_{\mathrm{hold}}$"),
        ("fixed_band",  COLOR_FB,   r"$\pi_{\mathrm{FB}}$"),
        ("safe_greedy", COLOR_SG,   r"$\pi_{\mathrm{SG}}$"),
        ("CPAC",        COLOR_CPAC, r"$\pi^{\mathrm{CPAC}}_\theta$"),
    ]

    labels = []
    means = {code: [] for code, _, _ in METHODS}
    los = {code: [] for code, _, _ in METHODS}
    his = {code: [] for code, _, _ in METHODS}
    n_total = 0
    for label, cfg in REGIMES:
        labels.append(label)
        for code, _, _ in METHODS:
            if code == "CPAC":
                w, _ = load_cpac_wcrt_and_rate(cfg)
            else:
                w, _ = load_baseline_wcrt_and_rate(cfg, code)
            if w:
                mu = float(np.mean(w))
                lo, hi = bootstrap_ci(w)
                n_total += len(w)
            else:
                mu, lo, hi = 0.0, 0.0, 0.0
            means[code].append(mu)
            los[code].append(mu - lo)
            his[code].append(hi - mu)

    style.apply()
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6.8,
    })

    n = len(labels)
    fig, ax = plt.subplots(
        1, 1, figsize=(3.35, 2.35),
        gridspec_kw={"left": 0.18, "right": 0.99,
                     "top": 0.80, "bottom": 0.18},
    )
    bar_w = 0.18
    x = np.arange(n)

    for j, (code, color, lbl) in enumerate(METHODS):
        offset = (j - 1.5) * bar_w
        ax.bar(
            x + offset, means[code], width=bar_w * 0.9,
            yerr=[los[code], his[code]], label=lbl,
            color=color, edgecolor="black", linewidth=0.35,
            error_kw=dict(ecolor="black", capsize=1.8, elinewidth=0.7,
                          capthick=0.7),
            zorder=2,
        )

    ax.axhline(UBAR_REF, color="black", linestyle="--", linewidth=1.0,
               label=r"$\overline{U}$", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel("Max response (epochs)")
    ax.set_ylim(0, UBAR_REF * 1.14)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16),
              ncol=5, frameon=True, framealpha=0.95,
              handlelength=0.9, handletextpad=0.28,
              columnspacing=0.60, borderpad=0.22)

    out = FIG_DIR / "fig_phase2_certification_v6.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"[write] {out}  clean grouped WCRT, n_wcrt_samples={n_total}")
    plt.close(fig)


if __name__ == "__main__":
    figure_flagship_v2()
    figure_certification_clean()
    print(f"\nPhase-2 v2 figures written to {FIG_DIR}")
