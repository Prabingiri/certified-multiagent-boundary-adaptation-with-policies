"""Aggregate deterministic-policy rollouts into the Phase-1 summary files.

Walks all runs/* JSONLs, extracts the metrics
and produces:

  runs/_master_phase1.csv  
  runs/_master_phase1.json     

Bootstrap 95% CIs use B=1000 resamples per group. Wilcoxon paired
signed-rank test is computed for each (regime, policy_pair) on the
seed-by-seed deltas.

Usage:
    python scripts/phase1_master_aggregate.py
"""
from __future__ import annotations

import os
import json
import csv
import math
import statistics
from typing import Any, Dict, List, Tuple

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")
OUT_CSV = os.path.join(RUNS_DIR, "_master_phase1.csv")
OUT_JSON = os.path.join(RUNS_DIR, "_master_phase1.json")


def _fmean(xs):
    """statistics.fmean fallback for Python 3.7 base env."""
    if hasattr(statistics, "fmean"):
        return statistics.fmean(xs)
    return statistics.mean(xs)


REGIME_TAXONOMY = [
    # label, tier, [path_substrings], [allowed_config_values]

    ("A1_uniform",          1, ["p1/exp00_g4_c8/"],
                              ["exp00_g4_c8"]),
    ("A2_single_hotspot",   1, ["p1/f4i_g4_c16_beta0_true_baselines/"],
                              ["f4i_g4_c16_beta0_true_baselines"]),
    ("A5_2corner",          1, ["p1/a5_2corner_a50/"],
                              ["a5_2corner_a50"]),
    ("A5_2corner_heavy",    1, ["p1/a5_2corner_a100/"],
                              ["a5_2corner_a100"]),
    ("A5_2edge",            1, ["p1/a5_2edge_a50/"],
                              ["a5_2edge_a50"]),
    ("A5_4corner",          1, ["p1/a5_4corner_a50/"],
                              ["a5_4corner_a50"]),
    ("A5_4corner_g6",       1, ["p1/a5_4corner_g6/"],
                              ["a5_4corner_g6"]),
    ("A5_4corner_g8",       1, ["p1/a5_4corner_g8/"],
                              ["a5_4corner_g8"]),
    ("A5_4corner_g10",      1, ["p1/a5_4corner_g10/"],
                              ["a5_4corner_g10"]),
    ("A5_4corner_g12",      1, ["p1/a5_4corner_g12/"],
                              ["a5_4corner_g12"]),
    ("A4_shifting_slow",    2, ["p1/a4_shifting_slow/"],
                              ["a4_shifting_slow"]),
    ("A4_shifting_medium",  2, ["p1/a4_shifting_medium/"],
                              ["a4_shifting_medium"]),
    ("A4_shifting_fast",    2, ["p1/a4_shifting_fast/"],
                              ["a4_shifting_fast"]),
    ("A_adv_boundary",      2, ["p1/adv_b_boundary_stress/"],
                              ["adv_b_boundary_stress"]),
    ("B1_chicago_911",      3, ["p1/chicago_real_replay/"],
                              ["chicago_real_replay"]),
]


# Deterministic shielded policies.
PHASE1_POLICIES = ("hold", "fixed_band", "safe_greedy")


# ---- metric extraction ------------------------------------------------------

def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def extract_metrics(row: Dict) -> Dict[str, float]:
    """Pull the reported metrics from one rollout row."""
    res = row.get("results", {}) or {}
    prs = row.get("per_region_stats", {}) or {}
    admitted = safe_get(res, "throughput", "admitted", default=0)
    rejected = safe_get(res, "throughput", "rejected", default=0)
    # Cross-service counts events delivered through a certified interface
    # band (the buffer-admission path); pi_hold has none by construction.
    cross = safe_get(res, "throughput", "buffer_admitted_total", default=0)
    arrivals = float(admitted) + float(rejected)
    cross_serve_rate = float(cross) / float(admitted) if admitted else 0.0
    rejection_rate = float(prs.get("rejection_count", rejected)) / arrivals if arrivals else 0.0
    admit_rate = float(admitted) / arrivals if arrivals else 0.0
    return {
        "mean": safe_get(res, "tail", "mean", default=float("nan")),
        "p95": safe_get(res, "tail", "p95", default=float("nan")),
        "max": safe_get(res, "tail", "max", default=float("nan")),
        "admitted": admitted,
        "completed": safe_get(res, "throughput", "completed", default=0),
        "rejected": rejected,
        "cross_adm": cross,
        "buffer_adm": safe_get(res, "throughput", "buffer_admitted_total", default=0),
        "same_owner_buffer_adm": safe_get(res, "throughput", "same_owner_buffer_admitted_total", default=0),
        "buffer_rej": safe_get(res, "throughput", "buffer_rejected_total", default=0),
        "viol_total": safe_get(res, "safety", "total", default=0),
        "viol_cert": safe_get(res, "safety", "cert", default=0),
        "viol_geom": safe_get(res, "safety", "geom", default=0),
        "viol_ker": safe_get(res, "safety", "ker", default=0),
        "viol_srv": safe_get(res, "safety", "srv", default=0),
        "viol_team": safe_get(res, "safety", "team", default=0),
        "coef_var": safe_get(res, "imbalance", "mean_coef_var", default=float("nan")),
        "cross_serve_rate": cross_serve_rate,
        "rejection_rate": rejection_rate,
        "admit_rate": admit_rate,
        "rejection": safe_get(prs, "rejection_count", default=rejected),
        "wcrt_per_region_max": safe_get(prs, "max_of_per_region_max", default=safe_get(prs, "global_max_T", default=float("nan"))),
        "wcrt_per_region_mean": safe_get(prs, "mean_of_per_region_max", default=float("nan")),
        "global_max_T": safe_get(prs, "global_max_T", default=float("nan")),
    }


# ---- bootstrap + Wilcoxon ---------------------------------------------------

def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05,
                 rng_seed: int = 0) -> Tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi). Skips NaN."""
    import random
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    rng = random.Random(rng_seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(_fmean(sample))
    means.sort()
    lo = means[int(n_boot * (alpha / 2))]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (_fmean(vals), lo, hi)


def wilcoxon_paired(a: List[float], b: List[float]) -> float:
    """Wilcoxon paired signed-rank test. Returns approximate two-sided p-value
    using the normal approximation. Returns NaN if not enough samples.
    """
    if len(a) != len(b) or len(a) < 6:
        return float("nan")
    diffs = [(ai - bi) for ai, bi in zip(a, b) if not math.isnan(ai) and not math.isnan(bi)]
    diffs = [d for d in diffs if d != 0.0]
    n = len(diffs)
    if n < 6:
        return float("nan")
    abs_diffs = sorted(((abs(d), 1 if d > 0 else -1) for d in diffs), key=lambda x: x[0])
    # rank with average for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    W_plus = sum(r for r, (_, s) in zip(ranks, abs_diffs) if s > 0)
    W_minus = sum(r for r, (_, s) in zip(ranks, abs_diffs) if s < 0)
    W = min(W_plus, W_minus)
    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24
    z = (W - mean_W) / math.sqrt(var_W)
    # two-sided normal p
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p


# ---- aggregation ------------------------------------------------------------

def collect_rows(regime_label: str, prefixes: List[str], allowed_configs: List[str]) -> List[Dict]:
    """Walk runs/ collecting rows matching path prefixes AND (if allowed_configs
    is non-empty) the row's config field. Dedup by (policy, seed) preferring
    rows with travel data."""
    candidates: Dict[Tuple[str, Any], List[Dict]] = {}
    for root, _, files in os.walk(RUNS_DIR):
        rel_root = os.path.relpath(root, RUNS_DIR).replace(os.sep, "/") + "/"
        path_matches = any(p in rel_root for p in prefixes)
        config_path_matches = any(ac in rel_root for ac in allowed_configs)
        if prefixes and allowed_configs and not (path_matches or config_path_matches):
            continue
        if prefixes and not allowed_configs and not path_matches:
            continue
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            try:
                with open(os.path.join(root, fn)) as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        pol = r.get("policy")
                        if pol not in PHASE1_POLICIES:
                            continue
                        cfg = r.get("config", "") or ""
                        # If allowed_configs is provided, require row config to
                        # match (substring or exact). This is what filters
                        # bundled multi-regime JSONLs like p1_travel_pt2/.
                        if allowed_configs:
                            if not any(ac == cfg or ac in cfg for ac in allowed_configs):
                                continue
                        seed = r.get("seed")
                        m = extract_metrics(r)
                        m["regime"] = regime_label
                        m["policy"] = pol
                        m["seed"] = seed
                        m["source_file"] = os.path.join(rel_root, fn)
                        m["_cfg"] = cfg
                        candidates.setdefault((pol, seed), []).append(m)
            except Exception:
                continue
    # Dedup: prefer rows with the reported per-region response witness.
    # Ties are broken by source-file string to keep the choice deterministic.
    def _score(r):
        s = 0
        gm = r.get("global_max_T")
        if isinstance(gm, (int, float)) and not math.isnan(float(gm)):
            s += 1
        return s
    out = []
    for _key, rows in candidates.items():
        rows.sort(key=lambda r: (_score(r), r.get("source_file", "")), reverse=True)
        out.append(rows[0])
    return out


def aggregate():
    all_rows: List[Dict] = []
    for label, tier, prefixes, allowed_cfgs in REGIME_TAXONOMY:
        rows = collect_rows(label, prefixes, allowed_cfgs or [])
        all_rows.extend(rows)

    # ----- write CSV (raw rows) -----
    metric_keys = [
        "regime", "policy", "seed",
        "rejection", "admitted", "completed", "rejected", "cross_adm",
        "buffer_adm", "same_owner_buffer_adm", "buffer_rej",
        "mean", "p95", "max",
        "wcrt_per_region_max", "wcrt_per_region_mean", "global_max_T",
        "viol_total", "viol_cert", "viol_geom", "viol_ker", "viol_srv", "viol_team",
        "coef_var", "cross_serve_rate", "rejection_rate", "admit_rate",
        "source_file",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metric_keys, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"wrote {OUT_CSV}  ({len(all_rows)} rows)")

    # ----- aggregate by (regime, policy) -----
    agg = {}  # (regime, policy) -> dict of metric -> (mean, lo, hi)
    metrics_for_agg = [
        "rejection", "admitted", "rejected", "mean", "p95", "max",
        "wcrt_per_region_max", "global_max_T", "viol_total", "coef_var",
        "cross_serve_rate", "rejection_rate", "admit_rate",
    ]
    for (label, tier, _, _) in REGIME_TAXONOMY:
        for pol in PHASE1_POLICIES:
            sub = [r for r in all_rows if r["regime"] == label and r["policy"] == pol]
            if not sub:
                continue
            entry = {"n_seeds": len(sub), "tier": tier}
            for m in metrics_for_agg:
                vals = [r[m] for r in sub if r.get(m) is not None]
                mu, lo, hi = bootstrap_ci(vals)
                entry[m] = {"mean": mu, "ci_lo": lo, "ci_hi": hi}
            agg[(label, pol)] = entry

    # ----- Wilcoxon SG vs hold per regime -----
    wilcox = {}
    for (label, tier, _, _) in REGIME_TAXONOMY:
        sub_h = sorted([r for r in all_rows if r["regime"] == label and r["policy"] == "hold"],
                       key=lambda r: r["seed"])
        sub_s = sorted([r for r in all_rows if r["regime"] == label and r["policy"] == "safe_greedy"],
                       key=lambda r: r["seed"])
        common = sorted(set(r["seed"] for r in sub_h) & set(r["seed"] for r in sub_s))
        if len(common) < 6:
            continue
        h_by_seed = {r["seed"]: r for r in sub_h}
        s_by_seed = {r["seed"]: r for r in sub_s}
        regime_w = {}
        for m in ["rejection", "p95", "max", "wcrt_per_region_max", "admitted"]:
            a = [h_by_seed[s][m] for s in common if h_by_seed[s].get(m) is not None]
            b = [s_by_seed[s][m] for s in common if s_by_seed[s].get(m) is not None]
            if len(a) == len(b) and len(a) >= 6:
                regime_w[m] = {"p_value": wilcoxon_paired(a, b),
                               "n_pairs": len(a)}
        wilcox[label] = regime_w

    out = {
        "regimes": [{"label": l, "tier": t} for (l, t, _, _) in REGIME_TAXONOMY],
        "by_regime_policy": {f"{l}|{p}": v for (l, p), v in agg.items()},
        "wilcoxon_sg_vs_hold": wilcox,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {OUT_JSON}")


    # ----- summary console -----
    print("\n" + "=" * 100)
    print(f"{'regime':<22s}  {'grp':<4s}  {'pol':<12s}  {'n':<3s}  {'rejection':>9s}  {'p95':>8s}  {'admit':>7s}  {'viol':>5s}  {'travel':>9s}  {'tr/evt':>7s}")
    print("=" * 100)
    for (label, tier, _, _) in REGIME_TAXONOMY:
        for pol in PHASE1_POLICIES:
            entry = agg.get((label, pol))
            if not entry:
                continue
            def fmt(m, w=6, dec=1):
                v = entry[m]["mean"] if m in entry else float("nan")
                if isinstance(v, float) and math.isnan(v):
                    return f"{'-':>{w}s}"
                return f"{v:>{w}.{dec}f}"
            print(f"{label:<22s}  G{tier:<3d}  {pol:<12s}  {entry['n_seeds']:<3d}  "
                  f"{fmt('rejection', 9, 1)}  {fmt('p95', 8, 2)}  {fmt('admitted', 7, 0)}  "
                  f"{fmt('viol_total', 5, 0)}")


if __name__ == "__main__":
    aggregate()
