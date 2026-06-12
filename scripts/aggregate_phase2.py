r"""Aggregate CPAC eval rollouts into a master CSV.

Walks the CPAC output tree (runs/cpac/<config>/eval*/*.jsonl) and emits a
single CSV with the same metric columns as runs/_master_phase1.csv.

Per-row schema mirrors phase1_master_aggregate.extract_metrics() so the
deterministic-policy figure scripts can render CPAC data from this CSV.

Usage
-----
    python scripts/aggregate_phase2.py
    python scripts/aggregate_phase2.py --root runs/cpac \\
                                       --out runs/_master_phase2.csv

Output
------
    runs/_master_phase2.csv      one row per (paper_regime, policy, seed)
    runs/_master_phase2.json     summary (paired vs deterministic SG)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Re-use Phase-1 metric extraction helpers.
sys.path.insert(0, str(HERE))
from phase1_master_aggregate import (   # noqa: E402
    bootstrap_ci, extract_metrics, safe_get, wilcoxon_paired,
)


def _fmean(xs):
    """statistics.fmean fallback for Python 3.7 base env."""
    if hasattr(statistics, "fmean"):
        return statistics.fmean(xs)
    return statistics.mean(xs)


def collect_phase2_rows(root: Path) -> List[Dict]:

    rows: List[Dict] = []
    if not root.exists():
        return rows
    # Find all `eval` directories under the root, at any depth up to 2.
    eval_dirs: list[tuple[Path, str, str]] = []
    for d in sorted(root.glob("**/eval*")):
        if not d.is_dir():
            continue
        regime_dir = d.parent
        regime = regime_dir.name
        # variant is the grandparent IFF it isn't `root` itself.
        variant_dir = regime_dir.parent
        variant = variant_dir.name if variant_dir != root else ""
        eval_dirs.append((d, regime, variant))
    for eval_dir, regime, variant in eval_dirs:
        for jsonl in sorted(eval_dir.glob("*.jsonl")):
            try:
                with open(jsonl) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        m = extract_metrics(rec)

                        m["regime"] = rec.get("config", regime)
                        m["variant"] = variant
                        m["policy"] = rec.get("policy",
                                              variant or "masked_ppo")
                        m["seed"] = rec.get("seed")
                        try:
                            m["source_file"] = str(
                                jsonl.resolve().relative_to(REPO_ROOT)
                            )
                        except ValueError:
                            m["source_file"] = str(jsonl.resolve())
                        rows.append(m)
            except Exception as e:
                print(f"  WARN: failed to parse {jsonl}: {e}",
                      file=sys.stderr)
    return rows


def apply_variant_aliases(rows: List[Dict], aliases: Dict[str, str]) -> None:
    """Apply paper-facing aliases without mutating run directories."""
    if not aliases:
        return
    for row in rows:
        variant = row.get("variant", "")
        policy = row.get("policy", "")
        if variant in aliases:
            row["variant"] = aliases[variant]
        if policy in aliases:
            row["policy"] = aliases[policy]


def write_csv(rows: List[Dict], out_path: Path) -> None:
    metric_keys = [
        "regime", "variant", "policy", "seed",
        "rejection", "admitted", "completed", "rejected", "cross_adm",
        "buffer_adm", "same_owner_buffer_adm", "buffer_rej",
        "mean", "p95", "max",
        "wcrt_per_region_max", "wcrt_per_region_mean", "global_max_T",
        "viol_total", "viol_cert", "viol_geom", "viol_ker", "viol_srv", "viol_team",
        "coef_var", "cross_serve_rate", "rejection_rate", "admit_rate",
        "source_file",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metric_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_phase1_sg_by_regime(phase1_csv: Path,
                              regime_alias_map: Dict[str, str]
                              ) -> Dict[str, Dict[int, Dict[str, float]]]:

    if not phase1_csv.exists():
        return {}
    by_regime: Dict[str, Dict[int, Dict[str, float]]] = {}
    with open(phase1_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("policy") != "safe_greedy":
                continue
            csv_regime = row.get("regime", "")
            paper_regime = regime_alias_map.get(csv_regime)
            if paper_regime is None:
                continue
            try:
                seed = int(row.get("seed") or -1)
            except Exception:
                continue
            if seed < 0:
                continue
            d = {}
            for k, v in row.items():
                if k in ("regime", "policy", "seed", "source_file"):
                    continue
                try:
                    d[k] = float(v) if v not in (None, "") else float("nan")
                except Exception:
                    d[k] = float("nan")
            by_regime.setdefault(paper_regime, {})[seed] = d
    return by_regime


def _load_phase1_all_policies_by_regime(
    phase1_csv: Path,
    regime_alias_map: Dict[str, str],
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:

    if not phase1_csv.exists():
        return {}
    target_policies = {"hold", "fixed_band", "safe_greedy"}
    out: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    with open(phase1_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy = row.get("policy", "")
            if policy not in target_policies:
                continue
            csv_regime = row.get("regime", "")
            paper_regime = regime_alias_map.get(csv_regime)
            if paper_regime is None:
                continue
            try:
                seed = int(row.get("seed") or -1)
            except Exception:
                continue
            if seed < 0:
                continue
            d = {}
            for k, v in row.items():
                if k in ("regime", "policy", "seed", "source_file"):
                    continue
                try:
                    d[k] = float(v) if v not in (None, "") else float("nan")
                except Exception:
                    d[k] = float("nan")
            out.setdefault(paper_regime, {}) \
               .setdefault(policy, {})[seed] = d
    return out


def aggregate_summary(phase2_rows: List[Dict],
                      phase1_sg_by_regime: Dict[str, Dict[int, Dict]],
                      ) -> Dict:
    """Compute per-regime mean + bootstrap CI for CPAC plus paired
    Wilcoxon comparison against Phase-1 SG on the same (regime, seed)."""
    metrics_to_agg = [
        "rejection_rate", "admit_rate", "admitted", "p95", "max",
        "wcrt_per_region_max", "cross_serve_rate", "coef_var", "viol_total",
    ]
    by_regime: Dict[str, Dict] = {}
    regimes = sorted({r["regime"] for r in phase2_rows})
    for regime in regimes:
        sub = [r for r in phase2_rows if r["regime"] == regime]
        entry = {"n_seeds": len({r["seed"] for r in sub})}
        for m in metrics_to_agg:
            vals = [r[m] for r in sub if r.get(m) is not None]
            mu, lo, hi = bootstrap_ci(vals)
            entry[m] = {"mean": mu, "ci_lo": lo, "ci_hi": hi}
        # Paired comparison vs Phase-1 SG on same seeds.
        sg_seed_map = phase1_sg_by_regime.get(regime, {})
        wilcox: Dict[str, Dict] = {}
        for m in metrics_to_agg:
            paired_cpac: List[float] = []
            paired_sg: List[float] = []
            for r in sub:
                seed = r.get("seed")
                if seed not in sg_seed_map:
                    continue
                v_cpac = r.get(m)
                v_sg = sg_seed_map[seed].get(m)
                if v_cpac is None or v_sg is None:
                    continue
                if (isinstance(v_cpac, float) and math.isnan(v_cpac)) \
                   or (isinstance(v_sg, float) and math.isnan(v_sg)):
                    continue
                paired_cpac.append(float(v_cpac))
                paired_sg.append(float(v_sg))
            if len(paired_cpac) >= 6:
                p = wilcoxon_paired(paired_cpac, paired_sg)
                wilcox[m] = {
                    "p_value": p, "n_pairs": len(paired_cpac),
                    "mean_cpac": _fmean(paired_cpac),
                    "mean_sg": _fmean(paired_sg),
                    "delta": (_fmean(paired_cpac)
                              - _fmean(paired_sg)),
                }
        entry["wilcoxon_vs_sg"] = wilcox
        by_regime[regime] = entry
    return {"by_regime": by_regime,
            "n_total_rows": len(phase2_rows)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", nargs="+", default=["runs/cpac"],
                   help="one or more Phase-2 output roots containing eval jsonl files")
    p.add_argument("--out", default="runs/_master_phase2.csv",
                   help="output CSV path")
    p.add_argument("--phase1-csv", default="runs/_master_phase1.csv",
                   help="Phase-1 master CSV for paired SG comparison")
    p.add_argument("--variant-alias", action="append", default=[],
                   metavar="OLD=NEW",
                   help="rename a variant/policy label in the emitted CSV; "
                        "repeatable, e.g. old_label=new_label")
    p.add_argument("--no-summary", action="store_true",
                   help="write only the rollout CSV and lightweight JSON metadata")
    args = p.parse_args()


    roots = [Path(r).resolve() for r in args.root]
    out_csv = Path(args.out).resolve()
    out_json = out_csv.with_suffix(".json")

    aliases: Dict[str, str] = {}
    for item in args.variant_alias:
        if "=" not in item:
            print(f"ERROR: --variant-alias must be OLD=NEW, got {item!r}",
                  file=sys.stderr)
            return 2
        old, new = item.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            print(f"ERROR: --variant-alias must be OLD=NEW, got {item!r}",
                  file=sys.stderr)
            return 2
        aliases[old] = new

    rows: List[Dict] = []
    for root in roots:
        root_rows = collect_phase2_rows(root)
        print(f"collected {len(root_rows)} Phase-2 rollouts from {root}")
        rows.extend(root_rows)

    config_to_regime = {
        "exp00_g4_c8": "U",
        "f4i_g4_c16_beta0_true_baselines": "S1",
        "a5_4corner_a50": "SB",
        "a4_shifting_slow": "N",
        "adv_b_boundary_stress": "A",
        "chicago_real_replay": "R",
    }
    for r in rows:
        r["regime"] = config_to_regime.get(r.get("regime"), r.get("regime"))
    apply_variant_aliases(rows, aliases)
    write_csv(rows, out_csv)
    print(f"wrote {out_csv}")

    if args.no_summary:
        with open(out_json, "w") as f:
            json.dump({
                "n_total_rows": len(rows),
                "roots": [str(r) for r in roots],
                "variant_aliases": aliases,
            }, f, indent=2)
        print(f"wrote {out_json}")
        return 0

    alias_map = {
        # paper_regime  ->  Phase-1 CSV regime (multiple may map to one paper)
        "A2_single_hotspot":   "S1",
        "A5_2corner":          "SB",
        "A5_2corner_heavy":    "SB",
        "A5_2edge":            "SB",
        "A5_4corner":          "SB",
        "A4_shifting_medium":  "N",
        "A4_shifting_fast":    "N",
        "A4_shifting_slow":    "N",
        "A_adv_boundary":      "A",
        "B1_chicago_911":      "R",
    }
    phase1_csv = Path(args.phase1_csv)
    sg_map = _load_phase1_sg_by_regime(phase1_csv, alias_map)
    summary = aggregate_summary(rows, sg_map)

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {out_json}")

    # ---- console summary ----
    print()
    print("Phase-2 summary (CPAC vs Phase-1 SG, paired Wilcoxon):")
    for regime, entry in summary["by_regime"].items():
        n = entry.get("n_seeds", 0)
        rejection = entry.get("rejection_rate", {}).get("mean", float("nan"))
        admit = entry.get("admit_rate", {}).get("mean", float("nan"))
        wilc_rejection = entry.get("wilcoxon_vs_sg", {}).get("rejection_rate", {})
        delta_rejection = wilc_rejection.get("delta", float("nan"))
        p_rejection = wilc_rejection.get("p_value", float("nan"))
        print(f"  {regime:<5} n={n:2d} CPAC rejection={rejection:.3f} admit={admit:.3f} "
              f"vs SG: d_rejection={delta_rejection:+.4f} p={p_rejection:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
