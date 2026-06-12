# Reproducing the Results

This repository provides code and configurations for the reported experiments.
Run outputs and datasets are not committed. Commands below assume `pip install -e .`
from the repository root.

## 1. One-time setup

Synthetic-regime sweep configs:

```bash
python scripts/make_a5_4corner_scale_configs.py
python scripts/make_hotspot_density_configs.py
```

Chicago data. Regime R uses the first week of January 2023:

```bash
python scripts/fetch_chicago_data.py
```

The load-sensitivity study replays twelve weekly traces (first week of each
2023 month) through a 4-district subset at four event scales:

```bash
for MM in 01 02 03 04 05 06 07 08 09 10 11 12; do
  python scripts/fetch_chicago_data.py --start 2023-$MM-01 \
      --out data/chicago_911_week_2023-$MM.json
done
python scripts/prep_chicago_loop4.py --batch
python scripts/scale_chicago_loop4.py
```

All data files are written under `data/`. The derivation scripts are seeded
and deterministic given the fetched inputs.

## 2. Deterministic policies

Run one configuration across matched seeds:

```bash
python scripts/run_experiment.py --configs CONFIG_NAME \
    --policies hold fixed_band safe_greedy \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir runs/p1/CONFIG_NAME
```

Then aggregate:

```bash
python scripts/phase1_master_aggregate.py
```

The aggregator scans the `runs/p1/<config>` directories written by the
command above and produces `runs/_master_phase1.csv`, which the regime
scorecard and region-scaling figures read.

### Regime-to-config map

| Regime | Config |
|---|---|
| U, uniform | `exp00_g4_c8` |
| S1, single hotspot | `f4i_g4_c16_beta0_true_baselines` |
| SB, corner hotspots | `a5_4corner_a50` |
| N, shifting hotspot | `a4_shifting_slow` |
| A, adaptive stress | `adv_b_boundary_stress` |
| R, Chicago replay | `chicago_real_replay` |

The six-regime panel groups raw variants within each matched seed: SB
averages `a5_2corner_a50`, `a5_2corner_a100`, `a5_2edge_a50`, and
`a5_4corner_a50`; N averages `a4_shifting_slow` (period 1000),
`a4_shifting_medium` (period 500), and `a4_shifting_fast` (period 200).
Run each variant with the command above.

### Sweeps

| Study | Configs | Output directory | Seeds |
|---|---|---|---|
| Region-count scaling | `a5_4corner_a50`, `a5_4corner_g{6,8,10,12}` | `runs/p1/<config>` | 0-9 |
| Hotspot concentration | `sb_density_h{1,5,10,15,20,32}` | `runs/p1_hotspot_density/<config>` | 0-9 |
| Response-budget tolerance | `tu18_rho{10,15,20,25,30}` | `runs/tu18_sensitivity/<config>` | 0-4 |
| Chicago load sensitivity | `chicago_loop4_{v3,v54x,v5,v510x}_2023_{01..12}` | `runs/chicago_loop4_<family>_full` | 0-9 |

The load-sensitivity families map to event-scaling multipliers: `v3` is the
1x compressed weekly replay, `v54x` is 4x, `v5` is 5x, and `v510x` is 10x.
Run one family per invocation so each output directory holds one JSONL:

```bash
python scripts/run_experiment.py \
    --configs chicago_loop4_v3_2023_01 ... chicago_loop4_v3_2023_12 \
    --policies hold fixed_band safe_greedy \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir runs/chicago_loop4_v3_full
```

The band-width trade-off figure is closed-form and needs no runs.

## 3. CPAC

Train and evaluate CPAC inside the same feasibility kernel:

```bash
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/train_cpac_phase2.py --configs CONFIG --seed $SEED \
      --num-updates 100 --rollout-length 256 --hidden 64 --device cpu \
      --out runs/cpac/CONFIG/seed$SEED
  python scripts/eval_cpac_phase2.py \
      --checkpoint runs/cpac/CONFIG/seed$SEED/ckpt.pt \
      --config CONFIG --seeds 0 1 2 3 4 5 6 7 8 9 \
      --out runs/cpac/CONFIG/eval_trainseed$SEED --device cpu
done
python scripts/aggregate_phase2.py
```

CONFIG ranges over the six regime configs in the map above. The training
defaults match the CPAC settings reported in the paper appendix.
`--no-reward-normalize` and `--no-nstep-fix` switch those two components off
for ablations.

## 4. Figures

| Figure family | Script |
|---|---|
| Regime taxonomy | `_make_fig_regime_taxonomy_4pane_v2.py` |
| S1 deterministic result | `make_fig_s1_deterministic_result.py` |
| Regime scorecard | `make_fig07_regime_scorecard.py` |
| Chicago load sensitivity | `_make_fig_load_sensitivity_bars.py` |
| CPAC ladder and certification | `make_phase2_figures_v2.py` |
| Band-width trade-off | `make_fig_part1_trilemma_clean.py` |
| Region-count scaling | `make_fig_region_scaling.py` |
| Hotspot concentration | `make_fig_hotspot_density.py` |
| Response-budget tolerance | `make_fig_rho_tolerance_sweep_clean_s1.py` |
| Load-signal dispersion | `make_fig_hotspot_dispersion.py` |


## 5. Tests

```bash
python -m pytest tests -q
```

The tests cover shield clauses, endpoint-disjoint matching, controller-action
orientation, certificate-consistent admission, dynamic eligibility, invariance,
seed determinism, and unit conversion.
