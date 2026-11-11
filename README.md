# Pretrained 3-D networks and classical methods for individual tree detection from airborne LiDAR

Evaluation code and aggregated results by Manuel Josue Malla Campoverde, Walter Andres
Osorio Tinitana and Eduardo Tusa (Universidad Tecnica de Machala).

Five individual tree detection methods are scored on 22 alpine ALS plots, in France and
Italy, against a field inventory of 1437 trees and under a single evaluation protocol:
three public pretrained networks (SegmentAnyTree, ForestFormer3D, TreeLite3D), run from
their released weights without retraining, and two classical methods tuned on each plot
(an adaptive 3-D mean shift, AMS3D, and a canopy height model watershed). Detections are
matched one to one with stem positions within 3.5 m, and confidence intervals come from
resampling plots.

## What is and is not here

This repository holds **code and aggregated results only**.

The ALS point clouds and the field inventories belong to the IRSTEA and FEM teams and are
**not redistributed here**, nor is anything from which tree positions could be
reconstructed. That rules out the point clouds, the per-tree inventory records and the
detection coordinates. Request the data from the original teams; every script reads it
from the path given in `TREELEARN_DATA`.

The adaptive mean shift is third-party work and is not redistributed either: the scripts
that score it read its stored output from `ams_port/out/` under `TREELEARN_DATA`.

`results/` holds the aggregated and per-plot numbers: recalls, precisions, F1, bootstrap
intervals, correlations, strata, species and runtimes, plus the configuration grids the
classical methods were tuned over and the whole TreeLite3D sweep. None of it contains
coordinates.

## Layout

```
analysis/   the evaluation itself: one protocol for all five methods
modal/      inference runners for the three networks and the watershed baseline
viewer/     a local browser viewer for the plots and the results
figures/    scripts that draw summary figures and write summary tables
results/    aggregated and per-plot numbers, no coordinates
```

## What each analysis is and where it lives

| Analysis | Scripts | Results |
|---|---|---|
| Overall comparison: recall, precision and F1 by site and pooled, with plot bootstrap intervals and paired differences | `analysis/main_evaluation.py`, `analysis/paired_comparison.py` | `main_evaluation.json`, `paired_comparison.json` |
| Crown strata and species | `analysis/main_evaluation.py`, `analysis/species_recall.py`, `analysis/species_table.py` | `main_evaluation.json`, `species_recall.json` |
| Configuration: the 25 AMS3D settings per plot and AMS3D chosen on the other plots, the 16 watershed settings, TreeLite3D's published defaults and its sweep | `analysis/ams3d_grid.py`, `analysis/review_checks.py`, `modal/watershed_sweep.py`, `analysis/treelite_results.py`, `modal/run_treelite.py` | `ams3d_grid.json`, `review_checks.json`, `watershed_sweep.json`, `treelite_evaluation.json` |
| Evaluation rules: matching tolerance, optimal assignment, a single matching protocol, scored region against the whole tile | `analysis/main_evaluation.py`, `analysis/review_checks2.py`, `modal/single_protocol_metrics.py` | `main_evaluation.json`, `consistent_results.json` |
| Crown overlap index | `analysis/crown_jaccard.py`, `modal/jaccard.py` | `crown_jaccard.json`, `jaccard.json` |
| Computing cost | `analysis/measure_runtimes.py`, `analysis/runtime_table.py` | `measure_runtimes.json`, `runtime_summary.json` |
| What a uniform random detector recovers, and control of multiple tests | `analysis/chance_floor.py`, `analysis/multiplicity.py` | `chance_floor.json`, `multiplicity.json` |
| Return density: controlled thinning of the dense site | see below | `thinning_*.json` |

`analysis/measure_runtimes.py` times the mean shift itself, so it needs the AMS3D
implementation, which is not here; its output is in `results/measure_runtimes.json`.

## Reproducing the numbers

Set up the environment once:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Two variables locate what is not in the repository: `TREELEARN_DATA` points at the
directory that holds `IRSTEA_dataset/` and `trento_dataset/`, and `TREELEARN_ROOT` at the
one that holds each model's output folder (`sat_output_all/`, `ff3d_output/`,
`treelite_output/`). Both default to the clone and its parent.

First rebuild the per-plot reference under `viewer/data/`. It is not committed, because
it carries tree positions, and everything downstream reads it:

```
python3 viewer/export_chm.py                  # canopy model, region and per-plot statistics
python3 viewer/export_trees.py                # inventory, detections, pairs and metrics
python3 viewer/patch_exports.py               # coverage, and the detections outside the region
python3 viewer/export_species.py              # species of every inventory tree
```

Then, in this order:

```
python3 analysis/gather_detections.py         # all five methods' detections in one file
python3 analysis/evaluate_ff3d.py             # ForestFormer3D, plot by plot
python3 analysis/treelite_results.py          # the TreeLite3D sweep and its chosen setting
python3 viewer/add_viewer_methods.py          # adds those two to the per-plot reference
python3 analysis/main_evaluation.py           # main table, matching, region, density, strata
python3 analysis/species_recall.py            # recall by species
python3 analysis/crown_jaccard.py             # crown overlap index
python3 analysis/paired_comparison.py         # paired differences against SegmentAnyTree
python3 analysis/chance_floor.py              # what a uniform detector recovers, per method
python3 analysis/multiplicity.py              # declared families, BH, paired thinning test
python3 analysis/ams3d_grid.py                # the mean shift's 25 configurations per plot
python3 figures/draw_figures.py               # two summary figures
python3 figures/write_tables.py               # summary tables in LaTeX
```

Each script writes its result next to `TREELEARN_ROOT`, and reads the ones before it from
there, falling back to the copy published in `results/`. So you can run any single step
against the published inputs without rerunning the whole chain.

The summary tables that `figures/write_tables.py` writes come from `results/`; no number
in them is typed by hand.

## Return density: controlled thinning

The two sites differ at once in sensor, stand and return density. To isolate density,
laser pulses are removed from the dense site and the three networks are run again, so
that density changes while site, sensor, stand and inventory stay fixed. The first step
writes the thinned clouds, the networks are then re-run on them with `modal/run_sat.py`,
`modal/run_ff3d.py` and `modal/run_treelite.py`, and the last three score the result
against the same inventory and the same protocol:

```
python3 modal/thin_clouds.py                  # removes pulses: 60, 40 and 25 % kept
python3 analysis/thinning_density.py          # density of every cloud, one definition
python3 analysis/sat_thinning_detections.py   # SegmentAnyTree detections
python3 analysis/ff3d_thinning_detections.py  # ForestFormer3D detections
python3 analysis/thinning_evaluation.py SAT      # and FF3D, and t4_p30 for TreeLite3D
```

Pulses are removed rather than points, so that the returns of one pulse stay together,
and the levels are nested draws of a single seeded permutation, so that they differ only
by what is removed. Density is always measured the same way, on the cloud the model
actually received: points inside the scored region divided by the area of that region.
Never from a model's output, since the networks differ in whether they keep ground
returns.

## Notebook

`notebooks/reproduce_results.ipynb` recomputes the reported quantities from the stored
detections and the field inventory: the main comparison, the results by site, the chance
floor, the multiplicity control, the strata, the thinning experiment, the TreeLite3D
sweep, the evaluation frame, the crown overlap index and species. It redraws two
figures and ends by checking its own output against the reported values. GPU steps
appear as the command that produced the stored artifact.

## The three checkpoints

Run from the authors' released pipelines and weights, unchanged except for TreeLite3D:
SegmentAnyTree and ForestFormer3D use their published defaults, while TreeLite3D is
scored both with its published defaults and with its linking radius and minimum proposal
size swept. The sweep is in `modal/run_treelite.py`.

## License

Code released under the MIT License (`LICENSE`). The data are not covered by it.
