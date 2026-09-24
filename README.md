# Density robustness and the transfer of pretrained 3-D networks to airborne LiDAR

Evaluation code for the letter *Density Robustness Governs the Transfer of Pretrained 3-D
Networks to Individual Tree Detection in Airborne LiDAR*, currently under review.

Five individual tree detection methods are scored on 22 alpine ALS plots against field
inventory, under a single evaluator: three public pretrained networks
(SegmentAnyTree, ForestFormer3D, TreeLite3D) and two per-plot tuned classical baselines
(an adaptive 3-D mean shift and a canopy height model watershed).

## What is and is not here

This repository holds **code and aggregated results only**.

The ALS point clouds and the field inventories belong to the IRSTEA and FEM teams and are
**not redistributed here**, nor is anything from which tree positions could be
reconstructed. That rules out the per-point clouds, the per-tree inventory records and the
detection coordinates. Request the data from the original teams; the scripts read it from
the path given in `TREELEARN_DATA`.

`results/` holds the aggregated and per-plot figures the letter reports: recalls,
precisions, F1, bootstrap intervals, correlations, strata and runtimes. None of it
contains coordinates.

## Layout

```
analysis/   the evaluation itself: one protocol for all five methods
modal/      inference runners for the three networks and the two baselines
viewer/     a local browser viewer for the plots and the results
figures/    the scripts that draw the figures and write the tables of the letter
results/    aggregated and per-plot numbers, no coordinates
```

## Reproducing the numbers

With the data in place and `TREELEARN_DATA` pointing at it:

```
python3 analysis/unificar.py            # gathers all five methods' detections
python3 analysis/robustez_cinco.py      # main table, matching, region, density, strata
python3 analysis/especie_cinco.py       # recall by species
python3 analysis/jaccard_cinco.py       # crown overlap index
python3 analysis/comparacion_modelos.py # paired differences against SegmentAnyTree
python3 figures/figuras.py              # figure 1
python3 figures/tablas.py               # the LaTeX tables
```

Every number in the letter's tables is generated from `results/` by `figures/tablas.py`;
none is typed by hand.

## The three checkpoints

Run from the authors' released pipelines and weights, unchanged except where the letter
says otherwise. SegmentAnyTree and ForestFormer3D use their published defaults;
TreeLite3D's linking radius and minimum proposal size were swept, and the sweep is in
`modal/run_treelite.py`.

## Licence

Code released under the MIT licence (`LICENSE`). The data are not covered by it. The
copyright holders are named once the manuscript is out of review.
