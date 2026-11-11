#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The AMS3D configuration grid, plot by plot, and which cell of it won.

The study scores the adaptive mean shift with the best of 25 configurations
chosen on the plot being scored. This writes that grid out so the claim can be checked
without the mean shift itself, which is third-party work and is not redistributed here.

The two bandwidths are plot specific: they come from the crown model fitted to that
plot, so each plot has its own 5x5 grid of (m1, m2) in millimetres. What is published
here is the grid that was tried and the cell that was kept, never a detection position.

Usage: python3 analysis/ams3d_grid.py
Reads   ams_port/out/ under TREELEARN_DATA
Writes  ams3d_grid.json next to TREELEARN_ROOT
"""
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import SITES, BASE as DATA  # noqa: E402

AMSDIR = os.path.join(DATA, "ams_port", "out")
PAIR = re.compile(r"_m1_(\d+)_m2_(\d+)\.csv$")


def grid_of(plot):
    """The (m1, m2) pairs tried on this plot, in millimetres."""
    out = set()
    for f in glob.glob(os.path.join(AMSDIR, f"e1_dets_{plot}_AMS3DE_las_{plot}_m1_*_m2_*.csv")):
        m = PAIR.search(f)
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return sorted(out)


def chosen():
    """The configuration kept per plot, with the score that chose it."""
    out = {}
    for site, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        path = os.path.join(AMSDIR, f)
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            m = re.search(r"_m1_(\d+)_m2_(\d+)$", r["cfg"])
            out[f"{site}/{r['plot']}"] = {
                "m1": int(m.group(1)), "m2": int(m.group(2)),
                "F1": float(r["F1"]), "recall": float(r["Rec"]),
                "precision": float(r["Pre"]), "TP": int(r["TP"]),
                "ND": int(r["ND"]), "NR": int(r["NT"])}
    return out


def main():
    if not os.path.isdir(AMSDIR):
        raise SystemExit(f"{AMSDIR} is not here: the mean shift output is not redistributed, "
                         "point TREELEARN_DATA at a copy that has ams_port/out/")
    best = chosen()
    output = {"note": "bandwidths in millimetres, one 5x5 grid per plot, chosen by F1 "
                      "on the plot itself; the mean shift code is not redistributed",
              "per_plot": {}}
    sizes = []
    print(f"{'plot':12} {'grid':>6} {'m1':>6} {'m2':>6} {'F1':>7}")
    for site, _, plots in SITES:
        for p in plots:
            k = f"{site}/{p}"
            g = grid_of(p)
            row = {"grid": [list(c) for c in g], "n_configurations": len(g)}
            row.update(best.get(k, {}))
            output["per_plot"][k] = row
            sizes.append(len(g))
            print(f"{k:12} {len(g):>6} {row.get('m1', 0):>6} {row.get('m2', 0):>6} "
                  f"{row.get('F1', float('nan')):>7.3f}")
    output["n_configurations"] = sorted(set(sizes))
    print(f"\ngrid size per plot: {sorted(set(sizes))}")
    with open(os.path.join(ROOT, "ams3d_grid.json"), "w") as f:
        json.dump(output, f, indent=2)
    print("saved to ams3d_grid.json")


if __name__ == "__main__":
    main()
