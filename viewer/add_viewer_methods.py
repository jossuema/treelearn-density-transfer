"""Adds ForestFormer3D and TreeLite3D to the data of the viewer.

The viewer was built with three methods. The detections of the two new models are
already in all_detections.json, uncropped, so here they are only cropped to the
region and matched with the SAME protocol the paper uses (one to one by distance), so
that what is seen on screen is exactly what is counted in the tables.

Usage: python3 viewer/add_viewer_methods.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
sys.path.insert(0, os.path.join(BASE, "analysis"))
from evaluation_core import SITES, reference, region  # noqa: E402
from main_evaluation import greedy_pairs, metrics  # noqa: E402

NEW = ("FF3D", "TreeLite3D")
DIST = ("2.5", "3.5", "5.0")


def main():
    all_available = json.load(open(os.path.join(ROOT, "all_detections.json")))["detections"]
    path = os.path.join(HERE, "data", "trees.json")
    arb = json.load(open(path))

    print(f"{'plot':12} " + " ".join(f"{m:>22}" for m in NEW))
    for site, _, ps in SITES:
        for p in ps:
            k = f"{site}/{p}"
            ref = reference(site, p)
            reg = region(ref)
            row = f"{k:12} "
            for m in NEW:
                d = all_available[k][m]
                x, y = np.array(d["x"]), np.array(d["y"])
                inside = reg.contains_points(np.c_[x, y]) if len(x) else np.zeros(0, bool)
                sel = {c: np.array(d[c])[inside] for c in ("x", "y", "h", "area")}
                outside_det = {c: np.array(d[c])[~inside] for c in ("x", "y", "h", "area")}
                arb[k]["det"][m] = {c: [round(float(v), 2) for v in sel[c]]
                                    for c in ("x", "y", "h", "area")}
                arb[k]["det"][m]["outside"] = int((~inside).sum())
                arb[k]["det_outside"][m] = {c: [round(float(v), 2) for v in outside_det[c]]
                                          for c in ("x", "y", "h", "area")}
                arb[k]["pairs"][m], arb[k]["met"][m] = {}, {}
                for ds in DIST:
                    pr = greedy_pairs(sel["x"], sel["y"], ref["x"], ref["y"], float(ds))
                    arb[k]["pairs"][m][ds] = [[int(i), int(j)] for i, j in pr]
                    TP, ND, NR = len(pr), int(inside.sum()), len(ref["x"])
                    r, pc, f1 = metrics(TP, NR, ND)
                    arb[k]["met"][m][ds] = dict(TP=TP, ND=ND, NR=NR, recall=round(r, 3),
                                                precision=round(pc, 3), f1=round(f1, 3))
                row += f"  {arb[k]['met'][m]['3.5']['TP']:>3}/{inside.sum():<3} of {len(ref['x']):<4}"
            print(row)

    with open(path, "w") as f:
        json.dump(arb, f)
    print(f"\ntrees.json extended to {len(arb['cham/1']['met'])} methods")


if __name__ == "__main__":
    main()
