"""Turns the SegmentAnyTree outputs on the thinned clouds into detections.

Same rule as in the whole study (det_sat): one detection per instance with at least
MIN_PTS points, placed at the apex, with the convex hull area in plan view.

Usage: python3 analysis/sat_thinning_detections.py
"""
import glob
import json
import os
import sys

import laspy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import hull_area, MIN_PTS  # noqa: E402

INPUT = os.path.join(ROOT, "sat_output_thinned")


def main():
    output = {}
    paths = sorted(glob.glob(os.path.join(INPUT, "**", "*.la[sz]"), recursive=True))
    print(f"{len(paths)} files in {INPUT}")
    for f in paths:
        base = os.path.basename(os.path.dirname(f))
        L = laspy.read(f)
        dims = [d.name for d in L.point_format.dimensions]
        field = next((c for c in ("PredInstance", "preds_instance_segmentation",
                                  "instance", "treeID") if c in dims), None)
        if field is None:
            print(f"  {base}: no instance field ({dims[:6]}...)")
            continue
        x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
        inst = np.asarray(getattr(L, field)).astype(np.int64)
        X, Y, H, A = [], [], [], []
        for i in np.unique(inst[inst > 0]):
            m = inst == i
            if m.sum() < MIN_PTS:
                continue
            k = int(np.argmax(z[m]))
            X.append(float(x[m][k])); Y.append(float(y[m][k]))
            H.append(float(z[m].max())); A.append(hull_area(np.c_[x[m], y[m]]))
        output[base] = dict(x=X, y=Y, h=H, area=A, points=len(x))
        print(f"  {base:20} {len(x):>8} pts  {len(X):>4} instances")
    path = os.path.join(ROOT, "treelite_output", "thinning_detections_SAT.json")
    json.dump(output, open(path, "w"))
    print(f"\n{len(output)} plots -> {path}")


if __name__ == "__main__":
    main()
