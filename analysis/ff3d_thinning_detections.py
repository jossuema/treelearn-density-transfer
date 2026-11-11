"""Turns the ForestFormer3D outputs on the thinned clouds into detections.

Same rule as in the whole study (det_sat): one detection per instance with at least
MIN_PTS points, placed at the apex, with the area of the convex hull in plan view. As in
evaluate_ff3d.py the recentering done by the pipeline has to be undone, and here the
reference for doing it is the THINNED cloud, which is the one the model received.

Usage: python3 analysis/ff3d_thinning_detections.py
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
sys.path.insert(0, HERE)
from evaluation_core import hull_area, MIN_PTS  # noqa: E402
from evaluate_ff3d import read_ply  # noqa: E402

INPUT = os.path.join(ROOT, "ff3d_output_thinned")
CLOUDS = os.path.join(ROOT, "thinning")


def main():
    import zipfile
    output = {}
    zips = sorted(glob.glob(os.path.join(INPUT, "*__results_*.zip")))
    print(f"{len(zips)} files in {INPUT}")
    for z in zips:
        base = os.path.basename(z).split("__results_")[0]
        with zipfile.ZipFile(z) as f:
            plys = [i for i in f.namelist() if i.endswith(".ply")]
            if not plys:
                print(f"  {base}: no PLY inside the zip")
                continue
            arr = read_ply(f.read(plys[0]))
        x, y, zz = (arr[c].astype(float) for c in ("x", "y", "z"))

        L = laspy.read(os.path.join(CLOUDS, f"{base}.las"))
        lx, ly, lz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
        for name, a, b in (("x", x, lx), ("y", y, ly), ("z", zz, lz)):
            if abs((a.max() - a.min()) - (b.max() - b.min())) > 0.05:
                print(f"  WARNING {base} {name}: extent {a.max()-a.min():.2f} "
                      f"against {b.max()-b.min():.2f}")
        x = x + (lx.min() - x.min())
        y = y + (ly.min() - y.min())
        zz = zz + (lz.min() - zz.min())

        inst = arr["instance_pred"].astype(np.int64)
        X, Y, H, A = [], [], [], []
        for i in np.unique(inst[inst >= 0]):
            m = inst == i
            if m.sum() < MIN_PTS:
                continue
            k = int(np.argmax(zz[m]))
            X.append(float(x[m][k])); Y.append(float(y[m][k]))
            H.append(float(zz[m].max())); A.append(hull_area(np.c_[x[m], y[m]]))
        output[base] = dict(x=X, y=Y, h=H, area=A, points=len(lx))
        print(f"  {base:20} {len(lx):>8} pts  {len(X):>4} instances")

    path = os.path.join(ROOT, "treelite_output", "thinning_detections_FF3D.json")
    json.dump(output, open(path, "w"))
    print(f"\n{len(output)} clouds -> {path}")


if __name__ == "__main__":
    main()
