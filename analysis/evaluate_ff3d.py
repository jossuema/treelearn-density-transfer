"""Step 4: turns the ForestFormer3D output into detections and evaluates it.

The output of each plot is a PLY with x, y, z, semantic_pred, instance_pred and score,
but note: the pipeline recenters the coordinates to a local origin and normalizes z so
that the minimum is zero. They have to be put back in their original system, which is
recovered from the input LAS by comparing the minima, because the transformation is a pure
translation. It is turned into detections with the SAME rule we use with SegmentAnyTree in
evaluation_core.py (det_sat): one detection per instance with at least MIN_PTS points, placed at
the apex, and with the area of the convex hull in plan view. That way the comparison is
direct.

Usage: python3 analysis/evaluate_ff3d.py
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import (reference, region, hull_area, MIN_PTS, SITES,
                             BASE as DATA)  # noqa: E402

ZIPS = os.path.join(ROOT, "ff3d_output")
D = 3.5
NB = 2000


def read_ply(data):
    """Minimal reader for the binary PLY the pipeline writes."""
    end = data.index(b"end_header") + len(b"end_header") + 1
    header = data[:end].decode("ascii", "replace").splitlines()
    n = next(int(l.split()[2]) for l in header if l.startswith("element vertex"))
    types = {"double": "f8", "float": "f4", "int": "i4", "uchar": "u1"}
    fields = [(l.split()[2], types[l.split()[1]]) for l in header if l.startswith("property")]
    arr = np.frombuffer(data[end:], dtype=np.dtype(fields), count=n)
    return arr


def input_las(site, p):
    """The same file the model received, and that SegmentAnyTree received."""
    if site == "cham":
        return os.path.join(DATA, "IRSTEA_dataset", "las", "full_las", f"las{p}.las")
    return os.path.join(DATA, "trento_dataset", "las", f"{p}.las")


def detections(key):
    """ForestFormer3D detections with the same criterion as det_sat."""
    z = sorted(glob.glob(os.path.join(ZIPS, f"{key}__results_*.zip")))[-1]
    with zipfile.ZipFile(z) as f:
        name = [i for i in f.namelist() if i.endswith(".ply")][0]
        arr = read_ply(f.read(name))
    x, y, zz = arr["x"].astype(float), arr["y"].astype(float), arr["z"].astype(float)

    # Put the coordinates back in their original system
    import laspy
    site, p = key.split("_", 1)
    L = laspy.read(input_las(site, p))
    lx, ly, lz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    for name, a, b in (("x", x, lx), ("y", y, ly), ("z", zz, lz)):
        ext_ply = a.max() - a.min(); las_ext = b.max() - b.min()
        if abs(ext_ply - las_ext) > 0.05:
            print(f"  WARNING {key} {name}: extent {ext_ply:.2f} vs LAS {las_ext:.2f}")
    x = x + (lx.min() - x.min())
    y = y + (ly.min() - y.min())
    zz = zz + (lz.min() - zz.min())

    inst = arr["instance_pred"].astype(np.int64)
    X, Y, H, A = [], [], [], []
    for i in np.unique(inst[inst >= 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        k = np.argmax(zz[m])
        X.append(x[m][k]); Y.append(y[m][k])
        H.append(float(zz[m].max())); A.append(hull_area(np.c_[x[m], y[m]]))
    return dict(x=np.array(X), y=np.array(Y), h=np.array(H), area=np.array(A)), len(arr)


def greedy(dx, dy, rx, ry, dmax):
    if not len(dx) or not len(rx):
        return 0
    Dm = np.hypot(np.asarray(dx)[:, None] - np.asarray(rx)[None, :],
                  np.asarray(dy)[:, None] - np.asarray(ry)[None, :])
    cand = np.argwhere(Dm <= dmax)
    order = np.argsort(Dm[cand[:, 0], cand[:, 1]], kind="stable")
    ud, ur, n = set(), set(), 0
    for i, j in cand[order]:
        if i in ud or j in ur:
            continue
        ud.add(i); ur.add(j); n += 1
    return n


def main():
    arb = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))
    rows, per_plot = {}, {}
    print(f"{'plot':14} {'points':>8} {'instances':>11} {'inside':>7} {'TP':>5} {'NR':>5}")
    for site, name, plots in SITES:
        for p in plots:
            key = f"{site}_{p}"
            k = f"{site}/{p}"
            det, npts = detections(key)
            ref = reference(site, p)
            inside = region(ref).contains_points(np.c_[det["x"], det["y"]]) if len(det["x"]) \
                else np.zeros(0, bool)
            dd = {c: det[c][inside] for c in ("x", "y", "h", "area")}
            tp = greedy(dd["x"], dd["y"], ref["x"], ref["y"], D)
            rows.setdefault(name, []).append((tp, len(dd["x"]), len(ref["x"])))
            per_plot[k] = dict(instances=len(det["x"]), inside=int(inside.sum()),
                                  TP=tp, NR=len(ref["x"]),
                                  x=[round(float(v), 2) for v in dd["x"]],
                                  y=[round(float(v), 2) for v in dd["y"]],
                                  h=[round(float(v), 2) for v in dd["h"]],
                                  area=[round(float(v), 2) for v in dd["area"]])
            print(f"{k:14} {npts:>8} {len(det['x']):>11} {int(inside.sum()):>7} {tp:>5} {len(ref['x']):>5}")

    print(f"\n{'site':12} {'recall':>14} {'95% CI':>16} {'precision':>10} {'F1':>7}")
    summary = {}
    for name, fs in list(rows.items()) + [("Both", [f for v in rows.values() for f in v])]:
        tp = sum(f[0] for f in fs); nd = sum(f[1] for f in fs); nr = sum(f[2] for f in fs)
        r, pr = tp / nr, (tp / nd if nd else 0)
        f1 = 2 * pr * r / (pr + r) if pr + r else 0
        rng = np.random.default_rng(0); v = []
        for _ in range(NB):
            idx = rng.integers(0, len(fs), len(fs))
            a = sum(fs[i][0] for i in idx); b = sum(fs[i][2] for i in idx)
            v.append(a / b if b else 0)
        ci = [round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)]
        summary[name] = dict(TP=tp, ND=nd, NR=nr, recall=round(r, 3),
                               precision=round(pr, 3), f1=round(f1, 3), recall_ci=ci)
        print(f"{name:12} {r:>14.3f} {str(ci):>16} {pr:>10.3f} {f1:>7.3f}")

    json.dump({"summary": summary, "per_plot": per_plot},
              open(os.path.join(ROOT, "ff3d_evaluation.json"), "w"), indent=2, ensure_ascii=False)
    print("\nsaved to ff3d_evaluation.json")


if __name__ == "__main__":
    main()
