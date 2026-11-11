"""Evaluates the SegmentAnyTree outputs against the field inventory.

Protocol (the same for every method, so that the comparison is clean):
  - A detected tree = one predicted instance; its position is the apex (highest point).
  - It is cropped to the convex hull of the inventory stems, dilated 1.5 m. The instances
    outside count neither as a hit nor as a false positive: there is real canopy there that
    nobody measured.
  - Greedy one to one matching by planimetric distance, capped at d_match.
  - Recall = TP / inventory trees. Precision = TP / instances inside the hull.
  - Aggregated per site by summing TP, NR and ND over plots (the same as Tusa does).

Usage: python3 modal/evaluate_sat.py [output_folder]
"""
import csv
import json
import os
import sys

import laspy
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

# The datasets are not redistributed, so where they live is configurable: TREELEARN_DATA
# points at the directory holding IRSTEA_dataset/ and trento_dataset/.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TREELEARN_DATA") or os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(DATA)
SAT_OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "sat_output_all")
DMATCH = (2.5, 3.5, 5.0)
BUF = 1.5
MIN_PTS = 10          # instances with fewer points are discarded as noise

CHAM = ["1", "1b", "2", "3", "3b", "4", "Premol"]
PELL = [f"p{i}" for i in range(1, 16)]


def inventory(site, plot):
    """Returns x, y, height and number of ALS points per tree (the last is not in the CSV)."""
    if site == "cham":
        path, columns = os.path.join(DATA, "IRSTEA_dataset/inventory_csv/Chamrousse.csv"), ("x", "y", "height")
    else:
        path, columns = os.path.join(DATA, "trento_dataset/csv/Pellizzano.csv"), ("x", "y", "height")
    xs, ys, hs = [], [], []
    for r in csv.DictReader(open(path)):
        if r["plotid"] == plot:
            try:
                xs.append(float(r[columns[0]])); ys.append(float(r[columns[1]])); hs.append(float(r[columns[2]]))
            except ValueError:
                pass
    return np.array(xs), np.array(ys), np.array(hs)


def dilate(poly, buf):
    c = poly.mean(0)
    d = np.linalg.norm(poly - c, axis=1)
    d[d == 0] = 1e-9
    return c + (poly - c) * ((d + buf) / d)[:, None]


def match(det_xy, ref_xy, dmax):
    if len(det_xy) == 0 or len(ref_xy) == 0:
        return 0
    D = np.linalg.norm(det_xy[:, None, :] - ref_xy[None, :, :], axis=2)
    tp = 0
    while D.size and D.min() <= dmax:
        i, j = np.unravel_index(np.argmin(D), D.shape)
        tp += 1
        D = np.delete(np.delete(D, i, 0), j, 1)
    return tp


def evaluate(site, plot):
    laz = os.path.join(SAT_OUT, f"{site}_{plot}_out.laz")
    if not os.path.exists(laz):
        return None
    L = laspy.read(laz)
    x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    inst = np.asarray(L.PredInstance).astype(np.int64)

    ix, iy, ih = inventory(site, plot)
    if len(ix) < 3:
        return None

    # apex of each instance
    ap = []
    for i in np.unique(inst[inst > 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        k = np.argmax(z[m])
        ap.append([x[m][k], y[m][k], float(z[m].max()), int(m.sum())])
    ap = np.array(ap) if ap else np.zeros((0, 4))

    hull = ConvexHull(np.c_[ix, iy])
    polyd = dilate(np.c_[ix, iy][hull.vertices], BUF)
    inside = MplPath(polyd).contains_points(ap[:, :2]) if len(ap) else np.zeros(0, bool)
    det = ap[inside][:, :2] if len(ap) else np.zeros((0, 2))

    row = {"site": site, "plot": plot, "NR": int(len(ix)),
            "n_instances_tile": int(len(ap)), "ND": int(len(det)),
            "hull_area_m2": round(float(hull.volume)),
            "pct_tile_inventoried": round(100 * hull.volume / ((x.max() - x.min()) * (y.max() - y.min())), 1)}
    for dm in DMATCH:
        tp = match(det, np.c_[ix, iy], dm)
        row[f"TP@{dm}"] = tp
    return row


def main():
    rows = [f for s, ps in (("cham", CHAM), ("pell", PELL)) for p in ps
             if (f := evaluate(s, p)) is not None]

    print(f"{'site':6} {'plot':7} {'NR':>4} {'tile':>5} {'ND':>4} {'%inv':>5} "
          + " ".join(f"{'R@'+str(d):>7}" for d in DMATCH))
    for f in rows:
        print(f"{f['site']:6} {f['plot']:7} {f['NR']:>4} {f['n_instances_tile']:>5} "
              f"{f['ND']:>4} {f['pct_tile_inventoried']:>5} "
              + " ".join(f"{f['TP@'+str(d)]/f['NR']:>7.3f}" for d in DMATCH))

    print("\n" + "=" * 74)
    print("PER SITE AGGREGATE (sums of TP, NR and ND, as in Tusa's protocol)")
    print(f"{'site':12} {'NR':>5} {'ND':>5} {'dmatch':>7} {'TP':>5} {'recall':>8} {'precis':>8} {'F1':>7}")
    summary = {}
    for site, label in (("cham", "Chamrousse"), ("pell", "Pellizzano")):
        sub = [f for f in rows if f["site"] == site]
        NR, ND = sum(f["NR"] for f in sub), sum(f["ND"] for f in sub)
        summary[label] = {"NR": NR, "ND": ND, "plots": len(sub), "dmatch": {}}
        for dm in DMATCH:
            TP = sum(f[f"TP@{dm}"] for f in sub)
            R, P = TP / NR, (TP / ND if ND else 0)
            F1 = 2 * P * R / (P + R) if P + R else 0
            summary[label]["dmatch"][str(dm)] = {"TP": TP, "recall": round(R, 3),
                                                    "precision": round(P, 3), "f1": round(F1, 3)}
            print(f"{label:12} {NR:>5} {ND:>5} {dm:>7} {TP:>5} {R:>8.3f} {P:>8.3f} {F1:>7.3f}")

    json.dump({"per_plot": rows, "per_site": summary},
              open(os.path.join(ROOT, "sat_evaluation.json"), "w"), indent=2, ensure_ascii=False)
    print("\nsaved to sat_evaluation.json")


if __name__ == "__main__":
    main()
