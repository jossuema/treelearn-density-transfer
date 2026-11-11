"""Crown Jaccard index for the five methods, with equivalent circles.

Only AMS3D and the watershed publish a crown area, not a shape, so to keep the comparison
fair all five are represented the same way: a circle of equivalent area centered on the
detection. The reference one comes from the crown area of the inventory.

The pairs are the same ones as in the detection table: one to one by distance, 3.5 m,
inside the region. It is checked that it reproduces jaccard.json for the three methods
that were already there, before extending it to the two new ones.

Usage: python3 analysis/crown_jaccard.py
"""
import json
import os
import sys

import numpy as np
from shapely.geometry import Point

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
sys.path.insert(0, HERE)
from evaluation_core import SITES, region, results_path  # noqa: E402
from main_evaluation import MET, greedy_pairs, ref_of  # noqa: E402


def circle(x, y, a):
    return Point(x, y).buffer(max(np.sqrt(max(a, 1e-6) / np.pi), 0.05), quad_segs=16)


def main():
    data = json.load(open(os.path.join(ROOT, "all_detections.json")))
    keys = [f"{s}/{p}" for s, _, ps in SITES for p in ps]
    sites = {n: [f"{s}/{p}" for p in ps] for s, n, ps in SITES}

    js = {m: {k: [] for k in keys} for m in MET}
    for k in keys:
        ref = ref_of(k)
        pr = [circle(ref["x"][i], ref["y"][i], ref["area"][i])
              for i in range(len(ref["x"]))]
        for m in MET:
            d = data["detections"][k][m]
            x, y, a = (np.array(d[c]) for c in ("x", "y", "area"))
            inside = region(ref).contains_points(np.c_[x, y]) if len(x) \
                else np.zeros(0, bool)
            xd, yd, ad = x[inside], y[inside], a[inside]
            for i, j in greedy_pairs(xd, yd, ref["x"], ref["y"], 3.5):
                pd_ = circle(xd[i], yd[i], ad[i])
                inter = pr[j].intersection(pd_).area
                union = pr[j].area + pd_.area - inter
                js[m][k].append(inter / union if union else 0.0)

    output = {}
    print(f"{'method':12} " + " ".join(f"{n:>22}" for n in
          list(sites) + ["Both"]))
    for m in MET:
        row = f"{m:12} "
        for n, ks in list(sites.items()) + [("Both", keys)]:
            v = [x for k in ks for x in js[m][k]]
            output.setdefault(n, {})[m] = dict(J=round(float(np.mean(v)), 3),
                                            pairs=len(v))
            row += f" {np.mean(v):.3f} ({len(v):>4} pairs)  "
        print(row)

    # check against what was already there, which was computed by another route
    pub = json.load(open(results_path("jaccard.json")))
    print(f"\ncheck against jaccard.json")
    print(f"{'site':12} {'method':10} {'here':>7} {'before':>7} {'diff':>7}")
    for n in sites:
        for m, key in (("SAT", "SAT_circle"), ("AMS3D", "AMS3D"), ("WS", "Watershed")):
            if n in pub and key in pub[n]:
                a, b = output[n][m]["J"], pub[n][key]["J_2D"]
                print(f"{n:12} {m:10} {a:>7.3f} {b:>7.3f} {a - b:>+7.3f}")

    json.dump(output, open(os.path.join(ROOT, "crown_jaccard.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\nsaved to crown_jaccard.json")


if __name__ == "__main__":
    main()
