"""Return density of each cloud of the thinning experiment, always measured the same way.

Points inside the region of the plot, divided by the area of that region, over the cloud
the model really received: the original one for 100 % and the pulse thinned one for 60,
40 and 25 %. It is the same definition gather_detections.py uses for the rest of the study, and
it serves the three models alike, so that the correlations between density and recall can
be compared between them.

Usage: python3 analysis/thinning_density.py
"""
import json
import os
import sys

import laspy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import reference, region, BASE as DATA  # noqa: E402

FRAC = (100, 60, 40, 25)
PS = [f"p{i}" for i in range(1, 16)]


def path(p, f):
    if f == 100:
        return os.path.join(DATA, "trento_dataset", "las", f"{p}.las")
    return os.path.join(ROOT, "thinning", f"pell_{p}_f{f}.las")


def main():
    per_plot, cols = {}, {f: [] for f in FRAC}
    print(f"{'plot':8} {'area m2':>9} " + " ".join(f"{str(f)+' %':>9}" for f in FRAC))
    for p in PS:
        ref = reference("pell", p)
        reg = region(ref)
        area = abs(np.sum(reg.vertices[:, 0] * np.roll(reg.vertices[:, 1], -1)
                          - np.roll(reg.vertices[:, 0], -1) * reg.vertices[:, 1]) / 2)
        per_plot[p] = {"area": round(float(area), 1)}
        row = f"{p:8} {area:>9.0f} "
        for f in FRAC:
            L = laspy.read(path(p, f))
            inside = reg.contains_points(np.c_[np.asarray(L.x), np.asarray(L.y)])
            d = float(inside.sum()) / area
            per_plot[p][str(f)] = round(d, 1)
            cols[f].append(d)
            row += f" {d:>9.1f}"
        print(row)

    output = {str(f): round(float(np.mean(cols[f])), 1) for f in FRAC}
    output["per_plot"] = per_plot
    json.dump(output, open(os.path.join(ROOT, "thinning_density.json"), "w"), indent=2)
    print("\nmean   " + " ".join(f"{output[str(f)]:>9.1f}" for f in FRAC))
    print("saved to thinning_density.json")


if __name__ == "__main__":
    main()
