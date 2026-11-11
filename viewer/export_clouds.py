"""Module C of the viewer: subsampled point clouds for the 3D view.

Reads the SegmentAnyTree outputs (one .laz per plot, with the PredInstance field)
and writes viewer/data/clouds.json with the coordinates compressed as 16 bit
integers in base64. The idea is that the browser can paint the 22 plots without
downloading hundreds of megabytes of raw cloud.

Usage:
  python3 viewer/export_clouds.py
"""
import base64
import json
import os
import sys

import laspy
import numpy as np

# We reuse the definition of sites and plots from the analysis module
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import SITES, reference, region, MIN_PTS   # noqa: E402

SAT_OUT = os.path.join(ROOT, "sat_output_all")
OUT = os.path.join(BASE, "viewer", "data", "clouds.json")
OUT_INST = os.path.join(BASE, "viewer", "data", "instances.json")

MAX_PTS = 40000         # cap per plot; you need density to see a single tree
LIM = 32767             # cap of a signed 16 bit integer


def b64_int16(arr):
    """Packs an array as little endian int16 and returns it in base64."""
    return base64.b64encode(np.asarray(arr).astype("<i2").tobytes()).decode("ascii")


def fits_int16(*arrays):
    """True if every value fits in a signed 16 bit integer."""
    for a in arrays:
        if a.size == 0:
            continue
        if a.min() < -LIM - 1 or a.max() > LIM:
            return False
    return True


def thin_clouds(n, limit):
    """Reproducible indices without replacement. If there are few points, all of them."""
    if n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=limit, replace=False)
    idx.sort()          # sorted so that reading the laz is predictable
    return idx


def export_plot(site, plot, limit):
    """Returns the compressed dictionary of a plot, or None if the laz is missing."""
    laz = os.path.join(SAT_OUT, f"{site}_{plot}_out.laz")
    if not os.path.exists(laz):
        return None

    L = laspy.read(laz)
    x = np.asarray(L.x, dtype=np.float64)
    y = np.asarray(L.y, dtype=np.float64)
    z = np.asarray(L.z, dtype=np.float64)
    inst = np.asarray(L.PredInstance).astype(np.int64)

    # Instance of each SegmentAnyTree detection, in the same order produced by
    # det_sat(site, plot, "apex") of evaluation_core.py: walk the instances in order,
    # skip the ones with fewer than MIN_PTS points, and keep the highest point.
    det_id, det_x, det_y = [], [], []
    for iid in np.unique(inst[inst > 0]):
        m = inst == iid
        if int(m.sum()) < MIN_PTS:
            continue
        k = int(np.argmax(z[m]))
        det_id.append(int(iid)); det_x.append(float(x[m][k])); det_y.append(float(y[m][k]))
    ref = reference(site, plot)
    if det_x:
        inside = region(ref).contains_points(np.c_[det_x, det_y])
    else:
        inside = np.zeros(0, bool)
    inst_map = {"inside": [det_id[i] for i in range(len(det_id)) if inside[i]],
                 "outside": [det_id[i] for i in range(len(det_id)) if not inside[i]]}

    n_original = int(x.size)
    idx = thin_clouds(n_original, limit)
    x, y, z, inst = x[idx], y[idx], z[idx], inst[idx]

    # Local origin: minimum x and y of the chosen points, rounded to 2 decimals.
    # We round BEFORE encoding so that the reconstruction is exact.
    x0 = round(float(x.min()), 2)
    y0 = round(float(y.min()), 2)

    scale = 100                     # centimeters
    xi = np.rint((x - x0) * scale)
    yi = np.rint((y - y0) * scale)
    zi = np.rint(z * scale)
    if not fits_int16(xi, yi, zi):
        # The plot is too wide or too tall for centimeters: we drop to decimeters
        scale = 10
        xi = np.rint((x - x0) * scale)
        yi = np.rint((y - y0) * scale)
        zi = np.rint(z * scale)
        if not fits_int16(xi, yi, zi):
            raise ValueError(f"{site}/{plot}: does not fit even in decimeters")

    # The ORIGINAL PredInstance identifier is kept, without reindexing. The maximum
    # in these 22 files is 646, so it fits in int16 with room to spare, and keeping it
    # lets each detection be linked to its points without any intermediate table.
    # Any value <= 0 is treated as a point with no instance.
    present = np.unique(inst[inst > 0])
    ii = np.where(inst > 0, inst, 0)
    if not fits_int16(ii):
        raise ValueError(f"{site}/{plot}: instance identifiers outside int16")

    return {
        "n": int(x.size),
        "n_original": n_original,
        "origin": [x0, y0],
        "scale": scale,
        "x": b64_int16(xi),
        "y": b64_int16(yi),
        "z": b64_int16(zi),
        "inst": b64_int16(ii),
        "n_inst": int(present.size),
    }, inst_map


def build(limit, inst_per_det):
    """Walks the 22 plots and builds the complete dictionary."""
    output = {}
    for site, _name, plots in SITES:
        for plot in plots:
            key = f"{site}/{plot}"
            r = export_plot(site, plot, limit)
            if r is None:
                print(f"  the laz of {key} is missing")
                continue
            d, mi = r
            inst_per_det[key] = mi
            if d is None:
                print(f"  the laz of {key} is missing")
                continue
            output[key] = d
            print(f"  {key}: {d['n']} of {d['n_original']} points, "
                  f"{d['n_inst']} instances, scale {d['scale']}")
    return output


def main():
    limit = MAX_PTS
    for attempt in range(2):
        print(f"Exporting with a cap of {limit} points per plot")
        inst_per_det = {}
        data = build(limit, inst_per_det)
        text = json.dumps(data, ensure_ascii=False)
        mb = len(text.encode("utf-8")) / 1e6
        print(f"JSON size: {mb:.2f} MB with {len(data)} plots")
        if mb <= 40 or attempt == 1:
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Written to {OUT}")
            with open(OUT_INST, "w", encoding="utf-8") as f:
                json.dump(inst_per_det, f, separators=(",", ":"))
            nd = sum(len(v["inside"]) for v in inst_per_det.values())
            nf = sum(len(v["outside"]) for v in inst_per_det.values())
            print(f"Written to {OUT_INST}: {nd} detections inside and {nf} outside")
            return data
        print("Over 40 MB, repeating with 6000 points per plot")
        limit = 6000


if __name__ == "__main__":
    main()
