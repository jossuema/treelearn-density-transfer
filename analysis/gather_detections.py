"""Gathers the detections of the five methods into a single file, uncropped.

Until now each method saved its own output in its own format and some already came
cropped to the region, which makes it impossible to redo the region sensitivity. Here
they all end up the same: every detection of each plot, with apex, height and area,
before any filter. Everything else in the study is computed from this file.

Sources:
    SAT, AMS3D, WS   viewer/data/trees.json, adding det and det_outside
    FF3D             the zips in ff3d_output, re-read with evaluate_ff3d.detections
    TreeLite3D       treelite_output, with the setting the leave-one-out protocol picks

Usage: python3 analysis/gather_detections.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
sys.path.insert(0, HERE)
from evaluation_core import SITES, reference, region, BASE as DATA  # noqa: E402

FIELDS = ("x", "y", "h", "area")


def main():
    arb = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))
    keys = [f"{s}/{p}" for s, _, ps in SITES for p in ps]

    all_dets = {k: {} for k in keys}
    for k in keys:
        for m in ("SAT", "AMS3D", "WS"):
            d, f = arb[k]["det"][m], arb[k]["det_outside"][m]
            all_dets[k][m] = {c: list(d[c]) + list(f[c]) for c in FIELDS}

    from evaluate_ff3d import detections as det_ff3d
    for k in keys:
        site, p = k.split("/")
        det, _ = det_ff3d(f"{site}_{p}")
        all_dets[k]["FF3D"] = {c: [float(v) for v in det[c]] for c in FIELDS}

    from treelite_results import load, f1_of
    settings = load()
    best = max(settings, key=lambda e: f1_of(settings[e], keys))
    tl = json.load(open(os.path.join(ROOT, "treelite_output",
                                     f"detections_{best}.json")))
    for k in keys:
        site, p = k.split("/")
        d = tl[f"{site}_{p}"]
        if str(d.get("error", "")).startswith("0 labels"):
            d = {c: [] for c in FIELDS}
        all_dets[k]["TreeLite3D"] = {c: list(d.get(c, [])) for c in FIELDS}

    # return density inside the inventoried region, which is where the scoring happens.
    # NOTE: reference() points at the LAS cropped above 1 m in Chamrousse, which is NOT
    # the one the models received. Measuring density there underestimated it and exaggerated
    # the contrast between sites. It is measured on the same cloud that each model processed.
    import laspy

    def model_cloud(site, p):
        if site == "cham":
            return os.path.join(DATA, "IRSTEA_dataset", "las",
                                "full_las", f"las{p}.las")
        return os.path.join(DATA, "trento_dataset", "las", f"{p}.las")

    dens = {}
    print(f"{'plot':12} {'trees':>8} {'region m2':>10} {'pts/m2':>8} " +
          " ".join(f"{m:>11}" for m in ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")))
    for k in keys:
        site, p = k.split("/")
        ref = reference(site, p)
        L = laspy.read(model_cloud(site, p))
        xy = np.c_[np.asarray(L.x), np.asarray(L.y)]
        reg = region(ref)
        inside = reg.contains_points(xy)
        area = abs(np.sum(reg.vertices[:, 0] * np.roll(reg.vertices[:, 1], -1)
                          - np.roll(reg.vertices[:, 0], -1) * reg.vertices[:, 1]) / 2)
        dens[k] = dict(points=int(inside.sum()), area=round(float(area), 1),
                       density=round(float(inside.sum() / area), 1),
                       trees=len(ref["x"]))
        print(f"{k:12} {len(ref['x']):>8} {area:>10.0f} {dens[k]['density']:>8.1f} " +
              " ".join(f"{len(all_dets[k][m]['x']):>11}"
                       for m in ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")))

    output = {"treelite_setting": best, "density": dens, "detections": all_dets}
    with open(os.path.join(ROOT, "all_detections.json"), "w") as f:
        json.dump(output, f)
    print(f"\nsaved to all_detections.json "
          f"({os.path.getsize(os.path.join(ROOT, 'all_detections.json')) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
