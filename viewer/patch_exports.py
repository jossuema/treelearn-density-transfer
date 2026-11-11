"""Two fixes on what was exported.

1. pct_region was computed against the CHM rectangle, which carries half a cell too
   much on each side. It is recomputed against the exact box of the points, which is
   the figure the study uses.
2. The detections that fall OUTSIDE the inventoried region are added. They are the
   direct evidence of the framing bias: uncropped, all of them would count as false
   positives even though there is real canopy there that nobody measured.
"""
import csv
import json
import os
import sys

import laspy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import (reference, region, det_sat, det_ams, det_watershed,
                             SITES, BASE as DATA_ROOT, results_path)  # noqa: E402
from shapely.geometry import Polygon                                              # noqa: E402

DATA = os.path.join(HERE, "data")
geo = json.load(open(os.path.join(DATA, "geo.json")))
arb = json.load(open(os.path.join(DATA, "trees.json")))

cfgs = {}
for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
    p = os.path.join(DATA_ROOT, "ams_port", "out", f)
    for r in csv.DictReader(open(p)):
        cfgs[(s, r["plot"])] = r["cfg"]

wb = json.load(open(results_path("watershed_sweep.json")))
wcfg = {}
for name, d in wb.items():
    sit = "cham" if name == "Chamrousse" else "pell"
    for pl, c in d.get("configs", {}).items():
        wcfg[(sit, pl)] = (float(c["hmin"]), float(c["mindist"]))

for site, name, plots in SITES:
    for p in plots:
        key = f"{site}/{p}"
        ref = reference(site, p)
        R = region(ref)

        # 1. coverage over the exact box of the points
        L = laspy.read(ref["las"])
        x, y = np.asarray(L.x), np.asarray(L.y)
        tile_area = float((x.max() - x.min()) * (y.max() - y.min()))
        region_area = float(Polygon(R.vertices).area)
        geo[key]["area_tile_m2"] = round(tile_area, 2)
        geo[key]["area_region_m2"] = round(region_area, 2)
        geo[key]["pct_region"] = round(100.0 * region_area / tile_area, 2)
        geo[key]["density_pts_m2"] = round(len(x) / tile_area, 2)

        # 2. detections outside the region
        hm, md = wcfg.get((site, p), (2.0, 2.0))
        dets = {"SAT": det_sat(site, p, "apex"),
                "AMS3D": det_ams(site, p, cfgs),
                "WS": det_watershed(ref, hmin=hm, mindist=md)}
        arb[key].setdefault("det_outside", {})
        for met, d in dets.items():
            if d is None or not len(d["x"]):
                arb[key]["det_outside"][met] = {"x": [], "y": [], "h": [], "area": []}
                continue
            inside = R.contains_points(np.c_[d["x"], d["y"]])
            f = ~inside
            arb[key]["det_outside"][met] = {
                k: [round(float(v), 2) for v in np.asarray(d[k])[f]]
                for k in ("x", "y", "h", "area")}
    print(f"  {site} done", flush=True)

json.dump(geo, open(os.path.join(DATA, "geo.json"), "w"), separators=(",", ":"))
json.dump(arb, open(os.path.join(DATA, "trees.json"), "w"), separators=(",", ":"))

for s, tag in (("cham", "Chamrousse"), ("pell", "Pellizzano")):
    v = [geo[k]["pct_region"] for k in geo if k.startswith(s + "/")]
    print(f"{tag}: region {min(v):.0f} to {max(v):.0f} % of the tile")
for s, tag in (("cham", "Chamrousse"), ("pell", "Pellizzano")):
    for met in ("SAT", "AMS3D", "WS"):
        inside = sum(len(arb[k]["det"][met]["x"]) for k in arb if k.startswith(s + "/"))
        outside = sum(len(arb[k]["det_outside"][met]["x"]) for k in arb if k.startswith(s + "/"))
        print(f"{tag:12s} {met:6s} inside={inside:5d} outside={outside:5d} "
              f"({100*outside/max(1,inside+outside):.0f} % falls outside)")
