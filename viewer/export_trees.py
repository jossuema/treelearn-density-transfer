"""Viewer module B: inventory trees, detections and matchings per plot.

Writes viewer/data/trees.json with, for each of the 22 plots:
  - inv:   inventory trees (x, y, height, area, radius, family, returns, stratum)
  - det:   each method's detections cropped to the inventoried region
  - pairs: the actual pairs (detection, tree) at 2.5, 3.5 and 5.0 m
  - met:   TP, ND, NR, recall, precision and f1 for that plot

Reuses evaluation_core.py without touching it. The only calculation rewritten here is the
matching, because the original function returns only the tree indices and the viewer
needs to know which detection was paired with which tree.

Usage:
    python3 viewer/export_trees.py
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))

from evaluation_core import (SITES, AMS, region, reference, det_sat, det_ams,
                      det_watershed, match, results_path)

OUTPUT = os.path.join(BASE, "viewer", "data", "trees.json")
DMATCH = (2.5, 3.5, 5.0)
METHODS = ("SAT", "AMS3D", "WS")


# ------------------------------------------------------------------ utilities
def load_cfgs_ams():
    """AMS3D setting chosen for each plot, same as main() in evaluation_core.py."""
    cfgs = {}
    for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        p = os.path.join(AMS, f)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cfgs[(s, r["plot"])] = r["cfg"]
    return cfgs


def load_cfgs_ws():
    """Watershed setting per plot, from the sweep already done."""
    w = json.load(open(results_path("watershed_sweep.json")))
    cfgs = {}
    for site, name, plots in SITES:
        for p in plots:
            c = w[name]["configs"][p]
            cfgs[(site, p)] = (float(c["hmin"]), float(c["mindist"]))
    return cfgs


def empty():
    return dict(x=np.array([]), y=np.array([]), h=np.array([]), area=np.array([]))


# ------------------------------------------------------------------ matching
def match_pairs(det, ref, dmax):
    """Same greedy logic as match(..., with_overlap=False), but returning pairs.

    2D distance matrix, only d <= dmax is admitted, the global minimum is taken,
    that row and that column are deleted, and it repeats until nothing admissible is left.
    Returns a list of [detection_index, tree_index].
    """
    nd, nr = len(det["x"]), len(ref["x"])
    if nd == 0 or nr == 0:
        return []
    D = np.hypot(det["x"][:, None] - ref["x"][None, :],
                 det["y"][:, None] - ref["y"][None, :])
    C = np.where(D <= dmax, D, np.inf)
    di, rj = np.arange(nd), np.arange(nr)
    pairs = []
    while C.size and np.isfinite(C).any():
        i, j = np.unravel_index(np.argmin(C), C.shape)
        pairs.append([int(di[i]), int(rj[j])])
        C = np.delete(np.delete(C, i, 0), j, 1)
        di, rj = np.delete(di, i), np.delete(rj, j)
    return pairs


def crop(det, pat):
    """Keeps only the detections inside the region and counts the ones left outside."""
    if det is None or len(det["x"]) == 0:
        return empty(), 0
    inside = pat.contains_points(np.c_[det["x"], det["y"]])
    dd = {k: v[inside] for k, v in det.items() if isinstance(v, np.ndarray)}
    return dd, int((~inside).sum())


def metrics(TP, NR, ND):
    R = TP / NR if NR else 0.0
    P = TP / ND if ND else 0.0
    return R, P, (2 * P * R / (P + R) if P + R else 0.0)


def r2(v):
    return [round(float(t), 2) for t in v]


# ------------------------------------------------------------------ main
def main():
    cfgs_ams = load_cfgs_ams()
    cfgs_ws = load_cfgs_ws()

    output = {}
    verified = []          # plots where match_pairs was compared with match
    agree = 0

    for site, name, plots in SITES:
        for p in plots:
            key = f"{site}/{p}"
            ref = reference(site, p)
            pat = region(ref)

            raw = {
                "SAT": det_sat(site, p, pos="apex"),
                "AMS3D": det_ams(site, p, cfgs_ams),
                # careful: always by keyword, the first positional of det_watershed is cell
                "WS": det_watershed(ref, hmin=cfgs_ws[(site, p)][0],
                                    mindist=cfgs_ws[(site, p)][1]),
            }

            dets, pairs, mets = {}, {}, {}
            for met in METHODS:
                d, outside = crop(raw[met], pat)
                dets[met] = {"x": r2(d["x"]), "y": r2(d["y"]),
                             "h": r2(d["h"]), "area": r2(d["area"]), "outside": outside}
                pairs[met], mets[met] = {}, {}
                for dm in DMATCH:
                    pr = match_pairs(d, ref, dm)
                    pairs[met][str(dm)] = pr
                    TP, ND, NR = len(pr), len(d["x"]), len(ref["x"])
                    R, P, F = metrics(TP, NR, ND)
                    mets[met][str(dm)] = {"TP": TP, "ND": ND, "NR": NR,
                                          "recall": round(R, 3),
                                          "precision": round(P, 3),
                                          "f1": round(F, 3)}
                    # cross check against the original function on a few plots
                    if dm == 3.5 and len(verified) < 5:
                        orig = sorted(int(t) for t in match(d, ref, dm, False))
                        ours = sorted(t[1] for t in pr)
                        verified.append((key, met, orig == ours, len(orig), len(ours)))
                        agree += int(orig == ours)

            output[key] = {
                "inv": {
                    "x": r2(ref["x"]), "y": r2(ref["y"]),
                    "h": r2(ref["h"]), "area": r2(ref["area"]),
                    "r": r2(np.sqrt(np.maximum(ref["area"], 0.0) / np.pi)),
                    "fam": [str(t) for t in ref["fam"]],
                    "npts": [int(t) for t in ref["npts"]],
                    "stratum": [str(t) for t in ref["stratum"]],
                },
                "det": dets,
                "pairs": pairs,
                "met": mets,
            }
            print(f"  {key}: NR={len(ref['x'])} "
                  + " ".join(f"{m}={mets[m]['3.5']['TP']}/{mets[m]['3.5']['ND']}"
                             for m in METHODS), flush=True)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    json.dump(output, open(OUTPUT, "w"), ensure_ascii=False)
    print(f"\nsaved to {OUTPUT} ({os.path.getsize(OUTPUT)} bytes, {len(output)} plots)")

    # ------------------------------------------------------------ checks
    print("\n--- matching check (my pairs against match())")
    for cl, met, ok, a, b in verified:
        print(f"  {cl:12} {met:6} {'same' if ok else 'DIFFERENT'}  ({a} vs {b})")
    print(f"  {agree} of {len(verified)} agree")

    print("\n--- NR per site")
    nr_by_site = {}
    for site, name, plots in SITES:
        nr_by_site[name] = sum(output[f"{site}/{p}"]["met"]["SAT"]["3.5"]["NR"] for p in plots)
        print(f"  {name}: {nr_by_site[name]}")

    # Note on the watershed: watershed_sweep.json came out of evaluate_plot(..., True),
    # that is with Tusa's overlap term, while here everything runs on distance alone so that
    # the three methods use the same criterion. The only consequence is one tree of difference
    # in Chamrousse (386 against 387 of 894). Pellizzano comes out identical in both.
    check = os.path.join(ROOT, "full_analysis.json")
    if not os.path.exists(check):
        print("\n--- full_analysis.json is not here, the cross-check is skipped")
        return
    print("\n--- aggregate at 3.5 m against the results already published")
    ac = json.load(open(check))["per_site"]["distance_only"]
    wb = json.load(open(results_path("watershed_sweep.json")))
    failures = []
    for site, name, plots in SITES:
        for met in METHODS:
            TP = sum(output[f"{site}/{p}"]["met"][met]["3.5"]["TP"] for p in plots)
            ND = sum(output[f"{site}/{p}"]["met"][met]["3.5"]["ND"] for p in plots)
            NR = nr_by_site[name]
            R, P, F = metrics(TP, NR, ND)
            if met == "WS":
                sp = wb[name]["3.5"]
            else:
                sp = ac[name][met]["3.5"]
            dr, dp = abs(R - sp["recall"]), abs(P - sp["precision"])
            ok = dr <= 0.002 and dp <= 0.002
            # known and explained case: the watershed sweep used the protocol with overlap
            known = (name == "Chamrousse" and met == "WS"
                        and TP == 386 and sp["TP"] == 387 and ND == sp["ND"])
            if not ok and not known:
                failures.append((name, met, round(R, 4), round(P, 4),
                               sp["recall"], sp["precision"]))
            label = "OK" if ok else ("known, 1 tree from the protocol" if known else "FAIL")
            print(f"  {name:11} {met:6} TP={TP:4} ND={ND:4} NR={NR:4} "
                  f"recall={R:.3f} (exp {sp['recall']:.3f}, d={dr:.4f})  "
                  f"prec={P:.3f} (exp {sp['precision']:.3f}, d={dp:.4f})  {label}")
    print("\nfailures:", failures if failures else "none")


if __name__ == "__main__":
    main()
