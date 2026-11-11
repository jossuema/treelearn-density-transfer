"""Module C of the viewer: crown geometry per plot.

Produces viewer/data/crowns.json with, for each of the 22 plots:
  - sat_inside: convex hull in plan view of each SegmentAnyTree detection that falls
                inside the inventoried region, in the same order as det["SAT"]
  - sat_outside:  the same for those that fall outside, in the order of det_outside["SAT"]
  - inv:        inventory crown polygon built from the four radii N, S, E, O,
                in the same order as inv. Chamrousse only; in Pellizzano the list is empty

Every polygon is stored as a list of vertices [dx, dy] in whole decimeters
RELATIVE to the position of the tree or of the detection, so the JSON does not blow up.

Reuses evaluation_core.py without touching it: the logic of det_sat and of reference is copied
so the geometry of each instance can be attached without changing the original function.

Usage:
    python3 viewer/export_crowns.py
"""
import csv
import json
import math
import os
import sys

import laspy
import numpy as np
from scipy.spatial import ConvexHull, QhullError

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))

from evaluation_core import (SITES, SAT_OUT, MIN_PTS, reference, region,
                             BASE as DATA)  # noqa: E402

OUTPUT = os.path.join(BASE, "viewer", "data", "crowns.json")
TREES = os.path.join(BASE, "viewer", "data", "trees.json")
CSV_CHAM = os.path.join(DATA, "IRSTEA_dataset/inventory_csv/Chamrousse.csv")

MAX_VERT = 12        # vertices per convex hull, drops to 8 if the JSON goes over 6 MB
N_ANG = 24           # angles of the four radii polygon
LIMIT_MB = 6.0


# ------------------------------------------------------------------ utilities
def simplify(vertices, maxv):
    """Keeps at most maxv vertices spread evenly along the traversal.

    ConvexHull already returns the vertices in counterclockwise order, so it is enough
    to take evenly spaced indices over that list to keep the general shape.
    """
    n = len(vertices)
    if n <= maxv:
        return vertices
    idx = np.round(np.linspace(0, n, maxv, endpoint=False)).astype(int) % n
    return vertices[idx]


def to_decimeters(vertices, cx, cy):
    """Turns absolute vertices in meters into whole decimeters relative to (cx, cy)."""
    return [[int(round((float(vx) - cx) * 10.0)),
             int(round((float(vy) - cy) * 10.0))] for vx, vy in vertices]


def polygon_area(poly):
    """Area by the shoelace formula. Comes in as decimeters, goes out as m2."""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for k in range(len(poly)):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0 / 100.0        # dm2 to m2


def dist_to_polygon(poly):
    """Distance in meters from the origin (0,0) to the polygon, 0 if the origin is inside.

    The polygon comes in decimeters relative to the detection, so the detection is
    always at the origin and it is enough to measure against that point.
    """
    if len(poly) < 3:
        return float("inf")
    p = np.asarray(poly, float)
    # inside or outside, by the number of crossings of the horizontal ray
    inside = False
    n = len(p)
    for k in range(n):
        x1, y1 = p[k]
        x2, y2 = p[(k + 1) % n]
        if (y1 > 0) != (y2 > 0):
            xc = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
            if xc > 0:
                inside = not inside
    if inside:
        return 0.0
    # distance to the nearest segment
    best = float("inf")
    for k in range(n):
        a = p[k]
        b = p[(k + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else float(np.clip((-a) @ ab / L2, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(a + t * ab)))
    return best / 10.0                # decimeters to meters


# ------------------------------------------------------------------ SegmentAnyTree
def hulls_sat(site, plot, maxv):
    """Repeats det_sat(site, plot, "apex") and also keeps the hull of each instance.

    Same walk: np.unique(inst[inst>0]) in order, anything with fewer than MIN_PTS
    points is skipped, and the position is the highest point of the instance.
    """
    laz = os.path.join(SAT_OUT, f"{site}_{plot}_out.laz")
    if not os.path.exists(laz):
        return None
    L = laspy.read(laz)
    x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    inst = np.asarray(L.PredInstance).astype(np.int64)
    X, Y, hulls = [], [], []
    failures = 0
    for i in np.unique(inst[inst > 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        xm, ym, zm = x[m], y[m], z[m]
        k = np.argmax(zm)
        cx, cy = float(xm[k]), float(ym[k])
        X.append(cx)
        Y.append(cy)
        xy = np.c_[xm, ym]
        try:
            v = xy[ConvexHull(xy).vertices]
            hulls.append(to_decimeters(simplify(v, maxv), cx, cy))
        except (QhullError, ValueError):
            # degenerate instance: repeated points or points aligned on a line
            hulls.append([])
            failures += 1
    return dict(x=np.array(X), y=np.array(Y), hulls=hulls, failures=failures)


# ------------------------------------------------------------------ inventory
def cham_inventory_rows(plot):
    """Rows of the Chamrousse CSV accepted by reference(), in the same order.

    The filter of evaluation_core.reference is copied as is: it tries to convert x, y,
    height and area, and if anything fails the row is dropped.
    """
    rows = []
    for r in csv.DictReader(open(CSV_CHAM)):
        if r["plotid"] != plot:
            continue
        try:
            float(r["x"]); float(r["y"]); float(r["height"]); float(r["area"])
        except ValueError:
            continue
        rows.append(r)
    return rows


def num(value):
    """Converts to a positive float, or returns None if missing, zero or not a number."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def polygon_four_radii(N, S, E, O, n=N_ANG):
    """Four quarter ellipses glued together, each with the two semiaxes that apply.

    For an angle t measured from the east and growing towards the north:
        r(t) = sqrt( 1 / ( (cos t / a)^2 + (sin t / b)^2 ) )
    with a = E if cos t >= 0 and a = O if not, and b = N if sin t >= 0 and b = S if not.
    Returns the vertices in meters, relative to the trunk.
    """
    pts = []
    for k in range(n):
        t = 2.0 * math.pi * k / n
        c, s = math.cos(t), math.sin(t)
        a = E if c >= 0 else O
        b = N if s >= 0 else S
        r = math.sqrt(1.0 / ((c / a) ** 2 + (s / b) ** 2))
        pts.append((r * c, r * s))
    return pts


def polygon_circle(radius, n=N_ANG):
    """Fallback when radii are missing: circle of the equivalent area."""
    return [(radius * math.cos(2.0 * math.pi * k / n),
             radius * math.sin(2.0 * math.pi * k / n)) for k in range(n)]


def inventory_crowns(plot):
    """Crown polygons of the Chamrousse inventory, aligned with reference()."""
    rows = cham_inventory_rows(plot)
    polys, csv_areas, fallbacks = [], [], 0
    for r in rows:
        N, S, E, O = (num(r.get(c)) for c in ("N", "S", "E", "O"))
        area = float(r["area"])
        if None in (N, S, E, O):
            radius = math.sqrt(max(area, 0.0) / math.pi)
            pts = polygon_circle(radius)
            fallbacks += 1
        else:
            pts = polygon_four_radii(N, S, E, O)
        polys.append(to_decimeters(pts, 0.0, 0.0))
        csv_areas.append(area)
    return polys, csv_areas, fallbacks


# ------------------------------------------------------------------ main
def builds(maxv):
    """Builds the complete dictionary with the given vertex limit."""
    output = {}
    aux = {}      # loose data for the checks
    for site, name, plots in SITES:
        for p in plots:
            key = f"{site}/{p}"
            ref = reference(site, p)
            pat = region(ref)

            sat = hulls_sat(site, p, maxv)
            if sat is None or len(sat["x"]) == 0:
                inside = np.zeros(0, bool)
                hulls = []
                failures = 0
            else:
                inside = pat.contains_points(np.c_[sat["x"], sat["y"]])
                hulls = sat["hulls"]
                failures = sat["failures"]

            sat_inside = [hulls[i] for i in range(len(hulls)) if inside[i]]
            sat_outside = [hulls[i] for i in range(len(hulls)) if not inside[i]]

            if site == "cham":
                inv, csv_areas, fallbacks = inventory_crowns(p)
            else:
                inv, csv_areas, fallbacks = [], [], 0

            output[key] = {"sat_inside": sat_inside,
                             "sat_outside": sat_outside,
                             "inv": inv}
            aux[key] = {"n_ref": int(len(ref["x"])), "areas_csv": csv_areas,
                          "fallbacks": fallbacks, "hull_failures": failures}
            print(f"  {key}: sat inside={len(sat_inside)} outside={len(sat_outside)} "
                  f"inv={len(inv)} (NR={len(ref['x'])})", flush=True)
    return output, aux


def saved(output):
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    json.dump(output, open(OUTPUT, "w"), separators=(",", ":"))
    return os.path.getsize(OUTPUT)


def main():
    maxv = MAX_VERT
    print(f"Building crowns with at most {maxv} vertices per hull...", flush=True)
    output, aux = builds(maxv)
    nbytes = saved(output)
    print(f"\nsaved in {OUTPUT} ({nbytes} bytes, {nbytes/1e6:.2f} MB)")

    if nbytes > LIMIT_MB * 1e6:
        maxv = 8
        print(f"\nover {LIMIT_MB} MB, repeating with at most {maxv} vertices", flush=True)
        output, aux = builds(maxv)
        nbytes = saved(output)
        print(f"\nsaved in {OUTPUT} ({nbytes} bytes, {nbytes/1e6:.2f} MB)")

    # ------------------------------------------------------------ checks
    arb = json.load(open(TREES))
    print("\n" + "=" * 78)
    print("CHECKS")

    # 1. the 22 keys
    missing = [k for k in arb if k not in output]
    extra = [k for k in output if k not in arb]
    print(f"\n1. keys: crowns.json has {len(output)}, trees.json has {len(arb)}. "
          f"missing={missing or 'none'} extra={extra or 'none'}")

    # 2. lengths against trees.json
    print("\n2. lengths of sat_inside and sat_outside against trees.json")
    bad = []
    for k in sorted(output):
        nd_sp = len(arb[k]["det"]["SAT"]["x"])
        nf_sp = len(arb[k]["det_outside"]["SAT"]["x"])
        nd, nf = len(output[k]["sat_inside"]), len(output[k]["sat_outside"])
        if nd != nd_sp or nf != nf_sp:
            bad.append((k, nd, nd_sp, nf, nf_sp))
    if bad:
        for k, nd, nde, nf, nfe in bad:
            print(f"   MISMATCH {k}: inside {nd} vs {nde}, outside {nf} vs {nfe}")
    else:
        tot_d = sum(len(output[k]["sat_inside"]) for k in output)
        tot_f = sum(len(output[k]["sat_outside"]) for k in output)
        print(f"   the 22 plots agree. total inside={tot_d}, outside={tot_f}")

    # 3. the hull wraps around its detection
    print("\n3. the detection falls inside the hull or within 2 m (0,0 in hull coordinates)")
    sample = ["cham/1", "cham/3", "pell/p1", "pell/p7", "cham/Premol"]
    glob_ok = glob_n = 0
    for k in sorted(output):
        ok = n = 0
        for poly in output[k]["sat_inside"] + output[k]["sat_outside"]:
            if len(poly) < 3:
                continue
            n += 1
            ok += int(dist_to_polygon(poly) <= 2.0)
        glob_ok += ok
        glob_n += n
        if k in sample:
            print(f"   {k:12} {ok}/{n} = {100.0*ok/max(1,n):.1f} %")
    print(f"   ALL the plots: {glob_ok}/{glob_n} = {100.0*glob_ok/max(1,glob_n):.2f} %")

    # 4. Chamrousse inventory: lengths and area
    print("\n4. Chamrousse inventory: length and area of the four radii polygon")
    ax, ay, fallbacks, bad_inv = [], [], 0, []
    for k in sorted(output):
        if k.startswith("pell/"):
            if output[k]["inv"]:
                bad_inv.append((k, len(output[k]["inv"]), 0))
            continue
        n_sp = len(arb[k]["inv"]["x"])
        n = len(output[k]["inv"])
        if n != n_sp:
            bad_inv.append((k, n, n_sp))
        fallbacks += aux[k]["fallbacks"]
        for poly, a_csv in zip(output[k]["inv"], aux[k]["areas_csv"]):
            ax.append(a_csv)
            ay.append(polygon_area(poly))
        print(f"   {k:12} inv={n} (trees.json {n_sp}) fallbacks={aux[k]['fallbacks']}")
    print(f"   length mismatches: {bad_inv or 'none'}")
    print(f"   times it fell back to the equivalent circle: {fallbacks} of {len(ax)}")

    ax_, ay_ = np.array(ax), np.array(ay)
    ss_res = float(((ay_ - ax_) ** 2).sum())
    ss_tot = float(((ax_ - ax_.mean()) ** 2).sum())
    r2_ident = 1.0 - ss_res / ss_tot                      # R2 against the line y = x
    r_pear = float(np.corrcoef(ax_, ay_)[0, 1])
    err_rel = float(np.mean(np.abs(ay_ - ax_) / np.maximum(ax_, 1e-9)))
    print(f"   n={len(ax_)}  R2 against y=x: {r2_ident:.4f}  "
          f"R2 of correlation: {r_pear**2:.4f}")
    print(f"   mean relative error: {100*err_rel:.2f} %  "
          f"mean bias: {100*float(np.mean((ay_-ax_)/np.maximum(ax_,1e-9))):.2f} %")
    print(f"   mean area: inventory {ax_.mean():.2f} m2, polygon {ay_.mean():.2f} m2")

    # 5. size
    print(f"\n5. JSON size: {nbytes} bytes = {nbytes/1e6:.2f} MB "
          f"(limit {LIMIT_MB} MB, maximum vertices used {maxv})")


if __name__ == "__main__":
    main()
