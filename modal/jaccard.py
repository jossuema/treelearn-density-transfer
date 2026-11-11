"""Jaccard index in the style of the AMS3D paper, with its three variants.

Tusa reports in his Table III a J_i column next to recall, precision and F1. Going through his
code, THREE different definitions of that J_i showed up, all of them with a real polygon
intersection but with a different vertical factor:

  J_2D    = a_i / (a_r + a_d - a_i)                                  (pyc from December 2019)
  J_vol_h = a_i*min(h_r,h_d) / (a_r*h_r + a_d*h_d - a_i*min(h))      (pyc from January 2020;
            it is also what the text of the paper describes: "volume from the area of the
            ellipse and the height of the tree")
  J_vol_d = the same but with crown DEPTH instead of height          (final version, whose
            code is not kept; it reproduces the saved Chamrousse outputs)

On top of that, the published J_i (53.7 %) does NOT reproduce from Tusa's own saved outputs:
those 848 pairs average 0.438. That is why the three variants are computed here and the
recalculated reference is reported too, not the published one.

Polygons: the reference one is the real one from the inventory. For the detections,
SegmentAnyTree gives the real convex hull of each instance; AMS3D and the watershed only
publish area, so a circle of equivalent area is used. The error of that approximation is
measured with SegmentAnyTree, which lets us compute both.

Usage: python3 modal/jaccard.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation_core import (SITES, BASE, ROOT, AMS, SAT_OUT, MIN_PTS, region, hull_area,
                      reference, det_watershed, match)


def det_ams(site, plot, cfgs):
    """Like the one in evaluation_core.py but loading the crown depth too."""
    c = cfgs.get((site, plot))
    if c is None:
        return None
    f = os.path.join(AMS, f"e1_dets_{plot}_{c}.csv")
    if not os.path.exists(f):
        return None
    x, y, h, a, d = [], [], [], [], []
    for r in csv.DictReader(open(f)):
        x.append(float(r["x"])); y.append(float(r["y"]))
        h.append(float(r["h"])); a.append(float(r["area"]))
        d.append(float(r.get("depth", "nan") or "nan"))
    return dict(x=np.array(x), y=np.array(y), h=np.array(h),
                area=np.array(a), depth=np.array(d))

import laspy
from scipy.spatial import ConvexHull, QhullError

DM = 3.5


def _pol(v):
    if len(v) < 3:
        return None
    p = Polygon(v)
    if not p.is_valid:
        p = p.buffer(0)
    return p if (p.is_valid and p.area > 1e-6) else None


def reference_polygons(site, plot, ids):
    """Inventory crown polygons, indexed by tree id."""
    if site == "cham":
        f = os.path.join(BASE, f"IRSTEA_dataset/inventory_csv/placette_{plot}_houppiers_coords.csv")
    else:
        f = os.path.join(BASE, f"trento_dataset/csv/{plot}.csv")
    if not os.path.exists(f):
        return {}
    by_id = {}
    for r in csv.DictReader(open(f)):
        by_id.setdefault(str(r["id"]), []).append((float(r["x"]), float(r["y"])))
    return {i: _pol(np.array(by_id[i])) for i in ids if i in by_id}


def ref_extended(site, plot):
    """Reference with id, crown depth and the real polygon."""
    ref = reference(site, plot)
    if site == "cham":
        path = os.path.join(BASE, "IRSTEA_dataset/inventory_csv/Chamrousse.csv")
    else:
        path = os.path.join(BASE, "trento_dataset/csv/Pellizzano.csv")
    ids, prof, cx, cy = [], [], [], []
    for r in csv.DictReader(open(path)):
        if r["plotid"] != plot:
            continue
        try:
            float(r["x"]); float(r["y"]); float(r["height"]); float(r["area"])
        except ValueError:
            continue
        ids.append(str(r["id"]))
        try:
            d = float(r.get("depth", "") or "nan")
        except ValueError:
            d = float("nan")
        prof.append(d)
    ref["id"] = np.array(ids)
    ref["depth"] = np.array(prof, float)
    poly_list = reference_polygons(site, plot, set(ids))
    ref["poly"] = [poly_list.get(i) for i in ids]
    # polygon centroid, for the stem versus crown centroid sensitivity
    ref["cx"] = np.array([p.centroid.x if p else x for p, x in zip(ref["poly"], ref["x"])])
    ref["cy"] = np.array([p.centroid.y if p else y for p, y in zip(ref["poly"], ref["y"])])
    return ref


def det_sat_pol(site, plot):
    """SegmentAnyTree detections with the real convex hull and the equivalent circle."""
    laz = os.path.join(SAT_OUT, f"{site}_{plot}_out.laz")
    if not os.path.exists(laz):
        return None
    L = laspy.read(laz)
    x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    inst = np.asarray(L.PredInstance).astype(np.int64)
    X, Y, H, A, POL, DEP = [], [], [], [], [], []
    for i in np.unique(inst[inst > 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        xy = np.c_[x[m], y[m]]
        k = np.argmax(z[m])
        X.append(x[m][k]); Y.append(y[m][k])
        H.append(float(z[m].max())); DEP.append(float(z[m].max() - z[m].min()))
        A.append(hull_area(xy))
        try:
            POL.append(_pol(xy[ConvexHull(xy).vertices]))
        except (QhullError, Exception):
            POL.append(None)
    return dict(x=np.array(X), y=np.array(Y), h=np.array(H), area=np.array(A),
                depth=np.array(DEP), poly=POL)


def circle(x, y, a):
    return Point(x, y).buffer(max(np.sqrt(max(a, 1e-6) / np.pi), 0.05), quad_segs=16)


def jaccards(pr, pd_, ar, ad, hr, hd, dr, dd):
    """Tusa's three variants. Returns (J_2D, J_vol_height, J_vol_depth)."""
    if pr is None or pd_ is None:
        return None
    try:
        ai = pr.intersection(pd_).area
    except Exception:
        return None
    has_depth = np.isfinite(dr) and dr > 0 and np.isfinite(dd) and dd > 0
    if ai <= 0:
        return (0.0, 0.0, 0.0 if has_depth else float("nan"))
    j2 = ai / (ar + ad - ai) if (ar + ad - ai) > 0 else 0.0
    mh = min(hr, hd)
    den_h = ar * hr + ad * hd - ai * mh
    jh = (ai * mh) / den_h if den_h > 0 else 0.0
    if has_depth:
        md = min(dr, dd)
        den_d = ar * dr + ad * dd - ai * md
        jd = (ai * md) / den_d if den_d > 0 else 0.0
    else:
        jd = float("nan")
    return (j2, jh, jd)


def evaluate(ref, det, use_real_polygon, pos_ref="stem"):
    """Matches and returns the J per pair, plus the areas for the fit."""
    rx = ref["x"] if pos_ref == "stem" else ref["cx"]
    ry = ref["y"] if pos_ref == "stem" else ref["cy"]
    refm = dict(x=rx, y=ry, h=ref["h"], area=ref["area"])
    inside = region(dict(x=ref["x"], y=ref["y"])).contains_points(np.c_[det["x"], det["y"]]) \
        if len(det["x"]) else np.zeros(0, bool)
    idx = np.nonzero(inside)[0]
    dd = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
          for k, v in det.items()}
    nd, nr = len(dd["x"]), len(refm["x"])
    if nd == 0 or nr == 0:
        return [], 0, nd
    D = np.hypot(dd["x"][:, None] - refm["x"][None, :], dd["y"][:, None] - refm["y"][None, :])
    Vd = np.maximum(dd["area"], 1e-6) * np.maximum(dd["h"], 1e-6)
    Vr = np.maximum(refm["area"], 1e-6) * np.maximum(refm["h"], 1e-6)
    JIv = np.minimum(Vd[:, None], Vr[None, :]) / np.maximum(Vd[:, None], Vr[None, :])
    C = np.where(D <= DM, (D / DM) * (1.0 - JIv), np.inf)
    di, rj = np.arange(nd), np.arange(nr)
    pairs = []
    while C.size and np.isfinite(C).any():
        i, j = np.unravel_index(np.argmin(C), C.shape)
        pairs.append((int(di[i]), int(rj[j])))
        C = np.delete(np.delete(C, i, 0), j, 1)
        di, rj = np.delete(di, i), np.delete(rj, j)

    rows = []
    for i, j in pairs:
        pd_ = (dd["poly"][i] if (use_real_polygon and dd.get("poly") and dd["poly"][i] is not None)
               else circle(dd["x"][i], dd["y"][i], dd["area"][i]))
        ad = pd_.area
        jj = jaccards(ref["poly"][j], pd_, ref["area"][j], ad,
                      ref["h"][j], dd["h"][i], ref["depth"][j],
                      dd["depth"][i] if "depth" in dd else float("nan"))
        if jj is None:
            continue
        rows.append(dict(j2=jj[0], jh=jj[1], jd=jj[2],
                          area_ref=float(ref["area"][j]), area_det=float(ad)))
    return rows, len(pairs), nd


def r2_rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), float("nan")
    a, b = a[m], b[m]
    ss_res = float(((a - b) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), float(np.sqrt(ss_res / len(a)))


def tusa_saved_matches():
    """AMS3D's index recomputed from the pairs its authors saved, under their matching.

    Their crown_analysis/inputs/det/<plot>/match.csv carries one row per matched pair
    with the JI they computed. Averaging that column is their own protocol applied to
    their own output, so it is the figure the study compares against the published
    53.7 %. Only the aggregate is stored: those files carry coordinates.

    Returns None when the folder is not here, which is the normal case: it is
    third-party material and is not redistributed.
    """
    for root in (ROOT, BASE):
        det = os.path.join(root, "AMS", "crown_analysis", "inputs", "det")
        if os.path.isdir(det):
            break
    else:
        return None
    by_site, every = {"Chamrousse": [], "Pellizzano": []}, []
    for f in sorted(glob.glob(os.path.join(det, "*", "match.csv"))):
        plot = os.path.basename(os.path.dirname(f))
        site = "Pellizzano" if plot.startswith("p") and plot != "Premol" else "Chamrousse"
        v = [float(r["JI"]) for r in csv.DictReader(open(f)) if r.get("JI")]
        by_site[site] += v
        every += v
    if not every:
        return None
    return {"n_pairs": len(every), "J_mean": round(float(np.mean(every)), 4),
            **{s: {"n_pairs": len(v), "J_mean": round(float(np.mean(v)), 4)}
               for s, v in by_site.items() if v},
            "published_in_the_AMS3D_paper": 0.537}


def main():
    cfgs = {}
    for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        p = os.path.join(AMS, f)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cfgs[(s, r["plot"])] = r["cfg"]

    # Watershed setting chosen per plot in the sweep, so that the three methods are
    # compared at their best setting and not one of them at its default.
    wcfg = {}
    pb = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "watershed_sweep.json")
    if os.path.exists(pb):
        import json as _j
        _b = _j.load(open(pb))
        for _sit, _d in _b.items():
            for _pl, _c in _d.get("configs", {}).items():
                wcfg[(_sit, _pl)] = (float(_c["hmin"]), float(_c["mindist"]))
    print(f"watershed settings loaded: {len(wcfg)}", flush=True)

    print("Computing (takes a few minutes)...", flush=True)
    out = {}
    for site, name, plots in SITES:
        acc = {}
        for p in plots:
            ref = ref_extended(site, p)
            dets = {"SAT": det_sat_pol(site, p),
                    "AMS3D": det_ams(site, p, cfgs),
                    "Watershed": det_watershed(ref, hmin=wcfg.get((name, p), (2.0, 2.0))[0],
                                               mindist=wcfg.get((name, p), (2.0, 2.0))[1])}
            for met, d in dets.items():
                if d is None:
                    continue
                real = (met == "SAT")
                rows, tp, nd = evaluate(ref, d, use_real_polygon=real)
                acc.setdefault(met, []).extend(rows)
                if met == "SAT":     # same matching, polygon approximated by a circle
                    f2, _, _ = evaluate(ref, d, use_real_polygon=False)
                    acc.setdefault("SAT_circle", []).extend(f2)
        out[name] = {}
        print(f"\n{'='*80}\n{name}: mean Jaccard index over the matched pairs (d=3.5 m)")
        print(f"{'method':14} {'pairs':>8} {'J_2D':>8} {'J_vol_h':>9} {'J_vol_d':>9} "
              f"{'area R2':>8} {'RMSE m2':>8} {'%no overlap':>8}")
        for met in ("SAT", "SAT_circle", "AMS3D", "Watershed"):
            f = acc.get(met, [])
            if not f:
                continue
            j2 = np.nanmean([r["j2"] for r in f])
            jh = np.nanmean([r["jh"] for r in f])
            jd_v = [r["jd"] for r in f if np.isfinite(r["jd"])]
            jd = np.mean(jd_v) if jd_v else float("nan")
            r2, rmse = r2_rmse([r["area_ref"] for r in f], [r["area_det"] for r in f])
            no_overlap = round(100.0 * float(np.mean([r["j2"] <= 0 for r in f])), 1)
            out[name][met] = dict(n_pairs=len(f), J_2D=round(float(j2), 3),
                                    J_vol_height=round(float(jh), 3),
                                    J_vol_depth=(round(float(jd), 3) if jd_v else None),
                                    area_R2=round(r2, 3), area_RMSE=round(rmse, 2),
                                    pct_no_overlap=no_overlap)
            print(f"{met:14} {len(f):>8} {j2:>8.3f} {jh:>9.3f} "
                  f"{(f'{jd:.3f}' if jd_v else '-'):>9} {r2:>8.3f} {rmse:>8.2f} "
                  f"{no_overlap:>8.1f}")

    # sensitivity: reference position, stem versus crown centroid
    print(f"\n{'='*80}\nSENSITIVITY: reference position (stem versus polygon centroid)")
    print(f"{'site':12} {'method':10} {'pos_ref':10} {'pairs':>8} {'J_vol_h':>9}")
    for site, name, plots in SITES:
        for met in ("SAT", "AMS3D"):
            for pos in ("stem", "centroid"):
                tot, js = 0, []
                for p in plots:
                    ref = ref_extended(site, p)
                    d = det_sat_pol(site, p) if met == "SAT" else det_ams(site, p, cfgs)
                    if d is None:
                        continue
                    rows, tp, _ = evaluate(ref, d, use_real_polygon=(met == "SAT"), pos_ref=pos)
                    tot += tp
                    js += [r["jh"] for r in rows]
                out.setdefault("sensitivity_ref_pos", {}).setdefault(name, {}) \
                    .setdefault(met, {})[pos] = dict(n_pairs=tot,
                                                     J_vol_height=round(float(np.nanmean(js)), 3) if js else None)
                print(f"{name:12} {met:10} {pos:10} {tot:>8} "
                      f"{(np.nanmean(js) if js else float('nan')):>9.3f}")

    tusa = tusa_saved_matches()
    if tusa:
        out["tusa_saved_matches"] = tusa
        print(f"\nAMS3D under its own matching, from its saved pairs: "
              f"{tusa['J_mean']:.3f} over {tusa['n_pairs']} pairs "
              f"(the paper publishes 53.7 %)")

    json.dump(out, open(os.path.join(ROOT, "jaccard.json"), "w"), indent=2, ensure_ascii=False)
    print("\nsaved to jaccard.json")


if __name__ == "__main__":
    main()
