"""Full analysis: SegmentAnyTree zero-shot, AMS3D and watershed under a single evaluator.

All local, no GPU. Produces:
  1. Detection per site with confidence intervals from a bootstrap resample of plots.
  2. Breakdown by detectability stratum (dominant, intermediate, suppressed).
  3. Two matching protocols: distance only, and with Tusa's overlap term.
  4. Apex versus centroid sensitivity for SegmentAnyTree.
  5. Watershed on the CHM as a third method, computed right here.

Usage: python3 modal/evaluation_core.py
"""
import csv
import json
import os
import sys

import laspy
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, QhullError

# The datasets are not redistributed with the code, so where they live is configurable:
# TREELEARN_DATA points at the directory holding IRSTEA_dataset/ and trento_dataset/.
# Without it, the layout of the working copy is assumed.
BASE = os.environ.get("TREELEARN_DATA") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
SAT_OUT = os.path.join(ROOT, "sat_output_all")
AMS = os.path.join(BASE, "ams_port", "out")
# Where the code itself lives, which is not where the data lives.
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_path(name):
    """Where to read an aggregated result from.

    A file just regenerated next to the project wins; the copy published in results/
    is the fallback, so a clone works without rerunning anything first.
    """
    fresh = os.path.join(ROOT, name)
    return fresh if os.path.exists(fresh) else os.path.join(CODE, "results", name)

DMATCH = (2.5, 3.5, 5.0)
BUF = 1.5
MIN_PTS = 10
NBOOT = 2000
RNG = np.random.default_rng(0)
# Strata by number of ALS returns inside the crown (the same rule we have been using)
DOM, SUP = 300, 50

CHAM = ["1", "1b", "2", "3", "3b", "4", "Premol"]
PELL = [f"p{i}" for i in range(1, 16)]
SITES = (("cham", "Chamrousse", CHAM), ("pell", "Pellizzano", PELL))


# ----------------------------------------------------------------- reference
def reference(site, plot):
    """Inventory trees: position, height, crown area, family."""
    if site == "cham":
        path = os.path.join(BASE, "IRSTEA_dataset/inventory_csv/Chamrousse.csv")
        las = os.path.join(BASE, f"IRSTEA_dataset/las/las{plot}.las")
    else:
        path = os.path.join(BASE, "trento_dataset/csv/Pellizzano.csv")
        las = os.path.join(BASE, f"trento_dataset/las/{plot}.las")
    x, y, h, a, fam = [], [], [], [], []
    for r in csv.DictReader(open(path)):
        if r["plotid"] != plot:
            continue
        try:
            x.append(float(r["x"])); y.append(float(r["y"]))
            h.append(float(r["height"])); a.append(float(r["area"]))
            fam.append(r.get("family", "?"))
        except ValueError:
            pass
    ref = dict(x=np.array(x), y=np.array(y), h=np.array(h), area=np.array(a),
               fam=np.array(fam), las=las)
    ref["npts"] = returns_per_tree(ref)
    n = ref["npts"]
    ref["stratum"] = np.where(n >= DOM, "dominant",
                              np.where(n < SUP, "suppressed", "intermediate"))
    return ref


def returns_per_tree(ref):
    """ALS returns inside each tree's crown cylinder: a proxy for visibility."""
    L = laspy.read(ref["las"])
    px, py, pz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    m = pz >= 1.5
    px, py, pz = px[m], py[m], pz[m]
    rad = np.sqrt(np.maximum(ref["area"], 0.1) / np.pi)
    base = 0.4 * ref["h"]                       # approximate crown base
    out = np.zeros(len(ref["x"]), int)
    for i in range(len(ref["x"])):
        d2 = (px - ref["x"][i]) ** 2 + (py - ref["y"][i]) ** 2
        out[i] = int(np.count_nonzero((d2 <= rad[i] ** 2) &
                                      (pz >= base[i]) & (pz <= ref["h"][i] + 2.0)))
    return out


def region(ref, buf=BUF):
    poly = np.c_[ref["x"], ref["y"]][ConvexHull(np.c_[ref["x"], ref["y"]]).vertices]
    c = poly.mean(0)
    d = np.linalg.norm(poly - c, axis=1)
    d[d == 0] = 1e-9
    return MplPath(c + (poly - c) * ((d + buf) / d)[:, None])


# ----------------------------------------------------------------- detections
def hull_area(xy):
    if len(xy) < 3:
        return 0.5
    try:
        return float(ConvexHull(xy).volume)
    except (QhullError, Exception):
        return max(float(np.ptp(xy[:, 0]) * np.ptp(xy[:, 1])), 0.5)


def det_sat(site, plot, pos="apex"):
    laz = os.path.join(SAT_OUT, f"{site}_{plot}_out.laz")
    if not os.path.exists(laz):
        return None
    L = laspy.read(laz)
    x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    inst = np.asarray(L.PredInstance).astype(np.int64)
    X, Y, H, A = [], [], [], []
    for i in np.unique(inst[inst > 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        xy = np.c_[x[m], y[m]]
        if pos == "apex":
            k = np.argmax(z[m]); X.append(x[m][k]); Y.append(y[m][k])
        else:
            X.append(xy[:, 0].mean()); Y.append(xy[:, 1].mean())
        H.append(float(z[m].max())); A.append(hull_area(xy))
    return dict(x=np.array(X), y=np.array(Y), h=np.array(H), area=np.array(A))


def det_ams(site, plot, cfgs):
    c = cfgs.get((site, plot))
    if c is None:
        return None
    f = os.path.join(AMS, f"e1_dets_{plot}_{c}.csv")
    if not os.path.exists(f):
        return None
    x, y, h, a = [], [], [], []
    for r in csv.DictReader(open(f)):
        x.append(float(r["x"])); y.append(float(r["y"]))
        h.append(float(r["h"])); a.append(float(r["area"]))
    return dict(x=np.array(x), y=np.array(y), h=np.array(h), area=np.array(a))


def det_watershed(ref, cell=0.5, hmin=2.0, mindist=2.0, sigma=1.0):
    """Marker-controlled watershed on the CHM. Third method, no training."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    L = laspy.read(ref["las"])
    px, py, pz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    m = pz >= 0
    px, py, pz = px[m], py[m], pz[m]
    x0, y0 = px.min(), py.min()
    nx = int((px.max() - x0) / cell) + 1
    ny = int((py.max() - y0) / cell) + 1
    chm = np.zeros((ny, nx), np.float32)
    j = np.clip(((px - x0) / cell).astype(int), 0, nx - 1)
    i = np.clip(((py - y0) / cell).astype(int), 0, ny - 1)
    np.maximum.at(chm, (i, j), pz)
    chm = gaussian_filter(chm, sigma)
    mask = chm >= hmin
    if not mask.any():
        return dict(x=np.array([]), y=np.array([]), h=np.array([]), area=np.array([]))
    peaks = peak_local_max(chm, min_distance=max(1, int(mindist / cell)),
                           threshold_abs=hmin, labels=mask)
    if not len(peaks):
        return dict(x=np.array([]), y=np.array([]), h=np.array([]), area=np.array([]))
    markers = np.zeros_like(chm, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    lab = watershed(-chm, markers, mask=mask)
    X, Y, H, A = [], [], [], []
    for k in range(1, len(peaks) + 1):
        s = lab == k
        n = int(s.sum())
        if n < 2:
            continue
        ii, jj = np.nonzero(s)
        top = np.argmax(chm[ii, jj])
        X.append(x0 + (jj[top] + 0.5) * cell)
        Y.append(y0 + (ii[top] + 0.5) * cell)
        H.append(float(chm[ii, jj].max()))
        A.append(n * cell * cell)
    return dict(x=np.array(X), y=np.array(Y), h=np.array(H), area=np.array(A))


# ----------------------------------------------------------------- match
def match(det, ref, dmax, with_overlap):
    """Returns the indices of the matched reference trees.

    with_overlap=False: greedy by 2D distance.
    with_overlap=True : greedy by (d/dmax)*(1-JI), JI = Jaccard of the area*height volume
                        (the term from Tusa's protocol). Same admission condition d<=dmax.
    """
    nd, nr = len(det["x"]), len(ref["x"])
    if nd == 0 or nr == 0:
        return np.zeros(0, int)
    D = np.hypot(det["x"][:, None] - ref["x"][None, :],
                 det["y"][:, None] - ref["y"][None, :])
    if with_overlap:
        Vd = np.maximum(det["area"], 1e-6) * np.maximum(det["h"], 1e-6)
        Vr = np.maximum(ref["area"], 1e-6) * np.maximum(ref["h"], 1e-6)
        JI = np.minimum(Vd[:, None], Vr[None, :]) / np.maximum(Vd[:, None], Vr[None, :])
        C = (D / dmax) * (1.0 - JI)
    else:
        C = D.copy()
    C = np.where(D <= dmax, C, np.inf)
    di, rj = np.arange(nd), np.arange(nr)
    matched = []
    while C.size and np.isfinite(C).any():
        i, j = np.unravel_index(np.argmin(C), C.shape)
        matched.append(int(rj[j]))
        C = np.delete(np.delete(C, i, 0), j, 1)
        di, rj = np.delete(di, i), np.delete(rj, j)
    return np.array(matched, int)


def evaluate_plot(ref, det, dmax, with_overlap):
    inside = region(ref).contains_points(np.c_[det["x"], det["y"]]) if len(det["x"]) else np.zeros(0, bool)
    dd = {k: v[inside] for k, v in det.items() if isinstance(v, np.ndarray)}
    matched = match(dd, ref, dmax, with_overlap)
    return dict(NR=len(ref["x"]), ND=int(inside.sum()), TP=len(matched), matched=matched)


# ----------------------------------------------------------------- aggregation
def metrics(TP, NR, ND):
    R = TP / NR if NR else 0.0
    P = TP / ND if ND else 0.0
    return R, P, (2 * P * R / (P + R) if P + R else 0.0)


def bootstrap(per_plot, n=NBOOT):
    k = len(per_plot)
    if k < 2:
        return {}
    Rs, Ps, Fs = [], [], []
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        TP = sum(per_plot[i]["TP"] for i in idx)
        NR = sum(per_plot[i]["NR"] for i in idx)
        ND = sum(per_plot[i]["ND"] for i in idx)
        R, P, F = metrics(TP, NR, ND)
        Rs.append(R); Ps.append(P); Fs.append(F)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {"recall_ci": q(Rs), "precision_ci": q(Ps), "f1_ci": q(Fs)}


def main():
    cfgs = {}
    for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        p = os.path.join(AMS, f)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cfgs[(s, r["plot"])] = r["cfg"]

    print("Loading references and detections (may take a couple of minutes)...", flush=True)
    data = {}
    for site, _, plots in SITES:
        for p in plots:
            ref = reference(site, p)
            data[(site, p)] = {
                "ref": ref,
                "SAT": det_sat(site, p, "apex"),
                "SAT_centroid": det_sat(site, p, "centroid"),
                "AMS3D": det_ams(site, p, cfgs),
                "Watershed": det_watershed(ref),
            }
        print(f"  {site} done", flush=True)

    methods = ["SAT", "AMS3D", "Watershed"]
    output = {"per_site": {}, "strata": {}, "sensitivity": {}}

    # ---------- 1 and 3: detection per site, two protocols, with CIs
    for with_overlap, label_of in ((False, "distance_only"), (True, "with_overlap_Tusa")):
        print(f"\n{'='*86}\nPROTOCOL: {label_of.replace('_',' ')}")
        print(f"{'site':11} {'method':10} {'dm':>4} {'NR':>5} {'ND':>5} {'TP':>5} "
              f"{'recall':>8} {'95% CI':>16} {'precis':>7} {'F1':>6}")
        for site, name, plots in SITES:
            for met in methods:
                for dm in DMATCH:
                    pp = [evaluate_plot(data[(site, p)]["ref"], data[(site, p)][met], dm, with_overlap)
                          for p in plots if data[(site, p)][met] is not None]
                    TP, NR, ND = (sum(e[k] for e in pp) for k in ("TP", "NR", "ND"))
                    R, P, F = metrics(TP, NR, ND)
                    ci = bootstrap(pp)
                    output["per_site"].setdefault(label_of, {}).setdefault(name, {}) \
                        .setdefault(met, {})[str(dm)] = dict(
                            NR=NR, ND=ND, TP=TP, recall=round(R, 3), precision=round(P, 3),
                            f1=round(F, 3), **ci)
                    print(f"{name:11} {met:10} {dm:>4} {NR:>5} {ND:>5} {TP:>5} "
                          f"{R:>8.3f} {str(ci.get('recall_ci','')):>16} {P:>7.3f} {F:>6.3f}")
            print()

    # ---------- 2: strata, at 3.5 m, protocol with overlap
    print("=" * 86)
    print("DETECTABILITY STRATA (ALS returns in the crown: dominant >=300, suppressed <50)")
    print(f"{'site':11} {'stratum':12} {'trees':>8} " + " ".join(f"{m:>10}" for m in methods))
    for site, name, plots in SITES:
        acc = {e: {"n": 0, **{m: 0 for m in methods}} for e in ("dominant", "intermediate", "suppressed")}
        for p in plots:
            ref = data[(site, p)]["ref"]
            for e in acc:
                acc[e]["n"] += int((ref["stratum"] == e).sum())
            for met in methods:
                d = data[(site, p)][met]
                if d is None:
                    continue
                cas = evaluate_plot(ref, d, 3.5, True)["matched"]
                for e in acc:
                    acc[e][met] += int((ref["stratum"][cas] == e).sum()) if len(cas) else 0
        output["strata"][name] = acc
        for e in ("dominant", "intermediate", "suppressed"):
            n = acc[e]["n"]
            print(f"{name:11} {e:12} {n:>8} " +
                  " ".join(f"{(acc[e][m]/n if n else 0):>10.3f}" for m in methods))
        print()

    # ---------- 4: apex versus centroid
    print("=" * 86)
    print("SegmentAnyTree SENSITIVITY: apex versus instance centroid (3.5 m, with overlap)")
    print(f"{'site':11} {'position':12} {'TP':>5} {'recall':>8} {'precis':>8} {'F1':>7}")
    for site, name, plots in SITES:
        for key, label in (("SAT", "apex"), ("SAT_centroid", "centroid")):
            pp = [evaluate_plot(data[(site, p)]["ref"], data[(site, p)][key], 3.5, True)
                  for p in plots]
            TP, NR, ND = (sum(e[k] for e in pp) for k in ("TP", "NR", "ND"))
            R, P, F = metrics(TP, NR, ND)
            output["sensitivity"].setdefault(name, {})[label] = dict(
                TP=TP, ND=ND, recall=round(R, 3), precision=round(P, 3), f1=round(F, 3))
            print(f"{name:11} {label:12} {TP:>5} {R:>8.3f} {P:>8.3f} {F:>7.3f}")
        print()

    json.dump(output, open(os.path.join(ROOT, "full_analysis.json"), "w"),
              indent=2, ensure_ascii=False, default=str)
    print("saved to full_analysis.json")


if __name__ == "__main__":
    main()
