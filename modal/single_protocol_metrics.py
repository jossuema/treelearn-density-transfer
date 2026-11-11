"""Recomputes every figure of the study under ONE SINGLE matching protocol.

Reason: evaluation_core.py mixed criteria without saying so. The per site table uses
matching by distance only, but the strata and the sensitivity of apex against
centroid used the overlap term of Tusa, and watershed_sweep.json too. Here
everything is redone with distance only, which is what the methods section
describes, and the protocol with overlap stays as a robustness check.

The source is viewer/data/trees.json, which already has the actual pairs per
method and distance, cropped to the inventoried region, and which reproduces the per
site table of full_analysis.json with a difference smaller than 0.0005.

Usage: python3 modal/single_protocol_metrics.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)

TREES = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))
SITES = (("cham", "Chamrousse", ["1", "1b", "2", "3", "3b", "4", "Premol"]),
          ("pell", "Pellizzano", [f"p{i}" for i in range(1, 16)]))
METHODS = ("SAT", "AMS3D", "WS")
DMATCH = ("2.5", "3.5", "5.0")
STRATA = ("dominant", "intermediate", "suppressed")
NBOOT = 2000


def metrics(TP, NR, ND):
    R = TP / NR if NR else 0.0
    P = TP / ND if ND else 0.0
    return R, P, (2 * P * R / (P + R) if P + R else 0.0)


def aggregate(keys, met, d):
    """Adds TP, ND and NR over a set of plots and gets the interval by bootstrap.

    The seed is reset on every call so that the result does not depend on the
    order in which the aggregates are asked for.
    """
    ps = [TREES[k]["met"][met][d] for k in keys if k in TREES]
    TP = sum(p["TP"] for p in ps); ND = sum(p["ND"] for p in ps); NR = sum(p["NR"] for p in ps)
    R, P, F = metrics(TP, NR, ND)
    rng = np.random.default_rng(0)
    Rs, Ps, Fs = [], [], []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(ps), len(ps))
        tp = sum(ps[i]["TP"] for i in idx)
        nd = sum(ps[i]["ND"] for i in idx)
        nr = sum(ps[i]["NR"] for i in idx)
        r, p, f = metrics(tp, nr, nd)
        Rs.append(r); Ps.append(p); Fs.append(f)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return dict(TP=TP, ND=ND, NR=NR, recall=round(R, 3), precision=round(P, 3), f1=round(F, 3),
                recall_ci=q(Rs), precision_ci=q(Ps), f1_ci=q(Fs), n_plots=len(ps))


def strata(keys, met, d):
    out = {e: {"n": 0, "TP": 0} for e in STRATA}
    for k in keys:
        if k not in TREES:
            continue
        dd = TREES[k]
        ok = {r for _, r in dd["pairs"][met][d]}
        for i, e in enumerate(dd["inv"]["stratum"]):
            if e not in out:
                continue
            out[e]["n"] += 1
            if i in ok:
                out[e]["TP"] += 1
    for e in out:
        out[e]["recall"] = round(out[e]["TP"] / out[e]["n"], 3) if out[e]["n"] else 0.0
    return out


def uncropped_metrics(keys, met, d):
    """Metrics if it were not cropped to the inventory: detections from outside count.

    They have to be rematched, it is not enough to add to the denominator, because a
    detection from outside can fall closer than d to a tree at the edge.
    """
    TP = ND = NR = 0
    dm = float(d)
    for k in keys:
        if k not in TREES:
            continue
        a = TREES[k]
        X = np.array(a["det"][met]["x"] + a["det_outside"][met]["x"])
        Y = np.array(a["det"][met]["y"] + a["det_outside"][met]["y"])
        rx, ry = np.array(a["inv"]["x"]), np.array(a["inv"]["y"])
        NR += len(rx); ND += len(X)
        if not len(X) or not len(rx):
            continue
        D = np.hypot(X[:, None] - rx[None, :], Y[:, None] - ry[None, :])
        C = np.where(D <= dm, D, np.inf)
        while C.size and np.isfinite(C).any():
            i, j = np.unravel_index(np.argmin(C), C.shape)
            TP += 1
            C = np.delete(np.delete(C, i, 0), j, 1)
    R, P, F = metrics(TP, NR, ND)
    return dict(TP=TP, ND=ND, NR=NR, recall=round(R, 3), precision=round(P, 3), f1=round(F, 3))


def main():
    output = {"note": "greedy matching by distance only; AMS3D and watershed with their "
                   "best setting per plot; SegmentAnyTree with no tuning",
           "per_site": {}, "global": {}, "strata": {}}
    all_available = [f"{s}/{p}" for s, _, ps in SITES for p in ps]

    for s, name, ps in SITES:
        keys = [f"{s}/{p}" for p in ps]
        output["per_site"][name] = {m: {d: aggregate(keys, m, d) for d in DMATCH} for m in METHODS}
        output["strata"][name] = {m: strata(keys, m, "3.5") for m in METHODS}
    output["global"] = {m: {d: aggregate(all_available, m, d) for d in DMATCH} for m in METHODS}
    output["uncropped"] = {}
    for s_, name, ps in SITES:
        keys = [f"{s_}/{p}" for p in ps]
        output["uncropped"][name] = {m: uncropped_metrics(keys, m, "3.5") for m in METHODS}

    with open(os.path.join(ROOT, "consistent_results.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("DETECTION at 3.5 m, one single protocol")
    print(f"{'site':12} {'method':7} {'recall':>8} {'95% CI':>16} {'precis':>8} {'F1':>7} {'TP/ND/NR':>16}")
    for name in list(output["per_site"]) + ["Both"]:
        src = output["per_site"].get(name, output["global"])
        for m in METHODS:
            a = src[m]["3.5"] if name != "Both" else output["global"][m]["3.5"]
            print(f"{name:12} {m:7} {a['recall']:>8.3f} {str(a['recall_ci']):>16} "
                  f"{a['precision']:>8.3f} {a['f1']:>7.3f} {a['TP']:>5}/{a['ND']}/{a['NR']}")
        print()

    print("STRATA at 3.5 m")
    for name in output["strata"]:
        for m in METHODS:
            e = output["strata"][name][m]
            print(f"{name:12} {m:7} " + "  ".join(
                f"{k[:3]} {e[k]['recall']:.3f} ({e[k]['TP']}/{e[k]['n']})" for k in STRATA))
        print()

    print("EFFECT OF CROPPING at 3.5 m: precision with and without cropping to the inventory")
    for name in output["uncropped"]:
        for m in METHODS:
            cropped = output["per_site"][name][m]["3.5"]
            uncropped = output["uncropped"][name][m]
            print(f"{name:12} {m:7} precision {cropped['precision']:.3f} -> {uncropped['precision']:.3f}   "
                  f"recall {cropped['recall']:.3f} -> {uncropped['recall']:.3f}   "
                  f"ND {cropped['ND']} -> {uncropped['ND']}")
        print()

    print("RECALL against distance")
    for name in output["per_site"]:
        for m in METHODS:
            print(f"{name:12} {m:7} " + "  ".join(
                f"{d} m {output['per_site'][name][m][d]['recall']:.3f}" for d in DMATCH))
    print("\nsaved in consistent_results.json")


if __name__ == "__main__":
    main()
