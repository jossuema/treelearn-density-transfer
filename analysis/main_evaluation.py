"""All the main numbers of the study, from all_detections.json.

A single evaluator for the five methods, and three robustness checks: optimal matching instead of greedy, sensitivity to how the region is
defined, and the relation between return density and accuracy plot by plot, which is
what holds the argument up.

Usage: python3 analysis/main_evaluation.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import SITES, reference, region  # noqa: E402

MET = ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")
DIST = (2.5, 3.5, 5.0)
NB = 2000
STRATA = ("dominant", "intermediate", "suppressed")
REF = {}


def ref_of(k):
    if k not in REF:
        REF[k] = reference(*k.split("/"))
    return REF[k]


def metrics(TP, NR, ND):
    r = TP / NR if NR else 0.0
    p = TP / ND if ND else 0.0
    return r, p, (2 * p * r / (p + r) if p + r else 0.0)


def greedy_pairs(dx, dy, rx, ry, dmax):
    """One to one by increasing distance. It is what we had been using."""
    if not len(dx) or not len(rx):
        return []
    D = np.hypot(np.asarray(dx)[:, None] - rx[None, :],
                 np.asarray(dy)[:, None] - ry[None, :])
    cand = np.argwhere(D <= dmax)
    order = np.argsort(D[cand[:, 0], cand[:, 1]], kind="stable")
    ud, ur, out = set(), set(), []
    for i, j in cand[order]:
        if i in ud or j in ur:
            continue
        ud.add(i); ur.add(j); out.append((int(i), int(j)))
    return out


def optimal_pairs(dx, dy, rx, ry, dmax):
    """Minimum cost matching. Maximizes the number of valid pairs."""
    from scipy.optimize import linear_sum_assignment
    if not len(dx) or not len(rx):
        return []
    D = np.hypot(np.asarray(dx)[:, None] - rx[None, :],
                 np.asarray(dy)[:, None] - ry[None, :])
    C = np.where(D <= dmax, D, 1e6)
    fi, co = linear_sum_assignment(C)
    return [(int(i), int(j)) for i, j in zip(fi, co) if D[i, j] <= dmax]


def count(data, keys, met, dmax, buf=1.5, optimal=False, whole_tile=False):
    """TP, ND and NR per plot with a given definition of region and matching."""
    out = []
    pair_up = optimal_pairs if optimal else greedy_pairs
    for k in keys:
        ref = ref_of(k)
        d = data["detections"][k][met]
        x, y = np.array(d["x"]), np.array(d["y"])
        if whole_tile:
            inside = np.ones(len(x), bool)
        else:
            inside = region(ref, buf).contains_points(np.c_[x, y]) if len(x) \
                else np.zeros(0, bool)
        pr = pair_up(x[inside], y[inside], ref["x"], ref["y"], dmax)
        out.append(dict(TP=len(pr), ND=int(inside.sum()), NR=len(ref["x"]),
                        ok={j for _, j in pr}))
    return out


def aggregate(ps, seed=0):
    TP, ND, NR = (sum(p[c] for p in ps) for c in ("TP", "ND", "NR"))
    r, p, f1 = metrics(TP, NR, ND)
    rng = np.random.default_rng(seed)
    R, P, F = [], [], []
    for _ in range(NB):
        i = rng.integers(0, len(ps), len(ps))
        a, b, c = (sum(ps[j][x] for j in i) for x in ("TP", "ND", "NR"))
        rr, pp, ff = metrics(a, c, b)
        R.append(rr); P.append(pp); F.append(ff)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3),
                   round(float(np.percentile(v, 97.5)), 3)]
    return dict(TP=TP, ND=ND, NR=NR, recall=round(r, 3), precision=round(p, 3),
                f1=round(f1, 3), recall_ci=q(R), precision_ci=q(P), f1_ci=q(F))


def spearman(a, b, nperm=10000, seed=0):
    """Rank correlation with a permutation p, without depending on scipy.stats."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    r = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    n = sum(abs(float(np.corrcoef(rng.permutation(ra), rb)[0, 1])) >= abs(r)
            for _ in range(nperm))
    # the p is not rounded: its floor is 1/(nperm+1), and rounding to four places
    # turns that floor into 0 for nperm >= 20000, which would state an impossibility
    return round(r, 3), (n + 1) / (nperm + 1)


def main():
    data = json.load(open(os.path.join(ROOT, "all_detections.json")))
    keys = [f"{s}/{p}" for s, _, ps in SITES for p in ps]
    sites = {n: [f"{s}/{p}" for p in ps] for s, n, ps in SITES}
    groups = list(sites.items()) + [("Both", keys)]
    output = {"treelite_setting": data["treelite_setting"],
           "density": data["density"]}

    print("=" * 78)
    print("1. Detection at 3.5 m, region = hull dilated 1.5 m, greedy matching")
    print(f"{'method':12} " + " ".join(f"{n:>24}" for n, _ in groups))
    output["main"] = {}
    base = {m: {n: count(data, ks, m, 3.5) for n, ks in groups} for m in MET}
    for m in MET:
        row = f"{m:12} "
        for n, _ in groups:
            v = aggregate(base[m][n])
            output["main"].setdefault(n, {})[m] = v
            row += f" {v['recall']:.3f}/{v['precision']:.3f}/{v['f1']:.3f}      "
        print(row)
    print("   (recall / precision / F1)")

    print("\n" + "=" * 78)
    print("2. Optimal versus greedy matching, at 3.5 m, total hits")
    print(f"{'method':12} {'greedy':>10} {'optimal':>9} {'diff':>5} {'rec gre':>8} {'rec opt':>8}")
    output["matching"] = {}
    for m in MET:
        g = sum(p["TP"] for p in base[m]["Both"])
        h = sum(p["TP"] for p in count(data, keys, m, 3.5, optimal=True))
        nr = sum(p["NR"] for p in base[m]["Both"])
        output["matching"][m] = dict(greedy=g, optimal=h, diff=h - g,
                                        recall_greedy=round(g / nr, 3),
                                        recall_optimal=round(h / nr, 3))
        print(f"{m:12} {g:>10} {h:>9} {h - g:>+5} {g / nr:>8.3f} {h / nr:>8.3f}")

    print("\n" + "=" * 78)
    print("3. How the region is defined, at 3.5 m, both sites")
    print(f"{'method':12} " + " ".join(f"{t:>22}" for t in
          ("hull +0 m", "hull +1.5 m", "hull +3 m", "whole tile")))
    output["region"] = {}
    for m in MET:
        row = f"{m:12} "
        for et, kw in (("hull +0 m", dict(buf=0.0)), ("hull +1.5 m", dict(buf=1.5)),
                       ("hull +3 m", dict(buf=3.0)),
                       ("whole tile", dict(whole_tile=True))):
            ps = count(data, keys, m, 3.5, **kw)
            TP, ND, NR = (sum(p[c] for p in ps) for c in ("TP", "ND", "NR"))
            r, p, f1 = metrics(TP, NR, ND)
            output["region"].setdefault(m, {})[et] = dict(
                TP=TP, ND=ND, recall=round(r, 3), precision=round(p, 3), f1=round(f1, 3))
            row += f" {r:.3f}/{p:.3f}/{f1:.3f}   "
        print(row)

    print("\n" + "=" * 78)
    print("4. Matching tolerance")
    print(f"{'method':12} " + " ".join(f"{d:>22}" for d in ("2.5 m", "3.5 m", "5.0 m")))
    output["distance"] = {}
    for m in MET:
        row = f"{m:12} "
        for d in DIST:
            ps = count(data, keys, m, d)
            TP, ND, NR = (sum(p[c] for p in ps) for c in ("TP", "ND", "NR"))
            r, p, f1 = metrics(TP, NR, ND)
            output["distance"].setdefault(m, {})[str(d)] = dict(
                recall=round(r, 3), precision=round(p, 3), f1=round(f1, 3))
            row += f" {r:.3f}/{p:.3f}/{f1:.3f}   "
        print(row)

    print("\n" + "=" * 78)
    print("5. Return density versus accuracy, plot by plot (22 plots)")
    dens = np.array([data["density"][k]["density"] for k in keys])
    per_plot_of = {m: np.array([p["TP"] / p["NR"] if p["NR"] else np.nan
                            for p in base[m]["Both"]]) for m in MET}
    output["density_vs_recall"] = {}
    print(f"{'method':12} {'rho recall':>12} {'p':>8} | {'rho gap vs SAT':>16} {'p':>8}")
    for m in MET:
        rho, pv = spearman(dens, per_plot_of[m])
        if m == "SAT":
            print(f"{m:12} {rho:>9.3f} {pv:>8.4f} |")
            output["density_vs_recall"][m] = dict(rho=rho, p=pv)
            continue
        d = per_plot_of["SAT"] - per_plot_of[m]
        rd, pd = spearman(dens, d)
        output["density_vs_recall"][m] = dict(rho=rho, p=pv, rho_diff=rd, p_diff=pd,
                                             mean_diff=round(float(np.mean(d)), 3))
        print(f"{m:12} {rho:>9.3f} {pv:>8.4f} | {rd:>16.3f} {pd:>8.4f}")
    output["per_plot"] = {k: {m: round(float(per_plot_of[m][i]), 3) for m in MET}
                          for i, k in enumerate(keys)}

    print("\n" + "=" * 78)
    print("6. Detectability strata, at 3.5 m")
    print(f"{'site':12} {'stratum':12} {'n':>5} " + " ".join(f"{m:>11}" for m in MET))
    output["strata"] = {}
    for n, ks in groups:
        idx = {k: i for i, k in enumerate(keys)}
        for est in STRATA:
            tot = 0
            tps = {m: 0 for m in MET}
            for k in ks:
                ref = ref_of(k)
                m_est = ref["stratum"] == est
                tot += int(m_est.sum())
                for m in MET:
                    ok = base[m]["Both"][idx[k]]["ok"]
                    tps[m] += sum(1 for j in ok if ref["stratum"][j] == est)
            output["strata"].setdefault(n, {})[est] = dict(
                n=tot, **{m: dict(TP=tps[m], recall=round(tps[m] / tot, 3) if tot else 0.0)
                          for m in MET})
            print(f"{n:12} {est:12} {tot:>5} " +
                  " ".join(f"{tps[m] / tot if tot else 0:>11.3f}" for m in MET))

    json.dump(output, open(os.path.join(ROOT, "main_evaluation.json"), "w"),
              indent=2, ensure_ascii=False, default=str)
    print("\nsaved to main_evaluation.json")


if __name__ == "__main__":
    main()
