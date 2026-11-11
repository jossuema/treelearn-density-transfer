"""The five methods in the same table, with differences paired by plot.

Sources, all under the same protocol (inventoried region, one to one matching by
distance alone, D = 3.5 m):
    SAT, AMS3D, WS   viewer/data/trees.json, which is what single_protocol_metrics.py uses
    FF3D             ff3d_evaluation.json
    TreeLite3D       the sweep, with the setting the leave-one-out protocol picks

The differences are computed paired: plots are resampled, not methods, so the interval
captures that the two methods saw exactly the same plots.

Usage: python3 analysis/paired_comparison.py
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
from evaluation_core import SITES  # noqa: E402
from treelite_results import load, f1_of, metrics  # noqa: E402

NB = 2000
D = "3.5"
ORDER = ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")


def gather():
    """Per method and plot: TP, ND, NR."""
    arb = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))
    keys = [f"{s}/{p}" for s, _, ps in SITES for p in ps]
    data = {m: {} for m in ORDER}
    for k in keys:
        for m in ("SAT", "AMS3D", "WS"):
            v = arb[k]["met"][m][D]
            data[m][k] = dict(TP=v["TP"], ND=v["ND"], NR=v["NR"])
    ff = json.load(open(os.path.join(ROOT, "ff3d_evaluation.json")))["per_plot"]
    for k, v in ff.items():
        data["FF3D"][k] = dict(TP=v["TP"], ND=v["inside"], NR=v["NR"])

    settings = load()
    best = max(settings, key=lambda e: f1_of(settings[e], keys))
    for k in keys:
        v = settings[best][k]
        data["TreeLite3D"][k] = dict(TP=v["TP"], ND=v["ND"], NR=v["NR"])
    return data, keys, best


def aggregate(ps, seed=0):
    TP, ND, NR = (sum(p[x] for p in ps) for x in ("TP", "ND", "NR"))
    r, pr, f1 = metrics(TP, NR, ND)
    rng = np.random.default_rng(seed)
    Rs, Fs = [], []
    for _ in range(NB):
        i = rng.integers(0, len(ps), len(ps))
        a, b, c = (sum(ps[j][x] for j in i) for x in ("TP", "ND", "NR"))
        rr, _, ff = metrics(a, c, b)
        Rs.append(rr); Fs.append(ff)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3),
                   round(float(np.percentile(v, 97.5)), 3)]
    return dict(TP=TP, ND=ND, NR=NR, recall=round(r, 3), precision=round(pr, 3),
                f1=round(f1, 3), recall_ci=q(Rs), f1_ci=q(Fs))


def difference(a, b, ks, seed=0):
    """Recall of a minus that of b, resampling the same plots."""
    ra = sum(a[k]["TP"] for k in ks) / sum(a[k]["NR"] for k in ks)
    rb = sum(b[k]["TP"] for k in ks) / sum(b[k]["NR"] for k in ks)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(NB):
        i = rng.integers(0, len(ks), len(ks))
        kk = [ks[j] for j in i]
        na = sum(a[k]["NR"] for k in kk); nb = sum(b[k]["NR"] for k in kk)
        v.append((sum(a[k]["TP"] for k in kk) / na if na else 0)
                 - (sum(b[k]["TP"] for k in kk) / nb if nb else 0))
    return dict(diff=round(ra - rb, 3),
                ci=[round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)])


def main():
    data, keys, best = gather()
    sites = {n: [f"{s}/{p}" for p in ps] for s, n, ps in SITES}
    groups = list(sites.items()) + [("Both", keys)]
    print(f"TreeLite3D with the setting {best}; D = {D} m\n")

    output = {"D": D, "treelite_setting": best, "per_site": {}, "diff_vs_SAT": {}}
    print(f"{'method':12} " + " | ".join(f"{n:^30}" for n, _ in groups))
    print(f"{'':12} " + " | ".join(" rec    95 % CI      prec   F1 " for _ in groups))
    for m in ORDER:
        row = f"{m:12} "
        for n, ks in groups:
            v = aggregate([data[m][k] for k in ks])
            output["per_site"].setdefault(n, {})[m] = v
            row += (f"{v['recall']:.3f} {str(v['recall_ci']):>14} "
                     f"{v['precision']:.3f} {v['f1']:.3f} | ")
        print(row)

    print(f"\nrecall difference versus SegmentAnyTree, paired by plot")
    print(f"{'method':12} " + " | ".join(f"{n:^24}" for n, _ in groups))
    for m in ORDER[1:]:
        row = f"{m:12} "
        for n, ks in groups:
            d = difference(data["SAT"], data[m], ks)
            output["diff_vs_SAT"].setdefault(n, {})[m] = d
            mark = " *" if d["ci"][0] > 0 or d["ci"][1] < 0 else "  "
            row += f"{d['diff']:>+7.3f} {str(d['ci']):>15}{mark} | "
        print(row)
    print("\n* the interval does not cross zero")

    json.dump(output, open(os.path.join(ROOT, "paired_comparison.json"), "w"),
              indent=2, ensure_ascii=False)
    print("saved to paired_comparison.json")


if __name__ == "__main__":
    main()
