"""Robustness checks, with numbers and not with opinions.

  A. AMS3D without looking at the test plot. The study uses the best of 25 settings,
     chosen on the same plot that is scored (oracle). Here the operational version is
     added: cell (i, j) of the 5x5 grid is chosen on the OTHER plots of the same site
     and applied to the excluded plot. The grids belong to each plot, because they
     come out of its allometry, so what transfers is the position in the grid, the
     same as eval_controlled.py did.
  B. Non-inferiority of SegmentAnyTree against AMS3D with a margin of 0.05. The 5th
     percentile of the bootstrap of the paired difference is used, which is the bound
     of a one-sided interval at 95 %.
  C. Greedy matching against the optimal one (Hungarian with maximum cardinality).
  D. Pellizzano restricted to the hyperspectral image footprint, a 40 m square read
     from the .hdr files. It is not a surveyed plot boundary, and no such boundary
     exists for these plots; it is one more framing to test, against the dilated hull and against the whole tile.

Usage: python3 analysis/review_checks.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import region, BASE as DATA  # noqa: E402

TREES = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))
AMSDIR = os.path.join(DATA, "ams_port", "out")
SITES = {"cham": ["1", "1b", "2", "3", "3b", "4", "Premol"],
          "pell": [f"p{i}" for i in range(1, 16)]}
D = 3.5
NB = 2000
DELTA = 0.05


# ----------------------------------------------------------------- matching
def greedy(dx, dy, rx, ry, dmax=D):
    if not len(dx) or not len(rx):
        return 0
    Dm = np.hypot(np.asarray(dx)[:, None] - np.asarray(rx)[None, :],
                  np.asarray(dy)[:, None] - np.asarray(ry)[None, :])
    cand = np.argwhere(Dm <= dmax)
    order = np.argsort(Dm[cand[:, 0], cand[:, 1]], kind="stable")
    ud, ur, n = set(), set(), 0
    for i, j in cand[order]:
        if i in ud or j in ur:
            continue
        ud.add(i); ur.add(j); n += 1
    return n


def optimal(dx, dy, rx, ry, dmax=D):
    """Maximum cardinality and, among those, minimum total distance."""
    if not len(dx) or not len(rx):
        return 0
    Dm = np.hypot(np.asarray(dx)[:, None] - np.asarray(rx)[None, :],
                  np.asarray(dy)[:, None] - np.asarray(ry)[None, :])
    BIG = 1e6       # larger than any admissible sum of distances
    C = np.where(Dm <= dmax, Dm, BIG)
    f, c = linear_sum_assignment(C)
    return int((C[f, c] < BIG).sum())


# ----------------------------------------------------------------- utilities
def boot(rows, fn, q=(2.5, 97.5)):
    """rows: one tuple of counts per plot. fn(sum of rows) -> scalar."""
    rows = [np.asarray(f, float) for f in rows]
    rng = np.random.default_rng(0)
    v = []
    for _ in range(NB):
        idx = rng.integers(0, len(rows), len(rows))
        v.append(fn(sum(rows[i] for i in idx)))
    return [round(float(np.percentile(v, x)), 3) for x in q]


def f1(tp, nd, nr):
    r = tp / nr if nr else 0.0
    p = tp / nd if nd else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# ----------------------------------------------------------------- A. AMS3D
def dets_ams(plot):
    """The 25 settings of a plot with their position (i, j) in the grid."""
    out = []
    for f in glob.glob(os.path.join(AMSDIR, f"e1_dets_{plot}_AMS3DE_las_{plot}_m1_*_m2_*.csv")):
        m = re.search(r"_m1_(\d+)_m2_(\d+)\.csv$", f)
        x, y = [], []
        for r in csv.DictReader(open(f)):
            x.append(float(r["x"])); y.append(float(r["y"]))
        out.append({"m1": int(m.group(1)), "m2": int(m.group(2)),
                    "x": np.array(x), "y": np.array(y), "cfg": os.path.basename(f)})
    m1s = sorted({o["m1"] for o in out}); m2s = sorted({o["m2"] for o in out})
    for o in out:
        o["i"], o["j"] = m1s.index(o["m1"]), m2s.index(o["m2"])
    return out


def count(key, x, y):
    a = TREES[key]
    R = region({"x": np.array(a["inv"]["x"]), "y": np.array(a["inv"]["y"])})
    if len(x):
        inside = R.contains_points(np.c_[x, y])
        x, y = x[inside], y[inside]
    tp = greedy(x, y, a["inv"]["x"], a["inv"]["y"])
    return (tp, len(x), len(a["inv"]["x"]))


print("A. AMS3D: oracle per plot against operational choice")
table = {}          # (site, plot) -> {(i,j): (tp, nd, nr)}
for s, ps in SITES.items():
    for p in ps:
        table[(s, p)] = {(o["i"], o["j"]): count(f"{s}/{p}", o["x"], o["y"]) for o in dets_ams(p)}
        assert len(table[(s, p)]) == 25, (s, p, len(table[(s, p)]))

# setting the study uses: the best one according to the F1 of Tusa's protocol
carta = {}
for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
    for r in csv.DictReader(open(os.path.join(AMSDIR, f))):
        m = re.search(r"_m1_(\d+)_m2_(\d+)$", r["cfg"])
        carta[(s, r["plot"])] = (int(m.group(1)), int(m.group(2)))

sel = {"reported oracle": {}, "oracle, our F1": {}, "leave-one-out validated": {}}
for (s, p), configs in table.items():
    # the reported oracle
    obj = carta[(s, p)]
    o = [c for c in dets_ams(p) if (c["m1"], c["m2"]) == obj][0]
    sel["reported oracle"][(s, p)] = configs[(o["i"], o["j"])]
    # oracle with our F1
    sel["oracle, our F1"][(s, p)] = max(configs.values(), key=lambda t: f1(*t))
    # leaving one out within the same site
    others = [q for q in SITES[s] if q != p]
    best = max(configs, key=lambda ij: f1(*[sum(table[(s, q)][ij][k] for q in others) for k in range(3)]))
    sel["leave-one-out validated"][(s, p)] = configs[best]

res_A = {}
for name, d in sel.items():
    res_A[name] = {}
    for s in list(SITES) + ["global"]:
        keys = [k for k in d if s == "global" or k[0] == s]
        tp, nd, nr = [sum(d[k][i] for k in keys) for i in range(3)]
        res_A[name][s] = dict(TP=tp, ND=nd, NR=nr, recall=round(tp / nr, 3),
                                precision=round(tp / nd, 3), f1=round(f1(tp, nd, nr), 3))
    g = res_A[name]
    print(f"  {name:28} recall cham {g['cham']['recall']:.3f} pell {g['pell']['recall']:.3f} "
          f"global {g['global']['recall']:.3f} | global F1 {g['global']['f1']:.3f}")


# ----------------------------------------------------------------- B. non inferiority
print(f"\nB. Non-inferiority of SegmentAnyTree against AMS3D, margin {DELTA}")
print("   p5 is the lower bound of the one sided 95 % interval; if it is greater than "
      f"-{DELTA}, SAT is not worse by more than {DELTA}")


def sat_rows(s):
    out = []
    for p in (SITES[s] if s != "global" else SITES["cham"] + SITES["pell"]):
        k = (f"cham/{p}" if p in SITES["cham"] else f"pell/{p}")
        m = TREES[k]["met"]["SAT"]["3.5"]
        out.append((k, (m["TP"], m["ND"], m["NR"])))
    return out


res_B = {}
for name in ("reported oracle", "leave-one-out validated"):
    res_B[name] = {}
    for s in ("cham", "pell", "global"):
        sat = sat_rows(s)
        rows = []
        for k, (tp, nd, nr) in sat:
            sit, p = k.split("/")
            atp, and_, anr = sel[name][(sit, p)]
            rows.append((tp, nd, nr, atp, and_))
        dr = boot(rows, lambda t: t[0] / t[2] - t[3] / t[2], q=(5, 2.5, 50, 95, 97.5))
        df1 = boot(rows, lambda t: f1(t[0], t[1], t[2]) - f1(t[3], t[4], t[2]), q=(5, 2.5, 50, 95, 97.5))
        tot = np.sum(np.array(rows, float), 0)
        pr = tot[0] / tot[2] - tot[3] / tot[2]
        pf = f1(tot[0], tot[1], tot[2]) - f1(tot[3], tot[4], tot[2])
        res_B[name][s] = {"recall": dict(diff=round(pr, 3), p5=dr[0], ci95=[dr[1], dr[4]],
                                           non_inferior=dr[0] > -DELTA,
                                           equivalent=dr[0] > -DELTA and dr[3] < DELTA),
                            "f1": dict(diff=round(pf, 3), p5=df1[0], ci95=[df1[1], df1[4]],
                                       non_inferior=df1[0] > -DELTA,
                                       equivalent=df1[0] > -DELTA and df1[3] < DELTA)}
        for met in ("recall", "f1"):
            r = res_B[name][s][met]
            print(f"  against {name:26} {s:6} {met:6} dif {r['diff']:+.3f}  p5 {r['p5']:+.3f}  "
                  f"not inferior {'YES' if r['non_inferior'] else 'no'}  "
                  f"equivalent {'YES' if r['equivalent'] else 'no'}")


# ----------------------------------------------------------------- C. greedy against optimal
print("\nC. Greedy matching against maximum cardinality Hungarian, at 3.5 m")
res_C = {}
for m in ("SAT", "AMS3D", "WS"):
    g = h = 0
    for k, a in TREES.items():
        dd = a["det"][m]
        g += greedy(dd["x"], dd["y"], a["inv"]["x"], a["inv"]["y"])
        h += optimal(dd["x"], dd["y"], a["inv"]["x"], a["inv"]["y"])
    res_C[m] = dict(greedy=g, optimal=h, diff=h - g)
    print(f"  {m:6} greedy {g:5d}   Hungarian {h:5d}   difference {h-g:+d} trees "
          f"({100*(h-g)/1437:+.2f} recall points)")


# ----------------------------------------------------------------- D. Pellizzano real limit
print("\nD. Pellizzano: evaluation region")
tables = {}
imgs = {}
for h in glob.glob(os.path.join(DATA, "trento_dataset", "img", "*.hdr")):
    t = open(h).read()
    s_ = int(re.search(r"samples\s*=\s*(\d+)", t).group(1))
    l_ = int(re.search(r"lines\s*=\s*(\d+)", t).group(1))
    mi = re.search(r"map info\s*=\s*\{([^}]*)\}", t).group(1).split(",")
    x0, y0, px = float(mi[3]), float(mi[4]), float(mi[5])
    imgs[os.path.basename(h)[:-4]] = (x0, y0 - l_ * px, x0 + s_ * px, y0)
for p in SITES["pell"]:
    a = TREES[f"pell/{p}"]
    for name, (x0, y0, x1, y1) in imgs.items():
        if all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in zip(a["inv"]["x"], a["inv"]["y"])):
            tables[p] = (x0, y0, x1, y1)
            break
assert len(tables) == 15, len(tables)

res_D = {}
for m in ("SAT", "AMS3D", "WS"):
    res_D[m] = {}
    for mode in ("hyperspectral footprint", "hull dilated 1.5 m", "whole tile"):
        rows = []
        for p in SITES["pell"]:
            a = TREES[f"pell/{p}"]
            X = np.array(a["det"][m]["x"] + a["det_outside"][m]["x"])
            Y = np.array(a["det"][m]["y"] + a["det_outside"][m]["y"])
            if mode == "hyperspectral footprint":
                x0, y0, x1, y1 = tables[p]
                k = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
                X, Y = X[k], Y[k]
            elif mode == "hull dilated 1.5 m":
                X, Y = np.array(a["det"][m]["x"]), np.array(a["det"][m]["y"])
            tp = greedy(X, Y, a["inv"]["x"], a["inv"]["y"])
            rows.append((tp, len(X), len(a["inv"]["x"])))
        tp, nd, nr = [sum(f[i] for f in rows) for i in range(3)]
        cip = boot(rows, lambda t: t[0] / t[1] if t[1] else 0.0)
        res_D[m][mode] = dict(TP=tp, ND=nd, recall=round(tp / nr, 3),
                              precision=round(tp / nd, 3), precision_ci=cip,
                              f1=round(f1(tp, nd, nr), 3))
        r = res_D[m][mode]
        print(f"  {m:6} {mode:22} recall {r['recall']:.3f}  prec {r['precision']:.3f} {cip}  "
              f"F1 {r['f1']:.3f}  ND {nd}")

json.dump({"A_ams3d": res_A, "B_non_inferiority": res_B, "C_matching": res_C,
           "D_region_pellizzano": res_D},
          open(os.path.join(ROOT, "review_checks.json"), "w"), indent=2, ensure_ascii=False,
          default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
print("\nsaved to review_checks.json")
