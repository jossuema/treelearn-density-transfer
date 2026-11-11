"""Qualitative figure: two plots that show the inversion, seen from above.

cham/4 is the sparsest plot (36 pts/m2) and it is where SegmentAnyTree gains most over
TreeLite3D (0.773 against 0.387). pell/p8 is dense (134 pts/m2) and there it inverts
(0.518 against 0.729). They are the two plots with the largest gap in each direction, not
a convenient choice.

It is drawn with the same data and the same matching as the tables, read from
trees.json, which is also what the viewer shows.

Usage: python3 figure_plot_maps.py
"""
import json
import os
import sys

import laspy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
# This file is kept in two layouts, so the code root is the one above it that
# holds the modal/ package, and the results are read from wherever they really are.
BASE = next(d for d in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)))
            if os.path.isdir(os.path.join(d, "modal")))
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import (reference, region, BASE as DATA,
                             results_path)  # noqa: E402

TREES = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))
PLOTS = [("cham/4", "Chamrousse 4, 36 pts/m$^2$"),
            ("pell/p8", "Pellizzano 8, 134 pts/m$^2$")]
MET = [("SAT", "SegmentAnyTree", "#1b3b6f"), ("TreeLite3D", "TreeLite3D", "#8e44ad")]
OK, MISS = "#1f7a5a", "#c2402a"
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                     "axes.linewidth": .6, "legend.frameon": False})


def cloud(site, p):
    r = (os.path.join(DATA, "IRSTEA_dataset", "las", "full_las", f"las{p}.las")
         if site == "cham" else
         os.path.join(DATA, "trento_dataset", "las", f"{p}.las"))
    L = laspy.read(r)
    return np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)


fig, axm = plt.subplots(2, 2, figsize=(3.45, 3.30))
ax = axm.T.ravel()
rng = np.random.default_rng(0)
for i, (k, tit) in enumerate(PLOTS):
    site, p = k.split("/")
    ref = reference(site, p)
    reg = region(ref)
    x, y, z = cloud(site, p)
    inside = reg.contains_points(np.c_[x, y])
    idx = rng.permutation(np.flatnonzero(inside))[:9000]   # only for the background
    v = reg.vertices
    for j, (m, name, col) in enumerate(MET):
        a = ax[i * 2 + j]
        a.scatter(x[idx], y[idx], s=.25, c="#c9ced4", lw=0, zorder=1, rasterized=True)
        a.plot(np.r_[v[:, 0], v[0, 0]], np.r_[v[:, 1], v[0, 1]], color="0.35", lw=.7,
               ls=(0, (4, 2)), zorder=4)
        d = TREES[k]["det"][m]
        emp = {r for _, r in TREES[k]["pairs"][m]["3.5"]}
        hit = np.array([t in emp for t in range(len(ref["x"]))])
        a.scatter(np.array(d["x"]), np.array(d["y"]), s=13, marker="+", c=col,
                  lw=.75, zorder=3)
        a.scatter(ref["x"][hit], ref["y"][hit], s=5.5, c=OK, lw=0, zorder=2)
        a.scatter(ref["x"][~hit], ref["y"][~hit], s=5.5, c=MISS, lw=0, zorder=2)
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        # scale bar: without it the reader cannot tell that the two plots
        # have different sizes
        x0, x1 = v[:, 0].min(), v[:, 0].max()
        y0, y1 = v[:, 1].min(), v[:, 1].max()
        bx, by = x0 + .04 * (x1 - x0), y0 - .055 * (y1 - y0)
        a.plot([bx, bx + 20], [by, by], color="0.3", lw=1.3, solid_capstyle="butt",
               clip_on=False, zorder=5)
        a.text(bx + 10, by - .035 * (y1 - y0), "20 m", fontsize=5.8, ha="center",
               va="top", color="0.3")
        r = TREES[k]["met"][m]["3.5"]["recall"]
        a.set_title(name, fontsize=6.8, loc="left", color=col, pad=2)
        a.text(.98, .97, f"recall {r:.3f}", transform=a.transAxes, fontsize=6.4,
               ha="right", va="top", color="0.2")
        a.margins(.02, .10)
    ax[i * 2].text(0, 1.20, tit, transform=ax[i * 2].transAxes, fontsize=7.0)

h = [Line2D([], [], marker="o", ls="", color=OK, ms=3, label="inventoried tree, matched"),
     Line2D([], [], marker="o", ls="", color=MISS, ms=3, label="inventoried tree, missed"),
     Line2D([], [], marker="+", ls="", color="0.3", ms=5, label="detection"),
     Line2D([], [], ls=(0, (4, 2)), color="0.35", lw=.7, label="scored region")]
fig.legend(handles=h, loc="lower center", ncol=2, fontsize=5.8, handlelength=1.3,
           columnspacing=1.2, bbox_to_anchor=(.5, -.008))
fig.subplots_adjust(left=.02, right=.98, top=.90, bottom=.12, wspace=.04, hspace=.28)
fig.savefig("fig3.pdf", dpi=400); fig.savefig("fig3.png", dpi=300)
print("fig3 written")
