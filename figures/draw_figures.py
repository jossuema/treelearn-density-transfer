"""Two summary figures of the study, from main_evaluation.json and crown_jaccard.json.

Figure 1, the mechanism: recall against density plot by plot, the paired difference
with SegmentAnyTree by site, and recall by detectability stratum.
Figure 2, the evaluation framework: what happens when the region changes and why the
overlap index does not rank the methods.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
# This file is kept in two layouts, so the code root is the one above it that
# holds the modal/ package, and the results are read from wherever they really are.
BASE = next(d for d in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)))
            if os.path.isdir(os.path.join(d, "modal")))
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import results_path  # noqa: E402

B = json.load(open(results_path("main_evaluation.json")))
J = json.load(open(results_path("crown_jaccard.json")))
CMP = json.load(open(results_path("paired_comparison.json")))

from decimal import Decimal, ROUND_HALF_UP


def r2(v):
    """Rounding to two decimals half up, not the one floating point does."""
    return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


MET = ["SAT", "AMS3D", "FF3D", "TreeLite3D", "WS"]
LABEL = {"SAT": "SegmentAnyTree", "AMS3D": "AMS3D", "FF3D": "ForestFormer3D",
       "TreeLite3D": "TreeLite3D", "WS": "CHM watershed"}
COL = {"SAT": "#1b3b6f", "AMS3D": "#d1701a", "FF3D": "#2e7d5b",
       "TreeLite3D": "#8e44ad", "WS": "#7a8b7f"}
MRK = {"SAT": "o", "AMS3D": "s", "FF3D": "D", "TreeLite3D": "v", "WS": "^"}
SIT = ["Chamrousse", "Pellizzano"]

plt.rcParams["pdf.fonttype"] = 42   # TrueType, not the Type 3 matplotlib uses by default
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False})

keys = list(B["per_plot"])
dens = np.array([B["density"][k]["density"] for k in keys])
rec = {m: np.array([B["per_plot"][k][m] for k in keys]) for m in MET}
cham = np.array([k.startswith("cham") for k in keys])

# =============================================================== figure 1
# Three panels: the slope between sites, the strata and the 3D views. The paired
# difference panel that was here before was removed: it was a drawing of Table II, which
# already gives those same differences with their intervals.
import laspy
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import reference, region, BASE as DATA  # noqa: E402

SAT_OUT = os.path.join(ROOT, "sat_output_all")
TL_OUT = os.path.join(ROOT, "treelite_output", "t4_p30")
P3D = [("cham", "4", "Chamrousse 4, 36 returns/m$^2$"),
       ("pell", "p8", "Pellizzano 8, 134 returns/m$^2$")]
MAXP, GREY = 26000, "#c2c6cc"
TREES = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))


def _cloud(site, p):
    r = (os.path.join(DATA, "IRSTEA_dataset", "las", "full_las",
                      f"las{p}.las") if site == "cham" else
         os.path.join(DATA, "trento_dataset", "las", f"{p}.las"))
    L = laspy.read(r)
    return np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)])


def inst_sat(site, p):
    L = laspy.read(os.path.join(SAT_OUT, f"{site}_{p}_out.laz"))
    return (np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)]),
            np.asarray(L.PredInstance).astype(np.int64))


def inst_tl(site, p):
    xyz = _cloud(site, p)
    e = np.load(os.path.join(TL_OUT, f"instance_pred_{site}_{p}.npy")).reshape(-1)
    a = np.load(os.path.join(TL_OUT, f"assigned_{site}_{p}.npy")).reshape(-1)
    return xyz, np.where(a, e + 1, 0)   # the unassigned carry 0 because of the argmax


# Figure 1 is written as six separate images with no lettering of their own. The panel
# titles, the plot names and the counts under each 3-D view are set in LaTeX instead, so
# that they come out in the manuscript's own typeface and can be edited without redrawing
# anything. The counts still come from the data: they are emitted as macros below.

SUB = {m: json.load(open(results_path(f"thinning_{f}.json")))
       for m, f in (("SAT", "SAT"), ("TreeLite3D", "t4_p30"))
       if os.path.exists(results_path(f"thinning_{f}.json"))}
if os.path.exists(results_path("thinning_FF3D.json")):
    SUB["FF3D"] = json.load(open(results_path("thinning_FF3D.json")))
DENS = json.load(open(results_path("thinning_density.json")))
FR = ["100", "60", "40", "25"]
xs = [DENS[f] for f in FR]

# --------------------- (a) controlled thinning: pulses are removed and it is run again
# It is a manipulation, not a correlation: the same 15 Pellizzano plots, the same sensor,
# the same stand and the same inventory. The only thing that changes is density.
figA, a = plt.subplots(figsize=(2.86, 1.86))
# the band marks the density range the Chamrousse plots fall in
cd = [B["density"][k]["density"] for k in B["density"] if k.startswith("cham")]
a.axvspan(min(cd), max(cd), color="0.92", zorder=0, lw=0)
a.text((min(cd) + max(cd)) / 2, 0.055, "Chamrousse\nrange", fontsize=6.2, ha="center",
       color="0.45", va="bottom")
ORD = [m for m in ("SAT", "FF3D", "TreeLite3D") if m in SUB]
for j, m in enumerate(ORD):
    v = SUB[m]
    y = [v[f]["recall"] for f in FR]
    lo = [v[f]["ci"][0] for f in FR]
    hi = [v[f]["ci"][1] for f in FR]
    a.fill_between(xs, lo, hi, color=COL[m], alpha=.13, lw=0, zorder=1)
    a.plot(xs, y, marker=MRK[m], ms=4.2, color=COL[m], lw=1.9, zorder=3, label=LABEL[m])
    # stacked in the free corner: the curves cross and labelling them alongside does not fit
    a.text(.985, .045 + .088 * (len(ORD) - 1 - j), LABEL[m], transform=a.transAxes,
           fontsize=6.6, color=COL[m], ha="right", va="bottom")
a.set_xlabel("Return density (returns/m$^2$)", fontsize=7.5)
a.set_ylabel("Plot recall", fontsize=7.5)
a.set_xlim(20, 128); a.set_ylim(0, .82)
a.grid(lw=.3, color="0.9"); a.set_axisbelow(True); a.tick_params(labelsize=6.8)
figA.tight_layout(pad=0.15)
figA.savefig("fig1a.pdf", dpi=400); figA.savefig("fig1a.png", dpi=250)

# ------------------------------------------------------- (b) bars by stratum
figB, c = plt.subplots(figsize=(3.94, 1.86))
STRATA = ["dominant", "intermediate", "suppressed"]
NAMES = ["Dominant\n$n$ = 848", "Intermediate\n$n$ = 381", "Suppressed\n$n$ = 208"]
w, x0 = .165, np.arange(3)
for j, m in enumerate(MET):
    c.bar(x0 + (j - 2) * w, [B["strata"]["Both"][e][m]["recall"] for e in STRATA],
          w * .88, color=COL[m], edgecolor="white", linewidth=.4, label=LABEL[m], zorder=2)
c.set_xticks(x0); c.set_xticklabels(NAMES, fontsize=7)
c.set_ylabel("Recall", fontsize=7.5); c.set_ylim(0, .98)
c.legend(fontsize=6.2, handlelength=1.0, labelspacing=.2, loc="upper right", ncol=2,
         columnspacing=.8)
c.grid(axis="y", lw=.3, color="0.9"); c.set_axisbelow(True); c.tick_params(labelsize=6.8)
figB.tight_layout(pad=0.15)
figB.savefig("fig1b.pdf", dpi=400); figB.savefig("fig1b.png", dpi=250)

# --------------------------------------------- (c) the clouds, by predicted instance
cm20 = plt.get_cmap("tab20")
macros = []
for i, (site, pp, tit) in enumerate(P3D):
    ref = reference(site, pp)
    reg = region(ref)
    for j, (name, fn, colm) in enumerate([("SegmentAnyTree", inst_sat, COL["SAT"]),
                                         ("TreeLite3D", inst_tl, COL["TreeLite3D"])]):
        figC = plt.figure(figsize=(1.74, 1.24))
        ax = figC.add_subplot(111, projection="3d")
        xyz, inst = fn(site, pp)
        d = reg.contains_points(xyz[:, :2])
        xyz, inst = xyz[d], inst[d]
        if len(xyz) > MAXP:
            q = np.random.default_rng(0).permutation(len(xyz))[:MAXP]
            xyz, inst = xyz[q], inst[q]
        col = np.where((inst > 0)[:, None], cm20((inst % 20) / 20.0)[:, :3],
                       np.array(matplotlib.colors.to_rgb(GREY))[None, :])
        o = np.argsort(xyz[:, 2])
        ax.scatter(xyz[o, 0], xyz[o, 1], xyz[o, 2], c=col[o], s=.55, lw=0,
                   depthshade=False, rasterized=True)
        ax.set_axis_off(); ax.view_init(elev=16, azim=-58)
        for lim, k3 in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            lo, hi = xyz[:, k3].min(), xyz[:, k3].max()
            lim(lo, hi)
        ax.set_box_aspect((1, 1, .60), zoom=1.34)
        figC.subplots_adjust(left=0, right=1, top=1, bottom=0)
        name = f"fig1c{i * 2 + j + 1}"
        figC.savefig(f"{name}.pdf", dpi=400); figC.savefig(f"{name}.png", dpi=250)
        plt.close(figC)
        v = TREES[f"{site}/{pp}"]["met"]["SAT" if j == 0 else "TreeLite3D"]["3.5"]
        macros.append((f"\\figc{'abcd'[i * 2 + j]}count",
                       f"{v['TP']}/{v['NR']} trees, recall {v['recall']:.3f}"))
        macros.append((f"\\figc{'abcd'[i * 2 + j]}name", name))
    macros.append((f"\\figsite{'ab'[i]}", tit))

# the counts are not typed into the manuscript: they are written here, from the data
with open("fig1_labels.tex", "w") as f:
    f.write("% generated by draw_figures.py, do not edit\n")
    for name, value in macros:
        f.write(f"\\newcommand{{{name}}}{{{value}}}\n")
print("fig1a, fig1b, fig1c1..4 and fig1_labels.tex written")

# =============================================================== figure 2
fig2, ax2 = plt.subplots(1, 2, figsize=(7.16, 2.25))

a2 = ax2[0]
LABELS = ["hull +0 m", "hull +1.5 m", "hull +3 m", "whole tile"]
NAMES = ["hull", "hull +1.5 m", "hull +3 m", "whole tile"]
for m in MET:
    a2.plot(np.arange(4), [B["region"][m][e]["precision"] for e in LABELS],
            marker=MRK[m], color=COL[m], lw=1.1, ms=3.2, label=LABEL[m],
            ls=(0, (3, 1.6)) if m == "FF3D" else "-",
            zorder=4 if m == "SAT" else 3)
a2.set_xticks(np.arange(4)); a2.set_xticklabels(NAMES, fontsize=6.2)
a2.set_ylabel("Precision"); a2.set_ylim(0, 1.02)
a2.set_xlabel("Region over which detections are counted")
a2.set_title("(a) The scoring region moves precision", fontsize=7.5, loc="left")
a2.legend(loc="lower left", fontsize=5.6, handlelength=1.2, borderpad=0.2,
          labelspacing=0.25)
a2.grid(lw=0.3, color="0.88"); a2.set_axisbelow(True)

b2 = ax2[1]
for m in MET:
    for i, s in enumerate(SIT):
        b2.scatter(B["main"][s][m]["recall"], J[s][m]["J"], s=42,
                   marker=MRK[m], color=COL[m], edgecolor="white", linewidth=0.6,
                   zorder=3, alpha=1.0 if i == 0 else 0.5)
b2.set_xlabel("Recall"); b2.set_ylabel("Mean 2-D crown Jaccard, matched pairs")
b2.set_title("(b) Overlap index vs recall", fontsize=7.5, loc="left")
b2.grid(lw=0.3, color="0.88"); b2.set_axisbelow(True)
b2.set_ylim(0.31, 0.58); b2.set_xlim(0.34, 0.74)
b2.annotate("watershed: highest $J$,\nlowest recall", xy=(0.376, 0.531),
            xytext=(0.43, 0.505), fontsize=5.8, color="0.25", ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.55"))
b2.annotate("TreeLite3D: lowest $J$,\nover-segments", xy=(0.678, 0.340),
            xytext=(0.47, 0.345), fontsize=5.8, color="0.25", ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.55"))
h2 = [Line2D([], [], marker="o", ls="", color="0.35", ms=3.6, label="Chamrousse"),
      Line2D([], [], marker="o", ls="", color="0.35", ms=3.6, alpha=0.5,
             label="Pellizzano")]
b2.legend(handles=h2, loc="upper right", fontsize=5.6, handlelength=1.0,
          borderpad=0.2, labelspacing=0.25)

fig2.tight_layout(pad=0.35, w_pad=1.2)
fig2.savefig("fig2.pdf"); fig2.savefig("fig2.png", dpi=400)
print("fig1 and fig2 written")
