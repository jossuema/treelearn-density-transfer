"""3D views of two plots, coloured by the instance each model predicted.

It is what an instance segmentation model really does, and it shows TreeLite3D's failure
far better than a point map: in the sparse plot it leaves most of the cloud unassigned to
any tree (grey), and in the dense one it shares it out well.

cham/4 is the sparsest plot in the study and pell/p8 one of the densest.

Usage: python3 figure_3d.py
"""
import glob
import json
import os
import sys
import zipfile

import laspy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# This file is kept in two layouts, so the code root is the one above it that
# holds the modal/ package, and the results are read from wherever they really are.
BASE = next(d for d in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)))
            if os.path.isdir(os.path.join(d, "modal")))
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import (reference, region, BASE as DATA,
                             results_path)  # noqa: E402

SAT_OUT = os.path.join(ROOT, "sat_output_all")
TL_OUT = os.path.join(ROOT, "treelite_output", "t4_p30")
PLOTS = [("cham", "4", "Chamrousse 4, 36 pts/m$^2$"),
            ("pell", "p8", "Pellizzano 8, 134 pts/m$^2$")]
MAXP = 26000
GREY = "#c2c6cc"


def input_cloud(site, p):
    r = (os.path.join(DATA, "IRSTEA_dataset", "las", "full_las", f"las{p}.las")
         if site == "cham" else
         os.path.join(DATA, "trento_dataset", "las", f"{p}.las"))
    L = laspy.read(r)
    return np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)])


def inst_sat(site, p):
    L = laspy.read(os.path.join(SAT_OUT, f"{site}_{p}_out.laz"))
    xyz = np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)])
    return xyz, np.asarray(L.PredInstance).astype(np.int64)


def inst_treelite(site, p):
    xyz = input_cloud(site, p)
    e = np.load(os.path.join(TL_OUT, f"instance_pred_{site}_{p}.npy")).reshape(-1)
    a = np.load(os.path.join(TL_OUT, f"assigned_{site}_{p}.npy")).reshape(-1)
    # points with no proposal carry label 0 because of the argmax; they are marked apart
    return xyz, np.where(a, e + 1, 0)


TREES = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))


def draw(ax, xyz, inst, reg, title, title_color, caption):
    inside = reg.contains_points(xyz[:, :2])
    xyz, inst = xyz[inside], inst[inside]
    if len(xyz) > MAXP:
        i = np.random.default_rng(0).permutation(len(xyz))[:MAXP]
        xyz, inst = xyz[i], inst[i]
    cm = plt.get_cmap("tab20")
    col = np.where((inst > 0)[:, None], cm((inst % 20) / 20.0)[:, :3],
                   np.array(matplotlib.colors.to_rgb(GREY))[None, :])
    o = np.argsort(xyz[:, 2])
    ax.scatter(xyz[o, 0], xyz[o, 1], xyz[o, 2], c=col[o], s=.6, lw=0,
               depthshade=False, rasterized=True)
    ax.set_axis_off()
    ax.view_init(elev=16, azim=-58)
    ax.set_box_aspect((1, 1, .62))
    ax.dist = 7.2
    ax.set_title(title, fontsize=6.8, color=title_color, pad=-2, loc="center")
    # the share of assigned returns does NOT measure quality: SegmentAnyTree leaves the
    # ground out and that is why it assigns fewer. What matters is how many inventory
    # trees are found.
    ax.text2D(.5, .03, caption, transform=ax.transAxes, fontsize=6.3, ha="center",
              color="0.25")


plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})
fig = plt.figure(figsize=(7.16, 2.15))
for i, (site, p, tit) in enumerate(PLOTS):
    ref = reference(site, p)
    reg = region(ref)
    for j, (name, fn, col) in enumerate(
            [("SegmentAnyTree", inst_sat, "#1b3b6f"),
             ("TreeLite3D", inst_treelite, "#8e44ad")]):
        ax = fig.add_subplot(1, 4, i * 2 + j + 1, projection="3d")
        xyz, inst = fn(site, p)
        key = f"{site}/{p}"
        met = "SAT" if name == "SegmentAnyTree" else "TreeLite3D"
        v = TREES[key]["met"][met]["3.5"]
        draw(ax, xyz, inst, reg, name, col,
               f"{v['TP']} of {v['NR']} field trees found, recall {v['recall']:.3f}")
    fig.text(.045 + i * .5, .945, tit, fontsize=7.6)
fig.subplots_adjust(left=0, right=1, top=.99, bottom=-.05, wspace=-.03)
fig.savefig("fig4.pdf", dpi=400)
fig.savefig("fig4.png", dpi=300)
print("fig4 written")
