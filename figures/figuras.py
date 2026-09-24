"""Las dos figuras del articulo, desde robustez_cinco.json y jaccard_cinco.json.

Figura 1, el mecanismo: acierto frente a densidad parcela a parcela, la diferencia
pareada con SegmentAnyTree por sitio, y el acierto por estrato de detectabilidad.
Figura 2, el marco de evaluacion: que pasa al cambiar la region y por que el indice
de solape no ordena los metodos.
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

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
B = json.load(open(os.path.join(R, "robustez_cinco.json")))
J = json.load(open(os.path.join(R, "jaccard_cinco.json")))
CMP = json.load(open(os.path.join(R, "comparacion_modelos.json")))

from decimal import Decimal, ROUND_HALF_UP


def r2(v):
    """Redondeo a dos decimales al alza, no el de la coma flotante."""
    return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


MET = ["SAT", "AMS3D", "FF3D", "TreeLite3D", "WS"]
ETI = {"SAT": "SegmentAnyTree", "AMS3D": "AMS3D", "FF3D": "ForestFormer3D",
       "TreeLite3D": "TreeLite3D", "WS": "CHM watershed"}
COL = {"SAT": "#1b3b6f", "AMS3D": "#d1701a", "FF3D": "#2e7d5b",
       "TreeLite3D": "#8e44ad", "WS": "#7a8b7f"}
MRK = {"SAT": "o", "AMS3D": "s", "FF3D": "D", "TreeLite3D": "v", "WS": "^"}
SIT = ["Chamrousse", "Pellizzano"]

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False})

claves = list(B["por_parcela"])
dens = np.array([B["densidad"][k]["densidad"] for k in claves])
rec = {m: np.array([B["por_parcela"][k][m] for k in claves]) for m in MET}
cham = np.array([k.startswith("cham") for k in claves])

# =============================================================== figura 1
# Tres paneles: la pendiente entre sitios, los estratos y las vistas 3D. El panel de
# diferencias pareadas que habia antes se quito: era un dibujo del Cuadro II, que ya da
# esas mismas diferencias con sus intervalos.
import laspy
sys.path.insert(0, os.path.join(R, "respaldo_minimo", "modal_app"))
from analisis import referencia, region  # noqa: E402

SAL_SAT = os.path.join(R, "salida_sat_all")
SAL_TL = os.path.join(R, "salida_treelite", "t4_p30")
P3D = [("cham", "4", "Chamrousse 4, 36 pts/m$^2$"),
       ("pell", "p8", "Pellizzano 8, 134 pts/m$^2$")]
MAXP, GRIS = 26000, "#c2c6cc"
ARB = json.load(open(os.path.join(R, "respaldo_minimo", "visor", "data", "arboles.json")))


def _nube(sitio, p):
    r = (os.path.join(R, "respaldo_minimo", "IRSTEA_dataset", "las", "full_las",
                      f"las{p}.las") if sitio == "cham" else
         os.path.join(R, "respaldo_minimo", "trento_dataset", "las", f"{p}.las"))
    L = laspy.read(r)
    return np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)])


def inst_sat(sitio, p):
    L = laspy.read(os.path.join(SAL_SAT, f"{sitio}_{p}_out.laz"))
    return (np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)]),
            np.asarray(L.PredInstance).astype(np.int64))


def inst_tl(sitio, p):
    xyz = _nube(sitio, p)
    e = np.load(os.path.join(SAL_TL, f"instance_pred_{sitio}_{p}.npy")).reshape(-1)
    a = np.load(os.path.join(SAL_TL, f"asignado_{sitio}_{p}.npy")).reshape(-1)
    return xyz, np.where(a, e + 1, 0)   # los no asignados llevan 0 por el argmax


fig = plt.figure(figsize=(7.16, 3.62))
gsa = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.30,
                       left=0.075, right=0.985, top=0.955, bottom=0.690)
gsb = fig.add_gridspec(1, 4, wspace=-0.02, left=0.005, right=0.995, top=0.480,
                       bottom=0.055)

# ------------------------------------------ (a) pendiente entre el sitio ralo y el denso
a = fig.add_subplot(gsa[0, 0])
rng = np.random.default_rng(0)
fin_ = {m: rec[m][~cham].mean() for m in MET}
sep = {}
for i, m in enumerate(sorted(MET, key=lambda x: -fin_[x])):
    sep[m] = fin_[m] + (0.048 * (1 - i)) if abs(fin_[m] - 0.70) < .03 else fin_[m]
for m in MET:
    for s, xx in ((cham, 0), (~cham, 1)):
        a.scatter(xx + rng.uniform(-.055, .055, s.sum()), rec[m][s], s=5,
                  color=COL[m], alpha=.22, lw=0, zorder=1)
    a.plot([0, 1], [rec[m][cham].mean(), fin_[m]], marker="o", ms=4.4, color=COL[m],
           lw=1.9, zorder=3)
    a.annotate(ETI[m], xy=(1, fin_[m]), xytext=(1.10, sep[m]), fontsize=5.9,
               color=COL[m], va="center",
               arrowprops=dict(arrowstyle="-", lw=.5, color=COL[m], alpha=.6)
               if abs(sep[m] - fin_[m]) > 1e-9 else None)
a.set_xlim(-.18, 2.12); a.set_xticks([0, 1])
a.set_xticklabels(["Chamrousse\n36--71 pts/m$^2$", "Pellizzano\n91--173"], fontsize=6.2)
a.set_ylabel("Plot recall", fontsize=7); a.set_ylim(0.20, 1.03)
a.set_title("(a) Plot recall at each site", fontsize=7.5, loc="left")
a.grid(axis="y", lw=.3, color="0.9"); a.set_axisbelow(True); a.tick_params(labelsize=6)

# ------------------------------------------------------- (b) barras por estrato
c = fig.add_subplot(gsa[0, 1])
ESTR = ["dominante", "intermedio", "suprimido"]
NOM = ["Dominant\n$n$ = 848", "Intermediate\n$n$ = 381", "Suppressed\n$n$ = 208"]
w, x0 = .165, np.arange(3)
for j, m in enumerate(MET):
    c.bar(x0 + (j - 2) * w, [B["estratos"]["Ambos"][e][m]["recall"] for e in ESTR],
          w * .88, color=COL[m], edgecolor="white", linewidth=.4, label=ETI[m], zorder=2)
c.set_xticks(x0); c.set_xticklabels(NOM, fontsize=6.4)
c.set_ylabel("Recall", fontsize=7); c.set_ylim(0, .98)
c.set_title("(b) Recall by detectability stratum, both sites", fontsize=7.5, loc="left")
c.legend(fontsize=5.4, handlelength=1.0, labelspacing=.2, loc="upper right", ncol=2,
         columnspacing=.8)
c.grid(axis="y", lw=.3, color="0.9"); c.set_axisbelow(True); c.tick_params(labelsize=6)

# --------------------------------------------- (c) las nubes, por instancia predicha
cm20 = plt.get_cmap("tab20")
for i, (sitio, pp, tit) in enumerate(P3D):
    ref = referencia(sitio, pp)
    reg = region(ref)
    for j, (nom, fn, colm) in enumerate([("SegmentAnyTree", inst_sat, COL["SAT"]),
                                         ("TreeLite3D", inst_tl, COL["TreeLite3D"])]):
        ax = fig.add_subplot(gsb[0, i * 2 + j], projection="3d")
        xyz, inst = fn(sitio, pp)
        d = reg.contains_points(xyz[:, :2])
        xyz, inst = xyz[d], inst[d]
        if len(xyz) > MAXP:
            q = np.random.default_rng(0).permutation(len(xyz))[:MAXP]
            xyz, inst = xyz[q], inst[q]
        col = np.where((inst > 0)[:, None], cm20((inst % 20) / 20.0)[:, :3],
                       np.array(matplotlib.colors.to_rgb(GRIS))[None, :])
        o = np.argsort(xyz[:, 2])
        ax.scatter(xyz[o, 0], xyz[o, 1], xyz[o, 2], c=col[o], s=.55, lw=0,
                   depthshade=False, rasterized=True)
        ax.set_axis_off(); ax.view_init(elev=16, azim=-58)
        for lim, k3 in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            lo, hi = xyz[:, k3].min(), xyz[:, k3].max()
            lim(lo, hi)
        ax.set_box_aspect((1, 1, .60), zoom=1.28)
        v = ARB[f"{sitio}/{pp}"]["met"]["SAT" if j == 0 else "TreeLite3D"]["3.5"]
        ax.text2D(.5, 1.0, nom, transform=ax.transAxes, fontsize=6.6, color=colm,
                  ha="center", va="top")
        ax.text2D(.5, .015, f"{v['TP']} of {v['NR']} found, recall {v['recall']:.3f}",
                  transform=ax.transAxes, fontsize=6.1, ha="center", color="0.25")
    fig.text(.145 + i * .5, .505, tit, fontsize=7.2)
fig.text(.006, .578, "(c) Returns coloured by the instance each model predicted",
         fontsize=7.5)

fig.savefig("fig1.pdf", dpi=400); fig.savefig("fig1.png", dpi=250)

# =============================================================== figura 2
fig2, ax2 = plt.subplots(1, 2, figsize=(7.16, 2.25))

a2 = ax2[0]
ETQ = ["casco +0 m", "casco +1,5 m", "casco +3 m", "tile entero"]
NOM = ["hull", "hull +1.5 m", "hull +3 m", "whole tile"]
for m in MET:
    a2.plot(np.arange(4), [B["region"][m][e]["precision"] for e in ETQ],
            marker=MRK[m], color=COL[m], lw=1.1, ms=3.2, label=ETI[m],
            ls=(0, (3, 1.6)) if m == "FF3D" else "-",
            zorder=4 if m == "SAT" else 3)
a2.set_xticks(np.arange(4)); a2.set_xticklabels(NOM, fontsize=6.2)
a2.set_ylabel("Precision"); a2.set_ylim(0, 1.02)
a2.set_xlabel("Region over which detections are counted")
a2.set_title("(a) The scoring region moves precision", fontsize=7.5, loc="left")
a2.legend(loc="lower left", fontsize=5.6, handlelength=1.2, borderpad=0.2,
          labelspacing=0.25)
a2.grid(lw=0.3, color="0.88"); a2.set_axisbelow(True)

b2 = ax2[1]
for m in MET:
    for i, s in enumerate(SIT):
        b2.scatter(B["principal"][s][m]["recall"], J[s][m]["J"], s=42,
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
print("fig1 y fig2 escritas")
