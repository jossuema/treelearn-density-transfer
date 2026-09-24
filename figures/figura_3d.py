"""Vistas 3D de dos parcelas, coloreadas por la instancia que predijo cada modelo.

Es lo que de verdad hace un modelo de segmentacion de instancias, y ensena el fallo de
TreeLite3D mucho mejor que un mapa de puntos: en la parcela dispersa deja la mayor parte
de la nube sin asignar a ningun arbol (gris), y en la densa la reparte bien.

cham/4 es la parcela mas dispersa del estudio y pell/p8 una de las mas densas.

Uso: python3 figura_3d.py
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

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(AQUI))
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
from analisis import referencia, region  # noqa: E402

SAL_SAT = os.path.join(RAIZ, "salida_sat_all")
SAL_TL = os.path.join(RAIZ, "salida_treelite", "t4_p30")
PARCELAS = [("cham", "4", "Chamrousse 4, 36 pts/m$^2$"),
            ("pell", "p8", "Pellizzano 8, 134 pts/m$^2$")]
MAXP = 26000
GRIS = "#c2c6cc"


def nube_entrada(sitio, p):
    r = (os.path.join(BASE, "IRSTEA_dataset", "las", "full_las", f"las{p}.las")
         if sitio == "cham" else
         os.path.join(BASE, "trento_dataset", "las", f"{p}.las"))
    L = laspy.read(r)
    return np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)])


def inst_sat(sitio, p):
    L = laspy.read(os.path.join(SAL_SAT, f"{sitio}_{p}_out.laz"))
    xyz = np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)])
    return xyz, np.asarray(L.PredInstance).astype(np.int64)


def inst_treelite(sitio, p):
    xyz = nube_entrada(sitio, p)
    e = np.load(os.path.join(SAL_TL, f"instance_pred_{sitio}_{p}.npy")).reshape(-1)
    a = np.load(os.path.join(SAL_TL, f"asignado_{sitio}_{p}.npy")).reshape(-1)
    # los puntos sin propuesta llevan la etiqueta 0 por el argmax; se marcan aparte
    return xyz, np.where(a, e + 1, 0)


ARB = json.load(open(os.path.join(BASE, "visor", "data", "arboles.json")))


def dibuja(ax, xyz, inst, reg, titulo, color_tit, pie):
    dentro = reg.contains_points(xyz[:, :2])
    xyz, inst = xyz[dentro], inst[dentro]
    if len(xyz) > MAXP:
        i = np.random.default_rng(0).permutation(len(xyz))[:MAXP]
        xyz, inst = xyz[i], inst[i]
    cm = plt.get_cmap("tab20")
    col = np.where((inst > 0)[:, None], cm((inst % 20) / 20.0)[:, :3],
                   np.array(matplotlib.colors.to_rgb(GRIS))[None, :])
    o = np.argsort(xyz[:, 2])
    ax.scatter(xyz[o, 0], xyz[o, 1], xyz[o, 2], c=col[o], s=.6, lw=0,
               depthshade=False, rasterized=True)
    ax.set_axis_off()
    ax.view_init(elev=16, azim=-58)
    ax.set_box_aspect((1, 1, .62))
    ax.dist = 7.2
    ax.set_title(titulo, fontsize=6.8, color=color_tit, pad=-2, loc="center")
    # el porcentaje de retornos asignados NO mide calidad: SegmentAnyTree deja el suelo
    # fuera y por eso asigna menos. Lo que importa es cuantos arboles del inventario
    # se encuentran.
    ax.text2D(.5, .03, pie, transform=ax.transAxes, fontsize=6.3, ha="center",
              color="0.25")


plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})
fig = plt.figure(figsize=(7.16, 2.15))
for i, (sitio, p, tit) in enumerate(PARCELAS):
    ref = referencia(sitio, p)
    reg = region(ref)
    for j, (nom, fn, col) in enumerate(
            [("SegmentAnyTree", inst_sat, "#1b3b6f"),
             ("TreeLite3D", inst_treelite, "#8e44ad")]):
        ax = fig.add_subplot(1, 4, i * 2 + j + 1, projection="3d")
        xyz, inst = fn(sitio, p)
        clave = f"{sitio}/{p}"
        met = "SAT" if nom == "SegmentAnyTree" else "TreeLite3D"
        v = ARB[clave]["met"][met]["3.5"]
        dibuja(ax, xyz, inst, reg, nom, col,
               f"{v['TP']} of {v['NR']} field trees found, recall {v['recall']:.3f}")
    fig.text(.045 + i * .5, .945, tit, fontsize=7.6)
fig.subplots_adjust(left=0, right=1, top=.99, bottom=-.05, wspace=-.03)
fig.savefig("fig4.pdf", dpi=400)
fig.savefig("fig4.png", dpi=300)
print("fig4 escrita")
