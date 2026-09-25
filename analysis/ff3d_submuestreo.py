"""Convierte las salidas de ForestFormer3D sobre las nubes raleadas en detecciones.

Misma regla que en todo el articulo (det_sat): una deteccion por instancia con al menos
MIN_PTS puntos, situada en el apice, con area del casco convexo en planta. Igual que en
evaluar_ff3d.py hay que deshacer el recentrado del pipeline, y aqui la referencia para
hacerlo es la nube RALEADA, que es la que recibio el modelo.

Uso: python3 respaldo_minimo/analisis_paper/ff3d_submuestreo.py
"""
import glob
import json
import os
import sys

import laspy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
sys.path.insert(0, AQUI)
from analisis import area_hull, MIN_PTS  # noqa: E402
from evaluar_ff3d import leer_ply  # noqa: E402

ENTRADA = os.path.join(RAIZ, "salida_ff3d_sub")
NUBES = os.path.join(RAIZ, "submuestreo")


def main():
    import zipfile
    sal = {}
    zips = sorted(glob.glob(os.path.join(ENTRADA, "*__results_*.zip")))
    print(f"{len(zips)} ficheros en {ENTRADA}")
    for z in zips:
        base = os.path.basename(z).split("__results_")[0]
        with zipfile.ZipFile(z) as f:
            plys = [i for i in f.namelist() if i.endswith(".ply")]
            if not plys:
                print(f"  {base}: sin PLY dentro del zip")
                continue
            arr = leer_ply(f.read(plys[0]))
        x, y, zz = (arr[c].astype(float) for c in ("x", "y", "z"))

        L = laspy.read(os.path.join(NUBES, f"{base}.las"))
        lx, ly, lz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
        for nom, a, b in (("x", x, lx), ("y", y, ly), ("z", zz, lz)):
            if abs((a.max() - a.min()) - (b.max() - b.min())) > 0.05:
                print(f"  AVISO {base} {nom}: extension {a.max()-a.min():.2f} "
                      f"frente a {b.max()-b.min():.2f}")
        x = x + (lx.min() - x.min())
        y = y + (ly.min() - y.min())
        zz = zz + (lz.min() - zz.min())

        inst = arr["instance_pred"].astype(np.int64)
        X, Y, H, A = [], [], [], []
        for i in np.unique(inst[inst >= 0]):
            m = inst == i
            if m.sum() < MIN_PTS:
                continue
            k = int(np.argmax(zz[m]))
            X.append(float(x[m][k])); Y.append(float(y[m][k]))
            H.append(float(zz[m].max())); A.append(area_hull(np.c_[x[m], y[m]]))
        sal[base] = dict(x=X, y=Y, h=H, area=A, puntos=len(lx))
        print(f"  {base:20} {len(lx):>8} pts  {len(X):>4} instancias")

    ruta = os.path.join(RAIZ, "salida_treelite", "detecciones_sub_FF3D.json")
    json.dump(sal, open(ruta, "w"))
    print(f"\n{len(sal)} nubes -> {ruta}")


if __name__ == "__main__":
    main()
