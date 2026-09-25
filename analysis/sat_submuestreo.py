"""Convierte las salidas de SegmentAnyTree sobre las nubes raleadas en detecciones.

Misma regla que en todo el articulo (det_sat): una deteccion por instancia con al menos
MIN_PTS puntos, situada en el apice, con area del casco convexo en planta.

Uso: python3 respaldo_minimo/analisis_paper/sat_submuestreo.py
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
from analisis import area_hull, MIN_PTS  # noqa: E402

ENTRADA = os.path.join(RAIZ, "salida_sat_sub")


def main():
    sal = {}
    fich = sorted(glob.glob(os.path.join(ENTRADA, "**", "*.la[sz]"), recursive=True))
    print(f"{len(fich)} ficheros en {ENTRADA}")
    for f in fich:
        base = os.path.basename(os.path.dirname(f))
        L = laspy.read(f)
        dims = [d.name for d in L.point_format.dimensions]
        campo = next((c for c in ("PredInstance", "preds_instance_segmentation",
                                  "instance", "treeID") if c in dims), None)
        if campo is None:
            print(f"  {base}: sin campo de instancia ({dims[:6]}...)")
            continue
        x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
        inst = np.asarray(getattr(L, campo)).astype(np.int64)
        X, Y, H, A = [], [], [], []
        for i in np.unique(inst[inst > 0]):
            m = inst == i
            if m.sum() < MIN_PTS:
                continue
            k = int(np.argmax(z[m]))
            X.append(float(x[m][k])); Y.append(float(y[m][k]))
            H.append(float(z[m].max())); A.append(area_hull(np.c_[x[m], y[m]]))
        sal[base] = dict(x=X, y=Y, h=H, area=A, puntos=len(x))
        print(f"  {base:20} {len(x):>8} pts  {len(X):>4} instancias")
    ruta = os.path.join(RAIZ, "salida_treelite", "detecciones_sub_SAT.json")
    json.dump(sal, open(ruta, "w"))
    print(f"\n{len(sal)} parcelas -> {ruta}")


if __name__ == "__main__":
    main()
