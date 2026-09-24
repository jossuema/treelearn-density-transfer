"""Anade ForestFormer3D y TreeLite3D a los datos del visor.

El visor se construyo con tres metodos. Las detecciones de los dos modelos nuevos ya
estan en detecciones_todas.json, sin recortar, asi que aqui solo se recortan a la region
y se emparejan con el MISMO protocolo que usa el articulo (uno a uno por distancia), para
que lo que se ve en pantalla sea exactamente lo que se cuenta en los cuadros.

Uso: python3 respaldo_minimo/visor/ampliar_metodos.py
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
sys.path.insert(0, os.path.join(BASE, "analisis_paper"))
from analisis import SITIOS, referencia, region  # noqa: E402
from robustez_cinco import pares_greedy, metricas  # noqa: E402

NUEVOS = ("FF3D", "TreeLite3D")
DIST = ("2.5", "3.5", "5.0")


def main():
    todas = json.load(open(os.path.join(RAIZ, "detecciones_todas.json")))["detecciones"]
    ruta = os.path.join(AQUI, "data", "arboles.json")
    arb = json.load(open(ruta))

    print(f"{'parcela':12} " + " ".join(f"{m:>22}" for m in NUEVOS))
    for sitio, _, ps in SITIOS:
        for p in ps:
            k = f"{sitio}/{p}"
            ref = referencia(sitio, p)
            reg = region(ref)
            fila = f"{k:12} "
            for m in NUEVOS:
                d = todas[k][m]
                x, y = np.array(d["x"]), np.array(d["y"])
                dentro = reg.contains_points(np.c_[x, y]) if len(x) else np.zeros(0, bool)
                sel = {c: np.array(d[c])[dentro] for c in ("x", "y", "h", "area")}
                fue = {c: np.array(d[c])[~dentro] for c in ("x", "y", "h", "area")}
                arb[k]["det"][m] = {c: [round(float(v), 2) for v in sel[c]]
                                    for c in ("x", "y", "h", "area")}
                arb[k]["det"][m]["fuera"] = int((~dentro).sum())
                arb[k]["det_fuera"][m] = {c: [round(float(v), 2) for v in fue[c]]
                                          for c in ("x", "y", "h", "area")}
                arb[k]["pares"][m], arb[k]["met"][m] = {}, {}
                for ds in DIST:
                    pr = pares_greedy(sel["x"], sel["y"], ref["x"], ref["y"], float(ds))
                    arb[k]["pares"][m][ds] = [[int(i), int(j)] for i, j in pr]
                    TP, ND, NR = len(pr), int(dentro.sum()), len(ref["x"])
                    r, pc, f1 = metricas(TP, NR, ND)
                    arb[k]["met"][m][ds] = dict(TP=TP, ND=ND, NR=NR, recall=round(r, 3),
                                                precision=round(pc, 3), f1=round(f1, 3))
                fila += f"  {arb[k]['met'][m]['3.5']['TP']:>3}/{dentro.sum():<3} de {len(ref['x']):<4}"
            print(fila)

    with open(ruta, "w") as f:
        json.dump(arb, f)
    print(f"\narboles.json ampliado a {len(arb['cham/1']['met'])} metodos")


if __name__ == "__main__":
    main()
