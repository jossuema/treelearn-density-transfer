"""Densidad de retornos de cada nube del experimento de raleo, medida siempre igual.

Puntos dentro de la region de la parcela, divididos por el area de esa region, sobre la
nube que de verdad recibio el modelo: la original para el 100 % y la raleada por pulsos
para 60, 40 y 25 %. Es la misma definicion que usa unificar.py para el resto del
articulo, y sirve para los tres modelos por igual, de modo que las correlaciones entre
densidad y exhaustividad se puedan comparar entre ellos.

Uso: python3 respaldo_minimo/analisis_paper/submuestreo_densidades.py
"""
import json
import os
import sys

import laspy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
from analisis import referencia, region  # noqa: E402

FRAC = (100, 60, 40, 25)
PS = [f"p{i}" for i in range(1, 16)]


def ruta(p, f):
    if f == 100:
        return os.path.join(BASE, "trento_dataset", "las", f"{p}.las")
    return os.path.join(RAIZ, "submuestreo", f"pell_{p}_f{f}.las")


def main():
    por_parcela, cols = {}, {f: [] for f in FRAC}
    print(f"{'parcela':8} {'area m2':>9} " + " ".join(f"{str(f)+' %':>9}" for f in FRAC))
    for p in PS:
        ref = referencia("pell", p)
        reg = region(ref)
        area = abs(np.sum(reg.vertices[:, 0] * np.roll(reg.vertices[:, 1], -1)
                          - np.roll(reg.vertices[:, 0], -1) * reg.vertices[:, 1]) / 2)
        por_parcela[p] = {"area": round(float(area), 1)}
        fila = f"{p:8} {area:>9.0f} "
        for f in FRAC:
            L = laspy.read(ruta(p, f))
            dentro = reg.contains_points(np.c_[np.asarray(L.x), np.asarray(L.y)])
            d = float(dentro.sum()) / area
            por_parcela[p][str(f)] = round(d, 1)
            cols[f].append(d)
            fila += f" {d:>9.1f}"
        print(fila)

    sal = {str(f): round(float(np.mean(cols[f])), 1) for f in FRAC}
    sal["por_parcela"] = por_parcela
    json.dump(sal, open(os.path.join(RAIZ, "submuestreo_densidades.json"), "w"), indent=2)
    print("\nmedia  " + " ".join(f"{sal[str(f)]:>9.1f}" for f in FRAC))
    print("guardado en submuestreo_densidades.json")


if __name__ == "__main__":
    main()
