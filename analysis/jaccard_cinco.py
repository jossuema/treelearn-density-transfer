"""Indice de Jaccard de copa para los cinco metodos, con circulos equivalentes.

Solo AMS3D y el watershed publican un area de copa, no una forma, asi que para que la
comparacion sea justa los cinco se representan igual: un circulo de area equivalente
centrado en la deteccion. El de referencia sale del area de copa del inventario.

Las parejas son las mismas del cuadro de deteccion: uno a uno por distancia, 3,5 m,
dentro de la region. Se comprueba que reproduce jaccard.json para los tres metodos que
ya estaban, antes de extenderlo a los dos nuevos.

Uso: python3 respaldo_minimo/analisis_paper/jaccard_cinco.py
"""
import json
import os
import sys

import numpy as np
from shapely.geometry import Point

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
sys.path.insert(0, AQUI)
from analisis import SITIOS, region  # noqa: E402
from robustez_cinco import MET, pares_greedy, ref_de  # noqa: E402


def circulo(x, y, a):
    return Point(x, y).buffer(max(np.sqrt(max(a, 1e-6) / np.pi), 0.05), quad_segs=16)


def main():
    datos = json.load(open(os.path.join(RAIZ, "detecciones_todas.json")))
    claves = [f"{s}/{p}" for s, _, ps in SITIOS for p in ps]
    sitios = {n: [f"{s}/{p}" for p in ps] for s, n, ps in SITIOS}

    js = {m: {k: [] for k in claves} for m in MET}
    for k in claves:
        ref = ref_de(k)
        pr = [circulo(ref["x"][i], ref["y"][i], ref["area"][i])
              for i in range(len(ref["x"]))]
        for m in MET:
            d = datos["detecciones"][k][m]
            x, y, a = (np.array(d[c]) for c in ("x", "y", "area"))
            dentro = region(ref).contains_points(np.c_[x, y]) if len(x) \
                else np.zeros(0, bool)
            xd, yd, ad = x[dentro], y[dentro], a[dentro]
            for i, j in pares_greedy(xd, yd, ref["x"], ref["y"], 3.5):
                pd_ = circulo(xd[i], yd[i], ad[i])
                inter = pr[j].intersection(pd_).area
                union = pr[j].area + pd_.area - inter
                js[m][k].append(inter / union if union else 0.0)

    sal = {}
    print(f"{'metodo':12} " + " ".join(f"{n:>22}" for n in
          list(sitios) + ["Ambos"]))
    for m in MET:
        fila = f"{m:12} "
        for n, ks in list(sitios.items()) + [("Ambos", claves)]:
            v = [x for k in ks for x in js[m][k]]
            sal.setdefault(n, {})[m] = dict(J=round(float(np.mean(v)), 3),
                                            pares=len(v))
            fila += f" {np.mean(v):.3f} ({len(v):>4} pares)  "
        print(fila)

    # comprobacion contra lo que ya habia, que se calculo por otro camino
    pub = json.load(open(os.path.join(RAIZ, "jaccard.json")))
    print(f"\ncomprobacion contra jaccard.json")
    print(f"{'sitio':12} {'metodo':10} {'aqui':>7} {'antes':>7} {'dif':>7}")
    for n in sitios:
        for m, clave in (("SAT", "SAT_circulo"), ("AMS3D", "AMS3D"), ("WS", "Watershed")):
            if n in pub and clave in pub[n]:
                a, b = sal[n][m]["J"], pub[n][clave]["J_2D"]
                print(f"{n:12} {m:10} {a:>7.3f} {b:>7.3f} {a - b:>+7.3f}")

    json.dump(sal, open(os.path.join(RAIZ, "jaccard_cinco.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\nguardado en jaccard_cinco.json")


if __name__ == "__main__":
    main()
