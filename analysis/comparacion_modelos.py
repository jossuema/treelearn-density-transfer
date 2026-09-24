"""Los cinco metodos en la misma tabla, con diferencias pareadas por parcela.

Fuentes, todas bajo el mismo protocolo (region inventariada, emparejamiento uno a uno
solo por distancia, D = 3,5 m):
    SAT, AMS3D, WS   visor/data/arboles.json, que es lo que usa consolidar.py
    FF3D             ff3d_evaluacion.json
    TreeLite3D       el barrido, con el ajuste que elige el protocolo de una fuera

Las diferencias se calculan pareadas: se remuestrean parcelas, no metodos, asi que el
intervalo recoge que los dos metodos vieron exactamente las mismas parcelas.

Uso: python3 respaldo_minimo/analisis_paper/comparacion_modelos.py
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
sys.path.insert(0, AQUI)
from analisis import SITIOS  # noqa: E402
from treelite_final import cargar, f1_de, metricas  # noqa: E402

NB = 2000
D = "3.5"
ORDEN = ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")


def reunir():
    """Por metodo y parcela: TP, ND, NR."""
    arb = json.load(open(os.path.join(BASE, "visor", "data", "arboles.json")))
    claves = [f"{s}/{p}" for s, _, ps in SITIOS for p in ps]
    datos = {m: {} for m in ORDEN}
    for k in claves:
        for m in ("SAT", "AMS3D", "WS"):
            v = arb[k]["met"][m][D]
            datos[m][k] = dict(TP=v["TP"], ND=v["ND"], NR=v["NR"])
    ff = json.load(open(os.path.join(RAIZ, "ff3d_evaluacion.json")))["por_parcela"]
    for k, v in ff.items():
        datos["FF3D"][k] = dict(TP=v["TP"], ND=v["dentro"], NR=v["NR"])

    ajustes = cargar()
    mejor = max(ajustes, key=lambda e: f1_de(ajustes[e], claves))
    for k in claves:
        v = ajustes[mejor][k]
        datos["TreeLite3D"][k] = dict(TP=v["TP"], ND=v["ND"], NR=v["NR"])
    return datos, claves, mejor


def agrega(ps, semilla=0):
    TP, ND, NR = (sum(p[x] for p in ps) for x in ("TP", "ND", "NR"))
    r, pr, f1 = metricas(TP, NR, ND)
    rng = np.random.default_rng(semilla)
    Rs, Fs = [], []
    for _ in range(NB):
        i = rng.integers(0, len(ps), len(ps))
        a, b, c = (sum(ps[j][x] for j in i) for x in ("TP", "ND", "NR"))
        rr, _, ff = metricas(a, c, b)
        Rs.append(rr); Fs.append(ff)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3),
                   round(float(np.percentile(v, 97.5)), 3)]
    return dict(TP=TP, ND=ND, NR=NR, recall=round(r, 3), precision=round(pr, 3),
                f1=round(f1, 3), recall_ic=q(Rs), f1_ic=q(Fs))


def diferencia(a, b, ks, semilla=0):
    """Exhaustividad de a menos la de b, remuestreando las mismas parcelas."""
    ra = sum(a[k]["TP"] for k in ks) / sum(a[k]["NR"] for k in ks)
    rb = sum(b[k]["TP"] for k in ks) / sum(b[k]["NR"] for k in ks)
    rng = np.random.default_rng(semilla)
    v = []
    for _ in range(NB):
        i = rng.integers(0, len(ks), len(ks))
        kk = [ks[j] for j in i]
        na = sum(a[k]["NR"] for k in kk); nb = sum(b[k]["NR"] for k in kk)
        v.append((sum(a[k]["TP"] for k in kk) / na if na else 0)
                 - (sum(b[k]["TP"] for k in kk) / nb if nb else 0))
    return dict(dif=round(ra - rb, 3),
                ic=[round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)])


def main():
    datos, claves, mejor = reunir()
    sitios = {n: [f"{s}/{p}" for p in ps] for s, n, ps in SITIOS}
    grupos = list(sitios.items()) + [("Ambos", claves)]
    print(f"TreeLite3D con el ajuste {mejor}; D = {D} m\n")

    sal = {"D": D, "ajuste_treelite": mejor, "por_sitio": {}, "dif_vs_SAT": {}}
    print(f"{'metodo':12} " + " | ".join(f"{n:^30}" for n, _ in grupos))
    print(f"{'':12} " + " | ".join(" exh    IC 95 %      prec   F1 " for _ in grupos))
    for m in ORDEN:
        fila = f"{m:12} "
        for n, ks in grupos:
            v = agrega([datos[m][k] for k in ks])
            sal["por_sitio"].setdefault(n, {})[m] = v
            fila += (f"{v['recall']:.3f} {str(v['recall_ic']):>14} "
                     f"{v['precision']:.3f} {v['f1']:.3f} | ")
        print(fila)

    print(f"\ndiferencia de exhaustividad frente a SegmentAnyTree, pareada por parcela")
    print(f"{'metodo':12} " + " | ".join(f"{n:^24}" for n, _ in grupos))
    for m in ORDEN[1:]:
        fila = f"{m:12} "
        for n, ks in grupos:
            d = diferencia(datos["SAT"], datos[m], ks)
            sal["dif_vs_SAT"].setdefault(n, {})[m] = d
            marca = " *" if d["ic"][0] > 0 or d["ic"][1] < 0 else "  "
            fila += f"{d['dif']:>+7.3f} {str(d['ic']):>15}{marca} | "
        print(fila)
    print("\n* el intervalo no cruza el cero")

    json.dump(sal, open(os.path.join(RAIZ, "comparacion_modelos.json"), "w"),
              indent=2, ensure_ascii=False)
    print("guardado en comparacion_modelos.json")


if __name__ == "__main__":
    main()
