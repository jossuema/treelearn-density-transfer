"""Resultado final de TreeLite3D: por defecto, con ajuste oracular y con una parcela fuera.

TreeLite3D viene configurado para nubes de miles de puntos por metro cuadrado: agrupa
con un radio de cluster_thresh * voxel_size = 0,05 m y solo acepta propuestas de mas de
100 puntos. Con eso, sobre nuestras parcelas casi no propone nada. Se barrieron 22
combinaciones de radio y tamano minimo de propuesta.

Ese barrido se eligio mirando los mismos datos con los que se mide, que es el problema
que ya senalamos en AMS3D y en el watershed. Por eso se dan las tres cifras:
  por defecto   lo que da el modelo tal cual lo publican
  oracular      la mejor combinacion sobre las 22 parcelas, cota superior
  una fuera     para cada parcela se elige la combinacion con las otras 21

Uso: python3 respaldo_minimo/analisis_paper/treelite_final.py
"""
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
from analisis import referencia, region, SITIOS  # noqa: E402

SALIDA = os.path.join(RAIZ, "salida_treelite")
D = 3.5
NB = 2000
ESTRATOS = ("dominante", "intermedio", "suprimido")
REF = {}


def ref_de(sitio, p):
    if (sitio, p) not in REF:
        REF[(sitio, p)] = referencia(sitio, p)
    return REF[(sitio, p)]


def greedy_pares(dx, dy, rx, ry, dmax=D):
    """Emparejamiento uno a uno por distancia; devuelve los indices de referencia."""
    if not len(dx) or not len(rx):
        return set()
    Dm = np.hypot(np.asarray(dx)[:, None] - np.asarray(rx)[None, :],
                  np.asarray(dy)[:, None] - np.asarray(ry)[None, :])
    cand = np.argwhere(Dm <= dmax)
    orden = np.argsort(Dm[cand[:, 0], cand[:, 1]], kind="stable")
    ud, ur = set(), set()
    for i, j in cand[orden]:
        if i in ud or j in ur:
            continue
        ud.add(i); ur.add(j)
    return ur


def metricas(tp, nr, nd):
    r = tp / nr if nr else 0.0
    p = tp / nd if nd else 0.0
    return r, p, (2 * p * r / (p + r) if p + r else 0.0)


def cargar():
    """Para cada ajuste y parcela: TP, ND, NR y que arboles del inventario se acertaron."""
    ajustes = {}
    for f in sorted(glob.glob(os.path.join(SALIDA, "detecciones_*.json"))):
        et = os.path.basename(f)[len("detecciones_"):-5]
        det = json.load(open(f))
        d_aj = {}
        for sitio, _, parcelas in SITIOS:
            for p in parcelas:
                clave, k = f"{sitio}_{p}", f"{sitio}/{p}"
                d = det.get(clave, {})
                if str(d.get("error", "")).startswith("0 etiquetas"):
                    d = dict(x=[], y=[])
                x, y = np.array(d.get("x", [])), np.array(d.get("y", []))
                ref = ref_de(sitio, p)
                dentro = region(ref).contains_points(np.c_[x, y]) if len(x) \
                    else np.zeros(0, bool)
                ok = greedy_pares(x[dentro], y[dentro], ref["x"], ref["y"])
                d_aj[k] = dict(TP=len(ok), ND=int(dentro.sum()), NR=len(ref["x"]),
                               ok=ok, sitio=sitio)
        ajustes[et] = d_aj
    return ajustes


def agrega(ps, semilla=0):
    TP = sum(p["TP"] for p in ps); ND = sum(p["ND"] for p in ps)
    NR = sum(p["NR"] for p in ps)
    r, pr, f1 = metricas(TP, NR, ND)
    rng = np.random.default_rng(semilla)
    Rs, Ps, Fs = [], [], []
    for _ in range(NB):
        i = rng.integers(0, len(ps), len(ps))
        a, b, c = (sum(ps[j][x] for j in i) for x in ("TP", "ND", "NR"))
        rr, pp, ff = metricas(a, c, b)
        Rs.append(rr); Ps.append(pp); Fs.append(ff)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3),
                   round(float(np.percentile(v, 97.5)), 3)]
    return dict(TP=TP, ND=ND, NR=NR, recall=round(r, 3), precision=round(pr, 3),
                f1=round(f1, 3), recall_ic=q(Rs), precision_ic=q(Ps), f1_ic=q(Fs),
                nparc=len(ps))


def f1_de(d_aj, claves):
    TP = sum(d_aj[k]["TP"] for k in claves)
    ND = sum(d_aj[k]["ND"] for k in claves)
    NR = sum(d_aj[k]["NR"] for k in claves)
    return metricas(TP, NR, ND)[2]


def main():
    ajustes = cargar()
    claves = [f"{s}/{p}" for s, _, ps in SITIOS for p in ps]
    sitios = {n: [f"{s}/{p}" for p in ps] for s, n, ps in SITIOS}

    mejor = max(ajustes, key=lambda e: f1_de(ajustes[e], claves))
    print(f"{len(ajustes)} ajustes probados; mejor sobre las 22 parcelas: {mejor}\n")

    sal = {"ajustes_probados": sorted(ajustes), "mejor": mejor, "D": D}
    print(f"{'variante':22} {'sitio':12} {'exh':>6} {'IC 95 %':>16} "
          f"{'prec':>6} {'F1':>6} {'ND':>6}")
    for nombre, et in (("por defecto", "t1"), ("oracular", mejor)):
        sal[nombre] = {"ajuste": et}
        for sit, ks in list(sitios.items()) + [("Ambos", claves)]:
            v = agrega([ajustes[et][k] for k in ks])
            sal[nombre][sit] = v
            print(f"{nombre + ' (' + et + ')':22} {sit:12} {v['recall']:>6.3f} "
                  f"{str(v['recall_ic']):>16} {v['precision']:>6.3f} "
                  f"{v['f1']:>6.3f} {v['ND']:>6}")

    # una parcela fuera: la combinacion se elige con las otras 21
    elegidos, ps_lopo = {}, {}
    for k in claves:
        resto = [c for c in claves if c != k]
        e = max(ajustes, key=lambda a: f1_de(ajustes[a], resto))
        elegidos[k] = e
        ps_lopo[k] = ajustes[e][k]
    sal["una_fuera"] = {"elegidos": elegidos}
    for sit, ks in list(sitios.items()) + [("Ambos", claves)]:
        v = agrega([ps_lopo[k] for k in ks])
        sal["una_fuera"][sit] = v
        print(f"{'una fuera':22} {sit:12} {v['recall']:>6.3f} "
              f"{str(v['recall_ic']):>16} {v['precision']:>6.3f} "
              f"{v['f1']:>6.3f} {v['ND']:>6}")
    cuenta = {}
    for e in elegidos.values():
        cuenta[e] = cuenta.get(e, 0) + 1
    print(f"\najustes que salieron elegidos: {cuenta}")

    # estratos con el ajuste oracular, para ver donde se pierde
    print(f"\n{'sitio':12} {'estrato':12} {'n':>5} {'TP':>5} {'exh':>7}")
    sal["estratos"] = {}
    for sit, ks in list(sitios.items()) + [("Ambos", claves)]:
        sal["estratos"][sit] = {}
        for est in ESTRATOS:
            n = tp = 0
            for k in ks:
                s, p = k.split("/")
                ref = ref_de(s, p)
                m = ref["estrato"] == est
                n += int(m.sum())
                tp += sum(1 for i in ajustes[mejor][k]["ok"] if ref["estrato"][i] == est)
            sal["estratos"][sit][est] = dict(n=n, TP=tp,
                                             recall=round(tp / n, 3) if n else 0.0)
            print(f"{sit:12} {est:12} {n:>5} {tp:>5} {tp / n if n else 0:>7.3f}")

    for v in sal.get("una_fuera", {}).values():
        pass
    json.dump(sal, open(os.path.join(RAIZ, "treelite_evaluacion.json"), "w"),
              indent=2, ensure_ascii=False, default=str)
    print("\nguardado en treelite_evaluacion.json")


if __name__ == "__main__":
    main()
