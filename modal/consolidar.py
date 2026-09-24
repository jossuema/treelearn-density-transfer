"""Recalcula todas las cifras de la carta bajo UN SOLO protocolo de emparejamiento.

Motivo: analisis.py mezclaba criterios sin decirlo. La tabla por sitio usa
emparejamiento solo por distancia, pero los estratos y la sensibilidad de apice
frente a centroide usaban el termino de solape de Tusa, y watershed_barrido.json
tambien. Aqui se rehace todo con solo distancia, que es lo que describe la seccion
de metodos, y el protocolo con solape queda como comprobacion de robustez.

La fuente es visor/data/arboles.json, que ya trae las parejas concretas por metodo
y distancia, recortadas a la region inventariada, y que reproduce la tabla por sitio
de analisis_completo.json con una diferencia menor que 0.0005.

Uso: python3 respaldo_minimo/modal_app/consolidar.py
"""
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)

ARB = json.load(open(os.path.join(BASE, "visor", "data", "arboles.json")))
SITIOS = (("cham", "Chamrousse", ["1", "1b", "2", "3", "3b", "4", "Premol"]),
          ("pell", "Pellizzano", [f"p{i}" for i in range(1, 16)]))
METODOS = ("SAT", "AMS3D", "WS")
DMATCH = ("2.5", "3.5", "5.0")
ESTRATOS = ("dominante", "intermedio", "suprimido")
NBOOT = 2000


def metricas(TP, NR, ND):
    R = TP / NR if NR else 0.0
    P = TP / ND if ND else 0.0
    return R, P, (2 * P * R / (P + R) if P + R else 0.0)


def agregar(claves, met, d):
    """Suma TP, ND y NR sobre un conjunto de parcelas y saca el intervalo por bootstrap.

    La semilla se reinicia en cada llamada para que el resultado no dependa del
    orden en que se pidan los agregados.
    """
    ps = [ARB[k]["met"][met][d] for k in claves if k in ARB]
    TP = sum(p["TP"] for p in ps); ND = sum(p["ND"] for p in ps); NR = sum(p["NR"] for p in ps)
    R, P, F = metricas(TP, NR, ND)
    rng = np.random.default_rng(0)
    Rs, Ps, Fs = [], [], []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(ps), len(ps))
        tp = sum(ps[i]["TP"] for i in idx)
        nd = sum(ps[i]["ND"] for i in idx)
        nr = sum(ps[i]["NR"] for i in idx)
        r, p, f = metricas(tp, nr, nd)
        Rs.append(r); Ps.append(p); Fs.append(f)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return dict(TP=TP, ND=ND, NR=NR, recall=round(R, 3), precision=round(P, 3), f1=round(F, 3),
                recall_ic=q(Rs), precision_ic=q(Ps), f1_ic=q(Fs), nparc=len(ps))


def estratos(claves, met, d):
    out = {e: {"n": 0, "TP": 0} for e in ESTRATOS}
    for k in claves:
        if k not in ARB:
            continue
        dd = ARB[k]
        ok = {r for _, r in dd["pares"][met][d]}
        for i, e in enumerate(dd["inv"]["estrato"]):
            if e not in out:
                continue
            out[e]["n"] += 1
            if i in ok:
                out[e]["TP"] += 1
    for e in out:
        out[e]["recall"] = round(out[e]["TP"] / out[e]["n"], 3) if out[e]["n"] else 0.0
    return out


def sin_recorte(claves, met, d):
    """Metricas si no se recortara al inventario: las detecciones de fuera cuentan.

    Hay que reemparejar, no basta con sumar al denominador, porque una deteccion de
    fuera puede caer a menos de d de un arbol del borde.
    """
    TP = ND = NR = 0
    dm = float(d)
    for k in claves:
        if k not in ARB:
            continue
        a = ARB[k]
        X = np.array(a["det"][met]["x"] + a["det_fuera"][met]["x"])
        Y = np.array(a["det"][met]["y"] + a["det_fuera"][met]["y"])
        rx, ry = np.array(a["inv"]["x"]), np.array(a["inv"]["y"])
        NR += len(rx); ND += len(X)
        if not len(X) or not len(rx):
            continue
        D = np.hypot(X[:, None] - rx[None, :], Y[:, None] - ry[None, :])
        C = np.where(D <= dm, D, np.inf)
        while C.size and np.isfinite(C).any():
            i, j = np.unravel_index(np.argmin(C), C.shape)
            TP += 1
            C = np.delete(np.delete(C, i, 0), j, 1)
    R, P, F = metricas(TP, NR, ND)
    return dict(TP=TP, ND=ND, NR=NR, recall=round(R, 3), precision=round(P, 3), f1=round(F, 3))


def main():
    sal = {"nota": "emparejamiento greedy solo por distancia; AMS3D y watershed con su "
                   "mejor configuracion por parcela; SegmentAnyTree sin ajuste",
           "por_sitio": {}, "global": {}, "estratos": {}}
    todas = [f"{s}/{p}" for s, _, ps in SITIOS for p in ps]

    for s, nom, ps in SITIOS:
        claves = [f"{s}/{p}" for p in ps]
        sal["por_sitio"][nom] = {m: {d: agregar(claves, m, d) for d in DMATCH} for m in METODOS}
        sal["estratos"][nom] = {m: estratos(claves, m, "3.5") for m in METODOS}
    sal["global"] = {m: {d: agregar(todas, m, d) for d in DMATCH} for m in METODOS}
    sal["sin_recorte"] = {}
    for s_, nom, ps in SITIOS:
        claves = [f"{s_}/{p}" for p in ps]
        sal["sin_recorte"][nom] = {m: sin_recorte(claves, m, "3.5") for m in METODOS}

    with open(os.path.join(RAIZ, "resultados_consistentes.json"), "w") as f:
        json.dump(sal, f, indent=2, ensure_ascii=False)

    print("DETECCION a 3.5 m, un solo protocolo")
    print(f"{'sitio':12} {'metodo':7} {'exhaust':>8} {'IC 95 %':>16} {'precis':>8} {'F1':>7} {'TP/ND/NR':>16}")
    for nom in list(sal["por_sitio"]) + ["Ambos"]:
        src = sal["por_sitio"].get(nom, sal["global"])
        for m in METODOS:
            a = src[m]["3.5"] if nom != "Ambos" else sal["global"][m]["3.5"]
            print(f"{nom:12} {m:7} {a['recall']:>8.3f} {str(a['recall_ic']):>16} "
                  f"{a['precision']:>8.3f} {a['f1']:>7.3f} {a['TP']:>5}/{a['ND']}/{a['NR']}")
        print()

    print("ESTRATOS a 3.5 m")
    for nom in sal["estratos"]:
        for m in METODOS:
            e = sal["estratos"][nom][m]
            print(f"{nom:12} {m:7} " + "  ".join(
                f"{k[:3]} {e[k]['recall']:.3f} ({e[k]['TP']}/{e[k]['n']})" for k in ESTRATOS))
        print()

    print("EFECTO DEL RECORTE a 3.5 m: precision con y sin recortar al inventario")
    for nom in sal["sin_recorte"]:
        for m in METODOS:
            con = sal["por_sitio"][nom][m]["3.5"]
            sin = sal["sin_recorte"][nom][m]
            print(f"{nom:12} {m:7} precision {con['precision']:.3f} -> {sin['precision']:.3f}   "
                  f"exhaustividad {con['recall']:.3f} -> {sin['recall']:.3f}   "
                  f"ND {con['ND']} -> {sin['ND']}")
        print()

    print("EXHAUSTIVIDAD frente a distancia")
    for nom in sal["por_sitio"]:
        for m in METODOS:
            print(f"{nom:12} {m:7} " + "  ".join(
                f"{d} m {sal['por_sitio'][nom][m][d]['recall']:.3f}" for d in DMATCH))
    print("\nguardado en resultados_consistentes.json")


if __name__ == "__main__":
    main()
