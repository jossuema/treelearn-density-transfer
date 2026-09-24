"""Modulo B del visor: arboles de inventario, detecciones y emparejamientos por parcela.

Produce respaldo_minimo/visor/data/arboles.json con, para cada una de las 22 parcelas:
  - inv:   arboles del inventario (x, y, altura, area, radio, familia, retornos, estrato)
  - det:   detecciones de cada metodo recortadas a la region inventariada
  - pares: las parejas concretas (deteccion, arbol) a 2.5, 3.5 y 5.0 m
  - met:   TP, ND, NR, recall, precision y f1 de esa parcela

Reutiliza analisis.py sin tocarlo. El unico calculo que se reescribe aqui es el
emparejamiento, porque la funcion original devuelve solo los indices de arbol y el
visor necesita saber que deteccion se caso con que arbol.

Uso, siempre con cwd = respaldo_minimo:
    python3 visor/exportar_arboles.py
"""
import csv
import json
import os
import sys

import numpy as np

# La raiz de los datos se toma de la variable TREELEARN_DATA; por defecto se
# asume que este repositorio cuelga de ella.
RAIZ = os.environ.get("TREELEARN_DATA",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(RAIZ, "respaldo_minimo")
sys.path.insert(0, os.path.join(BASE, "modal_app"))

from analisis import (SITIOS, AMS, region, referencia, det_sat, det_ams,
                      det_watershed, emparejar)

SALIDA = os.path.join(BASE, "visor", "data", "arboles.json")
DMATCH = (2.5, 3.5, 5.0)
METODOS = ("SAT", "AMS3D", "WS")


# ------------------------------------------------------------------ utilidades
def carga_cfgs_ams():
    """Configuracion de AMS3D elegida para cada parcela, igual que main() en analisis.py."""
    cfgs = {}
    for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        p = os.path.join(AMS, f)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cfgs[(s, r["plot"])] = r["cfg"]
    return cfgs


def carga_cfgs_ws():
    """Configuracion de watershed por parcela, del barrido ya hecho."""
    w = json.load(open(os.path.join(RAIZ, "watershed_barrido.json")))
    cfgs = {}
    for sitio, nombre, parcelas in SITIOS:
        for p in parcelas:
            c = w[nombre]["configs"][p]
            cfgs[(sitio, p)] = (float(c["hmin"]), float(c["mindist"]))
    return cfgs


def vacia():
    return dict(x=np.array([]), y=np.array([]), h=np.array([]), area=np.array([]))


# ------------------------------------------------------------------ emparejar
def emparejar_pares(det, ref, dmax):
    """Misma logica greedy que emparejar(..., con_solape=False), pero devolviendo parejas.

    Matriz de distancias 2D, se admite solo d <= dmax, se toma el minimo global,
    se borra esa fila y esa columna, y se repite hasta que no queda nada admisible.
    Devuelve una lista de [indice_de_deteccion, indice_de_arbol].
    """
    nd, nr = len(det["x"]), len(ref["x"])
    if nd == 0 or nr == 0:
        return []
    D = np.hypot(det["x"][:, None] - ref["x"][None, :],
                 det["y"][:, None] - ref["y"][None, :])
    C = np.where(D <= dmax, D, np.inf)
    di, rj = np.arange(nd), np.arange(nr)
    pares = []
    while C.size and np.isfinite(C).any():
        i, j = np.unravel_index(np.argmin(C), C.shape)
        pares.append([int(di[i]), int(rj[j])])
        C = np.delete(np.delete(C, i, 0), j, 1)
        di, rj = np.delete(di, i), np.delete(rj, j)
    return pares


def recorta(det, pat):
    """Deja solo las detecciones dentro de la region y cuenta las que quedan fuera."""
    if det is None or len(det["x"]) == 0:
        return vacia(), 0
    dentro = pat.contains_points(np.c_[det["x"], det["y"]])
    dd = {k: v[dentro] for k, v in det.items() if isinstance(v, np.ndarray)}
    return dd, int((~dentro).sum())


def metricas(TP, NR, ND):
    R = TP / NR if NR else 0.0
    P = TP / ND if ND else 0.0
    return R, P, (2 * P * R / (P + R) if P + R else 0.0)


def r2(v):
    return [round(float(t), 2) for t in v]


# ------------------------------------------------------------------ principal
def main():
    cfgs_ams = carga_cfgs_ams()
    cfgs_ws = carga_cfgs_ws()

    salida = {}
    verificadas = []          # parcelas donde se comparo emparejar_pares con emparejar
    coinciden = 0

    for sitio, nombre, parcelas in SITIOS:
        for p in parcelas:
            clave = f"{sitio}/{p}"
            ref = referencia(sitio, p)
            pat = region(ref)

            crudas = {
                "SAT": det_sat(sitio, p, pos="apice"),
                "AMS3D": det_ams(sitio, p, cfgs_ams),
                # ojo: siempre con nombre, el primer posicional de det_watershed es cell
                "WS": det_watershed(ref, hmin=cfgs_ws[(sitio, p)][0],
                                    mindist=cfgs_ws[(sitio, p)][1]),
            }

            dets, pares, mets = {}, {}, {}
            for met in METODOS:
                d, fuera = recorta(crudas[met], pat)
                dets[met] = {"x": r2(d["x"]), "y": r2(d["y"]),
                             "h": r2(d["h"]), "area": r2(d["area"]), "fuera": fuera}
                pares[met], mets[met] = {}, {}
                for dm in DMATCH:
                    pr = emparejar_pares(d, ref, dm)
                    pares[met][str(dm)] = pr
                    TP, ND, NR = len(pr), len(d["x"]), len(ref["x"])
                    R, P, F = metricas(TP, NR, ND)
                    mets[met][str(dm)] = {"TP": TP, "ND": ND, "NR": NR,
                                          "recall": round(R, 3),
                                          "precision": round(P, 3),
                                          "f1": round(F, 3)}
                    # control cruzado contra la funcion original en unas cuantas parcelas
                    if dm == 3.5 and len(verificadas) < 5:
                        orig = sorted(int(t) for t in emparejar(d, ref, dm, False))
                        mio = sorted(t[1] for t in pr)
                        verificadas.append((clave, met, orig == mio, len(orig), len(mio)))
                        coinciden += int(orig == mio)

            salida[clave] = {
                "inv": {
                    "x": r2(ref["x"]), "y": r2(ref["y"]),
                    "h": r2(ref["h"]), "area": r2(ref["area"]),
                    "r": r2(np.sqrt(np.maximum(ref["area"], 0.0) / np.pi)),
                    "fam": [str(t) for t in ref["fam"]],
                    "npts": [int(t) for t in ref["npts"]],
                    "estrato": [str(t) for t in ref["estrato"]],
                },
                "det": dets,
                "pares": pares,
                "met": mets,
            }
            print(f"  {clave}: NR={len(ref['x'])} "
                  + " ".join(f"{m}={mets[m]['3.5']['TP']}/{mets[m]['3.5']['ND']}"
                             for m in METODOS), flush=True)

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    json.dump(salida, open(SALIDA, "w"), ensure_ascii=False)
    print(f"\nguardado en {SALIDA} ({os.path.getsize(SALIDA)} bytes, {len(salida)} parcelas)")

    # ------------------------------------------------------------ comprobaciones
    print("\n--- control del emparejamiento (mis pares frente a emparejar())")
    for cl, met, ok, a, b in verificadas:
        print(f"  {cl:12} {met:6} {'igual' if ok else 'DISTINTO'}  ({a} vs {b})")
    print(f"  coinciden {coinciden} de {len(verificadas)}")

    print("\n--- NR por sitio")
    nr_sitio = {}
    for sitio, nombre, parcelas in SITIOS:
        nr_sitio[nombre] = sum(salida[f"{sitio}/{p}"]["met"]["SAT"]["3.5"]["NR"] for p in parcelas)
        print(f"  {nombre}: {nr_sitio[nombre]}")

    # Nota sobre el watershed: watershed_barrido.json se genero con evaluar_parcela(..., True),
    # es decir con el termino de solape de Tusa, mientras que aqui todo va con distancia sola
    # para que los tres metodos usen el mismo criterio. La unica consecuencia es un arbol de
    # diferencia en Chamrousse (386 frente a 387 de 894). Pellizzano da identico en ambos.
    print("\n--- agregado a 3.5 m frente a los resultados ya publicados")
    ac = json.load(open(os.path.join(RAIZ, "analisis_completo.json")))["por_sitio"]["solo_distancia"]
    wb = json.load(open(os.path.join(RAIZ, "watershed_barrido.json")))
    fallos = []
    for sitio, nombre, parcelas in SITIOS:
        for met in METODOS:
            TP = sum(salida[f"{sitio}/{p}"]["met"][met]["3.5"]["TP"] for p in parcelas)
            ND = sum(salida[f"{sitio}/{p}"]["met"][met]["3.5"]["ND"] for p in parcelas)
            NR = nr_sitio[nombre]
            R, P, F = metricas(TP, NR, ND)
            if met == "WS":
                esp = wb[nombre]["3.5"]
            else:
                esp = ac[nombre][met]["3.5"]
            dr, dp = abs(R - esp["recall"]), abs(P - esp["precision"])
            ok = dr <= 0.002 and dp <= 0.002
            # caso conocido y explicado: el barrido del watershed uso el protocolo con solape
            conocido = (nombre == "Chamrousse" and met == "WS"
                        and TP == 386 and esp["TP"] == 387 and ND == esp["ND"])
            if not ok and not conocido:
                fallos.append((nombre, met, round(R, 4), round(P, 4),
                               esp["recall"], esp["precision"]))
            etq = "OK" if ok else ("conocido, 1 arbol por el protocolo" if conocido else "FALLA")
            print(f"  {nombre:11} {met:6} TP={TP:4} ND={ND:4} NR={NR:4} "
                  f"recall={R:.3f} (esp {esp['recall']:.3f}, d={dr:.4f})  "
                  f"prec={P:.3f} (esp {esp['precision']:.3f}, d={dp:.4f})  {etq}")
    print("\nfallos:", fallos if fallos else "ninguno")


if __name__ == "__main__":
    main()
