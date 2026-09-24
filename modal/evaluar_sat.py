"""Evalúa las salidas de SegmentAnyTree contra el inventario de campo.

Protocolo (el mismo para todos los métodos, para que la comparación sea limpia):
  - Un árbol detectado = una instancia predicha; su posición es el ápice (punto más alto).
  - Se recorta al casco convexo de los troncos del inventario, dilatado 1.5 m. Las instancias
    de fuera no cuentan ni como acierto ni como falso positivo: ahí hay dosel real que nadie midió.
  - Emparejamiento greedy uno a uno por distancia en planta, con tope d_match.
  - Exhaustividad = TP / árboles del inventario. Precisión = TP / instancias dentro del casco.
  - Agregado por sitio sumando TP, NR y ND sobre parcelas (igual que hace Tusa).

Uso: python3 respaldo_minimo/modal_app/evaluar_sat.py [carpeta_salidas]
"""
import csv
import json
import os
import sys

import laspy
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "salida_sat_all")
DMATCH = (2.5, 3.5, 5.0)
BUF = 1.5
MIN_PTS = 10          # instancias con menos puntos se descartan como ruido

CHAM = ["1", "1b", "2", "3", "3b", "4", "Premol"]
PELL = [f"p{i}" for i in range(1, 16)]


def inventario(sitio, parcela):
    """Devuelve x, y, altura y número de puntos ALS por árbol (este último no está en el CSV)."""
    if sitio == "cham":
        ruta, campos = os.path.join(RAIZ, "IRSTEA_dataset/inventory_csv/Chamrousse.csv"), ("x", "y", "height")
    else:
        ruta, campos = os.path.join(RAIZ, "trento_dataset/csv/Pellizzano.csv"), ("x", "y", "height")
    xs, ys, hs = [], [], []
    for r in csv.DictReader(open(ruta)):
        if r["plotid"] == parcela:
            try:
                xs.append(float(r[campos[0]])); ys.append(float(r[campos[1]])); hs.append(float(r[campos[2]]))
            except ValueError:
                pass
    return np.array(xs), np.array(ys), np.array(hs)


def dilatar(poly, buf):
    c = poly.mean(0)
    d = np.linalg.norm(poly - c, axis=1)
    d[d == 0] = 1e-9
    return c + (poly - c) * ((d + buf) / d)[:, None]


def emparejar(det_xy, ref_xy, dmax):
    if len(det_xy) == 0 or len(ref_xy) == 0:
        return 0
    D = np.linalg.norm(det_xy[:, None, :] - ref_xy[None, :, :], axis=2)
    tp = 0
    while D.size and D.min() <= dmax:
        i, j = np.unravel_index(np.argmin(D), D.shape)
        tp += 1
        D = np.delete(np.delete(D, i, 0), j, 1)
    return tp


def evaluar(sitio, parcela):
    laz = os.path.join(SAL, f"{sitio}_{parcela}_out.laz")
    if not os.path.exists(laz):
        return None
    L = laspy.read(laz)
    x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    inst = np.asarray(L.PredInstance).astype(np.int64)

    ix, iy, ih = inventario(sitio, parcela)
    if len(ix) < 3:
        return None

    # ápice de cada instancia
    ap = []
    for i in np.unique(inst[inst > 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        k = np.argmax(z[m])
        ap.append([x[m][k], y[m][k], float(z[m].max()), int(m.sum())])
    ap = np.array(ap) if ap else np.zeros((0, 4))

    hull = ConvexHull(np.c_[ix, iy])
    polyd = dilatar(np.c_[ix, iy][hull.vertices], BUF)
    dentro = MplPath(polyd).contains_points(ap[:, :2]) if len(ap) else np.zeros(0, bool)
    det = ap[dentro][:, :2] if len(ap) else np.zeros((0, 2))

    fila = {"sitio": sitio, "parcela": parcela, "NR": int(len(ix)),
            "instancias_tile": int(len(ap)), "ND": int(len(det)),
            "area_casco_m2": round(float(hull.volume)),
            "pct_tile_inventariado": round(100 * hull.volume / ((x.max() - x.min()) * (y.max() - y.min())), 1)}
    for dm in DMATCH:
        tp = emparejar(det, np.c_[ix, iy], dm)
        fila[f"TP@{dm}"] = tp
    return fila


def main():
    filas = [f for s, ps in (("cham", CHAM), ("pell", PELL)) for p in ps
             if (f := evaluar(s, p)) is not None]

    print(f"{'sitio':6} {'parc':7} {'NR':>4} {'tile':>5} {'ND':>4} {'%inv':>5} "
          + " ".join(f"{'R@'+str(d):>7}" for d in DMATCH))
    for f in filas:
        print(f"{f['sitio']:6} {f['parcela']:7} {f['NR']:>4} {f['instancias_tile']:>5} "
              f"{f['ND']:>4} {f['pct_tile_inventariado']:>5} "
              + " ".join(f"{f['TP@'+str(d)]/f['NR']:>7.3f}" for d in DMATCH))

    print("\n" + "=" * 74)
    print("AGREGADO POR SITIO (sumas de TP, NR y ND, como en el protocolo de Tusa)")
    print(f"{'sitio':12} {'NR':>5} {'ND':>5} {'dmatch':>7} {'TP':>5} {'exhaust':>8} {'precis':>8} {'F1':>7}")
    resumen = {}
    for sitio, etiqueta in (("cham", "Chamrousse"), ("pell", "Pellizzano")):
        sub = [f for f in filas if f["sitio"] == sitio]
        NR, ND = sum(f["NR"] for f in sub), sum(f["ND"] for f in sub)
        resumen[etiqueta] = {"NR": NR, "ND": ND, "parcelas": len(sub), "dmatch": {}}
        for dm in DMATCH:
            TP = sum(f[f"TP@{dm}"] for f in sub)
            R, P = TP / NR, (TP / ND if ND else 0)
            F1 = 2 * P * R / (P + R) if P + R else 0
            resumen[etiqueta]["dmatch"][str(dm)] = {"TP": TP, "recall": round(R, 3),
                                                    "precision": round(P, 3), "f1": round(F1, 3)}
            print(f"{etiqueta:12} {NR:>5} {ND:>5} {dm:>7} {TP:>5} {R:>8.3f} {P:>8.3f} {F1:>7.3f}")

    json.dump({"por_parcela": filas, "por_sitio": resumen},
              open(os.path.join(RAIZ, "sat_evaluacion.json"), "w"), indent=2, ensure_ascii=False)
    print("\nguardado en sat_evaluacion.json")


if __name__ == "__main__":
    main()
