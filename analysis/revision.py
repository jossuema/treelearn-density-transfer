"""Comprobaciones pedidas por la revision externa, con numeros y no con opiniones.

  A. AMS3D sin mirar la parcela de prueba. La carta usa la mejor de 25 configuraciones
     elegida sobre la misma parcela que se puntua (oraculo). Aqui se anade la version
     operacional: la celda (i, j) de la malla 5x5 se elige en las OTRAS parcelas del
     mismo sitio y se aplica a la parcela excluida. Las mallas son propias de cada
     parcela, porque salen de su alometria, asi que lo que se transfiere es la
     posicion en la malla, igual que hacia eval_controlled.py.
  B. No inferioridad de SegmentAnyTree frente a AMS3D con margen de 0.05. Se usa el
     percentil 5 del bootstrap de la diferencia emparejada, que es el limite de un
     intervalo unilateral al 95 %.
  C. Emparejamiento greedy frente al optimo (hungaro con maxima cardinalidad).
  D. Pellizzano con el limite real de la parcela, un cuadrado de 40 m que sale de las
     imagenes de inventario, frente al casco dilatado y frente al tile entero.

Uso: python3 respaldo_minimo/analisis_paper/revision.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
from analisis import region  # noqa: E402

ARB = json.load(open(os.path.join(BASE, "visor", "data", "arboles.json")))
AMSDIR = os.path.join(BASE, "ams_port", "out")
SITIOS = {"cham": ["1", "1b", "2", "3", "3b", "4", "Premol"],
          "pell": [f"p{i}" for i in range(1, 16)]}
D = 3.5
NB = 2000
DELTA = 0.05


# ----------------------------------------------------------------- emparejamiento
def greedy(dx, dy, rx, ry, dmax=D):
    if not len(dx) or not len(rx):
        return 0
    Dm = np.hypot(np.asarray(dx)[:, None] - np.asarray(rx)[None, :],
                  np.asarray(dy)[:, None] - np.asarray(ry)[None, :])
    cand = np.argwhere(Dm <= dmax)
    orden = np.argsort(Dm[cand[:, 0], cand[:, 1]], kind="stable")
    ud, ur, n = set(), set(), 0
    for i, j in cand[orden]:
        if i in ud or j in ur:
            continue
        ud.add(i); ur.add(j); n += 1
    return n


def hungaro(dx, dy, rx, ry, dmax=D):
    """Maxima cardinalidad y, entre esas, minima distancia total."""
    if not len(dx) or not len(rx):
        return 0
    Dm = np.hypot(np.asarray(dx)[:, None] - np.asarray(rx)[None, :],
                  np.asarray(dy)[:, None] - np.asarray(ry)[None, :])
    GRANDE = 1e6       # mayor que cualquier suma de distancias admisibles
    C = np.where(Dm <= dmax, Dm, GRANDE)
    f, c = linear_sum_assignment(C)
    return int((C[f, c] < GRANDE).sum())


# ----------------------------------------------------------------- utilidades
def boot(filas, fn, q=(2.5, 97.5)):
    """filas: una tupla de conteos por parcela. fn(suma de filas) -> escalar."""
    filas = [np.asarray(f, float) for f in filas]
    rng = np.random.default_rng(0)
    v = []
    for _ in range(NB):
        idx = rng.integers(0, len(filas), len(filas))
        v.append(fn(sum(filas[i] for i in idx)))
    return [round(float(np.percentile(v, x)), 3) for x in q]


def f1(tp, nd, nr):
    r = tp / nr if nr else 0.0
    p = tp / nd if nd else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# ----------------------------------------------------------------- A. AMS3D
def dets_ams(plot):
    """Las 25 configuraciones de una parcela con su posicion (i, j) en la malla."""
    out = []
    for f in glob.glob(os.path.join(AMSDIR, f"e1_dets_{plot}_AMS3DE_las_{plot}_m1_*_m2_*.csv")):
        m = re.search(r"_m1_(\d+)_m2_(\d+)\.csv$", f)
        x, y = [], []
        for r in csv.DictReader(open(f)):
            x.append(float(r["x"])); y.append(float(r["y"]))
        out.append({"m1": int(m.group(1)), "m2": int(m.group(2)),
                    "x": np.array(x), "y": np.array(y), "cfg": os.path.basename(f)})
    m1s = sorted({o["m1"] for o in out}); m2s = sorted({o["m2"] for o in out})
    for o in out:
        o["i"], o["j"] = m1s.index(o["m1"]), m2s.index(o["m2"])
    return out


def conteo(key, x, y):
    a = ARB[key]
    R = region({"x": np.array(a["inv"]["x"]), "y": np.array(a["inv"]["y"])})
    if len(x):
        dentro = R.contains_points(np.c_[x, y])
        x, y = x[dentro], y[dentro]
    tp = greedy(x, y, a["inv"]["x"], a["inv"]["y"])
    return (tp, len(x), len(a["inv"]["x"]))


print("A. AMS3D: oraculo por parcela frente a eleccion operacional")
tabla = {}          # (sitio, parcela) -> {(i,j): (tp, nd, nr)}
for s, ps in SITIOS.items():
    for p in ps:
        tabla[(s, p)] = {(o["i"], o["j"]): conteo(f"{s}/{p}", o["x"], o["y"]) for o in dets_ams(p)}
        assert len(tabla[(s, p)]) == 25, (s, p, len(tabla[(s, p)]))

# configuracion que usa la carta: la mejor segun el F1 del protocolo de Tusa
carta = {}
for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
    for r in csv.DictReader(open(os.path.join(AMSDIR, f))):
        m = re.search(r"_m1_(\d+)_m2_(\d+)$", r["cfg"])
        carta[(s, r["plot"])] = (int(m.group(1)), int(m.group(2)))

sel = {"oraculo de la carta": {}, "oraculo, F1 nuestro": {}, "validado dejando una fuera": {}}
for (s, p), celdas in tabla.items():
    # oraculo de la carta
    obj = carta[(s, p)]
    o = [c for c in dets_ams(p) if (c["m1"], c["m2"]) == obj][0]
    sel["oraculo de la carta"][(s, p)] = celdas[(o["i"], o["j"])]
    # oraculo con nuestro F1
    sel["oraculo, F1 nuestro"][(s, p)] = max(celdas.values(), key=lambda t: f1(*t))
    # dejando una fuera dentro del mismo sitio
    otras = [q for q in SITIOS[s] if q != p]
    mejor = max(celdas, key=lambda ij: f1(*[sum(tabla[(s, q)][ij][k] for q in otras) for k in range(3)]))
    sel["validado dejando una fuera"][(s, p)] = celdas[mejor]

res_A = {}
for nombre, d in sel.items():
    res_A[nombre] = {}
    for s in list(SITIOS) + ["global"]:
        claves = [k for k in d if s == "global" or k[0] == s]
        tp, nd, nr = [sum(d[k][i] for k in claves) for i in range(3)]
        res_A[nombre][s] = dict(TP=tp, ND=nd, NR=nr, recall=round(tp / nr, 3),
                                precision=round(tp / nd, 3), f1=round(f1(tp, nd, nr), 3))
    g = res_A[nombre]
    print(f"  {nombre:28} exh cham {g['cham']['recall']:.3f} pell {g['pell']['recall']:.3f} "
          f"global {g['global']['recall']:.3f} | F1 global {g['global']['f1']:.3f}")


# ----------------------------------------------------------------- B. no inferioridad
print(f"\nB. No inferioridad de SegmentAnyTree frente a AMS3D, margen {DELTA}")
print("   p5 es el limite inferior del intervalo unilateral al 95 %; si es mayor que "
      f"-{DELTA}, SAT no es peor en mas de {DELTA}")


def filas_sat(s):
    out = []
    for p in (SITIOS[s] if s != "global" else SITIOS["cham"] + SITIOS["pell"]):
        k = (f"cham/{p}" if p in SITIOS["cham"] else f"pell/{p}")
        m = ARB[k]["met"]["SAT"]["3.5"]
        out.append((k, (m["TP"], m["ND"], m["NR"])))
    return out


res_B = {}
for nombre in ("oraculo de la carta", "validado dejando una fuera"):
    res_B[nombre] = {}
    for s in ("cham", "pell", "global"):
        sat = filas_sat(s)
        filas = []
        for k, (tp, nd, nr) in sat:
            sit, p = k.split("/")
            atp, and_, anr = sel[nombre][(sit, p)]
            filas.append((tp, nd, nr, atp, and_))
        dr = boot(filas, lambda t: t[0] / t[2] - t[3] / t[2], q=(5, 2.5, 50, 95, 97.5))
        df1 = boot(filas, lambda t: f1(t[0], t[1], t[2]) - f1(t[3], t[4], t[2]), q=(5, 2.5, 50, 95, 97.5))
        tot = np.sum(np.array(filas, float), 0)
        pr = tot[0] / tot[2] - tot[3] / tot[2]
        pf = f1(tot[0], tot[1], tot[2]) - f1(tot[3], tot[4], tot[2])
        res_B[nombre][s] = {"recall": dict(dif=round(pr, 3), p5=dr[0], ic95=[dr[1], dr[4]],
                                           no_inferior=dr[0] > -DELTA,
                                           equivalente=dr[0] > -DELTA and dr[3] < DELTA),
                            "f1": dict(dif=round(pf, 3), p5=df1[0], ic95=[df1[1], df1[4]],
                                       no_inferior=df1[0] > -DELTA,
                                       equivalente=df1[0] > -DELTA and df1[3] < DELTA)}
        for met in ("recall", "f1"):
            r = res_B[nombre][s][met]
            print(f"  frente a {nombre:26} {s:6} {met:6} dif {r['dif']:+.3f}  p5 {r['p5']:+.3f}  "
                  f"no inferior {'SI' if r['no_inferior'] else 'no'}  "
                  f"equivalente {'SI' if r['equivalente'] else 'no'}")


# ----------------------------------------------------------------- C. greedy frente a optimo
print("\nC. Emparejamiento greedy frente a hungaro de maxima cardinalidad, a 3.5 m")
res_C = {}
for m in ("SAT", "AMS3D", "WS"):
    g = h = 0
    for k, a in ARB.items():
        dd = a["det"][m]
        g += greedy(dd["x"], dd["y"], a["inv"]["x"], a["inv"]["y"])
        h += hungaro(dd["x"], dd["y"], a["inv"]["x"], a["inv"]["y"])
    res_C[m] = dict(greedy=g, hungaro=h, dif=h - g)
    print(f"  {m:6} greedy {g:5d}   hungaro {h:5d}   diferencia {h-g:+d} arboles "
          f"({100*(h-g)/1437:+.2f} puntos de exhaustividad)")


# ----------------------------------------------------------------- D. limite real en Pellizzano
print("\nD. Pellizzano: region de evaluacion")
cuadros = {}
imgs = {}
for h in glob.glob(os.path.join(RAIZ, "trento_dataset", "img", "*.hdr")):
    t = open(h).read()
    s_ = int(re.search(r"samples\s*=\s*(\d+)", t).group(1))
    l_ = int(re.search(r"lines\s*=\s*(\d+)", t).group(1))
    mi = re.search(r"map info\s*=\s*\{([^}]*)\}", t).group(1).split(",")
    x0, y0, px = float(mi[3]), float(mi[4]), float(mi[5])
    imgs[os.path.basename(h)[:-4]] = (x0, y0 - l_ * px, x0 + s_ * px, y0)
for p in SITIOS["pell"]:
    a = ARB[f"pell/{p}"]
    for nom, (x0, y0, x1, y1) in imgs.items():
        if all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in zip(a["inv"]["x"], a["inv"]["y"])):
            cuadros[p] = (x0, y0, x1, y1)
            break
assert len(cuadros) == 15, len(cuadros)

res_D = {}
for m in ("SAT", "AMS3D", "WS"):
    res_D[m] = {}
    for modo in ("cuadro real de 40 m", "casco dilatado 1.5 m", "tile entero"):
        filas = []
        for p in SITIOS["pell"]:
            a = ARB[f"pell/{p}"]
            X = np.array(a["det"][m]["x"] + a["det_fuera"][m]["x"])
            Y = np.array(a["det"][m]["y"] + a["det_fuera"][m]["y"])
            if modo == "cuadro real de 40 m":
                x0, y0, x1, y1 = cuadros[p]
                k = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
                X, Y = X[k], Y[k]
            elif modo == "casco dilatado 1.5 m":
                X, Y = np.array(a["det"][m]["x"]), np.array(a["det"][m]["y"])
            tp = greedy(X, Y, a["inv"]["x"], a["inv"]["y"])
            filas.append((tp, len(X), len(a["inv"]["x"])))
        tp, nd, nr = [sum(f[i] for f in filas) for i in range(3)]
        icp = boot(filas, lambda t: t[0] / t[1] if t[1] else 0.0)
        res_D[m][modo] = dict(TP=tp, ND=nd, recall=round(tp / nr, 3),
                              precision=round(tp / nd, 3), precision_ic=icp,
                              f1=round(f1(tp, nd, nr), 3))
        r = res_D[m][modo]
        print(f"  {m:6} {modo:22} exh {r['recall']:.3f}  prec {r['precision']:.3f} {icp}  "
              f"F1 {r['f1']:.3f}  ND {nd}")

json.dump({"A_ams3d": res_A, "B_no_inferioridad": res_B, "C_emparejamiento": res_C,
           "D_region_pellizzano": res_D},
          open(os.path.join(RAIZ, "revision.json"), "w"), indent=2, ensure_ascii=False,
          default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
print("\nguardado en revision.json")
