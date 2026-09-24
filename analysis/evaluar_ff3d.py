"""Paso 4: convierte la salida de ForestFormer3D en detecciones y la evalua.

La salida de cada parcela es un PLY con x, y, z, semantic_pred, instance_pred y score,
pero OJO: el pipeline recentra las coordenadas a un origen local y normaliza z para que
el minimo sea cero. Hay que devolverlas a su sistema original, que se recupera del LAS
de entrada comparando los minimos, porque la transformacion es una traslacion pura.
Se convierte a detecciones con la MISMA regla que usamos con SegmentAnyTree en
analisis.py (det_sat): una deteccion por instancia con al menos MIN_PTS puntos, situada
en el apice, y con area del casco convexo en planta. Asi la comparacion es directa.

Uso: python3 respaldo_minimo/analisis_paper/evaluar_ff3d.py
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
from analisis import referencia, region, area_hull, MIN_PTS, SITIOS  # noqa: E402

ZIPS = os.path.join(RAIZ, "salida_ff3d")
D = 3.5
NB = 2000


def leer_ply(datos):
    """Lector minimo del PLY binario que escribe el pipeline."""
    fin = datos.index(b"end_header") + len(b"end_header") + 1
    cab = datos[:fin].decode("ascii", "replace").splitlines()
    n = next(int(l.split()[2]) for l in cab if l.startswith("element vertex"))
    tipos = {"double": "f8", "float": "f4", "int": "i4", "uchar": "u1"}
    campos = [(l.split()[2], tipos[l.split()[1]]) for l in cab if l.startswith("property")]
    arr = np.frombuffer(datos[fin:], dtype=np.dtype(campos), count=n)
    return arr


def las_de_entrada(sitio, p):
    """El mismo archivo que recibio el modelo, y que recibio SegmentAnyTree."""
    if sitio == "cham":
        return os.path.join(RAIZ, "IRSTEA_dataset", "las", "full_las", f"las{p}.las")
    return os.path.join(RAIZ, "trento_dataset", "las", f"{p}.las")


def detecciones(clave):
    """Detecciones de ForestFormer3D con el mismo criterio que det_sat."""
    z = sorted(glob.glob(os.path.join(ZIPS, f"{clave}__results_*.zip")))[-1]
    with zipfile.ZipFile(z) as f:
        nombre = [i for i in f.namelist() if i.endswith(".ply")][0]
        arr = leer_ply(f.read(nombre))
    x, y, zz = arr["x"].astype(float), arr["y"].astype(float), arr["z"].astype(float)

    # Devolver las coordenadas a su sistema original
    import laspy
    sitio, p = clave.split("_", 1)
    L = laspy.read(las_de_entrada(sitio, p))
    lx, ly, lz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    for nom, a, b in (("x", x, lx), ("y", y, ly), ("z", zz, lz)):
        ext_ply = a.max() - a.min(); ext_las = b.max() - b.min()
        if abs(ext_ply - ext_las) > 0.05:
            print(f"  AVISO {clave} {nom}: extension {ext_ply:.2f} frente a {ext_las:.2f}")
    x = x + (lx.min() - x.min())
    y = y + (ly.min() - y.min())
    zz = zz + (lz.min() - zz.min())

    inst = arr["instance_pred"].astype(np.int64)
    X, Y, H, A = [], [], [], []
    for i in np.unique(inst[inst >= 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        k = np.argmax(zz[m])
        X.append(x[m][k]); Y.append(y[m][k])
        H.append(float(zz[m].max())); A.append(area_hull(np.c_[x[m], y[m]]))
    return dict(x=np.array(X), y=np.array(Y), h=np.array(H), area=np.array(A)), len(arr)


def greedy(dx, dy, rx, ry, dmax):
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


def main():
    arb = json.load(open(os.path.join(BASE, "visor", "data", "arboles.json")))
    filas, por_parcela = {}, {}
    print(f"{'parcela':14} {'puntos':>8} {'instancias':>11} {'dentro':>7} {'TP':>5} {'NR':>5}")
    for sitio, nombre, parcelas in SITIOS:
        for p in parcelas:
            clave = f"{sitio}_{p}"
            k = f"{sitio}/{p}"
            det, npts = detecciones(clave)
            ref = referencia(sitio, p)
            dentro = region(ref).contains_points(np.c_[det["x"], det["y"]]) if len(det["x"]) \
                else np.zeros(0, bool)
            dd = {c: det[c][dentro] for c in ("x", "y", "h", "area")}
            tp = greedy(dd["x"], dd["y"], ref["x"], ref["y"], D)
            filas.setdefault(nombre, []).append((tp, len(dd["x"]), len(ref["x"])))
            por_parcela[k] = dict(instancias=len(det["x"]), dentro=int(dentro.sum()),
                                  TP=tp, NR=len(ref["x"]),
                                  x=[round(float(v), 2) for v in dd["x"]],
                                  y=[round(float(v), 2) for v in dd["y"]],
                                  h=[round(float(v), 2) for v in dd["h"]],
                                  area=[round(float(v), 2) for v in dd["area"]])
            print(f"{k:14} {npts:>8} {len(det['x']):>11} {int(dentro.sum()):>7} {tp:>5} {len(ref['x']):>5}")

    print(f"\n{'sitio':12} {'exhaustividad':>14} {'IC 95 %':>16} {'precision':>10} {'F1':>7}")
    resumen = {}
    for nombre, fs in list(filas.items()) + [("Ambos", [f for v in filas.values() for f in v])]:
        tp = sum(f[0] for f in fs); nd = sum(f[1] for f in fs); nr = sum(f[2] for f in fs)
        r, pr = tp / nr, (tp / nd if nd else 0)
        f1 = 2 * pr * r / (pr + r) if pr + r else 0
        rng = np.random.default_rng(0); v = []
        for _ in range(NB):
            idx = rng.integers(0, len(fs), len(fs))
            a = sum(fs[i][0] for i in idx); b = sum(fs[i][2] for i in idx)
            v.append(a / b if b else 0)
        ic = [round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)]
        resumen[nombre] = dict(TP=tp, ND=nd, NR=nr, recall=round(r, 3),
                               precision=round(pr, 3), f1=round(f1, 3), recall_ic=ic)
        print(f"{nombre:12} {r:>14.3f} {str(ic):>16} {pr:>10.3f} {f1:>7.3f}")

    json.dump({"resumen": resumen, "por_parcela": por_parcela},
              open(os.path.join(RAIZ, "ff3d_evaluacion.json"), "w"), indent=2, ensure_ascii=False)
    print("\nguardado en ff3d_evaluacion.json")


if __name__ == "__main__":
    main()
