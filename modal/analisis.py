"""Analisis completo: SegmentAnyTree zero-shot, AMS3D y watershed bajo un unico evaluador.

Todo local, sin GPU. Produce:
  1. Deteccion por sitio con intervalos de confianza por bootstrap de parcelas.
  2. Desglose por estrato de detectabilidad (dominante, intermedio, suprimido).
  3. Dos protocolos de emparejamiento: solo distancia, y con el termino de solape de Tusa.
  4. Sensibilidad apice frente a centroide para SegmentAnyTree.
  5. Watershed sobre CHM como tercer metodo, calculado aqui mismo.

Uso: python3 respaldo_minimo/modal_app/analisis.py
"""
import csv
import json
import os
import sys

import laspy
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, QhullError

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # respaldo_minimo
RAIZ = os.path.dirname(BASE)
SAL = os.path.join(RAIZ, "salida_sat_all")
AMS = os.path.join(BASE, "ams_port", "out")

DMATCH = (2.5, 3.5, 5.0)
BUF = 1.5
MIN_PTS = 10
NBOOT = 2000
RNG = np.random.default_rng(0)
# Estratos por numero de retornos ALS dentro de la copa (misma regla que veniamos usando)
DOM, SUP = 300, 50

CHAM = ["1", "1b", "2", "3", "3b", "4", "Premol"]
PELL = [f"p{i}" for i in range(1, 16)]
SITIOS = (("cham", "Chamrousse", CHAM), ("pell", "Pellizzano", PELL))


# ----------------------------------------------------------------- referencia
def referencia(sitio, parcela):
    """Arboles del inventario: posicion, altura, area de copa, familia."""
    if sitio == "cham":
        ruta = os.path.join(BASE, "IRSTEA_dataset/inventory_csv/Chamrousse.csv")
        las = os.path.join(BASE, f"IRSTEA_dataset/las/las{parcela}.las")
    else:
        ruta = os.path.join(BASE, "trento_dataset/csv/Pellizzano.csv")
        las = os.path.join(BASE, f"trento_dataset/las/{parcela}.las")
    x, y, h, a, fam = [], [], [], [], []
    for r in csv.DictReader(open(ruta)):
        if r["plotid"] != parcela:
            continue
        try:
            x.append(float(r["x"])); y.append(float(r["y"]))
            h.append(float(r["height"])); a.append(float(r["area"]))
            fam.append(r.get("family", "?"))
        except ValueError:
            pass
    ref = dict(x=np.array(x), y=np.array(y), h=np.array(h), area=np.array(a),
               fam=np.array(fam), las=las)
    ref["npts"] = puntos_por_arbol(ref)
    n = ref["npts"]
    ref["estrato"] = np.where(n >= DOM, "dominante",
                              np.where(n < SUP, "suprimido", "intermedio"))
    return ref


def puntos_por_arbol(ref):
    """Retornos ALS dentro del cilindro de copa de cada arbol: proxy de visibilidad."""
    L = laspy.read(ref["las"])
    px, py, pz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    m = pz >= 1.5
    px, py, pz = px[m], py[m], pz[m]
    rad = np.sqrt(np.maximum(ref["area"], 0.1) / np.pi)
    base = 0.4 * ref["h"]                       # base de copa aproximada
    out = np.zeros(len(ref["x"]), int)
    for i in range(len(ref["x"])):
        d2 = (px - ref["x"][i]) ** 2 + (py - ref["y"][i]) ** 2
        out[i] = int(np.count_nonzero((d2 <= rad[i] ** 2) &
                                      (pz >= base[i]) & (pz <= ref["h"][i] + 2.0)))
    return out


def region(ref, buf=BUF):
    poly = np.c_[ref["x"], ref["y"]][ConvexHull(np.c_[ref["x"], ref["y"]]).vertices]
    c = poly.mean(0)
    d = np.linalg.norm(poly - c, axis=1)
    d[d == 0] = 1e-9
    return MplPath(c + (poly - c) * ((d + buf) / d)[:, None])


# ----------------------------------------------------------------- detecciones
def area_hull(xy):
    if len(xy) < 3:
        return 0.5
    try:
        return float(ConvexHull(xy).volume)
    except (QhullError, Exception):
        return max(float(np.ptp(xy[:, 0]) * np.ptp(xy[:, 1])), 0.5)


def det_sat(sitio, parcela, pos="apice"):
    laz = os.path.join(SAL, f"{sitio}_{parcela}_out.laz")
    if not os.path.exists(laz):
        return None
    L = laspy.read(laz)
    x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    inst = np.asarray(L.PredInstance).astype(np.int64)
    X, Y, H, A = [], [], [], []
    for i in np.unique(inst[inst > 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        xy = np.c_[x[m], y[m]]
        if pos == "apice":
            k = np.argmax(z[m]); X.append(x[m][k]); Y.append(y[m][k])
        else:
            X.append(xy[:, 0].mean()); Y.append(xy[:, 1].mean())
        H.append(float(z[m].max())); A.append(area_hull(xy))
    return dict(x=np.array(X), y=np.array(Y), h=np.array(H), area=np.array(A))


def det_ams(sitio, parcela, cfgs):
    c = cfgs.get((sitio, parcela))
    if c is None:
        return None
    f = os.path.join(AMS, f"e1_dets_{parcela}_{c}.csv")
    if not os.path.exists(f):
        return None
    x, y, h, a = [], [], [], []
    for r in csv.DictReader(open(f)):
        x.append(float(r["x"])); y.append(float(r["y"]))
        h.append(float(r["h"])); a.append(float(r["area"]))
    return dict(x=np.array(x), y=np.array(y), h=np.array(h), area=np.array(a))


def det_watershed(ref, cell=0.5, hmin=2.0, mindist=2.0, sigma=1.0):
    """Watershed con marcadores sobre el CHM. Tercer metodo, sin entrenamiento."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    L = laspy.read(ref["las"])
    px, py, pz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    m = pz >= 0
    px, py, pz = px[m], py[m], pz[m]
    x0, y0 = px.min(), py.min()
    nx = int((px.max() - x0) / cell) + 1
    ny = int((py.max() - y0) / cell) + 1
    chm = np.zeros((ny, nx), np.float32)
    j = np.clip(((px - x0) / cell).astype(int), 0, nx - 1)
    i = np.clip(((py - y0) / cell).astype(int), 0, ny - 1)
    np.maximum.at(chm, (i, j), pz)
    chm = gaussian_filter(chm, sigma)
    mask = chm >= hmin
    if not mask.any():
        return dict(x=np.array([]), y=np.array([]), h=np.array([]), area=np.array([]))
    picos = peak_local_max(chm, min_distance=max(1, int(mindist / cell)),
                           threshold_abs=hmin, labels=mask)
    if not len(picos):
        return dict(x=np.array([]), y=np.array([]), h=np.array([]), area=np.array([]))
    marc = np.zeros_like(chm, int)
    marc[tuple(picos.T)] = np.arange(1, len(picos) + 1)
    lab = watershed(-chm, marc, mask=mask)
    X, Y, H, A = [], [], [], []
    for k in range(1, len(picos) + 1):
        s = lab == k
        n = int(s.sum())
        if n < 2:
            continue
        ii, jj = np.nonzero(s)
        top = np.argmax(chm[ii, jj])
        X.append(x0 + (jj[top] + 0.5) * cell)
        Y.append(y0 + (ii[top] + 0.5) * cell)
        H.append(float(chm[ii, jj].max()))
        A.append(n * cell * cell)
    return dict(x=np.array(X), y=np.array(Y), h=np.array(H), area=np.array(A))


# ----------------------------------------------------------------- emparejar
def emparejar(det, ref, dmax, con_solape):
    """Devuelve los indices de arboles de referencia emparejados.

    con_solape=False: greedy por distancia 2D.
    con_solape=True : greedy por (d/dmax)*(1-JI), JI = Jaccard de volumen area*altura
                      (el termino del protocolo de Tusa). Misma condicion de admision d<=dmax.
    """
    nd, nr = len(det["x"]), len(ref["x"])
    if nd == 0 or nr == 0:
        return np.zeros(0, int)
    D = np.hypot(det["x"][:, None] - ref["x"][None, :],
                 det["y"][:, None] - ref["y"][None, :])
    if con_solape:
        Vd = np.maximum(det["area"], 1e-6) * np.maximum(det["h"], 1e-6)
        Vr = np.maximum(ref["area"], 1e-6) * np.maximum(ref["h"], 1e-6)
        JI = np.minimum(Vd[:, None], Vr[None, :]) / np.maximum(Vd[:, None], Vr[None, :])
        C = (D / dmax) * (1.0 - JI)
    else:
        C = D.copy()
    C = np.where(D <= dmax, C, np.inf)
    di, rj = np.arange(nd), np.arange(nr)
    casados = []
    while C.size and np.isfinite(C).any():
        i, j = np.unravel_index(np.argmin(C), C.shape)
        casados.append(int(rj[j]))
        C = np.delete(np.delete(C, i, 0), j, 1)
        di, rj = np.delete(di, i), np.delete(rj, j)
    return np.array(casados, int)


def evaluar_parcela(ref, det, dmax, con_solape):
    dentro = region(ref).contains_points(np.c_[det["x"], det["y"]]) if len(det["x"]) else np.zeros(0, bool)
    dd = {k: v[dentro] for k, v in det.items() if isinstance(v, np.ndarray)}
    casados = emparejar(dd, ref, dmax, con_solape)
    return dict(NR=len(ref["x"]), ND=int(dentro.sum()), TP=len(casados), casados=casados)


# ----------------------------------------------------------------- agregacion
def metricas(TP, NR, ND):
    R = TP / NR if NR else 0.0
    P = TP / ND if ND else 0.0
    return R, P, (2 * P * R / (P + R) if P + R else 0.0)


def bootstrap(por_parcela, n=NBOOT):
    k = len(por_parcela)
    if k < 2:
        return {}
    Rs, Ps, Fs = [], [], []
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        TP = sum(por_parcela[i]["TP"] for i in idx)
        NR = sum(por_parcela[i]["NR"] for i in idx)
        ND = sum(por_parcela[i]["ND"] for i in idx)
        R, P, F = metricas(TP, NR, ND)
        Rs.append(R); Ps.append(P); Fs.append(F)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {"recall_ic": q(Rs), "precision_ic": q(Ps), "f1_ic": q(Fs)}


def main():
    cfgs = {}
    for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        p = os.path.join(AMS, f)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cfgs[(s, r["plot"])] = r["cfg"]

    print("Cargando referencias y detecciones (puede tardar un par de minutos)...", flush=True)
    datos = {}
    for sitio, _, parcelas in SITIOS:
        for p in parcelas:
            ref = referencia(sitio, p)
            datos[(sitio, p)] = {
                "ref": ref,
                "SAT": det_sat(sitio, p, "apice"),
                "SAT_centroide": det_sat(sitio, p, "centroide"),
                "AMS3D": det_ams(sitio, p, cfgs),
                "Watershed": det_watershed(ref),
            }
        print(f"  {sitio} listo", flush=True)

    metodos = ["SAT", "AMS3D", "Watershed"]
    salida = {"por_sitio": {}, "estratos": {}, "sensibilidad": {}}

    # ---------- 1 y 3: deteccion por sitio, dos protocolos, con IC
    for con_solape, etiq in ((False, "solo_distancia"), (True, "con_solape_Tusa")):
        print(f"\n{'='*86}\nPROTOCOLO: {etiq.replace('_',' ')}")
        print(f"{'sitio':11} {'metodo':10} {'dm':>4} {'NR':>5} {'ND':>5} {'TP':>5} "
              f"{'exhaust':>8} {'IC 95%':>16} {'precis':>7} {'F1':>6}")
        for sitio, nombre, parcelas in SITIOS:
            for met in metodos:
                for dm in DMATCH:
                    pp = [evaluar_parcela(datos[(sitio, p)]["ref"], datos[(sitio, p)][met], dm, con_solape)
                          for p in parcelas if datos[(sitio, p)][met] is not None]
                    TP, NR, ND = (sum(e[k] for e in pp) for k in ("TP", "NR", "ND"))
                    R, P, F = metricas(TP, NR, ND)
                    ic = bootstrap(pp)
                    salida["por_sitio"].setdefault(etiq, {}).setdefault(nombre, {}) \
                        .setdefault(met, {})[str(dm)] = dict(
                            NR=NR, ND=ND, TP=TP, recall=round(R, 3), precision=round(P, 3),
                            f1=round(F, 3), **ic)
                    print(f"{nombre:11} {met:10} {dm:>4} {NR:>5} {ND:>5} {TP:>5} "
                          f"{R:>8.3f} {str(ic.get('recall_ic','')):>16} {P:>7.3f} {F:>6.3f}")
            print()

    # ---------- 2: estratos, a 3.5 m, protocolo con solape
    print("=" * 86)
    print("ESTRATOS DE DETECTABILIDAD (retornos ALS en la copa: dominante >=300, suprimido <50)")
    print(f"{'sitio':11} {'estrato':12} {'arboles':>8} " + " ".join(f"{m:>10}" for m in metodos))
    for sitio, nombre, parcelas in SITIOS:
        acum = {e: {"n": 0, **{m: 0 for m in metodos}} for e in ("dominante", "intermedio", "suprimido")}
        for p in parcelas:
            ref = datos[(sitio, p)]["ref"]
            for e in acum:
                acum[e]["n"] += int((ref["estrato"] == e).sum())
            for met in metodos:
                d = datos[(sitio, p)][met]
                if d is None:
                    continue
                cas = evaluar_parcela(ref, d, 3.5, True)["casados"]
                for e in acum:
                    acum[e][met] += int((ref["estrato"][cas] == e).sum()) if len(cas) else 0
        salida["estratos"][nombre] = acum
        for e in ("dominante", "intermedio", "suprimido"):
            n = acum[e]["n"]
            print(f"{nombre:11} {e:12} {n:>8} " +
                  " ".join(f"{(acum[e][m]/n if n else 0):>10.3f}" for m in metodos))
        print()

    # ---------- 4: apice frente a centroide
    print("=" * 86)
    print("SENSIBILIDAD DE SegmentAnyTree: apice frente a centroide de la instancia (3.5 m, con solape)")
    print(f"{'sitio':11} {'posicion':12} {'TP':>5} {'exhaust':>8} {'precis':>8} {'F1':>7}")
    for sitio, nombre, parcelas in SITIOS:
        for clave, etq in (("SAT", "apice"), ("SAT_centroide", "centroide")):
            pp = [evaluar_parcela(datos[(sitio, p)]["ref"], datos[(sitio, p)][clave], 3.5, True)
                  for p in parcelas]
            TP, NR, ND = (sum(e[k] for e in pp) for k in ("TP", "NR", "ND"))
            R, P, F = metricas(TP, NR, ND)
            salida["sensibilidad"].setdefault(nombre, {})[etq] = dict(
                TP=TP, ND=ND, recall=round(R, 3), precision=round(P, 3), f1=round(F, 3))
            print(f"{nombre:11} {etq:12} {TP:>5} {R:>8.3f} {P:>8.3f} {F:>7.3f}")
        print()

    json.dump(salida, open(os.path.join(RAIZ, "analisis_completo.json"), "w"),
              indent=2, ensure_ascii=False, default=str)
    print("guardado en analisis_completo.json")


if __name__ == "__main__":
    main()
