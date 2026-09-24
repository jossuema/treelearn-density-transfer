"""Todas las cifras del articulo de JSTARS, desde detecciones_todas.json.

Un solo evaluador para los cinco metodos, y las tres cosas que la revision externa
pedia comprobar: emparejamiento optimo en vez de codicioso, sensibilidad a como se
define la region, y la relacion entre densidad de retornos y acierto parcela a parcela,
que es lo que sostiene el argumento.

Uso: python3 respaldo_minimo/analisis_paper/robustez_cinco.py
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
from analisis import SITIOS, referencia, region  # noqa: E402

MET = ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")
DIST = (2.5, 3.5, 5.0)
NB = 2000
ESTRATOS = ("dominante", "intermedio", "suprimido")
REF = {}


def ref_de(k):
    if k not in REF:
        REF[k] = referencia(*k.split("/"))
    return REF[k]


def metricas(TP, NR, ND):
    r = TP / NR if NR else 0.0
    p = TP / ND if ND else 0.0
    return r, p, (2 * p * r / (p + r) if p + r else 0.0)


def pares_greedy(dx, dy, rx, ry, dmax):
    """Uno a uno por distancia creciente. Es lo que veniamos usando."""
    if not len(dx) or not len(rx):
        return []
    D = np.hypot(np.asarray(dx)[:, None] - rx[None, :],
                 np.asarray(dy)[:, None] - ry[None, :])
    cand = np.argwhere(D <= dmax)
    orden = np.argsort(D[cand[:, 0], cand[:, 1]], kind="stable")
    ud, ur, out = set(), set(), []
    for i, j in cand[orden]:
        if i in ud or j in ur:
            continue
        ud.add(i); ur.add(j); out.append((int(i), int(j)))
    return out


def pares_hungaro(dx, dy, rx, ry, dmax):
    """Emparejamiento de coste minimo. Maximiza el numero de parejas validas."""
    from scipy.optimize import linear_sum_assignment
    if not len(dx) or not len(rx):
        return []
    D = np.hypot(np.asarray(dx)[:, None] - rx[None, :],
                 np.asarray(dy)[:, None] - ry[None, :])
    C = np.where(D <= dmax, D, 1e6)
    fi, co = linear_sum_assignment(C)
    return [(int(i), int(j)) for i, j in zip(fi, co) if D[i, j] <= dmax]


def contar(datos, claves, met, dmax, buf=1.5, hungaro=False, todo_el_tile=False):
    """TP, ND y NR por parcela con una definicion concreta de region y emparejamiento."""
    out = []
    empareja = pares_hungaro if hungaro else pares_greedy
    for k in claves:
        ref = ref_de(k)
        d = datos["detecciones"][k][met]
        x, y = np.array(d["x"]), np.array(d["y"])
        if todo_el_tile:
            dentro = np.ones(len(x), bool)
        else:
            dentro = region(ref, buf).contains_points(np.c_[x, y]) if len(x) \
                else np.zeros(0, bool)
        pr = empareja(x[dentro], y[dentro], ref["x"], ref["y"], dmax)
        out.append(dict(TP=len(pr), ND=int(dentro.sum()), NR=len(ref["x"]),
                        ok={j for _, j in pr}))
    return out


def agrega(ps, semilla=0):
    TP, ND, NR = (sum(p[c] for p in ps) for c in ("TP", "ND", "NR"))
    r, p, f1 = metricas(TP, NR, ND)
    rng = np.random.default_rng(semilla)
    R, P, F = [], [], []
    for _ in range(NB):
        i = rng.integers(0, len(ps), len(ps))
        a, b, c = (sum(ps[j][x] for j in i) for x in ("TP", "ND", "NR"))
        rr, pp, ff = metricas(a, c, b)
        R.append(rr); P.append(pp); F.append(ff)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3),
                   round(float(np.percentile(v, 97.5)), 3)]
    return dict(TP=TP, ND=ND, NR=NR, recall=round(r, 3), precision=round(p, 3),
                f1=round(f1, 3), recall_ic=q(R), precision_ic=q(P), f1_ic=q(F))


def spearman(a, b, nperm=10000, semilla=0):
    """Correlacion de rangos con p por permutacion, sin depender de scipy.stats."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    r = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(semilla)
    n = sum(abs(float(np.corrcoef(rng.permutation(ra), rb)[0, 1])) >= abs(r)
            for _ in range(nperm))
    return round(r, 3), round((n + 1) / (nperm + 1), 4)


def main():
    datos = json.load(open(os.path.join(RAIZ, "detecciones_todas.json")))
    claves = [f"{s}/{p}" for s, _, ps in SITIOS for p in ps]
    sitios = {n: [f"{s}/{p}" for p in ps] for s, n, ps in SITIOS}
    grupos = list(sitios.items()) + [("Ambos", claves)]
    sal = {"ajuste_treelite": datos["ajuste_treelite"],
           "densidad": datos["densidad"]}

    print("=" * 78)
    print("1. Deteccion a 3,5 m, region = casco dilatado 1,5 m, emparejamiento codicioso")
    print(f"{'metodo':12} " + " ".join(f"{n:>24}" for n, _ in grupos))
    sal["principal"] = {}
    base = {m: {n: contar(datos, ks, m, 3.5) for n, ks in grupos} for m in MET}
    for m in MET:
        fila = f"{m:12} "
        for n, _ in grupos:
            v = agrega(base[m][n])
            sal["principal"].setdefault(n, {})[m] = v
            fila += f" {v['recall']:.3f}/{v['precision']:.3f}/{v['f1']:.3f}      "
        print(fila)
    print("   (exhaustividad / precision / F1)")

    print("\n" + "=" * 78)
    print("2. Emparejamiento optimo frente a codicioso, a 3,5 m, aciertos totales")
    print(f"{'metodo':12} {'codicioso':>10} {'hungaro':>9} {'dif':>5} {'exh cod':>8} {'exh hun':>8}")
    sal["emparejamiento"] = {}
    for m in MET:
        g = sum(p["TP"] for p in base[m]["Ambos"])
        h = sum(p["TP"] for p in contar(datos, claves, m, 3.5, hungaro=True))
        nr = sum(p["NR"] for p in base[m]["Ambos"])
        sal["emparejamiento"][m] = dict(greedy=g, hungaro=h, dif=h - g,
                                        recall_greedy=round(g / nr, 3),
                                        recall_hungaro=round(h / nr, 3))
        print(f"{m:12} {g:>10} {h:>9} {h - g:>+5} {g / nr:>8.3f} {h / nr:>8.3f}")

    print("\n" + "=" * 78)
    print("3. Como se define la region, a 3,5 m, ambos sitios")
    print(f"{'metodo':12} " + " ".join(f"{t:>22}" for t in
          ("casco +0 m", "casco +1,5 m", "casco +3 m", "tile entero")))
    sal["region"] = {}
    for m in MET:
        fila = f"{m:12} "
        for et, kw in (("casco +0 m", dict(buf=0.0)), ("casco +1,5 m", dict(buf=1.5)),
                       ("casco +3 m", dict(buf=3.0)),
                       ("tile entero", dict(todo_el_tile=True))):
            ps = contar(datos, claves, m, 3.5, **kw)
            TP, ND, NR = (sum(p[c] for p in ps) for c in ("TP", "ND", "NR"))
            r, p, f1 = metricas(TP, NR, ND)
            sal["region"].setdefault(m, {})[et] = dict(
                TP=TP, ND=ND, recall=round(r, 3), precision=round(p, 3), f1=round(f1, 3))
            fila += f" {r:.3f}/{p:.3f}/{f1:.3f}   "
        print(fila)

    print("\n" + "=" * 78)
    print("4. Tolerancia de emparejamiento")
    print(f"{'metodo':12} " + " ".join(f"{d:>22}" for d in ("2.5 m", "3.5 m", "5.0 m")))
    sal["distancia"] = {}
    for m in MET:
        fila = f"{m:12} "
        for d in DIST:
            ps = contar(datos, claves, m, d)
            TP, ND, NR = (sum(p[c] for p in ps) for c in ("TP", "ND", "NR"))
            r, p, f1 = metricas(TP, NR, ND)
            sal["distancia"].setdefault(m, {})[str(d)] = dict(
                recall=round(r, 3), precision=round(p, 3), f1=round(f1, 3))
            fila += f" {r:.3f}/{p:.3f}/{f1:.3f}   "
        print(fila)

    print("\n" + "=" * 78)
    print("5. Densidad de retornos frente a acierto, parcela a parcela (22 parcelas)")
    dens = np.array([datos["densidad"][k]["densidad"] for k in claves])
    porparc = {m: np.array([p["TP"] / p["NR"] if p["NR"] else np.nan
                            for p in base[m]["Ambos"]]) for m in MET}
    sal["densidad_vs_acierto"] = {}
    print(f"{'metodo':12} {'rho exh':>9} {'p':>8} | {'rho dif con SAT':>16} {'p':>8}")
    for m in MET:
        rho, pv = spearman(dens, porparc[m])
        if m == "SAT":
            print(f"{m:12} {rho:>9.3f} {pv:>8.4f} |")
            sal["densidad_vs_acierto"][m] = dict(rho=rho, p=pv)
            continue
        d = porparc["SAT"] - porparc[m]
        rd, pd = spearman(dens, d)
        sal["densidad_vs_acierto"][m] = dict(rho=rho, p=pv, rho_dif=rd, p_dif=pd,
                                             dif_media=round(float(np.mean(d)), 3))
        print(f"{m:12} {rho:>9.3f} {pv:>8.4f} | {rd:>16.3f} {pd:>8.4f}")
    sal["por_parcela"] = {k: {m: round(float(porparc[m][i]), 3) for m in MET}
                          for i, k in enumerate(claves)}

    print("\n" + "=" * 78)
    print("6. Estratos de detectabilidad, a 3,5 m")
    print(f"{'sitio':12} {'estrato':12} {'n':>5} " + " ".join(f"{m:>11}" for m in MET))
    sal["estratos"] = {}
    for n, ks in grupos:
        idx = {k: i for i, k in enumerate(claves)}
        for est in ESTRATOS:
            tot = 0
            tps = {m: 0 for m in MET}
            for k in ks:
                ref = ref_de(k)
                m_est = ref["estrato"] == est
                tot += int(m_est.sum())
                for m in MET:
                    ok = base[m]["Ambos"][idx[k]]["ok"]
                    tps[m] += sum(1 for j in ok if ref["estrato"][j] == est)
            sal["estratos"].setdefault(n, {})[est] = dict(
                n=tot, **{m: dict(TP=tps[m], recall=round(tps[m] / tot, 3) if tot else 0.0)
                          for m in MET})
            print(f"{n:12} {est:12} {tot:>5} " +
                  " ".join(f"{tps[m] / tot if tot else 0:>11.3f}" for m in MET))

    json.dump(sal, open(os.path.join(RAIZ, "robustez_cinco.json"), "w"),
              indent=2, ensure_ascii=False, default=str)
    print("\nguardado en robustez_cinco.json")


if __name__ == "__main__":
    main()
