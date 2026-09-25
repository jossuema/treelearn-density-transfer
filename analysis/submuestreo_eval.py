"""Evalua el experimento de submuestreo controlado.

Las parcelas de Pellizzano raleadas por pulso a 60, 40 y 25 % se puntuan contra el MISMO
inventario y con el MISMO protocolo que el resto del articulo. Sitio, sensor, rodal y
referencia quedan fijos; lo unico que cambia es la densidad. Es la manipulacion que el
analisis observacional no podia hacer.

Uso: python3 respaldo_minimo/analisis_paper/submuestreo_eval.py [metodo]
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
from analisis import referencia, region  # noqa: E402
from robustez_cinco import pares_greedy, metricas, spearman  # noqa: E402

FRAC = (100, 60, 40, 25)
NB = 2000


# el 100 % de cada metodo sale de SU corrida original, no de la de otro
CLAVE = {"SAT": "SAT", "t4_p30": "TreeLite3D", "FF3D": "FF3D"}


def carga(met):
    """Detecciones por parcela y fraccion, con el 100 % de la corrida sin ralear."""
    d = {}
    todas = json.load(open(os.path.join(RAIZ, "detecciones_todas.json")))["detecciones"]
    clave = CLAVE.get(met, met)
    for k, v in todas.items():
        if k.startswith("pell/"):
            d[(k.split("/")[1], 100)] = dict(v[clave], puntos=0)
    sub = json.load(open(os.path.join(RAIZ, "salida_treelite",
                                      f"detecciones_sub_{met}.json")))
    for k, v in sub.items():
        p, f = k[5:].rsplit("_f", 1)
        d[(p, int(f))] = v
    return d


def main():
    met = sys.argv[1] if len(sys.argv) > 1 else "t4_p30"
    det = carga(met)
    DENS = json.load(open(os.path.join(RAIZ, "submuestreo_densidades.json")))["por_parcela"]
    ps = [f"p{i}" for i in range(1, 16)]
    faltan = [p for p in ps if any((p, fr) not in det for fr in FRAC)]
    if faltan:
        print(f"fuera por no tener las cuatro fracciones: {faltan}")
        ps = [p for p in ps if p not in faltan]
    filas, dens_rec = {}, []
    print(f"{'parcela':8} " + " ".join(f"{f:>5}%" for f in FRAC)
          + "    exhaustividad por fraccion de pulsos")
    for p in ps:
        ref = referencia("pell", p)
        reg = region(ref)
        area = abs(np.sum(reg.vertices[:, 0] * np.roll(reg.vertices[:, 1], -1)
                          - np.roll(reg.vertices[:, 0], -1) * reg.vertices[:, 1]) / 2)
        fila = f"{p:8} "
        for f in FRAC:
            d = det.get((p, f))
            if d is None:
                fila += "    -- "; continue
            x, y = np.array(d.get("x", [])), np.array(d.get("y", []))
            dentro = reg.contains_points(np.c_[x, y]) if len(x) else np.zeros(0, bool)
            pr = pares_greedy(x[dentro], y[dentro], ref["x"], ref["y"], 3.5)
            TP, ND, NR = len(pr), int(dentro.sum()), len(ref["x"])
            r = TP / NR if NR else 0
            filas.setdefault(f, []).append(dict(TP=TP, ND=ND, NR=NR))
            dens_rec.append((DENS[p][str(f)], r))
            fila += f" {r:>5.3f}"
        print(fila)

    print(f"\n{'fraccion':10} {'pts/m2 medio':>13} {'exhaustividad':>14} "
          f"{'IC 95 %':>16} {'precision':>10}")
    sal = {}
    for f in FRAC:
        if f not in filas:
            continue
        v = filas[f]
        TP, ND, NR = (sum(x[c] for x in v) for c in ("TP", "ND", "NR"))
        r, pc, f1 = metricas(TP, NR, ND)
        rng = np.random.default_rng(0); b = []
        for _ in range(NB):
            i = rng.integers(0, len(v), len(v))
            a = sum(v[j]["TP"] for j in i); c = sum(v[j]["NR"] for j in i)
            b.append(a / c if c else 0)
        ic = [round(float(np.percentile(b, q)), 3) for q in (2.5, 97.5)]
        sal[f] = dict(recall=round(r, 3), precision=round(pc, 3), f1=round(f1, 3),
                      ic=ic, TP=TP, ND=ND, NR=NR)
        print(f"{str(f)+' %':10} {'':>13} {r:>14.3f} {str(ic):>16} {pc:>10.3f}")

    d = np.array([x for x, _ in dens_rec]); rr = np.array([y for _, y in dens_rec])
    rho, pv = spearman(d, rr, nperm=20000)
    print(f"\nexhaustividad frente a densidad, dentro del MISMO sitio y sensor:")
    print(f"   {len(d)} observaciones, rho {rho:+.3f}, p {pv:.4f}")
    sal["correlacion"] = dict(n=len(d), rho=rho, p=pv)
    json.dump(sal, open(os.path.join(RAIZ, f"submuestreo_{met}.json"), "w"), indent=2)
    print(f"\nguardado en submuestreo_{met}.json")


if __name__ == "__main__":
    main()
