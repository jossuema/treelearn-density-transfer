"""Evalua TreeLite3D con el mismo protocolo que los demas metodos.

Las detecciones vienen de run_treelite.py::detecciones, que las calcula en el
contenedor con la MISMA regla que usamos para SegmentAnyTree y ForestFormer3D: una
deteccion por instancia con al menos MIN_PTS puntos, situada en el apice, con area del
casco convexo en planta. Aqui solo se filtra por la region y se empareja.

    python3 respaldo_minimo/analisis_paper/evaluar_treelite.py            # barrido
    python3 respaldo_minimo/analisis_paper/evaluar_treelite.py t5         # un ajuste
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
from evaluar_ff3d import greedy  # noqa: E402

SALIDA = os.path.join(RAIZ, "salida_treelite")
D = 3.5
NB = 2000


def evaluar(etiqueta, verboso=False):
    """Exhaustividad, precision y F1 por sitio para un ajuste del agrupamiento."""
    det = json.load(open(os.path.join(SALIDA, f"detecciones_{etiqueta}.json")))
    filas, por_parcela = {}, {}
    for sitio, nombre, parcelas in SITIOS:
        for p in parcelas:
            clave, k = f"{sitio}_{p}", f"{sitio}/{p}"
            d = det.get(clave)
            if d is not None and str(d.get("error", "")).startswith("0 etiquetas"):
                d = dict(x=[], y=[], h=[], area=[], n=[], asignados=0,
                         puntos=int(str(d["error"]).split()[-2]))
            if d is None or "error" in d:
                raise SystemExit(f"{etiqueta} {clave}: {d}")
            x, y = np.array(d["x"]), np.array(d["y"])
            ref = referencia(sitio, p)
            dentro = region(ref).contains_points(np.c_[x, y]) if len(x) \
                else np.zeros(0, bool)
            tp = greedy(x[dentro], y[dentro], ref["x"], ref["y"], D)
            filas.setdefault(nombre, []).append((tp, int(dentro.sum()), len(ref["x"])))
            por_parcela[k] = dict(instancias=len(x), dentro=int(dentro.sum()), TP=tp,
                                  NR=len(ref["x"]), asignados=d["asignados"],
                                  puntos=d["puntos"],
                                  x=[round(float(v), 2) for v in x[dentro]],
                                  y=[round(float(v), 2) for v in y[dentro]],
                                  h=[round(float(v), 2) for v in np.array(d["h"])[dentro]],
                                  area=[round(float(v), 2) for v in np.array(d["area"])[dentro]])
            if verboso:
                print(f"{k:14} {d['puntos']:>8} {d['asignados']:>9} "
                      f"{len(x):>6} {int(dentro.sum()):>7} {tp:>5} {len(ref['x']):>5}")

    resumen = {}
    for nombre, fs in list(filas.items()) + [("Ambos", [f for v in filas.values() for f in v])]:
        tp = sum(f[0] for f in fs); nd = sum(f[1] for f in fs); nr = sum(f[2] for f in fs)
        r, pr = tp / nr, (tp / nd if nd else 0.0)
        f1 = 2 * pr * r / (pr + r) if pr + r else 0.0
        rng = np.random.default_rng(0); v = []
        for _ in range(NB):
            idx = rng.integers(0, len(fs), len(fs))
            a = sum(fs[i][0] for i in idx); b = sum(fs[i][2] for i in idx)
            v.append(a / b if b else 0)
        resumen[nombre] = dict(
            TP=tp, ND=nd, NR=nr, recall=round(r, 3), precision=round(pr, 3),
            f1=round(f1, 3),
            recall_ic=[round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)])
    return resumen, por_parcela, filas


def main():
    if len(sys.argv) > 1:
        et = sys.argv[1]
        print(f"{'parcela':14} {'puntos':>8} {'asignad':>9} {'inst':>6} "
              f"{'dentro':>7} {'TP':>5} {'NR':>5}")
        resumen, por_parcela, _ = evaluar(et, verboso=True)
        print(f"\n{'sitio':12} {'exhaustividad':>14} {'IC 95 %':>16} "
              f"{'precision':>10} {'F1':>7}")
        for n, v in resumen.items():
            print(f"{n:12} {v['recall']:>14.3f} {str(v['recall_ic']):>16} "
                  f"{v['precision']:>10.3f} {v['f1']:>7.3f}")
        json.dump({"ajuste": et, "resumen": resumen, "por_parcela": por_parcela},
                  open(os.path.join(RAIZ, "treelite_evaluacion.json"), "w"),
                  indent=2, ensure_ascii=False)
        print("\nguardado en treelite_evaluacion.json")
        return

    ets = sorted((os.path.basename(f)[len("detecciones_"):-5]
                  for f in glob.glob(os.path.join(SALIDA, "detecciones_*.json"))),
                 key=lambda s: float(s[1:].split("_")[0]) if s[1:].split("_")[0].replace(".", "").isdigit() else 1e9)
    print(f"{'ajuste':10} {'radio':>7} | " + " | ".join(
        f"{n:^26}" for n in ("Chamrousse", "Pellizzano", "Ambos")))
    print(f"{'':10} {'':>7} | " + " | ".join(
        "  exh   prec     F1   ND" for _ in range(3)))
    for et in ets:
        resumen, _, _ = evaluar(et)
        radio = float(et[1:].split("_")[0]) * 0.05
        fila = f"{et:10} {radio:>6.2f}m |"
        for n in ("Chamrousse", "Pellizzano", "Ambos"):
            v = resumen[n]
            fila += f" {v['recall']:>5.3f} {v['precision']:>6.3f} {v['f1']:>6.3f} {v['ND']:>4} |"
        print(fila)


if __name__ == "__main__":
    main()
