"""Segunda ronda de comprobaciones de la revision externa.

  1. No inferioridad frente al oraculo mas exigente: AMS3D con la mejor de sus 25
     configuraciones elegida con NUESTRO F1, no con el del protocolo de Tusa.
  2. La tabla principal rehecha con emparejamiento optimo (hungaro de maxima
     cardinalidad), para ver si el orden entre metodos cambia.

Reutiliza las funciones de revision.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revision as R  # noqa: E402  (ejecuta revision.py; sus impresiones salen primero)

print("\n" + "=" * 90)
print("1. SegmentAnyTree frente al oraculo mas exigente, margen", R.DELTA)
nombre = "oraculo, F1 nuestro"
for s in ("cham", "pell", "global"):
    filas = []
    for k, (tp, nd, nr) in R.filas_sat(s):
        sit, p = k.split("/")
        atp, and_, _ = R.sel[nombre][(sit, p)]
        filas.append((tp, nd, nr, atp, and_))
    tot = np.sum(np.array(filas, float), 0)
    for met, fn in (("recall", lambda t: t[0] / t[2] - t[3] / t[2]),
                    ("f1", lambda t: R.f1(t[0], t[1], t[2]) - R.f1(t[3], t[4], t[2]))):
        q = R.boot(filas, fn, q=(5, 50, 95))
        print(f"  {s:6} {met:6} dif {fn(tot):+.3f}  p5 {q[0]:+.3f}  p95 {q[2]:+.3f}  "
              f"no inferior {'SI' if q[0] > -R.DELTA else 'no'}  "
              f"equivalente {'SI' if q[0] > -R.DELTA and q[2] < R.DELTA else 'no'}")

print("\n2. Tabla principal con emparejamiento optimo")
print(f"  {'sitio':12} {'metodo':6} {'greedy exh/prec/F1':>22} {'hungaro exh/prec/F1':>22}")
for s, ps in R.SITIOS.items():
    for m in ("SAT", "AMS3D", "WS"):
        g = [0, 0, 0]; h = [0, 0, 0]
        for p in ps:
            a = R.ARB[f"{s}/{p}"]; dd = a["det"][m]
            nr = len(a["inv"]["x"]); nd = len(dd["x"])
            g[0] += R.greedy(dd["x"], dd["y"], a["inv"]["x"], a["inv"]["y"]); g[1] += nd; g[2] += nr
            h[0] += R.hungaro(dd["x"], dd["y"], a["inv"]["x"], a["inv"]["y"]); h[1] += nd; h[2] += nr
        fg = f"{g[0]/g[2]:.3f}/{g[0]/g[1]:.3f}/{R.f1(*g):.3f}"
        fh = f"{h[0]/h[2]:.3f}/{h[0]/h[1]:.3f}/{R.f1(*h):.3f}"
        print(f"  {s:12} {m:6} {fg:>22} {fh:>22}")
