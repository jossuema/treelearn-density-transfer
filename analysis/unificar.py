"""Reune las detecciones de los cinco metodos en un solo fichero, sin recortar.

Hasta ahora cada metodo guardaba lo suyo con su propio formato y algunos ya venian
recortados a la region, lo que impide rehacer la sensibilidad de la region. Aqui
quedan todos igual: todas las detecciones de cada parcela, con apice, altura y area,
antes de cualquier filtro. Todo lo demas del articulo se calcula desde este fichero.

Fuentes:
    SAT, AMS3D, WS   visor/data/arboles.json, sumando det y det_fuera
    FF3D             los zip de salida_ff3d, releidos con evaluar_ff3d.detecciones
    TreeLite3D       salida_treelite, con el ajuste que elige el protocolo de una fuera

Uso: python3 respaldo_minimo/analisis_paper/unificar.py
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
from analisis import SITIOS, referencia, region  # noqa: E402

CAMPOS = ("x", "y", "h", "area")


def main():
    arb = json.load(open(os.path.join(BASE, "visor", "data", "arboles.json")))
    claves = [f"{s}/{p}" for s, _, ps in SITIOS for p in ps]

    todo = {k: {} for k in claves}
    for k in claves:
        for m in ("SAT", "AMS3D", "WS"):
            d, f = arb[k]["det"][m], arb[k]["det_fuera"][m]
            todo[k][m] = {c: list(d[c]) + list(f[c]) for c in CAMPOS}

    from evaluar_ff3d import detecciones as det_ff3d
    for k in claves:
        sitio, p = k.split("/")
        det, _ = det_ff3d(f"{sitio}_{p}")
        todo[k]["FF3D"] = {c: [float(v) for v in det[c]] for c in CAMPOS}

    from treelite_final import cargar, f1_de
    ajustes = cargar()
    mejor = max(ajustes, key=lambda e: f1_de(ajustes[e], claves))
    tl = json.load(open(os.path.join(RAIZ, "salida_treelite",
                                     f"detecciones_{mejor}.json")))
    for k in claves:
        sitio, p = k.split("/")
        d = tl[f"{sitio}_{p}"]
        if str(d.get("error", "")).startswith("0 etiquetas"):
            d = {c: [] for c in CAMPOS}
        todo[k]["TreeLite3D"] = {c: list(d.get(c, [])) for c in CAMPOS}

    # densidad de retornos dentro de la region inventariada, que es donde se puntua.
    # OJO: referencia() apunta al LAS recortado por encima de 1 m en Chamrousse, que NO
    # es el que recibieron los modelos. Medir ahi la densidad la subestimaba y exageraba
    # el contraste entre sitios. Se mide sobre la misma nube que proceso cada modelo.
    import laspy

    def nube_del_modelo(sitio, p):
        if sitio == "cham":
            return os.path.join(RAIZ, "respaldo_minimo", "IRSTEA_dataset", "las",
                                "full_las", f"las{p}.las")
        return os.path.join(RAIZ, "respaldo_minimo", "trento_dataset", "las", f"{p}.las")

    dens = {}
    print(f"{'parcela':12} {'arboles':>8} {'region m2':>10} {'pts/m2':>8} " +
          " ".join(f"{m:>11}" for m in ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")))
    for k in claves:
        sitio, p = k.split("/")
        ref = referencia(sitio, p)
        L = laspy.read(nube_del_modelo(sitio, p))
        xy = np.c_[np.asarray(L.x), np.asarray(L.y)]
        reg = region(ref)
        dentro = reg.contains_points(xy)
        area = abs(np.sum(reg.vertices[:, 0] * np.roll(reg.vertices[:, 1], -1)
                          - np.roll(reg.vertices[:, 0], -1) * reg.vertices[:, 1]) / 2)
        dens[k] = dict(puntos=int(dentro.sum()), area=round(float(area), 1),
                       densidad=round(float(dentro.sum() / area), 1),
                       arboles=len(ref["x"]))
        print(f"{k:12} {len(ref['x']):>8} {area:>10.0f} {dens[k]['densidad']:>8.1f} " +
              " ".join(f"{len(todo[k][m]['x']):>11}"
                       for m in ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")))

    sal = {"ajuste_treelite": mejor, "densidad": dens, "detecciones": todo}
    with open(os.path.join(RAIZ, "detecciones_todas.json"), "w") as f:
        json.dump(sal, f)
    print(f"\nguardado en detecciones_todas.json "
          f"({os.path.getsize(os.path.join(RAIZ, 'detecciones_todas.json')) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
