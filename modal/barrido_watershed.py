"""Barrido de parametros del watershed, para no compararlo en desventaja.

AMS3D corre con su mejor configuracion de 25 elegida POR PARCELA. Para que la comparacion sea
justa, el watershed tambien debe correr con su mejor configuracion. Se barre altura minima y
distancia minima entre picos, y se elige por F1 por parcela, igual que hace AMS3D.

Uso: python3 respaldo_minimo/modal_app/barrido_watershed.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analisis import (SITIOS, DMATCH, referencia, det_watershed, evaluar_parcela,
                      metricas, bootstrap, RAIZ)

HMIN = (1.0, 2.0, 3.0, 5.0)
MINDIST = (1.0, 1.5, 2.0, 3.0)


def main():
    resultado = {}
    for sitio, nombre, parcelas in SITIOS:
        print(f"\n{nombre}: barriendo {len(HMIN)*len(MINDIST)} configuraciones en {len(parcelas)} parcelas",
              flush=True)
        mejores = {}
        for p in parcelas:
            ref = referencia(sitio, p)
            mejor = None
            for hmin in HMIN:
                for md in MINDIST:
                    det = det_watershed(ref, hmin=hmin, mindist=md)
                    e = evaluar_parcela(ref, det, 3.5, True)
                    _, _, f1 = metricas(e["TP"], e["NR"], e["ND"])
                    if mejor is None or f1 > mejor[0]:
                        mejor = (f1, hmin, md, ref, det)
            f1, hmin, md, ref, det = mejor
            mejores[p] = dict(ref=ref, det=det, hmin=hmin, mindist=md, f1=round(f1, 3))
            print(f"  {p:7} mejor hmin={hmin} mindist={md}  F1={f1:.3f}", flush=True)

        print(f"\n{nombre}, watershed con mejor configuracion por parcela:")
        print(f"{'dm':>4} {'NR':>5} {'ND':>5} {'TP':>5} {'exhaust':>8} {'IC 95%':>16} {'precis':>7} {'F1':>6}")
        resultado[nombre] = {}
        for dm in DMATCH:
            pp = [evaluar_parcela(m["ref"], m["det"], dm, True) for m in mejores.values()]
            TP, NR, ND = (sum(e[k] for e in pp) for k in ("TP", "NR", "ND"))
            R, P, F = metricas(TP, NR, ND)
            ic = bootstrap(pp)
            resultado[nombre][str(dm)] = dict(NR=NR, ND=ND, TP=TP, recall=round(R, 3),
                                              precision=round(P, 3), f1=round(F, 3), **ic)
            print(f"{dm:>4} {NR:>5} {ND:>5} {TP:>5} {R:>8.3f} {str(ic.get('recall_ic','')):>16} "
                  f"{P:>7.3f} {F:>6.3f}")

        # estratos con la mejor configuracion
        acum = {e: {"n": 0, "TP": 0} for e in ("dominante", "intermedio", "suprimido")}
        for m in mejores.values():
            ref = m["ref"]
            cas = evaluar_parcela(ref, m["det"], 3.5, True)["casados"]
            for e in acum:
                acum[e]["n"] += int((ref["estrato"] == e).sum())
                acum[e]["TP"] += int((ref["estrato"][cas] == e).sum()) if len(cas) else 0
        resultado[nombre]["estratos"] = acum
        print("  estratos:", " ".join(
            f"{e} {acum[e]['TP']/acum[e]['n']:.3f}" for e in acum if acum[e]["n"]))
        resultado[nombre]["configs"] = {p: {"hmin": m["hmin"], "mindist": m["mindist"]}
                                        for p, m in mejores.items()}

    json.dump(resultado, open(os.path.join(RAIZ, "watershed_barrido.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\nguardado en watershed_barrido.json")


if __name__ == "__main__":
    main()
