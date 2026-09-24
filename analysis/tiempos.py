"""Mide el tiempo de ejecucion de AMS3D y del watershed, parcela a parcela.

Metodo, y hay que respetarlo para que los numeros signifiquen algo:
  - Todo SECUENCIAL en la misma maquina. Medir dos metodos a la vez los contamina
    por competencia de CPU.
  - Se cronometra el TIEMPO DE CPU DEL PROCESO, no el reloj de pared. La maquina es
    de trabajo y tiene otras aplicaciones encima: el mismo calculo medido con reloj
    de pared en dos momentos distintos varia mas del 50 %, mientras que el tiempo de
    CPU no cuenta lo que el proceso pasa desalojado. Se guardan los dos.
  - Se cronometra la configuracion que cada metodo acabo usando en el articulo, la
    mejor de su barrido para esa parcela.
  - De cada medida se toma la mediana de varias repeticiones cuando el metodo es
    rapido. AMS3D tarda minutos, asi que ahi va una sola pasada.
  - Se escribe el JSON despues de cada parcela, para poder parar y quedarse con lo
    medido hasta ese punto.

El tiempo de SegmentAnyTree NO se mide aqui: esta en sat_run_result.json, medido en
Modal sobre una GPU T4. La asimetria de hardware es real y hay que decirla, no
esconderla.

Uso: python3 respaldo_minimo/analisis_paper/tiempos.py [--ams N]
  --ams N  cuantas parcelas cronometrar con AMS3D, de menor a mayor tamano (0 = ninguna)
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
sys.path.insert(0, os.path.join(BASE, "ams_port"))
from analisis import referencia, det_watershed, SITIOS  # noqa: E402

SALIDA = os.path.join(RAIZ, "tiempos.json")


def maquina():
    """Lo que hace falta para que otro pueda situar estos numeros."""
    d = {"sistema": platform.platform(), "python": platform.python_version()}
    try:
        d["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        d["nucleos"] = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True))
    except Exception:
        d["cpu"] = platform.processor() or "desconocido"
        d["nucleos"] = os.cpu_count()
    return d


def configs_ams():
    out = {}
    for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        p = os.path.join(BASE, "ams_port", "out", f)
        for r in csv.DictReader(open(p)):
            # el nombre lleva m1 y m2 multiplicados por 1000
            partes = r["cfg"].split("_")
            m1 = int(partes[partes.index("m1") + 1]) / 1000.0
            m2 = int(partes[partes.index("m2") + 1]) / 1000.0
            out[(s, r["plot"])] = (m1, m2)
    return out


def configs_ws():
    wb = json.load(open(os.path.join(RAIZ, "watershed_barrido.json")))
    out = {}
    for nombre, d in wb.items():
        s = "cham" if nombre == "Chamrousse" else "pell"
        for pl, c in d.get("configs", {}).items():
            out[(s, pl)] = (float(c["hmin"]), float(c["mindist"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ams", type=int, default=6)
    args = ap.parse_args()

    cfa, cfw = configs_ams(), configs_ws()
    res = {"maquina": maquina(), "parcelas": {}}
    if os.path.exists(SALIDA):
        try:
            res = json.load(open(SALIDA))
            res["maquina"] = maquina()
        except Exception:
            pass
    res.setdefault("parcelas", {})

    # 1. inventario de tamanos, y el watershed, que es barato
    print("Watershed, mediana de 3 pasadas por parcela")
    print(f"{'parcela':14} {'puntos':>9} {'ws_s':>8}")
    orden = []
    for sitio, nombre, parcelas in SITIOS:
        for p in parcelas:
            clave = f"{sitio}/{p}"
            ref = referencia(sitio, p)
            import laspy
            n = int(len(laspy.read(ref["las"]).x))
            orden.append((n, sitio, p, clave))
            hm, md = cfw.get((sitio, p), (2.0, 2.0))
            ts, cs = [], []
            for _ in range(5):
                t0, p0 = time.perf_counter(), time.process_time()
                det_watershed(ref, hmin=hm, mindist=md)
                ts.append(time.perf_counter() - t0)
                cs.append(time.process_time() - p0)
            d = res["parcelas"].setdefault(clave, {})
            d.update(puntos=n, ws_s=round(float(np.median(ts)), 3),
                     ws_cpu=round(float(np.median(cs)), 4), ws_cfg=[hm, md])
            print(f"{clave:14} {n:>9} {np.median(ts):>8.2f}", flush=True)
            json.dump(res, open(SALIDA, "w"), indent=2, ensure_ascii=False)

    # 2. AMS3D, de la parcela mas pequena a la mas grande, una pasada
    orden.sort()
    objetivo = orden[:args.ams] if args.ams >= 0 else orden
    print(f"\nAMS3D, una pasada, {len(objetivo)} parcelas de menor a mayor")
    print(f"{'parcela':14} {'puntos':>9} {'reloj_s':>9} {'cpu_s':>9} {'pts/cpu_s':>9}")
    from ams_engine import run_e1
    for n, sitio, p, clave in objetivo:
        if res["parcelas"].get(clave, {}).get("ams_cpu"):
            print(f"{clave:14} ya medida, se salta", flush=True)
            continue
        m1, m2 = cfa[(sitio, p)]
        las = referencia(sitio, p)["las"]
        t0, p0 = time.perf_counter(), time.process_time()
        run_e1(las, m1, m2)
        dt, dc = time.perf_counter() - t0, time.process_time() - p0
        d = res["parcelas"].setdefault(clave, {})
        d.update(ams_s=round(dt, 2), ams_cpu=round(dc, 2), ams_cfg=[m1, m2])
        print(f"{clave:14} {n:>9} {dt:>9.1f} {dc:>9.1f} {n/dc:>9.0f}", flush=True)
        json.dump(res, open(SALIDA, "w"), indent=2, ensure_ascii=False)

    json.dump(res, open(SALIDA, "w"), indent=2, ensure_ascii=False)
    print(f"\nguardado en {SALIDA}")


if __name__ == "__main__":
    main()
