"""Dos correcciones sobre lo exportado.

1. pct_region se calculaba contra el rectangulo del CHM, que lleva media celda de
   sobra por lado. Se recalcula contra la caja exacta de los puntos, que es la cifra
   que usa el articulo.
2. Se anaden las detecciones que caen FUERA de la region inventariada. Son la
   evidencia directa del sesgo de encuadre: sin recorte, todas contarian como
   falsos positivos aunque ahi haya dosel real que nadie midio.
"""
import csv
import json
import os
import sys

import laspy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
from analisis import referencia, region, det_sat, det_ams, det_watershed, SITIOS  # noqa: E402
from shapely.geometry import Polygon                                              # noqa: E402

DATA = os.path.join(AQUI, "data")
geo = json.load(open(os.path.join(DATA, "geo.json")))
arb = json.load(open(os.path.join(DATA, "arboles.json")))

cfgs = {}
for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
    p = os.path.join(BASE, "ams_port", "out", f)
    for r in csv.DictReader(open(p)):
        cfgs[(s, r["plot"])] = r["cfg"]

wb = json.load(open(os.path.join(RAIZ, "watershed_barrido.json")))
wcfg = {}
for nombre, d in wb.items():
    sit = "cham" if nombre == "Chamrousse" else "pell"
    for pl, c in d.get("configs", {}).items():
        wcfg[(sit, pl)] = (float(c["hmin"]), float(c["mindist"]))

for sitio, nombre, parcelas in SITIOS:
    for p in parcelas:
        key = f"{sitio}/{p}"
        ref = referencia(sitio, p)
        R = region(ref)

        # 1. cobertura sobre la caja exacta de los puntos
        L = laspy.read(ref["las"])
        x, y = np.asarray(L.x), np.asarray(L.y)
        area_tile = float((x.max() - x.min()) * (y.max() - y.min()))
        area_reg = float(Polygon(R.vertices).area)
        geo[key]["area_tile_m2"] = round(area_tile, 2)
        geo[key]["area_region_m2"] = round(area_reg, 2)
        geo[key]["pct_region"] = round(100.0 * area_reg / area_tile, 2)
        geo[key]["densidad_pts_m2"] = round(len(x) / area_tile, 2)

        # 2. detecciones fuera de la region
        hm, md = wcfg.get((sitio, p), (2.0, 2.0))
        dets = {"SAT": det_sat(sitio, p, "apice"),
                "AMS3D": det_ams(sitio, p, cfgs),
                "WS": det_watershed(ref, hmin=hm, mindist=md)}
        arb[key].setdefault("det_fuera", {})
        for met, d in dets.items():
            if d is None or not len(d["x"]):
                arb[key]["det_fuera"][met] = {"x": [], "y": [], "h": [], "area": []}
                continue
            dentro = R.contains_points(np.c_[d["x"], d["y"]])
            f = ~dentro
            arb[key]["det_fuera"][met] = {
                k: [round(float(v), 2) for v in np.asarray(d[k])[f]]
                for k in ("x", "y", "h", "area")}
    print(f"  {sitio} listo", flush=True)

json.dump(geo, open(os.path.join(DATA, "geo.json"), "w"), separators=(",", ":"))
json.dump(arb, open(os.path.join(DATA, "arboles.json"), "w"), separators=(",", ":"))

for s, tag in (("cham", "Chamrousse"), ("pell", "Pellizzano")):
    v = [geo[k]["pct_region"] for k in geo if k.startswith(s + "/")]
    print(f"{tag}: region {min(v):.0f} a {max(v):.0f} % del tile")
for s, tag in (("cham", "Chamrousse"), ("pell", "Pellizzano")):
    for met in ("SAT", "AMS3D", "WS"):
        dentro = sum(len(arb[k]["det"][met]["x"]) for k in arb if k.startswith(s + "/"))
        fuera = sum(len(arb[k]["det_fuera"][met]["x"]) for k in arb if k.startswith(s + "/"))
        print(f"{tag:12s} {met:6s} dentro={dentro:5d} fuera={fuera:5d} "
              f"({100*fuera/max(1,dentro+fuera):.0f} % cae fuera)")
