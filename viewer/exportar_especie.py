"""Anade a arboles.json la especie de cada arbol del inventario.

referencia() en analisis.py lee las filas del CSV en orden, filtradas por plotid, y
descarta la fila si x, y, altura o area no son numeros. Aqui se repite ese mismo
filtro para que el indice i de la especie sea el arbol i de inv.
"""
import csv
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
from analisis import SITIOS  # noqa: E402

RUTAS = {"cham": os.path.join(BASE, "IRSTEA_dataset/inventory_csv/Chamrousse.csv"),
         "pell": os.path.join(BASE, "trento_dataset/csv/Pellizzano.csv")}

arb = json.load(open(os.path.join(AQUI, "data", "arboles.json")))
filas = {s: list(csv.DictReader(open(r))) for s, r in RUTAS.items()}

total = 0
for sitio, nombre, parcelas in SITIOS:
    for p in parcelas:
        key = f"{sitio}/{p}"
        sp, fam = [], []
        for r in filas[sitio]:
            if r["plotid"] != p:
                continue
            try:                       # el mismo filtro que referencia()
                float(r["x"]); float(r["y"]); float(r["height"]); float(r["area"])
            except ValueError:
                continue
            sp.append(r.get("specie", "?") or "?")
            fam.append(r.get("family", "?") or "?")
        n = len(arb[key]["inv"]["x"])
        assert len(sp) == n, f"{key}: {len(sp)} especies frente a {n} arboles"
        assert fam == arb[key]["inv"]["fam"], f"{key}: la familia no coincide con lo exportado"
        arb[key]["inv"]["specie"] = sp
        total += n

json.dump(arb, open(os.path.join(AQUI, "data", "arboles.json"), "w"), separators=(",", ":"))
print(f"especie anadida a {total} arboles, comprobada contra la familia ya exportada")

import collections
for sitio, nombre, parcelas in SITIOS:
    c = collections.Counter()
    for p in parcelas:
        c.update(arb[f"{sitio}/{p}"]["inv"]["specie"])
    print(f"{nombre}: " + ", ".join(f"{k} {v}" for k, v in c.most_common(6)))
