"""Adds to trees.json the species of each inventory tree.

reference() in evaluation_core.py reads the CSV rows in order, filtered by plotid, and
drops the row if x, y, height or area are not numbers. Here that same filter is
repeated so that index i of the species is tree i of inv.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import SITES, BASE as DATA  # noqa: E402

PATHS = {"cham": os.path.join(DATA, "IRSTEA_dataset/inventory_csv/Chamrousse.csv"),
         "pell": os.path.join(DATA, "trento_dataset/csv/Pellizzano.csv")}

arb = json.load(open(os.path.join(HERE, "data", "trees.json")))
rows = {s: list(csv.DictReader(open(r))) for s, r in PATHS.items()}

total = 0
for site, name, plots in SITES:
    for p in plots:
        key = f"{site}/{p}"
        sp, fam = [], []
        for r in rows[site]:
            if r["plotid"] != p:
                continue
            try:                       # the same filter as reference()
                float(r["x"]); float(r["y"]); float(r["height"]); float(r["area"])
            except ValueError:
                continue
            # "specie" is how the inventory CSV spells the column; it is their name, not ours
            sp.append(r.get("specie", "?") or "?")
            fam.append(r.get("family", "?") or "?")
        n = len(arb[key]["inv"]["x"])
        assert len(sp) == n, f"{key}: {len(sp)} species against {n} trees"
        assert fam == arb[key]["inv"]["fam"], f"{key}: the family does not match the export"
        arb[key]["inv"]["species"] = sp
        total += n

json.dump(arb, open(os.path.join(HERE, "data", "trees.json"), "w"), separators=(",", ":"))
print(f"species added to {total} trees, checked against the family already exported")

import collections
for site, name, plots in SITES:
    c = collections.Counter()
    for p in plots:
        c.update(arb[f"{site}/{p}"]["inv"]["species"])
    print(f"{name}: " + ", ".join(f"{k} {v}" for k, v in c.most_common(6)))
