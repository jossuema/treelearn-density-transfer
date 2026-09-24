"""Anade a nubes.json el numero real de instancias del archivo original.

El submuestreo a 9000 puntos pierde instancias enteras: las que tenian pocos puntos
pueden quedarse sin ninguno. El panel 3D anunciaba el conteo del submuestreo como si
fuera el total, lo que se presta a confusion con las detecciones que reporta el
panel de arboles, que ademas estan filtradas por numero minimo de puntos y por la
region inventariada. Aqui se guardan los dos numeros para poder decirlo claro.
"""
import json
import os

import laspy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(os.path.dirname(AQUI))
SAL = os.path.join(RAIZ, "salida_sat_all")

nub = json.load(open(os.path.join(AQUI, "data", "nubes.json")))
for key in nub:
    sitio, parcela = key.split("/")
    laz = os.path.join(SAL, f"{sitio}_{parcela}_out.laz")
    inst = np.asarray(laspy.read(laz).PredInstance).astype(np.int64)
    total = int(len(np.unique(inst[inst > 0])))
    nub[key]["n_inst_total"] = total
    nub[key]["n_inst_perdidas"] = max(0, total - nub[key]["n_inst"])
    print(f"{key:14s} submuestreo {nub[key]['n_inst']:4d} de {total:4d} instancias "
          f"({100*nub[key]['n_inst']/total:.0f} %)")

json.dump(nub, open(os.path.join(AQUI, "data", "nubes.json"), "w"), separators=(",", ":"))
print("\nnubes.json actualizado")
