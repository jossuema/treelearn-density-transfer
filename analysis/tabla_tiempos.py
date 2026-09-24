"""Arma la comparacion de tiempos de ejecucion de los tres metodos.

Lo que hace comparable a los tres no es el tiempo de una pasada, sino el tiempo que
cuesta llegar al resultado que se publica. SegmentAnyTree corre una sola vez, con su
checkpoint publico y sin ajustar nada. AMS3D publica la mejor de 25 configuraciones
por parcela, y el watershed la mejor de 16, asi que para reproducir su cifra hay que
correrlos 25 y 16 veces por parcela. Ese multiplicador es parte del coste y se
reporta aparte, no escondido.

Las medidas de CPU salen de tiempos.py, las 22 parcelas en una sola tirada y en la
misma maquina, usando tiempo de CPU del proceso. Importa que sea una sola tirada: al
medirlas en tandas separadas, las que coincidieron con otras tareas salieron entre un
21 y un 48 por ciento mas lentas, porque bajo carga el procesador reduce su frecuencia
y eso afecta por igual al reloj de pared y al tiempo de CPU. Las de
GPU salen de sat_run_result.json, medidas en Modal sobre una T4. La asimetria de
hardware es real y se dice.

Uso: python3 respaldo_minimo/analisis_paper/tabla_tiempos.py
"""
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)

T = json.load(open(os.path.join(RAIZ, "tiempos.json")))
SAT = json.load(open(os.path.join(RAIZ, "sat_run_result.json")))

# configuraciones por parcela que necesita cada metodo para dar su cifra publicada
CONFIGS = {"SAT": 1, "AMS3D": 25, "WS": 16}

# puntos de entrada de SegmentAnyTree: corrio sobre full_las, que trae el suelo, y no
# sobre el .las recortado a 1 m que usan los otros dos. Hay que decirlo.
sat = {}
for r in SAT:
    clave = r["archivo"].replace(".las", "").replace("_", "/", 1)
    res = list(r["resumen"].values())[0] if r.get("resumen") else {}
    sat[clave] = {"s": r["segundos"], "puntos": res.get("n_puntos")}

P = T["parcelas"]
claves = sorted(P, key=lambda k: P[k]["puntos"])

print(f"Maquina de CPU: {T['maquina']['cpu']}, {T['maquina']['nucleos']} nucleos, "
      f"{T['maquina']['sistema']}")
print("GPU: NVIDIA T4 en Modal, segun sat_run_result.json\n")

print(f"{'parcela':14} {'pts CPU':>9} {'pts GPU':>9} {'WS s':>7} {'AMS3D s':>9} {'SAT s':>8}")
for k in claves:
    d = P[k]
    a = d.get("ams_cpu")
    s = sat.get(k, {})
    print(f"{k:14} {d['puntos']:>9} {str(s.get('puntos','')):>9} {d.get('ws_cpu',0):>7.2f} "
          f"{(f'{a:.1f}' if a else '-'):>9} {(f'{s.get(chr(115),0):.1f}' if s else '-'):>8}")

# --------------------------------------------------------------- rendimiento
print("\nRENDIMIENTO, puntos por segundo, mediana sobre las parcelas medidas")
rend = {}
ws = [P[k]["puntos"] / P[k]["ws_cpu"] for k in claves if P[k].get("ws_cpu", 0) > 0]
rend["WS"] = float(np.median(ws))
am = [P[k]["puntos"] / P[k]["ams_cpu"] for k in claves if P[k].get("ams_cpu")]
rend["AMS3D"] = float(np.median(am)) if am else float("nan")
st = [v["puntos"] / v["s"] for v in sat.values() if v.get("puntos")]
rend["SAT"] = float(np.median(st))
for m in ("WS", "AMS3D", "SAT"):
    print(f"  {m:6} {rend[m]:>12,.0f} pts/s"
          + ("   (GPU T4)" if m == "SAT" else "   (CPU, 1 proceso)")
          + (f"   n={len(am)} parcelas" if m == "AMS3D" else ""))

# --------------------------------------------------------------- coste total
print("\nCOSTE DE UNA PASADA sobre las 22 parcelas, extrapolando con el rendimiento medido")
tot_cpu = sum(P[k]["puntos"] for k in claves)
tot_gpu = sum(v["puntos"] for v in sat.values() if v.get("puntos"))
print(f"  puntos totales: {tot_cpu:,} en los .las recortados, {tot_gpu:,} en los full_las de SAT")
una = {}
una["WS"] = sum(P[k]["ws_cpu"] for k in claves)
una["AMS3D"] = sum(P[k]["ams_cpu"] for k in claves)
una["SAT"] = sum(v["s"] for v in sat.values())
for m in ("WS", "AMS3D", "SAT"):
    med = "medido" if m != "AMS3D" else f"{len(am)} medidas, {22-len(am)} extrapoladas"
    print(f"  {m:6} {una[m]:>10.1f} s = {una[m]/60:>7.1f} min   ({med})")

print("\nCOSTE DE REPRODUCIR LA CIFRA PUBLICADA, con el barrido que cada metodo necesita")
for m in ("WS", "AMS3D", "SAT"):
    t = una[m] * CONFIGS[m]
    print(f"  {m:6} {CONFIGS[m]:>3} config/parcela  {t:>10.1f} s = {t/3600:>7.2f} h")

print("\nRelaciones frente a SegmentAnyTree, una pasada:")
for m in ("WS", "AMS3D"):
    print(f"  {m:6} una pasada {una['SAT']/una[m]:>8.1f} veces mas rapido que SAT; "
          f"con su barrido, {una['SAT']/(una[m]*CONFIGS[m]):>6.2f} veces")

json.dump({"maquina": T["maquina"], "rendimiento": rend, "una_pasada_s": una,
           "configs": CONFIGS, "puntos_cpu": tot_cpu, "puntos_gpu": tot_gpu,
           "ams_medidas": len(am)},
          open(os.path.join(RAIZ, "tiempos_resumen.json"), "w"), indent=2, ensure_ascii=False)
print("\nguardado en tiempos_resumen.json")
