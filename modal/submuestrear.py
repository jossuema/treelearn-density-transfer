"""Ralea las parcelas de Pellizzano quitando PULSOS enteros, no puntos sueltos.

Por que por pulso: un ALS mas ralo emite menos pulsos, pero cada pulso sigue devolviendo
todos sus retornos. Si se quitaran puntos al azar se romperia la estructura de retornos
multiples, que es justo una de las caracteristicas que consumen los modelos
(number_of_returns y return_number). Los LAS traen gps_time, asi que se puede agrupar.

Por que anidado: se permutan los pulsos UNA vez con semilla fija y cada nivel toma un
prefijo. Asi el 25 % es subconjunto del 40 % y este del 60 %, y las diferencias entre
niveles no llevan encima el ruido de tres sorteos distintos.

Pellizzano va de 91 a 173 pts/m2; al 40 % queda en 36 a 69, que es el rango de Chamrousse.
Eso permite comparar a igualdad de densidad sin cambiar sitio, sensor ni rodal.

Uso: python3 respaldo_minimo/modal_app/submuestrear.py
"""
import os

import laspy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
ORIGEN = os.path.join(BASE, "trento_dataset", "las")
DESTINO = os.path.join(RAIZ, "submuestreo")
FRACCIONES = (0.60, 0.40, 0.25)
PARCELAS = [f"p{i}" for i in range(1, 16)]


def main():
    os.makedirs(DESTINO, exist_ok=True)
    print(f"{'parcela':9} {'pulsos':>8} {'puntos':>8} " +
          " ".join(f"{int(f*100):>13}%" for f in FRACCIONES))
    for p in PARCELAS:
        L = laspy.read(os.path.join(ORIGEN, f"{p}.las"))
        t = np.asarray(L.gps_time)
        pulsos, inverso = np.unique(t, return_inverse=True)
        # una sola permutacion: los niveles son prefijos anidados
        orden = np.random.default_rng(0).permutation(len(pulsos))
        fila = f"{p:9} {len(pulsos):>8} {len(L.points):>8} "
        for f in FRACCIONES:
            k = int(round(len(pulsos) * f))
            quedan = np.zeros(len(pulsos), bool)
            quedan[orden[:k]] = True
            m = quedan[inverso]
            sal = laspy.LasData(L.header)
            sal.points = L.points[m].copy()
            nom = f"pell_{p}_f{int(f*100):02d}.las"
            sal.write(os.path.join(DESTINO, nom))
            fila += f" {int(m.sum()):>13}"
        print(fila)
    n = len([f for f in os.listdir(DESTINO) if f.endswith(".las")])
    mb = sum(os.path.getsize(os.path.join(DESTINO, f))
             for f in os.listdir(DESTINO)) / 1e6
    print(f"\n{n} ficheros, {mb:.0f} MB en {DESTINO}")


if __name__ == "__main__":
    main()
