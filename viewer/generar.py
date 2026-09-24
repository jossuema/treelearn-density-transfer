"""Genera todo lo que el visor necesita, en el orden correcto.

Importa porque los pasos no son independientes: parche_extra.py reescribe campos que
exportar_geo.py y exportar_arboles.py habian dejado de otra forma, y empaquetar.py
tiene que ir el ultimo. Si se ejecutan sueltos y en otro orden, los .js no
corresponden con los .json.

Uso: python3 generar.py
"""
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)

PASOS = [
    ("exportar_geo.py", "dosel, region y estadisticos por parcela"),
    ("exportar_arboles.py", "inventario, detecciones, parejas y metricas"),
    ("exportar_nubes.py", "nubes de puntos submuestreadas"),
    ("completar_nubes.py", "conteo real de instancias, antes del submuestreo"),
    ("parche_extra.py", "cobertura sobre la caja de puntos y detecciones de fuera"),
    ("exportar_copas.py", "cascos de copa de SegmentAnyTree y poligonos de cuatro radios"),
    ("exportar_especie.py", "especie de cada arbol del inventario"),
    ("empaquetar.py", "envolver los JSON en .js para poder abrir con file://"),
]


def main():
    for archivo, que in PASOS:
        ruta = os.path.join(AQUI, archivo)
        if not os.path.exists(ruta):
            print(f"FALTA {archivo}, se salta")
            continue
        print(f"\n=== {archivo}: {que}")
        r = subprocess.run([sys.executable, ruta], cwd=BASE)
        if r.returncode:
            print(f"\n{archivo} fallo con codigo {r.returncode}. Se para aqui.")
            return r.returncode
    print("\nlisto. Abre index.html, o sirvelo con:")
    print(f"  python3 -m http.server 8777 --directory '{AQUI}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
