"""Convierte los JSON exportados en archivos .js que el visor puede cargar con file://

Los navegadores bloquean fetch sobre file://, pero si permiten <script src>. Asi que
envolvemos cada JSON en una asignacion a window y lo guardamos como .js.

Uso: python3 empaquetar.py
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")

PIEZAS = [("geo.json", "geo.js", "GEO"),
          ("arboles.json", "arboles.js", "ARB"),
          ("nubes.json", "nubes.js", "NUB"),
          ("copas.json", "copas.js", "COP"),
          ("instancias.json", "instancias.js", "INS")]


def main():
    total = 0
    for entrada, salida, var in PIEZAS:
        pe = os.path.join(DATA, entrada)
        ps = os.path.join(DATA, salida)
        if not os.path.exists(pe):
            # Sin el JSON dejamos un objeto vacio para que el visor no reviente.
            with open(ps, "w") as f:
                f.write(f"window.{var} = {{}};\n")
            print(f"  {entrada:14s} no existe, {salida} queda vacio")
            continue
        with open(pe) as f:
            d = json.load(f)
        with open(ps, "w") as f:
            f.write(f"window.{var} = ")
            json.dump(d, f, separators=(",", ":"), ensure_ascii=False)
            f.write(";\n")
        n = os.path.getsize(ps)
        total += n
        print(f"  {entrada:14s} -> {salida:12s} {n/1e6:7.2f} MB  {len(d)} parcelas")
    print(f"total {total/1e6:.2f} MB")
    print(f"abre {os.path.join(AQUI, 'index.html')} en el navegador")


if __name__ == "__main__":
    main()
