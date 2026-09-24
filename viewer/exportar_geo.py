"""Modulo A del visor: geometria del terreno de las 22 parcelas.

Para cada parcela produce:
  - Un CHM (modelo de altura del dosel) de celda 0.5 m, rejilla del maximo de z,
    suavizado gaussiano sigma=1.0. Mismo procedimiento que det_watershed en analisis.py.
  - Ese CHM convertido a PNG en escala de grises de 8 bits, embebido como data URI.
  - El rectangulo del mundo que cubre exactamente esa imagen.
  - El poligono de la region inventariada (casco convexo dilatado 1.5 m).
  - Estadisticos basicos de la parcela.

Salida: respaldo_minimo/visor/data/geo.json

Uso: cd respaldo_minimo && python3 visor/exportar_geo.py
"""
import base64
import io
import json
import os
import sys
import time

import laspy
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# El modulo de analisis vive en respaldo_minimo/modal_app y es el que manda:
# de ahi salen las rutas de datos, la lista de parcelas y la funcion region().
# La raiz de los datos se toma de la variable TREELEARN_DATA; por defecto se
# asume que este repositorio cuelga de ella.
RAIZ_PROY = os.environ.get("TREELEARN_DATA",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RAIZ_PROY, "respaldo_minimo", "modal_app"))
from analisis import referencia, region, SITIOS, BUF  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "data", "geo.json")

CELDA = 0.5      # metros por celda del CHM
SIGMA = 1.0      # suavizado gaussiano, igual que en det_watershed
MAX_PNG = 120 * 1024   # limite por parcela: 120 KB


def construir_chm(px, py, pz, celda=CELDA, sigma=SIGMA):
    """Rejilla del maximo de z por celda, suavizada. Devuelve (chm, x0, y0).

    La fila i del array crece con y (la fila 0 es la y mas baja), igual que en
    det_watershed. El volteo para pantalla se hace despues, al pintar el PNG.
    """
    x0, y0 = float(px.min()), float(py.min())
    nx = int((px.max() - x0) / celda) + 1
    ny = int((py.max() - y0) / celda) + 1
    chm = np.zeros((ny, nx), np.float32)
    j = np.clip(((px - x0) / celda).astype(int), 0, nx - 1)
    i = np.clip(((py - y0) / celda).astype(int), 0, ny - 1)
    np.maximum.at(chm, (i, j), pz)
    return gaussian_filter(chm, sigma), x0, y0


def chm_a_png(chm, techo):
    """CHM a PNG gris de 8 bits, normalizado entre 0 y techo, fila 0 arriba.

    Devuelve (data_uri, ancho, alto). Si el PNG pasa de 120 KB, reduce la
    imagen a la mitad y reintenta hasta que quepa.
    """
    techo = max(float(techo), 1e-6)
    v = np.clip(chm / techo, 0.0, 1.0)
    # En el CHM la fila i crece con y; en pantalla la fila crece hacia abajo,
    # asi que hay que voltear verticalmente.
    v = np.flipud(v)
    img = Image.fromarray((v * 255.0 + 0.5).astype(np.uint8), mode="L")

    while True:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        crudo = buf.getvalue()
        if len(crudo) <= MAX_PNG or min(img.size) <= 8:
            break
        img = img.resize((max(1, img.width // 2), max(1, img.height // 2)),
                         Image.BILINEAR)
    uri = "data:image/png;base64," + base64.b64encode(crudo).decode("ascii")
    return uri, img.width, img.height


def area_poligono(v):
    """Area del poligono cerrado por la formula del cordon de zapato."""
    x, y = v[:, 0], v[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def procesar(sitio, parcela):
    ref = referencia(sitio, parcela)

    L = laspy.read(ref["las"])
    px, py, pz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    n_puntos = int(px.size)          # todos los retornos del archivo

    m = pz >= 0                      # solo alturas positivas, como en det_watershed
    px, py, pz = px[m], py[m], pz[m]

    chm, x0, y0 = construir_chm(px, py, pz)
    ny, nx = chm.shape
    x1, y1 = x0 + nx * CELDA, y0 + ny * CELDA

    h_max = float(pz.max())
    h_p99 = float(np.percentile(pz, 99))
    uri, w_png, h_png = chm_a_png(chm, h_p99)

    area_tile = float(nx * CELDA * ny * CELDA)
    verts = np.asarray(region(ref, buf=1.5).vertices, float)
    area_reg = area_poligono(verts)

    return {
        "chm_png": uri,
        "chm_extent": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "chm_max_h": round(h_p99, 2),
        "region": [[round(float(a), 2), round(float(b), 2)] for a, b in verts],
        "n_puntos": n_puntos,
        "area_tile_m2": round(area_tile, 2),
        "area_region_m2": round(area_reg, 2),
        "pct_region": round(100.0 * area_reg / area_tile, 2) if area_tile else 0.0,
        "densidad_pts_m2": round(n_puntos / area_tile, 2) if area_tile else 0.0,
        "h_max": round(h_max, 2),
        "h_p99": round(h_p99, 2),
    }, (nx, ny, w_png, h_png, len(uri))


def main():
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    salida = {}
    t0 = time.time()
    for sitio, nombre, parcelas in SITIOS:
        for p in parcelas:
            clave = f"{sitio}/{p}"
            t = time.time()
            salida[clave], info = procesar(sitio, p)
            nx, ny, w, h, nuri = info
            print(f"{clave:12} rejilla {nx:4}x{ny:<4} png {w:4}x{h:<4} "
                  f"{nuri/1024:7.1f} KB b64  pct_region "
                  f"{salida[clave]['pct_region']:5.2f}  ({time.time()-t:.1f}s)",
                  flush=True)

    with open(SALIDA, "w") as f:
        json.dump(salida, f, ensure_ascii=False)

    tam = os.path.getsize(SALIDA)
    print(f"\n{len(salida)} parcelas en {SALIDA}")
    print(f"tamano {tam} bytes = {tam/1024/1024:.2f} MB  ({time.time()-t0:.1f}s total)")

    # Comprobaciones: nada vacio y los porcentajes de region en rango.
    vacios = [k for k, v in salida.items() if not v["chm_png"]]
    print("chm_png vacios:", vacios if vacios else "ninguno")
    for pref, lo, hi in (("cham", 31, 46), ("pell", 23, 40)):
        vals = [(k, v["pct_region"]) for k, v in salida.items() if k.startswith(pref)]
        fuera = [(k, p) for k, p in vals if not (lo <= p <= hi)]
        arr = [p for _, p in vals]
        print(f"{pref}: pct_region min {min(arr):.2f} max {max(arr):.2f} "
              f"media {sum(arr)/len(arr):.2f}  fuera de [{lo},{hi}]: "
              f"{fuera if fuera else 'ninguna'}")


if __name__ == "__main__":
    main()
