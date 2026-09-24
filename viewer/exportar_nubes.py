"""Modulo C del visor: nubes de puntos submuestreadas para la vista 3D.

Lee las salidas de SegmentAnyTree (un .laz por parcela, con el campo PredInstance)
y escribe visor/data/nubes.json con las coordenadas comprimidas como enteros de
16 bits en base64. La idea es que el navegador pueda pintar las 22 parcelas sin
descargar cientos de megas de nube cruda.

Uso:
  cd respaldo_minimo && python3 visor/exportar_nubes.py
"""
import base64
import json
import os
import sys

import laspy
import numpy as np

# Reutilizamos la definicion de sitios y parcelas del modulo de analisis
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modal"))
from analisis import SITIOS  # noqa: E402

# La raiz de los datos se toma de la variable TREELEARN_DATA; por defecto se
# asume que este repositorio cuelga de ella.
RAIZ = os.environ.get("TREELEARN_DATA",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAL = os.path.join(RAIZ, "salida_sat_all")
DESTINO = os.path.join(RAIZ, "respaldo_minimo", "visor", "data", "nubes.json")
DESTINO_INST = os.path.join(RAIZ, "respaldo_minimo", "visor", "data", "instancias.json")

sys.path.insert(0, os.path.join(RAIZ, "respaldo_minimo", "modal_app"))
from analisis import referencia, region, MIN_PTS   # noqa: E402

MAX_PTS = 40000         # tope por parcela; hace falta densidad para ver un arbol suelto
LIM = 32767             # tope de un entero con signo de 16 bits


def b64_int16(arr):
    """Empaqueta un arreglo como int16 little endian y lo devuelve en base64."""
    return base64.b64encode(np.asarray(arr).astype("<i2").tobytes()).decode("ascii")


def cabe_int16(*arrays):
    """True si todos los valores caben en un entero con signo de 16 bits."""
    for a in arrays:
        if a.size == 0:
            continue
        if a.min() < -LIM - 1 or a.max() > LIM:
            return False
    return True


def submuestrear(n, tope):
    """Indices reproducibles sin reemplazo. Si hay pocos puntos los devuelve todos."""
    if n <= tope:
        return np.arange(n)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=tope, replace=False)
    idx.sort()          # ordenados para que la lectura del laz sea predecible
    return idx


def exportar_parcela(sitio, parcela, tope):
    """Devuelve el diccionario comprimido de una parcela, o None si falta el laz."""
    laz = os.path.join(SAL, f"{sitio}_{parcela}_out.laz")
    if not os.path.exists(laz):
        return None

    L = laspy.read(laz)
    x = np.asarray(L.x, dtype=np.float64)
    y = np.asarray(L.y, dtype=np.float64)
    z = np.asarray(L.z, dtype=np.float64)
    inst = np.asarray(L.PredInstance).astype(np.int64)

    # Instancia de cada deteccion de SegmentAnyTree, en el mismo orden que produce
    # det_sat(sitio, parcela, "apice") de analisis.py: recorrer las instancias en orden,
    # saltarse las de menos de MIN_PTS puntos, y quedarse con el punto mas alto.
    det_id, det_x, det_y = [], [], []
    for iid in np.unique(inst[inst > 0]):
        m = inst == iid
        if int(m.sum()) < MIN_PTS:
            continue
        k = int(np.argmax(z[m]))
        det_id.append(int(iid)); det_x.append(float(x[m][k])); det_y.append(float(y[m][k]))
    ref = referencia(sitio, parcela)
    if det_x:
        dentro = region(ref).contains_points(np.c_[det_x, det_y])
    else:
        dentro = np.zeros(0, bool)
    mapa_inst = {"dentro": [det_id[i] for i in range(len(det_id)) if dentro[i]],
                 "fuera": [det_id[i] for i in range(len(det_id)) if not dentro[i]]}

    n_original = int(x.size)
    idx = submuestrear(n_original, tope)
    x, y, z, inst = x[idx], y[idx], z[idx], inst[idx]

    # Origen local: minimo de x e y de los puntos elegidos, redondeado a 2 decimales.
    # Redondeamos ANTES de codificar para que la reconstruccion sea exacta.
    x0 = round(float(x.min()), 2)
    y0 = round(float(y.min()), 2)

    escala = 100                     # centimetros
    xi = np.rint((x - x0) * escala)
    yi = np.rint((y - y0) * escala)
    zi = np.rint(z * escala)
    if not cabe_int16(xi, yi, zi):
        # La parcela es demasiado grande o alta para centimetros: bajamos a decimetros
        escala = 10
        xi = np.rint((x - x0) * escala)
        yi = np.rint((y - y0) * escala)
        zi = np.rint(z * escala)
        if not cabe_int16(xi, yi, zi):
            raise ValueError(f"{sitio}/{parcela}: no cabe ni en decimetros")

    # Se conserva el identificador ORIGINAL de PredInstance, sin reindexar. El maximo
    # en estos 22 archivos es 646, asi que cabe de sobra en int16, y mantenerlo permite
    # enlazar cada deteccion con sus puntos sin ninguna tabla intermedia.
    # Cualquier valor <= 0 se trata como punto sin instancia.
    presentes = np.unique(inst[inst > 0])
    ii = np.where(inst > 0, inst, 0)
    if not cabe_int16(ii):
        raise ValueError(f"{sitio}/{parcela}: identificadores de instancia fuera de int16")

    return {
        "n": int(x.size),
        "n_original": n_original,
        "origen": [x0, y0],
        "escala": escala,
        "x": b64_int16(xi),
        "y": b64_int16(yi),
        "z": b64_int16(zi),
        "inst": b64_int16(ii),
        "n_inst": int(presentes.size),
    }, mapa_inst


def construir(tope, inst_por_det):
    """Recorre las 22 parcelas y arma el diccionario completo."""
    salida = {}
    for sitio, _nombre, parcelas in SITIOS:
        for parcela in parcelas:
            clave = f"{sitio}/{parcela}"
            r = exportar_parcela(sitio, parcela, tope)
            if r is None:
                print(f"  falta el laz de {clave}")
                continue
            d, mi = r
            inst_por_det[clave] = mi
            if d is None:
                print(f"  falta el laz de {clave}")
                continue
            salida[clave] = d
            print(f"  {clave}: {d['n']} de {d['n_original']} puntos, "
                  f"{d['n_inst']} instancias, escala {d['escala']}")
    return salida


def main():
    tope = MAX_PTS
    for intento in range(2):
        print(f"Exportando con tope de {tope} puntos por parcela")
        inst_por_det = {}
        datos = construir(tope, inst_por_det)
        texto = json.dumps(datos, ensure_ascii=False)
        mb = len(texto.encode("utf-8")) / 1e6
        print(f"Tamano del JSON: {mb:.2f} MB con {len(datos)} parcelas")
        if mb <= 40 or intento == 1:
            os.makedirs(os.path.dirname(DESTINO), exist_ok=True)
            with open(DESTINO, "w", encoding="utf-8") as f:
                f.write(texto)
            print(f"Escrito en {DESTINO}")
            with open(DESTINO_INST, "w", encoding="utf-8") as f:
                json.dump(inst_por_det, f, separators=(",", ":"))
            nd = sum(len(v["dentro"]) for v in inst_por_det.values())
            nf = sum(len(v["fuera"]) for v in inst_por_det.values())
            print(f"Escrito en {DESTINO_INST}: {nd} detecciones dentro y {nf} fuera")
            return datos
        print("Pasa de 40 MB, repito con 6000 puntos por parcela")
        tope = 6000


if __name__ == "__main__":
    main()
