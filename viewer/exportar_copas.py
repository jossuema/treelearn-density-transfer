"""Modulo C del visor: geometria de copa por parcela.

Produce respaldo_minimo/visor/data/copas.json con, para cada una de las 22 parcelas:
  - sat_dentro: casco convexo en planta de cada deteccion de SegmentAnyTree que cae
                dentro de la region inventariada, en el mismo orden que det["SAT"]
  - sat_fuera:  lo mismo para las detecciones que caen fuera, en el orden de det_fuera["SAT"]
  - inv:        poligono de copa del inventario a partir de los cuatro radios N, S, E, O,
                en el mismo orden que inv. Solo Chamrousse, en Pellizzano va vacio

Todos los poligonos se guardan como lista de vertices [dx, dy] en decimetros enteros
RELATIVOS a la posicion del arbol o de la deteccion, para que el JSON no se dispare.

Reutiliza analisis.py sin tocarlo: se copia la logica de det_sat y de referencia para
poder colgar la geometria de cada instancia sin cambiar la funcion original.

Uso, siempre con cwd = respaldo_minimo:
    python3 visor/exportar_copas.py
"""
import csv
import json
import math
import os
import sys

import laspy
import numpy as np
from scipy.spatial import ConvexHull, QhullError

# La raiz de los datos se toma de la variable TREELEARN_DATA; por defecto se
# asume que este repositorio cuelga de ella.
RAIZ = os.environ.get("TREELEARN_DATA",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(RAIZ, "respaldo_minimo")
sys.path.insert(0, os.path.join(BASE, "modal_app"))

from analisis import SITIOS, SAL, MIN_PTS, referencia, region  # noqa: E402

SALIDA = os.path.join(BASE, "visor", "data", "copas.json")
ARBOLES = os.path.join(BASE, "visor", "data", "arboles.json")
CSV_CHAM = os.path.join(BASE, "IRSTEA_dataset/inventory_csv/Chamrousse.csv")

MAX_VERT = 12        # vertices por casco convexo, se baja a 8 si el JSON pasa de 6 MB
N_ANG = 24           # angulos del poligono de cuatro radios
LIMITE_MB = 6.0


# ------------------------------------------------------------------ utilidades
def simplifica(vertices, maxv):
    """Deja como mucho maxv vertices repartidos de forma uniforme por el recorrido.

    ConvexHull ya devuelve los vertices en orden antihorario, asi que basta tomar
    indices igualmente espaciados sobre esa lista para conservar la forma general.
    """
    n = len(vertices)
    if n <= maxv:
        return vertices
    idx = np.round(np.linspace(0, n, maxv, endpoint=False)).astype(int) % n
    return vertices[idx]


def a_decimetros(vertices, cx, cy):
    """Pasa vertices absolutos en metros a decimetros enteros relativos a (cx, cy)."""
    return [[int(round((float(vx) - cx) * 10.0)),
             int(round((float(vy) - cy) * 10.0))] for vx, vy in vertices]


def area_poligono(poli):
    """Area por la formula del cordon de zapato. Entra en decimetros, sale en m2."""
    if len(poli) < 3:
        return 0.0
    s = 0.0
    for k in range(len(poli)):
        x1, y1 = poli[k]
        x2, y2 = poli[(k + 1) % len(poli)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0 / 100.0        # dm2 a m2


def dist_al_poligono(poli):
    """Distancia en metros del origen (0,0) al poligono, 0 si el origen cae dentro.

    El poligono viene en decimetros relativos a la deteccion, asi que la deteccion
    esta siempre en el origen y basta medir contra ese punto.
    """
    if len(poli) < 3:
        return float("inf")
    p = np.asarray(poli, float)
    # dentro o fuera, por el numero de cruces del rayo horizontal
    dentro = False
    n = len(p)
    for k in range(n):
        x1, y1 = p[k]
        x2, y2 = p[(k + 1) % n]
        if (y1 > 0) != (y2 > 0):
            xc = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
            if xc > 0:
                dentro = not dentro
    if dentro:
        return 0.0
    # distancia al segmento mas cercano
    mejor = float("inf")
    for k in range(n):
        a = p[k]
        b = p[(k + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else float(np.clip((-a) @ ab / L2, 0.0, 1.0))
        mejor = min(mejor, float(np.linalg.norm(a + t * ab)))
    return mejor / 10.0                # decimetros a metros


# ------------------------------------------------------------------ SegmentAnyTree
def cascos_sat(sitio, parcela, maxv):
    """Repite det_sat(sitio, parcela, "apice") y ademas guarda el casco de cada instancia.

    Mismo recorrido: np.unique(inst[inst>0]) en orden, se salta lo que tenga menos de
    MIN_PTS puntos, y la posicion es el punto mas alto de la instancia.
    """
    laz = os.path.join(SAL, f"{sitio}_{parcela}_out.laz")
    if not os.path.exists(laz):
        return None
    L = laspy.read(laz)
    x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    inst = np.asarray(L.PredInstance).astype(np.int64)
    X, Y, cascos = [], [], []
    fallos = 0
    for i in np.unique(inst[inst > 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        xm, ym, zm = x[m], y[m], z[m]
        k = np.argmax(zm)
        cx, cy = float(xm[k]), float(ym[k])
        X.append(cx)
        Y.append(cy)
        xy = np.c_[xm, ym]
        try:
            v = xy[ConvexHull(xy).vertices]
            cascos.append(a_decimetros(simplifica(v, maxv), cx, cy))
        except (QhullError, ValueError):
            # instancia degenerada: puntos repetidos o alineados en una recta
            cascos.append([])
            fallos += 1
    return dict(x=np.array(X), y=np.array(Y), cascos=cascos, fallos=fallos)


# ------------------------------------------------------------------ inventario
def filas_inventario_cham(parcela):
    """Filas del CSV de Chamrousse aceptadas por referencia(), en el mismo orden.

    Se copia tal cual el filtro de analisis.referencia: se prueba a convertir x, y,
    height y area, y si algo falla la fila se descarta.
    """
    filas = []
    for r in csv.DictReader(open(CSV_CHAM)):
        if r["plotid"] != parcela:
            continue
        try:
            float(r["x"]); float(r["y"]); float(r["height"]); float(r["area"])
        except ValueError:
            continue
        filas.append(r)
    return filas


def num(valor):
    """Convierte a float positivo, o devuelve None si falta, es cero o no es numero."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def poligono_cuatro_radios(N, S, E, O, n=N_ANG):
    """Cuatro cuartos de elipse pegados, cada uno con los dos semiejes que le tocan.

    Para un angulo t medido desde el este y creciendo hacia el norte:
        r(t) = sqrt( 1 / ( (cos t / a)^2 + (sin t / b)^2 ) )
    con a = E si cos t >= 0 y a = O si no, y b = N si sin t >= 0 y b = S si no.
    Devuelve los vertices en metros, relativos al tronco.
    """
    pts = []
    for k in range(n):
        t = 2.0 * math.pi * k / n
        c, s = math.cos(t), math.sin(t)
        a = E if c >= 0 else O
        b = N if s >= 0 else S
        r = math.sqrt(1.0 / ((c / a) ** 2 + (s / b) ** 2))
        pts.append((r * c, r * s))
    return pts


def poligono_circulo(radio, n=N_ANG):
    """Respaldo cuando faltan radios: circulo del area equivalente."""
    return [(radio * math.cos(2.0 * math.pi * k / n),
             radio * math.sin(2.0 * math.pi * k / n)) for k in range(n)]


def copas_inventario(parcela):
    """Poligonos de copa del inventario de Chamrousse, alineados con referencia()."""
    filas = filas_inventario_cham(parcela)
    polis, areas_csv, respaldos = [], [], 0
    for r in filas:
        N, S, E, O = (num(r.get(c)) for c in ("N", "S", "E", "O"))
        area = float(r["area"])
        if None in (N, S, E, O):
            radio = math.sqrt(max(area, 0.0) / math.pi)
            pts = poligono_circulo(radio)
            respaldos += 1
        else:
            pts = poligono_cuatro_radios(N, S, E, O)
        polis.append(a_decimetros(pts, 0.0, 0.0))
        areas_csv.append(area)
    return polis, areas_csv, respaldos


# ------------------------------------------------------------------ principal
def construye(maxv):
    """Genera el diccionario completo con el limite de vertices dado."""
    salida = {}
    aux = {}      # datos sueltos para las comprobaciones
    for sitio, nombre, parcelas in SITIOS:
        for p in parcelas:
            clave = f"{sitio}/{p}"
            ref = referencia(sitio, p)
            pat = region(ref)

            sat = cascos_sat(sitio, p, maxv)
            if sat is None or len(sat["x"]) == 0:
                dentro = np.zeros(0, bool)
                cascos = []
                fallos = 0
            else:
                dentro = pat.contains_points(np.c_[sat["x"], sat["y"]])
                cascos = sat["cascos"]
                fallos = sat["fallos"]

            sat_dentro = [cascos[i] for i in range(len(cascos)) if dentro[i]]
            sat_fuera = [cascos[i] for i in range(len(cascos)) if not dentro[i]]

            if sitio == "cham":
                inv, areas_csv, respaldos = copas_inventario(p)
            else:
                inv, areas_csv, respaldos = [], [], 0

            salida[clave] = {"sat_dentro": sat_dentro,
                             "sat_fuera": sat_fuera,
                             "inv": inv}
            aux[clave] = {"n_ref": int(len(ref["x"])), "areas_csv": areas_csv,
                          "respaldos": respaldos, "fallos_casco": fallos}
            print(f"  {clave}: sat dentro={len(sat_dentro)} fuera={len(sat_fuera)} "
                  f"inv={len(inv)} (NR={len(ref['x'])})", flush=True)
    return salida, aux


def guarda(salida):
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    json.dump(salida, open(SALIDA, "w"), separators=(",", ":"))
    return os.path.getsize(SALIDA)


def main():
    maxv = MAX_VERT
    print(f"Construyendo copas con maximo {maxv} vertices por casco...", flush=True)
    salida, aux = construye(maxv)
    tam = guarda(salida)
    print(f"\nguardado en {SALIDA} ({tam} bytes, {tam/1e6:.2f} MB)")

    if tam > LIMITE_MB * 1e6:
        maxv = 8
        print(f"\npasa de {LIMITE_MB} MB, se repite con maximo {maxv} vertices", flush=True)
        salida, aux = construye(maxv)
        tam = guarda(salida)
        print(f"\nguardado en {SALIDA} ({tam} bytes, {tam/1e6:.2f} MB)")

    # ------------------------------------------------------------ comprobaciones
    arb = json.load(open(ARBOLES))
    print("\n" + "=" * 78)
    print("COMPROBACIONES")

    # 1. las 22 claves
    faltan = [k for k in arb if k not in salida]
    sobran = [k for k in salida if k not in arb]
    print(f"\n1. claves: copas.json tiene {len(salida)}, arboles.json tiene {len(arb)}. "
          f"faltan={faltan or 'ninguna'} sobran={sobran or 'ninguna'}")

    # 2. longitudes frente a arboles.json
    print("\n2. longitudes de sat_dentro y sat_fuera frente a arboles.json")
    malas = []
    for k in sorted(salida):
        nd_esp = len(arb[k]["det"]["SAT"]["x"])
        nf_esp = len(arb[k]["det_fuera"]["SAT"]["x"])
        nd, nf = len(salida[k]["sat_dentro"]), len(salida[k]["sat_fuera"])
        if nd != nd_esp or nf != nf_esp:
            malas.append((k, nd, nd_esp, nf, nf_esp))
    if malas:
        for k, nd, nde, nf, nfe in malas:
            print(f"   DISCREPA {k}: dentro {nd} vs {nde}, fuera {nf} vs {nfe}")
    else:
        tot_d = sum(len(salida[k]["sat_dentro"]) for k in salida)
        tot_f = sum(len(salida[k]["sat_fuera"]) for k in salida)
        print(f"   las 22 parcelas coinciden. total dentro={tot_d}, fuera={tot_f}")

    # 3. el casco rodea a su deteccion
    print("\n3. la deteccion cae dentro del casco o a menos de 2 m (0,0 en coordenadas del casco)")
    muestra = ["cham/1", "cham/3", "pell/p1", "pell/p7", "cham/Premol"]
    glob_ok = glob_n = 0
    for k in sorted(salida):
        ok = n = 0
        for poli in salida[k]["sat_dentro"] + salida[k]["sat_fuera"]:
            if len(poli) < 3:
                continue
            n += 1
            ok += int(dist_al_poligono(poli) <= 2.0)
        glob_ok += ok
        glob_n += n
        if k in muestra:
            print(f"   {k:12} {ok}/{n} = {100.0*ok/max(1,n):.1f} %")
    print(f"   TODAS las parcelas: {glob_ok}/{glob_n} = {100.0*glob_ok/max(1,glob_n):.2f} %")

    # 4. inventario de Chamrousse: longitudes y area
    print("\n4. inventario de Chamrousse: longitud y area del poligono de cuatro radios")
    ax, ay, respaldos, malas_inv = [], [], 0, []
    for k in sorted(salida):
        if k.startswith("pell/"):
            if salida[k]["inv"]:
                malas_inv.append((k, len(salida[k]["inv"]), 0))
            continue
        n_esp = len(arb[k]["inv"]["x"])
        n = len(salida[k]["inv"])
        if n != n_esp:
            malas_inv.append((k, n, n_esp))
        respaldos += aux[k]["respaldos"]
        for poli, a_csv in zip(salida[k]["inv"], aux[k]["areas_csv"]):
            ax.append(a_csv)
            ay.append(area_poligono(poli))
        print(f"   {k:12} inv={n} (arboles.json {n_esp}) respaldos={aux[k]['respaldos']}")
    print(f"   discrepancias de longitud: {malas_inv or 'ninguna'}")
    print(f"   veces que se cayo al circulo equivalente: {respaldos} de {len(ax)}")

    ax_, ay_ = np.array(ax), np.array(ay)
    ss_res = float(((ay_ - ax_) ** 2).sum())
    ss_tot = float(((ax_ - ax_.mean()) ** 2).sum())
    r2_ident = 1.0 - ss_res / ss_tot                      # R2 frente a la recta y = x
    r_pear = float(np.corrcoef(ax_, ay_)[0, 1])
    err_rel = float(np.mean(np.abs(ay_ - ax_) / np.maximum(ax_, 1e-9)))
    print(f"   n={len(ax_)}  R2 frente a y=x: {r2_ident:.4f}  "
          f"R2 de correlacion: {r_pear**2:.4f}")
    print(f"   error relativo medio: {100*err_rel:.2f} %  "
          f"sesgo medio: {100*float(np.mean((ay_-ax_)/np.maximum(ax_,1e-9))):.2f} %")
    print(f"   area media: inventario {ax_.mean():.2f} m2, poligono {ay_.mean():.2f} m2")

    # 5. tamano
    print(f"\n5. tamano del JSON: {tam} bytes = {tam/1e6:.2f} MB "
          f"(limite {LIMITE_MB} MB, maximo de vertices usado {maxv})")


if __name__ == "__main__":
    main()
