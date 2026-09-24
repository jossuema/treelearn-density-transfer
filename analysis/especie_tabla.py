#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tabla base por especie: exhaustividad de SAT, AMS3D y watershed a 3.5 m.

Lo nuevo respecto de especie.py es el puente entre sitios. Chamrousse nombra las
especies con codigos de cuatro letras y Pellizzano con nombres en ingles, pero
varias son la misma especie. Al traducir los dos a nombre cientifico se puede
mirar la misma especie en dos sitios con densidades de vuelo que se doblan.

Unidad de remuestreo: la parcela. Bootstrap de 2000 replicas con
numpy.random.default_rng(0), semilla reiniciada en cada agregado.
Las comparaciones entre metodos son emparejadas: dentro de cada replica se
calcula la diferencia sobre los mismos arboles, no dos intervalos por separado.
Para los agregados que juntan los dos sitios el remuestreo es estratificado por
sitio, para no perder la mezcla de parcelas de cada uno.

Lee    respaldo_minimo/visor/data/arboles.json
Escribe respaldo_minimo/analisis_paper/especie_tabla.json
No modifica ningun archivo existente.
"""

import json
import os
import collections
import datetime

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
FUENTE = os.path.join(RAIZ, "visor", "data", "arboles.json")
SALIDA = os.path.join(AQUI, "especie_tabla.json")

METODOS = ["SAT", "AMS3D", "WS"]
DIST = "3.5"
NBOOT = 2000
MIN_SEPARADA = 30      # por debajo de esto la especie no va sola en la tabla
MIN_CONCLUYENTE = 30   # por debajo de esto el intervalo se marca no concluyente
MIN_CONTEO = 10        # por debajo de esto solo se da el conteo

NOMBRE_SITIO = {"cham": "Chamrousse", "pell": "Pellizzano"}

# codigo del inventario -> (nombre cientifico, nombre comun, hoja)
# hoja: conifera perenne, conifera caducifolia, frondosa
ESPECIES = {
    # Chamrousse, codigos de cuatro letras
    "PIAB": ("Picea abies", "picea comun", "conifera"),
    "ABAL": ("Abies alba", "abeto blanco", "conifera"),
    "FASY": ("Fagus sylvatica", "haya", "frondosa"),
    "PIUN": ("Pinus uncinata", "pino negro", "conifera"),
    "BEPE": ("Betula pendula", "abedul", "frondosa"),
    "ACPS": ("Acer pseudoplatanus", "arce blanco", "frondosa"),
    "SOAR": ("Sorbus aria", "mostajo", "frondosa"),
    "SOAU": ("Sorbus aucuparia", "serbal de cazadores", "frondosa"),
    "COAV": ("Corylus avellana", "avellano", "frondosa"),
    "POTR": ("Populus tremula", "chopo temblon", "frondosa"),
    "BEsp": ("Betula sp.", "abedul sin determinar", "frondosa"),
    "FREX": ("Fraxinus excelsior", "fresno", "frondosa"),
    "PICE": ("Picea sp.", "picea sin determinar", "conifera"),
    "ND": ("sin determinar", "sin determinar", "sin determinar"),
    # Pellizzano, nombres en ingles
    "Norway_spruce": ("Picea abies", "picea comun", "conifera"),
    "Silver_fir": ("Abies alba", "abeto blanco", "conifera"),
    "European_beech": ("Fagus sylvatica", "haya", "frondosa"),
    "Larch": ("Larix decidua", "alerce", "conifera caducifolia"),
    "Silver_birch": ("Betula pendula", "abedul", "frondosa"),
    "Sycamore_maple": ("Acer pseudoplatanus", "arce blanco", "frondosa"),
    "Rowan": ("Sorbus aucuparia", "serbal de cazadores", "frondosa"),
    "Common_hazel": ("Corylus avellana", "avellano", "frondosa"),
    "Common_alder": ("Alnus sp.", "aliso", "frondosa"),
    "Aspen": ("Populus tremula", "chopo temblon", "frondosa"),
    "Willows": ("Salix sp.", "sauce", "frondosa"),
    "European_barberry": ("Berberis vulgaris", "agracejo", "frondosa"),
}

# contador global de todas las comparaciones que se miran
CONTADOR = {"intervalos_exhaustividad": 0, "intervalos_diferencia": 0}


# ---------------------------------------------------------------------------
# carga
# ---------------------------------------------------------------------------
def cargar():
    """Devuelve una lista de filas, una por arbol de inventario."""
    with open(FUENTE) as f:
        datos = json.load(f)

    filas = []
    for clave, blo in datos.items():
        sitio, _ = clave.split("/")
        inv = blo["inv"]
        n = len(inv["x"])
        det = {}
        for m in METODOS:
            det[m] = set(int(p[1]) for p in blo["pares"][m][DIST])
        for i in range(n):
            cod = inv["specie"][i]
            cien, comun, hoja = ESPECIES.get(cod, (cod, cod, "sin determinar"))
            filas.append({
                "sitio": sitio,
                "parcela": clave,
                "codigo": cod,
                "cientifico": cien,
                "comun": comun,
                "hoja": hoja,
                "estrato": inv["estrato"][i],
                "npts": inv["npts"][i],
                "h": inv["h"][i],
                "det": {m: (i in det[m]) for m in METODOS},
            })
    return filas


# ---------------------------------------------------------------------------
# bootstrap de parcelas
# ---------------------------------------------------------------------------
def por_parcela(filas, metodo):
    """Agrupa: parcela -> (aciertos, total) para un metodo."""
    acc = collections.defaultdict(lambda: [0, 0])
    for f in filas:
        a = acc[f["parcela"]]
        a[0] += 1 if f["det"][metodo] else 0
        a[1] += 1
    return acc


def estratos_de(filas):
    """parcelas agrupadas por sitio, en orden estable."""
    d = collections.defaultdict(list)
    for f in filas:
        if f["parcela"] not in d[f["sitio"]]:
            d[f["sitio"]].append(f["parcela"])
    return d


def replicas_indices(filas, nboot=NBOOT, semilla=0):
    """
    Matriz nboot x k con los indices de parcela remuestreados.
    Si el grupo toca dos sitios el remuestreo es estratificado por sitio:
    dentro de cada sitio se sortean tantas parcelas como tenia.
    Devuelve tambien la lista de parcelas en el orden de las columnas.
    """
    rng = np.random.default_rng(semilla)
    estr = estratos_de(filas)
    parcelas = []
    for s in sorted(estr):
        parcelas.extend(estr[s])
    pos = {p: j for j, p in enumerate(parcelas)}
    bloques = []
    for s in sorted(estr):
        idx = np.array([pos[p] for p in estr[s]])
        bloques.append(rng.choice(idx, size=(nboot, len(idx)), replace=True))
    return np.hstack(bloques), parcelas


def ic_exhaustividad(filas, metodo, nboot=NBOOT):
    """Exhaustividad puntual y percentiles 2.5 y 97.5 del bootstrap de parcelas."""
    CONTADOR["intervalos_exhaustividad"] += 1
    acc = por_parcela(filas, metodo)
    reps, parcelas = replicas_indices(filas, nboot)
    aciertos = np.array([acc[p][0] for p in parcelas], dtype=float)
    totales = np.array([acc[p][1] for p in parcelas], dtype=float)
    punto = aciertos.sum() / totales.sum() if totales.sum() else float("nan")
    a = aciertos[reps].sum(axis=1)
    t = totales[reps].sum(axis=1)
    ok = t > 0
    v = a[ok] / t[ok]
    return {
        "recall": round(float(punto), 4),
        "ic_bajo": round(float(np.percentile(v, 2.5)), 4),
        "ic_alto": round(float(np.percentile(v, 97.5)), 4),
        "aciertos": int(aciertos.sum()),
        "n": int(totales.sum()),
    }


def ic_diferencia(filas, ma, mb, nboot=NBOOT):
    """
    Diferencia emparejada de exhaustividad ma menos mb sobre los MISMOS arboles.
    Se remuestrea la diferencia, no dos intervalos sueltos.
    """
    CONTADOR["intervalos_diferencia"] += 1
    dif = collections.defaultdict(lambda: [0.0, 0.0])
    for f in filas:
        d = dif[f["parcela"]]
        d[0] += (1 if f["det"][ma] else 0) - (1 if f["det"][mb] else 0)
        d[1] += 1
    reps, parcelas = replicas_indices(filas, nboot)
    num = np.array([dif[p][0] for p in parcelas], dtype=float)
    den = np.array([dif[p][1] for p in parcelas], dtype=float)
    punto = num.sum() / den.sum() if den.sum() else float("nan")
    a = num[reps].sum(axis=1)
    t = den[reps].sum(axis=1)
    ok = t > 0
    v = a[ok] / t[ok]
    lo = float(np.percentile(v, 2.5))
    hi = float(np.percentile(v, 97.5))
    return {
        "dif": round(float(punto), 4),
        "ic_bajo": round(lo, 4),
        "ic_alto": round(hi, 4),
        "cruza_cero": bool(lo <= 0.0 <= hi),
        "n": int(den.sum()),
    }


# ---------------------------------------------------------------------------
# descripcion del reparto por parcelas
# ---------------------------------------------------------------------------
def reparto(filas):
    c = collections.Counter(f["parcela"] for f in filas)
    n = sum(c.values())
    dom, ndom = c.most_common(1)[0]
    est = collections.Counter(f["estrato"] for f in filas)
    if len(c) == 1:
        aviso = ("una sola parcela: con bootstrap de parcelas el intervalo "
                 "sale de ancho cero y no informa de nada")
    elif len(c) == 2:
        aviso = "dos parcelas: el intervalo es casi ciego"
    else:
        aviso = ""
    return {
        "n": n,
        "n_parcelas": len(c),
        "parcela_dominante": dom,
        "frac_parcela_dominante": round(ndom / n, 3),
        "por_parcela": dict(sorted(c.items())),
        "estrato": {k: est.get(k, 0) for k in
                    ["dominante", "intermedio", "suprimido"]},
        "frac_suprimido": round(est.get("suprimido", 0) / n, 3),
        "aviso_parcelas": aviso,
    }


def marca(n):
    if n < MIN_CONTEO:
        return "solo_conteo"
    if n < MIN_CONCLUYENTE:
        return "no_concluyente"
    return "ok"


def bloque_grupo(filas, etiqueta, extra=None):
    """Fila completa de la tabla: reparto, exhaustividad de los tres metodos."""
    r = reparto(filas)
    fila = {"grupo": etiqueta}
    fila.update(r)
    fila["marca"] = marca(r["n"])
    if extra:
        fila.update(extra)
    if r["n"] >= MIN_CONTEO:
        fila["exhaustividad"] = {m: ic_exhaustividad(filas, m) for m in METODOS}
    else:
        fila["exhaustividad"] = {
            m: {"recall": round(sum(1 for f in filas if f["det"][m]) / r["n"], 4),
                "aciertos": sum(1 for f in filas if f["det"][m]),
                "n": r["n"], "ic_bajo": None, "ic_alto": None}
            for m in METODOS
        }
    return fila


def bloque_diferencias(filas, etiqueta):
    r = reparto(filas)
    if r["n"] < MIN_CONTEO:
        return None
    return {
        "grupo": etiqueta,
        "n": r["n"],
        "n_parcelas": r["n_parcelas"],
        "frac_parcela_dominante": r["frac_parcela_dominante"],
        "aviso_parcelas": r["aviso_parcelas"],
        "marca": marca(r["n"]),
        "SAT_menos_AMS3D": ic_diferencia(filas, "SAT", "AMS3D"),
        "SAT_menos_WS": ic_diferencia(filas, "SAT", "WS"),
        "AMS3D_menos_WS": ic_diferencia(filas, "AMS3D", "WS"),
    }


# ---------------------------------------------------------------------------
def main():
    filas = cargar()
    salida = {
        "generado": datetime.datetime.now().isoformat(timespec="seconds"),
        "fuente": FUENTE,
        "distancia_m": float(DIST),
        "nboot": NBOOT,
        "semilla": 0,
        "unidad_remuestreo": "parcela",
        "nota_bootstrap": ("remuestreo estratificado por sitio cuando el grupo "
                           "junta los dos sitios; percentiles 2.5 y 97.5"),
    }

    # -- 1. por especie y sitio ---------------------------------------------
    tabla1 = []
    resumen_otras = {}
    for sitio in ["cham", "pell"]:
        del_sitio = [f for f in filas if f["sitio"] == sitio]
        cuenta = collections.Counter(f["codigo"] for f in del_sitio)
        grandes = sorted([c for c, n in cuenta.items() if n >= MIN_SEPARADA],
                         key=lambda c: -cuenta[c])
        chicas = [c for c in cuenta if c not in grandes]
        for cod in grandes:
            sub = [f for f in del_sitio if f["codigo"] == cod]
            tabla1.append(bloque_grupo(
                sub, "%s/%s" % (sitio, cod),
                {"sitio": NOMBRE_SITIO[sitio], "sitio_cod": sitio,
                 "codigo": cod, "cientifico": sub[0]["cientifico"],
                 "comun": sub[0]["comun"], "hoja": sub[0]["hoja"]}))
        sub = [f for f in del_sitio if f["codigo"] in chicas]
        if sub:
            tabla1.append(bloque_grupo(
                sub, "%s/otras" % sitio,
                {"sitio": NOMBRE_SITIO[sitio], "sitio_cod": sitio,
                 "codigo": "otras", "cientifico": "varias", "comun": "varias",
                 "hoja": "varias",
                 "especies_dentro": sorted(chicas),
                 "n_especies_dentro": len(chicas)}))
            resumen_otras[sitio] = {
                "n_arboles": len(sub),
                "n_especies": len(chicas),
                "especies": {c: cuenta[c] for c in sorted(chicas,
                                                          key=lambda x: -cuenta[x])},
            }
        # total del sitio, para tener la referencia
        tabla1.append(bloque_grupo(
            del_sitio, "%s/TODAS" % sitio,
            {"sitio": NOMBRE_SITIO[sitio], "sitio_cod": sitio,
             "codigo": "TODAS", "cientifico": "todas", "comun": "todas",
             "hoja": "todas"}))
    salida["por_especie_y_sitio"] = tabla1
    salida["otras_detalle"] = resumen_otras

    # -- 2. especies presentes en los dos sitios ----------------------------
    presencia = collections.defaultdict(set)
    for f in filas:
        presencia[f["cientifico"]].add(f["sitio"])
    compartidas = sorted([c for c, s in presencia.items()
                          if len(s) == 2 and c != "sin determinar"],
                         key=lambda c: -sum(1 for f in filas
                                            if f["cientifico"] == c))
    tabla2 = []
    for cien in compartidas:
        sub = [f for f in filas if f["cientifico"] == cien]
        fila = bloque_grupo(sub, "ambos/%s" % cien,
                            {"cientifico": cien, "comun": sub[0]["comun"],
                             "hoja": sub[0]["hoja"], "sitio": "los dos",
                             "codigos": sorted(set(f["codigo"] for f in sub))})
        fila["desglose_sitio"] = {}
        for sitio in ["cham", "pell"]:
            ss = [f for f in sub if f["sitio"] == sitio]
            if not ss:
                continue
            fila["desglose_sitio"][sitio] = bloque_grupo(
                ss, "%s/%s" % (sitio, cien), {"sitio": NOMBRE_SITIO[sitio]})
        tabla2.append(fila)
    salida["compartidas"] = tabla2
    salida["especies_compartidas_lista"] = compartidas

    # -- 2b. agregado por tipo de hoja --------------------------------------
    # La lectura por especie apunta a un corte mas simple, asi que se mide
    # tambien de frente. El alerce va aparte porque es conifera que tira la
    # hoja y es el unico caso.
    tabla_hoja = []
    for hoja in ["conifera", "conifera caducifolia", "frondosa"]:
        for ambito in ["ambos", "cham", "pell"]:
            sub = [f for f in filas if f["hoja"] == hoja
                   and (ambito == "ambos" or f["sitio"] == ambito)]
            if not sub:
                continue
            fila = bloque_grupo(sub, "%s/%s" % (ambito, hoja),
                                {"hoja": hoja, "ambito": ambito})
            tabla_hoja.append(fila)
    salida["por_tipo_hoja"] = tabla_hoja

    # -- 3. diferencias emparejadas -----------------------------------------
    difs = []
    for fila in tabla1:
        sub = [f for f in filas
               if f["sitio"] == fila["sitio_cod"]
               and (fila["codigo"] == "TODAS"
                    or (fila["codigo"] == "otras"
                        and f["codigo"] in fila.get("especies_dentro", []))
                    or f["codigo"] == fila["codigo"])]
        b = bloque_diferencias(sub, fila["grupo"])
        if b:
            b["sitio"] = fila["sitio"]
            b["codigo"] = fila["codigo"]
            b["cientifico"] = fila["cientifico"]
            difs.append(b)
    for cien in compartidas:
        sub = [f for f in filas if f["cientifico"] == cien]
        b = bloque_diferencias(sub, "ambos/%s" % cien)
        if b:
            b["sitio"] = "los dos"
            b["codigo"] = "compartida"
            b["cientifico"] = cien
            difs.append(b)
        for sitio in ["cham", "pell"]:
            ss = [f for f in sub if f["sitio"] == sitio]
            b = bloque_diferencias(ss, "%s/%s" % (sitio, cien))
            if b:
                b["sitio"] = NOMBRE_SITIO[sitio]
                b["codigo"] = "compartida"
                b["cientifico"] = cien
                difs.append(b)
    for fila in tabla_hoja:
        sub = [f for f in filas if f["hoja"] == fila["hoja"]
               and (fila["ambito"] == "ambos" or f["sitio"] == fila["ambito"])]
        b = bloque_diferencias(sub, fila["grupo"])
        if b:
            b["sitio"] = fila["ambito"]
            b["codigo"] = "hoja"
            b["cientifico"] = fila["hoja"]
            difs.append(b)
    # los dos sitios juntos, todo
    b = bloque_diferencias(filas, "global/TODAS")
    b["sitio"] = "los dos"
    b["codigo"] = "TODAS"
    b["cientifico"] = "todas"
    difs.append(b)
    salida["diferencias_emparejadas"] = difs

    # -- 4. comprobaciones ---------------------------------------------------
    comp = {}
    for sitio, esperado in [("cham", 894), ("pell", 543)]:
        suma = sum(fila["n"] for fila in tabla1
                   if fila["sitio_cod"] == sitio and fila["codigo"] != "TODAS")
        comp["suma_%s" % sitio] = suma
        comp["cuadra_%s" % sitio] = (suma == esperado)
        comp["esperado_%s" % sitio] = esperado
    comp["total"] = sum(1 for _ in filas)
    comp["cuadra_total"] = (comp["total"] == 1437)
    # ningun arbol sin mapear a nombre cientifico
    sin_mapa = sorted(set(f["codigo"] for f in filas
                          if f["codigo"] not in ESPECIES))
    comp["codigos_sin_mapear"] = sin_mapa
    comp["n_parcelas"] = len(set(f["parcela"] for f in filas))
    salida["comprobaciones"] = comp
    salida["comparaciones_miradas"] = dict(CONTADOR)
    salida["comparaciones_miradas"]["total"] = sum(CONTADOR.values())

    with open(SALIDA, "w") as f:
        json.dump(salida, f, indent=1, ensure_ascii=False)

    # -- impresion legible ---------------------------------------------------
    def linea(fila, sangria=""):
        e = fila["exhaustividad"]
        def cel(m):
            v = e[m]
            if v["ic_bajo"] is None:
                return "%5.3f      (n/a)   " % v["recall"]
            return "%5.3f [%.3f-%.3f]" % (v["recall"], v["ic_bajo"], v["ic_alto"])
        print("%-34s n=%4d par=%2d dom=%.2f sup=%.2f  SAT %s  AMS3D %s  WS %s  %s%s" % (
            sangria + fila["grupo"], fila["n"], fila["n_parcelas"],
            fila["frac_parcela_dominante"], fila["frac_suprimido"],
            cel("SAT"), cel("AMS3D"), cel("WS"), fila["marca"],
            "  OJO" if fila["aviso_parcelas"] else ""))

    print("\n=== 1. exhaustividad por especie y sitio, 3.5 m ===")
    for fila in tabla1:
        linea(fila)
    print("\notras:", json.dumps(resumen_otras, ensure_ascii=False))

    print("\n=== 2. especies en los dos sitios ===")
    for fila in tabla2:
        linea(fila)
        for sitio in ["cham", "pell"]:
            if sitio in fila["desglose_sitio"]:
                linea(fila["desglose_sitio"][sitio], "   ")

    print("\n=== 2b. por tipo de hoja ===")
    for fila in tabla_hoja:
        linea(fila)

    print("\n=== 3. diferencias emparejadas (mismos arboles) ===")
    for b in difs:
        def cd(k):
            v = b[k]
            return "%+.3f [%+.3f,%+.3f]%s" % (
                v["dif"], v["ic_bajo"], v["ic_alto"],
                " ~0" if v["cruza_cero"] else "  *")
        print("%-34s n=%4d  SAT-AMS3D %s  SAT-WS %s  AMS3D-WS %s  %s" % (
            b["grupo"], b["n"], cd("SAT_menos_AMS3D"), cd("SAT_menos_WS"),
            cd("AMS3D_menos_WS"), b["marca"]))

    print("\n=== 4. comprobaciones ===")
    print(json.dumps(comp, ensure_ascii=False, indent=1))
    print("comparaciones miradas:", salida["comparaciones_miradas"])
    print("\nescrito:", SALIDA)


if __name__ == "__main__":
    main()
