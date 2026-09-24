"""Indice de Jaccard al estilo del articulo de AMS3D, con sus tres variantes.

Tusa reporta en su Tabla III una columna J_i junto a exhaustividad, precision y F1. Al revisar
su codigo aparecieron TRES definiciones distintas de esa J_i, todas con interseccion real de
poligonos pero con un factor vertical diferente:

  J_2D    = a_i / (a_r + a_d - a_i)                                  (pyc de diciembre de 2019)
  J_vol_h = a_i*min(h_r,h_d) / (a_r*h_r + a_d*h_d - a_i*min(h))      (pyc de enero de 2020; es
            tambien lo que describe el texto del articulo: "volumen a partir del area de la
            elipse y la altura del arbol")
  J_vol_d = igual pero con PROFUNDIDAD de copa en vez de altura      (version final, cuyo codigo
            no se conserva; reproduce las salidas guardadas de Chamrousse)

Ademas, el J_i publicado (53.7 %) NO se reproduce desde las salidas guardadas del propio Tusa:
esas 848 parejas dan 0.438 de media. Por eso aqui se calculan las tres variantes y se reporta
tambien la referencia recalculada, no la publicada.

Poligonos: el de referencia es el real del inventario. Para las detecciones, SegmentAnyTree
aporta el casco convexo real de cada instancia; AMS3D y el watershed solo publican area, asi que
se usa un circulo de area equivalente. El error de esa aproximacion se mide con SegmentAnyTree,
que permite calcular las dos cosas.

Uso: python3 respaldo_minimo/modal_app/jaccard.py
"""
import csv
import json
import os
import sys

import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analisis import (SITIOS, BASE, RAIZ, AMS, SAL, MIN_PTS, region, area_hull,
                      referencia, det_watershed, emparejar)


def det_ams(sitio, parcela, cfgs):
    """Como el de analisis.py pero cargando tambien la profundidad de copa."""
    c = cfgs.get((sitio, parcela))
    if c is None:
        return None
    f = os.path.join(AMS, f"e1_dets_{parcela}_{c}.csv")
    if not os.path.exists(f):
        return None
    x, y, h, a, d = [], [], [], [], []
    for r in csv.DictReader(open(f)):
        x.append(float(r["x"])); y.append(float(r["y"]))
        h.append(float(r["h"])); a.append(float(r["area"]))
        d.append(float(r.get("depth", "nan") or "nan"))
    return dict(x=np.array(x), y=np.array(y), h=np.array(h),
                area=np.array(a), depth=np.array(d))

import laspy
from scipy.spatial import ConvexHull, QhullError

DM = 3.5


def _pol(v):
    if len(v) < 3:
        return None
    p = Polygon(v)
    if not p.is_valid:
        p = p.buffer(0)
    return p if (p.is_valid and p.area > 1e-6) else None


def poligonos_ref(sitio, parcela, ids):
    """Poligonos de copa del inventario, indexados por id de arbol."""
    if sitio == "cham":
        f = os.path.join(BASE, f"IRSTEA_dataset/inventory_csv/placette_{parcela}_houppiers_coords.csv")
    else:
        f = os.path.join(BASE, f"trento_dataset/csv/{parcela}.csv")
    if not os.path.exists(f):
        return {}
    porid = {}
    for r in csv.DictReader(open(f)):
        porid.setdefault(str(r["id"]), []).append((float(r["x"]), float(r["y"])))
    return {i: _pol(np.array(porid[i])) for i in ids if i in porid}


def ref_extendida(sitio, parcela):
    """Referencia con id, profundidad de copa y poligono real."""
    ref = referencia(sitio, parcela)
    if sitio == "cham":
        ruta = os.path.join(BASE, "IRSTEA_dataset/inventory_csv/Chamrousse.csv")
    else:
        ruta = os.path.join(BASE, "trento_dataset/csv/Pellizzano.csv")
    ids, prof, cx, cy = [], [], [], []
    for r in csv.DictReader(open(ruta)):
        if r["plotid"] != parcela:
            continue
        try:
            float(r["x"]); float(r["y"]); float(r["height"]); float(r["area"])
        except ValueError:
            continue
        ids.append(str(r["id"]))
        try:
            d = float(r.get("depth", "") or "nan")
        except ValueError:
            d = float("nan")
        prof.append(d)
    ref["id"] = np.array(ids)
    ref["depth"] = np.array(prof, float)
    pols = poligonos_ref(sitio, parcela, set(ids))
    ref["pol"] = [pols.get(i) for i in ids]
    # centroide del poligono, para la sensibilidad tronco frente a centroide de copa
    ref["cx"] = np.array([p.centroid.x if p else x for p, x in zip(ref["pol"], ref["x"])])
    ref["cy"] = np.array([p.centroid.y if p else y for p, y in zip(ref["pol"], ref["y"])])
    return ref


def det_sat_pol(sitio, parcela):
    """Detecciones de SegmentAnyTree con casco convexo real y circulo equivalente."""
    laz = os.path.join(SAL, f"{sitio}_{parcela}_out.laz")
    if not os.path.exists(laz):
        return None
    L = laspy.read(laz)
    x, y, z = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    inst = np.asarray(L.PredInstance).astype(np.int64)
    X, Y, H, A, POL, DEP = [], [], [], [], [], []
    for i in np.unique(inst[inst > 0]):
        m = inst == i
        if m.sum() < MIN_PTS:
            continue
        xy = np.c_[x[m], y[m]]
        k = np.argmax(z[m])
        X.append(x[m][k]); Y.append(y[m][k])
        H.append(float(z[m].max())); DEP.append(float(z[m].max() - z[m].min()))
        A.append(area_hull(xy))
        try:
            POL.append(_pol(xy[ConvexHull(xy).vertices]))
        except (QhullError, Exception):
            POL.append(None)
    return dict(x=np.array(X), y=np.array(Y), h=np.array(H), area=np.array(A),
                depth=np.array(DEP), pol=POL)


def circulo(x, y, a):
    return Point(x, y).buffer(max(np.sqrt(max(a, 1e-6) / np.pi), 0.05), quad_segs=16)


def jaccards(pr, pd_, ar, ad, hr, hd, dr, dd):
    """Las tres variantes de Tusa. Devuelve (J_2D, J_vol_altura, J_vol_profundidad)."""
    if pr is None or pd_ is None:
        return None
    try:
        ai = pr.intersection(pd_).area
    except Exception:
        return None
    hay_prof = np.isfinite(dr) and dr > 0 and np.isfinite(dd) and dd > 0
    if ai <= 0:
        return (0.0, 0.0, 0.0 if hay_prof else float("nan"))
    j2 = ai / (ar + ad - ai) if (ar + ad - ai) > 0 else 0.0
    mh = min(hr, hd)
    den_h = ar * hr + ad * hd - ai * mh
    jh = (ai * mh) / den_h if den_h > 0 else 0.0
    if hay_prof:
        md = min(dr, dd)
        den_d = ar * dr + ad * dd - ai * md
        jd = (ai * md) / den_d if den_d > 0 else 0.0
    else:
        jd = float("nan")
    return (j2, jh, jd)


def evaluar(ref, det, usar_pol_real, pos_ref="tronco"):
    """Empareja y devuelve las J por pareja, ademas de las areas para el ajuste."""
    rx = ref["x"] if pos_ref == "tronco" else ref["cx"]
    ry = ref["y"] if pos_ref == "tronco" else ref["cy"]
    refm = dict(x=rx, y=ry, h=ref["h"], area=ref["area"])
    dentro = region(dict(x=ref["x"], y=ref["y"])).contains_points(np.c_[det["x"], det["y"]]) \
        if len(det["x"]) else np.zeros(0, bool)
    idx = np.nonzero(dentro)[0]
    dd = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
          for k, v in det.items()}
    nd, nr = len(dd["x"]), len(refm["x"])
    if nd == 0 or nr == 0:
        return [], 0, nd
    D = np.hypot(dd["x"][:, None] - refm["x"][None, :], dd["y"][:, None] - refm["y"][None, :])
    Vd = np.maximum(dd["area"], 1e-6) * np.maximum(dd["h"], 1e-6)
    Vr = np.maximum(refm["area"], 1e-6) * np.maximum(refm["h"], 1e-6)
    JIv = np.minimum(Vd[:, None], Vr[None, :]) / np.maximum(Vd[:, None], Vr[None, :])
    C = np.where(D <= DM, (D / DM) * (1.0 - JIv), np.inf)
    di, rj = np.arange(nd), np.arange(nr)
    pares = []
    while C.size and np.isfinite(C).any():
        i, j = np.unravel_index(np.argmin(C), C.shape)
        pares.append((int(di[i]), int(rj[j])))
        C = np.delete(np.delete(C, i, 0), j, 1)
        di, rj = np.delete(di, i), np.delete(rj, j)

    filas = []
    for i, j in pares:
        pd_ = (dd["pol"][i] if (usar_pol_real and dd.get("pol") and dd["pol"][i] is not None)
               else circulo(dd["x"][i], dd["y"][i], dd["area"][i]))
        ad = pd_.area
        jj = jaccards(ref["pol"][j], pd_, ref["area"][j], ad,
                      ref["h"][j], dd["h"][i], ref["depth"][j],
                      dd["depth"][i] if "depth" in dd else float("nan"))
        if jj is None:
            continue
        filas.append(dict(j2=jj[0], jh=jj[1], jd=jj[2],
                          area_ref=float(ref["area"][j]), area_det=float(ad)))
    return filas, len(pares), nd


def r2_rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), float("nan")
    a, b = a[m], b[m]
    ss_res = float(((a - b) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), float(np.sqrt(ss_res / len(a)))


def main():
    cfgs = {}
    for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        p = os.path.join(AMS, f)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cfgs[(s, r["plot"])] = r["cfg"]

    # Configuracion del watershed elegida por parcela en el barrido, para que los
    # tres metodos se comparen en su mejor ajuste y no uno de ellos por defecto.
    wcfg = {}
    pb = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "watershed_barrido.json")
    if os.path.exists(pb):
        import json as _j
        _b = _j.load(open(pb))
        for _sit, _d in _b.items():
            for _pl, _c in _d.get("configs", {}).items():
                wcfg[(_sit, _pl)] = (float(_c["hmin"]), float(_c["mindist"]))
    print(f"configuraciones de watershed cargadas: {len(wcfg)}", flush=True)

    print("Calculando (tarda unos minutos)...", flush=True)
    out = {}
    for sitio, nombre, parcelas in SITIOS:
        acum = {}
        for p in parcelas:
            ref = ref_extendida(sitio, p)
            dets = {"SAT": det_sat_pol(sitio, p),
                    "AMS3D": det_ams(sitio, p, cfgs),
                    "Watershed": det_watershed(ref, hmin=wcfg.get((nombre, p), (2.0, 2.0))[0],
                                               mindist=wcfg.get((nombre, p), (2.0, 2.0))[1])}
            for met, d in dets.items():
                if d is None:
                    continue
                real = (met == "SAT")
                filas, tp, nd = evaluar(ref, d, usar_pol_real=real)
                acum.setdefault(met, []).extend(filas)
                if met == "SAT":     # mismo emparejamiento, poligono aproximado por circulo
                    f2, _, _ = evaluar(ref, d, usar_pol_real=False)
                    acum.setdefault("SAT_circulo", []).extend(f2)
        out[nombre] = {}
        print(f"\n{'='*80}\n{nombre}: indice de Jaccard medio sobre las parejas emparejadas (d=3.5 m)")
        print(f"{'metodo':14} {'parejas':>8} {'J_2D':>8} {'J_vol_h':>9} {'J_vol_d':>9} "
              f"{'area R2':>8} {'RMSE m2':>8} {'%sin sol':>8}")
        for met in ("SAT", "SAT_circulo", "AMS3D", "Watershed"):
            f = acum.get(met, [])
            if not f:
                continue
            j2 = np.nanmean([r["j2"] for r in f])
            jh = np.nanmean([r["jh"] for r in f])
            jd_v = [r["jd"] for r in f if np.isfinite(r["jd"])]
            jd = np.mean(jd_v) if jd_v else float("nan")
            r2, rmse = r2_rmse([r["area_ref"] for r in f], [r["area_det"] for r in f])
            sin_solape = round(100.0 * float(np.mean([r["j2"] <= 0 for r in f])), 1)
            out[nombre][met] = dict(parejas=len(f), J_2D=round(float(j2), 3),
                                    J_vol_altura=round(float(jh), 3),
                                    J_vol_profundidad=(round(float(jd), 3) if jd_v else None),
                                    area_R2=round(r2, 3), area_RMSE=round(rmse, 2),
                                    pct_sin_solape=sin_solape)
            print(f"{met:14} {len(f):>8} {j2:>8.3f} {jh:>9.3f} "
                  f"{(f'{jd:.3f}' if jd_v else '-'):>9} {r2:>8.3f} {rmse:>8.2f} "
                  f"{sin_solape:>8.1f}")

    # sensibilidad: posicion de referencia tronco frente a centroide de la copa
    print(f"\n{'='*80}\nSENSIBILIDAD: posicion de referencia (tronco frente a centroide del poligono)")
    print(f"{'sitio':12} {'metodo':10} {'pos_ref':10} {'parejas':>8} {'J_vol_h':>9}")
    for sitio, nombre, parcelas in SITIOS:
        for met in ("SAT", "AMS3D"):
            for pos in ("tronco", "centroide"):
                tot, js = 0, []
                for p in parcelas:
                    ref = ref_extendida(sitio, p)
                    d = det_sat_pol(sitio, p) if met == "SAT" else det_ams(sitio, p, cfgs)
                    if d is None:
                        continue
                    filas, tp, _ = evaluar(ref, d, usar_pol_real=(met == "SAT"), pos_ref=pos)
                    tot += tp
                    js += [r["jh"] for r in filas]
                out.setdefault("sensibilidad_pos_ref", {}).setdefault(nombre, {}) \
                    .setdefault(met, {})[pos] = dict(parejas=tot,
                                                     J_vol_altura=round(float(np.nanmean(js)), 3) if js else None)
                print(f"{nombre:12} {met:10} {pos:10} {tot:>8} "
                      f"{(np.nanmean(js) if js else float('nan')):>9.3f}")

    json.dump(out, open(os.path.join(RAIZ, "jaccard.json"), "w"), indent=2, ensure_ascii=False)
    print("\nguardado en jaccard.json")


if __name__ == "__main__":
    main()
