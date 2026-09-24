"""Exhaustividad por especie para los cinco metodos, desde detecciones_todas.json.

Las parcelas, no los arboles, son la unidad de remuestreo, porque especie y parcela van
juntas: toda una especie puede vivir en una sola parcela. Se da el numero efectivo de
parcelas, n_eff = 1 / suma(p_i^2), para que se vea cuando una cifra descansa en una sola.

Uso: python3 respaldo_minimo/analisis_paper/especie_cinco.py
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)
RAIZ = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal_app"))
sys.path.insert(0, AQUI)
from analisis import SITIOS  # noqa: E402
from robustez_cinco import MET, contar, ref_de  # noqa: E402
from especie_tabla import ESPECIES  # noqa: E402

NB = 2000
MIN_N = 30


def main():
    datos = json.load(open(os.path.join(RAIZ, "detecciones_todas.json")))
    arb = json.load(open(os.path.join(BASE, "visor", "data", "arboles.json")))
    claves = [f"{s}/{p}" for s, _, ps in SITIOS for p in ps]
    base = {m: contar(datos, claves, m, 3.5) for m in MET}

    # por arbol: especie, parcela y si lo encontro cada metodo
    esp, hoja, parc, hit = [], [], [], {m: [] for m in MET}
    for i, k in enumerate(claves):
        especies = arb[k]["inv"]["specie"]
        n = len(ref_de(k)["x"])
        for j in range(n):
            cod = especies[j] if j < len(especies) else "ND"
            esp.append(ESPECIES.get(cod, ("?", "", "sin determinar"))[0])
            hoja.append(ESPECIES.get(cod, ("?", "", "sin determinar"))[2])
            parc.append(k)
            for m in MET:
                hit[m].append(j in base[m][i]["ok"])
    esp = np.array(esp); parc = np.array(parc); hoja = np.array(hoja)
    hit = {m: np.array(v) for m, v in hit.items()}

    co = np.char.startswith(hoja.astype(str), "conifera")
    fr = hoja == "frondosa"
    grupos = []
    for e in sorted(set(esp)):
        if e in ("?", "sin determinar") or (esp == e).sum() < MIN_N:
            continue
        grupos.append((e, esp == e))
    grupos.append(("Todas las coniferas", co))
    grupos.append(("Todas las frondosas", fr))

    def ic(m, sel, semilla=0):
        """Exhaustividad remuestreando parcelas, no arboles."""
        ks = sorted(set(parc[sel]))
        tp = [int(hit[m][sel & (parc == k)].sum()) for k in ks]
        nn = [int((sel & (parc == k)).sum()) for k in ks]
        rng = np.random.default_rng(semilla); v = []
        for _ in range(NB):
            i = rng.integers(0, len(ks), len(ks))
            a, b = sum(tp[j] for j in i), sum(nn[j] for j in i)
            v.append(a / b if b else 0)
        return (sum(tp) / sum(nn),
                [round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)])

    sal = {}
    print(f"{'especie':24} {'n':>5} {'parc':>5} {'n_eff':>6} " +
          " ".join(f"{m:>11}" for m in MET))
    for nombre, sel in grupos:
        ks = sorted(set(parc[sel]))
        p = np.array([(sel & (parc == k)).sum() for k in ks], float)
        p /= p.sum()
        neff = 1 / np.sum(p ** 2)
        fila = f"{nombre:24} {int(sel.sum()):>5} {len(ks):>5} {neff:>6.1f} "
        sal[nombre] = dict(n=int(sel.sum()), parcelas=len(ks), n_eff=round(float(neff), 1))
        for m in MET:
            r, i95 = ic(m, sel)
            sal[nombre][m] = dict(recall=round(r, 3), ic=i95)
            fila += f"{r:>11.3f} "
        print(fila)

    # la brecha entre frondosas y coniferas, solo en las parcelas que tienen las dos
    print(f"\nbrecha frondosas menos coniferas, en las parcelas con ambas")
    mixtas = [k for k in sorted(set(parc)) if (fr & (parc == k)).sum()
              and (co & (parc == k)).sum()]
    selm = np.isin(parc, mixtas)
    sal["brecha_frondosas"] = {"parcelas": len(mixtas)}
    for m in MET:
        a = hit[m][fr & selm]; b = hit[m][co & selm]
        # por parcela y de una vez: np.isin solo mira pertenencia, asi que una parcela
        # sorteada dos veces contaba una, y el remuestreo no era con reemplazo de verdad
        af = np.array([int(hit[m][fr & (parc == k)].sum()) for k in mixtas])
        nf = np.array([int((fr & (parc == k)).sum()) for k in mixtas])
        ac = np.array([int(hit[m][co & (parc == k)].sum()) for k in mixtas])
        nc = np.array([int((co & (parc == k)).sum()) for k in mixtas])
        rng = np.random.default_rng(0); v = []
        for _ in range(NB):
            i = rng.integers(0, len(mixtas), len(mixtas))
            na = nf[i].sum(); nb = nc[i].sum()
            v.append((af[i].sum() / na if na else 0) - (ac[i].sum() / nb if nb else 0))
        d = a.mean() - b.mean()
        i95 = [round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)]
        sal["brecha_frondosas"][m] = dict(dif=round(float(d), 3), ic=i95,
                                          n_frond=int(fr.sum()), n_conif=int(co.sum()))
        print(f"   {m:12} {d:>+7.3f} {i95}")

    json.dump(sal, open(os.path.join(RAIZ, "especie_cinco.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\nguardado en especie_cinco.json")


if __name__ == "__main__":
    main()
