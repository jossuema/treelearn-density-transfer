"""Watershed parameter sweep, so it is not compared at a disadvantage.

AMS3D runs with its best setting out of 25 chosen PER PLOT. For the comparison to be fair,
the watershed must also run with its best setting. Minimum height and minimum distance
between apexes are swept, and the best is picked by F1 per plot, just as AMS3D does.

Usage: python3 modal/watershed_sweep.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation_core import (SITES, DMATCH, reference, det_watershed, evaluate_plot,
                      metrics, bootstrap, ROOT)

HMIN = (1.0, 2.0, 3.0, 5.0)
MINDIST = (1.0, 1.5, 2.0, 3.0)


def main():
    result = {}
    for site, name, plots in SITES:
        print(f"\n{name}: sweeping {len(HMIN)*len(MINDIST)} settings over {len(plots)} plots",
              flush=True)
        best_per_plot = {}
        for p in plots:
            ref = reference(site, p)
            best = None
            for hmin in HMIN:
                for md in MINDIST:
                    det = det_watershed(ref, hmin=hmin, mindist=md)
                    e = evaluate_plot(ref, det, 3.5, True)
                    _, _, f1 = metrics(e["TP"], e["NR"], e["ND"])
                    if best is None or f1 > best[0]:
                        best = (f1, hmin, md, ref, det)
            f1, hmin, md, ref, det = best
            best_per_plot[p] = dict(ref=ref, det=det, hmin=hmin, mindist=md, f1=round(f1, 3))
            print(f"  {p:7} best hmin={hmin} mindist={md}  F1={f1:.3f}", flush=True)

        print(f"\n{name}, watershed with its best setting per plot:")
        print(f"{'dm':>4} {'NR':>5} {'ND':>5} {'TP':>5} {'recall':>8} {'95% CI':>16} {'precis':>7} {'F1':>6}")
        result[name] = {}
        for dm in DMATCH:
            pp = [evaluate_plot(m["ref"], m["det"], dm, True) for m in best_per_plot.values()]
            TP, NR, ND = (sum(e[k] for e in pp) for k in ("TP", "NR", "ND"))
            R, P, F = metrics(TP, NR, ND)
            ci = bootstrap(pp)
            result[name][str(dm)] = dict(NR=NR, ND=ND, TP=TP, recall=round(R, 3),
                                              precision=round(P, 3), f1=round(F, 3), **ci)
            print(f"{dm:>4} {NR:>5} {ND:>5} {TP:>5} {R:>8.3f} {str(ci.get('recall_ci','')):>16} "
                  f"{P:>7.3f} {F:>6.3f}")

        # strata with the best setting
        acc = {e: {"n": 0, "TP": 0} for e in ("dominant", "intermediate", "suppressed")}
        for m in best_per_plot.values():
            ref = m["ref"]
            cas = evaluate_plot(ref, m["det"], 3.5, True)["matched"]
            for e in acc:
                acc[e]["n"] += int((ref["stratum"] == e).sum())
                acc[e]["TP"] += int((ref["stratum"][cas] == e).sum()) if len(cas) else 0
        result[name]["strata"] = acc
        print("  strata:", " ".join(
            f"{e} {acc[e]['TP']/acc[e]['n']:.3f}" for e in acc if acc[e]["n"]))
        result[name]["configs"] = {p: {"hmin": m["hmin"], "mindist": m["mindist"]}
                                        for p, m in best_per_plot.items()}

    json.dump(result, open(os.path.join(ROOT, "watershed_sweep.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\nsaved to watershed_sweep.json")


if __name__ == "__main__":
    main()
