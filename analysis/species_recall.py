"""Recall by species for the five methods, from all_detections.json.

The plots, not the trees, are the resampling unit, because species and plot go together:
a whole species can live in a single plot. The effective number of plots is given,
n_eff = 1 / sum(p_i^2), so that it shows when a figure rests on one plot alone.

Usage: python3 analysis/species_recall.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
sys.path.insert(0, HERE)
from evaluation_core import SITES  # noqa: E402
from main_evaluation import MET, count, ref_of  # noqa: E402
from species_table import SPECIES  # noqa: E402

NB = 2000
MIN_N = 30


def main():
    data = json.load(open(os.path.join(ROOT, "all_detections.json")))
    arb = json.load(open(os.path.join(BASE, "viewer", "data", "trees.json")))
    keys = [f"{s}/{p}" for s, _, ps in SITES for p in ps]
    base = {m: count(data, keys, m, 3.5) for m in MET}

    # per tree: species, plot and whether each method found it
    sp, leaf, plot_of, hit = [], [], [], {m: [] for m in MET}
    for i, k in enumerate(keys):
        species = arb[k]["inv"]["species"]
        n = len(ref_of(k)["x"])
        for j in range(n):
            sp_code = species[j] if j < len(species) else "ND"
            sp.append(SPECIES.get(sp_code, ("?", "", "undetermined"))[0])
            leaf.append(SPECIES.get(sp_code, ("?", "", "undetermined"))[2])
            plot_of.append(k)
            for m in MET:
                hit[m].append(j in base[m][i]["ok"])
    sp = np.array(sp); plot_of = np.array(plot_of); leaf = np.array(leaf)
    hit = {m: np.array(v) for m, v in hit.items()}

    co = np.char.startswith(leaf.astype(str), "conifer")
    fr = leaf == "broadleaf"
    groups = []
    for e in sorted(set(sp)):
        if e in ("?", "undetermined") or (sp == e).sum() < MIN_N:
            continue
        groups.append((e, sp == e))
    groups.append(("All conifers", co))
    groups.append(("All broadleaves", fr))

    def ci(m, sel, seed=0):
        """Recall resampling plots, not trees."""
        ks = sorted(set(plot_of[sel]))
        tp = [int(hit[m][sel & (plot_of == k)].sum()) for k in ks]
        nn = [int((sel & (plot_of == k)).sum()) for k in ks]
        rng = np.random.default_rng(seed); v = []
        for _ in range(NB):
            i = rng.integers(0, len(ks), len(ks))
            a, b = sum(tp[j] for j in i), sum(nn[j] for j in i)
            v.append(a / b if b else 0)
        return (sum(tp) / sum(nn),
                [round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)])

    output = {}
    print(f"{'species':24} {'n':>5} {'plots':>5} {'n_eff':>6} " +
          " ".join(f"{m:>11}" for m in MET))
    for name, sel in groups:
        ks = sorted(set(plot_of[sel]))
        p = np.array([(sel & (plot_of == k)).sum() for k in ks], float)
        p /= p.sum()
        neff = 1 / np.sum(p ** 2)
        row = f"{name:24} {int(sel.sum()):>5} {len(ks):>5} {neff:>6.1f} "
        output[name] = dict(n=int(sel.sum()), plots=len(ks), n_eff=round(float(neff), 1))
        for m in MET:
            r, i95 = ci(m, sel)
            output[name][m] = dict(recall=round(r, 3), ci=i95)
            row += f"{r:>11.3f} "
        print(row)

    # the gap between broadleaves and conifers, only in the plots that have both
    print(f"\nbroadleaf minus conifer gap, in the plots with both")
    mixed_plots = [k for k in sorted(set(plot_of)) if (fr & (plot_of == k)).sum()
              and (co & (plot_of == k)).sum()]
    sel_mixed = np.isin(plot_of, mixed_plots)
    output["broadleaf_gap"] = {"plots": len(mixed_plots)}
    for m in MET:
        a = hit[m][fr & sel_mixed]; b = hit[m][co & sel_mixed]
        # per plot and in one go: np.isin only looks at membership, so a plot drawn
        # twice counted once, and the resample was not really with replacement
        af = np.array([int(hit[m][fr & (plot_of == k)].sum()) for k in mixed_plots])
        nf = np.array([int((fr & (plot_of == k)).sum()) for k in mixed_plots])
        ac = np.array([int(hit[m][co & (plot_of == k)].sum()) for k in mixed_plots])
        nc = np.array([int((co & (plot_of == k)).sum()) for k in mixed_plots])
        rng = np.random.default_rng(0); v = []
        for _ in range(NB):
            i = rng.integers(0, len(mixed_plots), len(mixed_plots))
            na = nf[i].sum(); nb = nc[i].sum()
            v.append((af[i].sum() / na if na else 0) - (ac[i].sum() / nb if nb else 0))
        d = a.mean() - b.mean()
        i95 = [round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)]
        output["broadleaf_gap"][m] = dict(diff=round(float(d), 3), ci=i95,
                                          n_broadleaf=int(fr.sum()), n_conifer=int(co.sum()))
        print(f"   {m:12} {d:>+7.3f} {i95}")

    json.dump(output, open(os.path.join(ROOT, "species_recall.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\nsaved to species_recall.json")


if __name__ == "__main__":
    main()
