"""Final TreeLite3D result: default, with oracle setting and with one plot left out.

TreeLite3D comes configured for clouds of thousands of points per square meter: it groups
with a radius of cluster_thresh * voxel_size = 0.05 m and only accepts proposals of more
than 100 points. With that, on our plots it proposes almost nothing. We swept 22
combinations of radius and minimum proposal size.

That sweep was chosen looking at the same data the measurement is made on, which is the
problem we already pointed out in AMS3D and in the watershed. Hence the three figures:
  default    what the model gives exactly as published
  oracle     the best combination over the 22 plots, an upper bound
  one out    for each plot the combination is chosen with the other 21

Usage: python3 analysis/treelite_results.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import reference, region, SITES  # noqa: E402

OUTPUT = os.path.join(ROOT, "treelite_output")
D = 3.5
NB = 2000
STRATA = ("dominant", "intermediate", "suppressed")
REF = {}


def ref_of(site, p):
    if (site, p) not in REF:
        REF[(site, p)] = reference(site, p)
    return REF[(site, p)]


def greedy_pairs(dx, dy, rx, ry, dmax=D):
    """One to one matching by distance; returns the reference indices."""
    if not len(dx) or not len(rx):
        return set()
    Dm = np.hypot(np.asarray(dx)[:, None] - np.asarray(rx)[None, :],
                  np.asarray(dy)[:, None] - np.asarray(ry)[None, :])
    cand = np.argwhere(Dm <= dmax)
    order = np.argsort(Dm[cand[:, 0], cand[:, 1]], kind="stable")
    ud, ur = set(), set()
    for i, j in cand[order]:
        if i in ud or j in ur:
            continue
        ud.add(i); ur.add(j)
    return ur


def metrics(tp, nr, nd):
    r = tp / nr if nr else 0.0
    p = tp / nd if nd else 0.0
    return r, p, (2 * p * r / (p + r) if p + r else 0.0)


def load():
    """For each setting and plot: TP, ND, NR and which inventory trees were hit."""
    settings = {}
    for f in sorted(glob.glob(os.path.join(OUTPUT, "detections_*.json"))):
        et = os.path.basename(f)[len("detections_"):-5]
        det = json.load(open(f))
        adjusted = {}
        for site, _, plots in SITES:
            for p in plots:
                key, k = f"{site}_{p}", f"{site}/{p}"
                d = det.get(key, {})
                if str(d.get("error", "")).startswith("0 labels"):
                    d = dict(x=[], y=[])
                x, y = np.array(d.get("x", [])), np.array(d.get("y", []))
                ref = ref_of(site, p)
                inside = region(ref).contains_points(np.c_[x, y]) if len(x) \
                    else np.zeros(0, bool)
                ok = greedy_pairs(x[inside], y[inside], ref["x"], ref["y"])
                adjusted[k] = dict(TP=len(ok), ND=int(inside.sum()), NR=len(ref["x"]),
                               n_instances=len(x), ok=ok, site=site)
        settings[et] = adjusted
    return settings


def aggregate(ps, seed=0):
    TP = sum(p["TP"] for p in ps); ND = sum(p["ND"] for p in ps)
    NR = sum(p["NR"] for p in ps)
    r, pr, f1 = metrics(TP, NR, ND)
    rng = np.random.default_rng(seed)
    Rs, Ps, Fs = [], [], []
    for _ in range(NB):
        i = rng.integers(0, len(ps), len(ps))
        a, b, c = (sum(ps[j][x] for j in i) for x in ("TP", "ND", "NR"))
        rr, pp, ff = metrics(a, c, b)
        Rs.append(rr); Ps.append(pp); Fs.append(ff)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3),
                   round(float(np.percentile(v, 97.5)), 3)]
    return dict(TP=TP, ND=ND, NR=NR, recall=round(r, 3), precision=round(pr, 3),
                f1=round(f1, 3), recall_ci=q(Rs), precision_ci=q(Ps), f1_ci=q(Fs),
                n_plots=len(ps))


def f1_of(adjusted, keys):
    TP = sum(adjusted[k]["TP"] for k in keys)
    ND = sum(adjusted[k]["ND"] for k in keys)
    NR = sum(adjusted[k]["NR"] for k in keys)
    return metrics(TP, NR, ND)[2]


def main():
    settings = load()
    keys = [f"{s}/{p}" for s, _, ps in SITES for p in ps]
    sites = {n: [f"{s}/{p}" for p in ps] for s, n, ps in SITES}

    best = max(settings, key=lambda e: f1_of(settings[e], keys))
    print(f"{len(settings)} settings tried; best over the 22 plots: {best}\n")

    output = {"tunings_tested": sorted(settings), "best": best, "D": D}
    # The whole sweep surface, not just its winner: this is what the study means
    # by "sweeping the linking radius and minimum proposal size".
    # Per plot with the published settings. n_instances is what the model returned,
    # before cropping to the inventoried region, which is where the reported "on two
    # of the seven Chamrousse plots no instance at all" can be checked; ND is what
    # survives the crop, and it can be 0 while the model did return something.
    output["default_per_plot"] = {
        k: {m: settings["t1"][k][m] for m in ("n_instances", "TP", "ND", "NR")}
        for k in keys if k in settings.get("t1", {})}
    output["sweep"] = {}
    for et in sorted(settings):
        row = {sit: aggregate([settings[et][k] for k in ks])
               for sit, ks in list(sites.items()) + [("Both", keys)]}
        output["sweep"][et] = {s: {m: v[m] for m in ("recall", "precision", "f1", "TP", "ND", "NR")}
                               for s, v in row.items()}
    print(f"{'variant':22} {'site':12} {'recall':>6} {'95% CI':>16} "
          f"{'prec':>6} {'F1':>6} {'ND':>6}")
    for name, et in (("default", "t1"), ("oracle", best)):
        output[name] = {"tuning": et}
        for sit, ks in list(sites.items()) + [("Both", keys)]:
            v = aggregate([settings[et][k] for k in ks])
            output[name][sit] = v
            print(f"{name + ' (' + et + ')':22} {sit:12} {v['recall']:>6.3f} "
                  f"{str(v['recall_ci']):>16} {v['precision']:>6.3f} "
                  f"{v['f1']:>6.3f} {v['ND']:>6}")

    # one plot out: the combination is chosen with the other 21
    chosen, ps_lopo = {}, {}
    for k in keys:
        rest = [c for c in keys if c != k]
        e = max(settings, key=lambda a: f1_of(settings[a], rest))
        chosen[k] = e
        ps_lopo[k] = settings[e][k]
    output["leave_one_out"] = {"selected": chosen}
    for sit, ks in list(sites.items()) + [("Both", keys)]:
        v = aggregate([ps_lopo[k] for k in ks])
        output["leave_one_out"][sit] = v
        print(f"{'one out':22} {sit:12} {v['recall']:>6.3f} "
              f"{str(v['recall_ci']):>16} {v['precision']:>6.3f} "
              f"{v['f1']:>6.3f} {v['ND']:>6}")
    counts = {}
    for e in chosen.values():
        counts[e] = counts.get(e, 0) + 1
    print(f"\nsettings that ended up chosen: {counts}")

    # strata with the oracle setting, to see where it loses
    print(f"\n{'site':12} {'stratum':12} {'n':>5} {'TP':>5} {'recall':>7}")
    output["strata"] = {}
    for sit, ks in list(sites.items()) + [("Both", keys)]:
        output["strata"][sit] = {}
        for est in STRATA:
            n = tp = 0
            for k in ks:
                s, p = k.split("/")
                ref = ref_of(s, p)
                m = ref["stratum"] == est
                n += int(m.sum())
                tp += sum(1 for i in settings[best][k]["ok"] if ref["stratum"][i] == est)
            output["strata"][sit][est] = dict(n=n, TP=tp,
                                             recall=round(tp / n, 3) if n else 0.0)
            print(f"{sit:12} {est:12} {n:>5} {tp:>5} {tp / n if n else 0:>7.3f}")

    for v in output.get("leave_one_out", {}).values():
        pass
    json.dump(output, open(os.path.join(ROOT, "treelite_evaluation.json"), "w"),
              indent=2, ensure_ascii=False, default=str)
    print("\nsaved to treelite_evaluation.json")


if __name__ == "__main__":
    main()
