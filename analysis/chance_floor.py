"""How much of the inventory a detector recovers without looking at the cloud.

The stems of these stands sit about 1.9 m apart, and the matching tolerance is 3.5 m, so a
detection does not have to be right to be counted: it has to land in a neighbourhood that
holds several candidates. This puts a floor under every recall in the study, and the floor
is not the same for every method, because it grows with how many detections a method emits.

The floor is estimated by giving a uniform detector the SAME number of detections each
method produced inside the same scoring region, and matching it the same way. What is left
after subtracting it is the part of the score that required looking at the returns.

Usage: python3 analysis/chance_floor.py [--tolerances]
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
from evaluation_core import reference, region  # noqa: E402
from main_evaluation import greedy_pairs  # noqa: E402

MET = ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")
NAME = {"SAT": "SegmentAnyTree", "AMS3D": "AMS3D", "FF3D": "ForestFormer3D",
        "TreeLite3D": "TreeLite3D", "WS": "CHM watershed"}
NSIM = 40


def uniform_recall(n, reg, ref, dmatch, rng, reps=NSIM):
    """Recall of n detections thrown uniformly inside the scoring region."""
    if not n:
        return 0.0
    v = reg.vertices
    x0, x1, y0, y1 = v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()
    total = 0
    for _ in range(reps):
        got = []
        while len(got) < n:
            c = np.c_[rng.uniform(x0, x1, n * 3), rng.uniform(y0, y1, n * 3)]
            got.extend(c[reg.contains_points(c)].tolist())
        c = np.array(got[:n])
        total += len(greedy_pairs(c[:, 0], c[:, 1], ref["x"], ref["y"], dmatch))
    return total / reps


def stem_spacing(keys):
    out = {}
    for site in ("cham", "pell"):
        d = []
        for k in keys:
            if not k.startswith(site):
                continue
            r = reference(*k.split("/"))
            x, y = np.asarray(r["x"]), np.asarray(r["y"])
            D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
            np.fill_diagonal(D, np.inf)
            d.extend(D.min(axis=1))
        d = np.array(d)
        out[site] = dict(median=round(float(np.median(d)), 2),
                         within_3_5=round(float((d <= 3.5).mean()), 3),
                         within_2_0=round(float((d <= 2.0).mean()), 3))
    return out


def table(det, keys, dmatch, rng):
    rows = {}
    for m in MET:
        tp = tp0 = nd_t = nr = 0
        for k in keys:
            ref = reference(*k.split("/"))
            reg = region(ref)
            d = det[k][m]
            x, y = np.array(d["x"]), np.array(d["y"])
            inside = reg.contains_points(np.c_[x, y]) if len(x) else np.zeros(0, bool)
            nd = int(inside.sum())
            tp += len(greedy_pairs(x[inside], y[inside], ref["x"], ref["y"], dmatch))
            tp0 += uniform_recall(nd, reg, ref, dmatch, rng)
            nd_t += nd
            nr += len(ref["x"])
        rows[m] = dict(recall=round(tp / nr, 3), chance=round(tp0 / nr, 3),
                       excess=round(tp / nr - tp0 / nr, 3),
                       precision=round(tp / nd_t, 3) if nd_t else 0.0, ND=nd_t)
    return rows


def main():
    A = json.load(open(os.path.join(ROOT, "all_detections.json")))
    det = A["detections"]
    keys = list(det)
    rng = np.random.default_rng(0)
    out = {"stem_spacing": stem_spacing(keys)}
    s = out["stem_spacing"]
    print(f"nearest stem: Chamrousse {s['cham']['median']} m, Pellizzano {s['pell']['median']} m; "
          f"{s['cham']['within_3_5']:.0%} and {s['pell']['within_3_5']:.0%} have a neighbour "
          f"within 3.5 m")

    tolerances = (3.5, 2.5, 2.0) if "--tolerances" in sys.argv else (3.5,)
    out["by_tolerance"] = {}
    for dm in tolerances:
        rows = table(det, keys, dm, rng)
        out["by_tolerance"][str(dm)] = rows
        print(f"\nd_match = {dm} m")
        print(f"{'method':16} {'recall':>7} {'chance':>7} {'excess':>7} {'prec.':>7} {'ND':>6}")
        for m in MET:
            r = rows[m]
            print(f"{NAME[m]:16} {r['recall']:>7.3f} {r['chance']:>7.3f} {r['excess']:>7.3f} "
                  f"{r['precision']:>7.3f} {r['ND']:>6}")
        order = sorted(MET, key=lambda m: -rows[m]["recall"])
        order_x = sorted(MET, key=lambda m: -rows[m]["excess"])
        print(f"  by recall: {' > '.join(NAME[m] for m in order)}")
        print(f"  by excess: {' > '.join(NAME[m] for m in order_x)}")
        out["by_tolerance"][str(dm)]["_order_recall"] = [NAME[m] for m in order]
        out["by_tolerance"][str(dm)]["_order_excess"] = [NAME[m] for m in order_x]

    # ---- the same floor under the thinning experiment, where it matters most
    LAB = {"SAT": "SAT", "FF3D": "FF3D", "t4_p30": "TreeLite3D"}
    PS = [f"p{i}" for i in range(1, 16)]
    out["thinning"] = {}
    print(f"\n{'model':16} {'pulses':>7} {'ND':>6} {'recall':>7} {'chance':>7} "
          f"{'excess':>7} {'prec.':>7}")
    for m in ("SAT", "FF3D", "t4_p30"):
        out["thinning"][LAB[m]] = {}
        for fr in (100, 25):
            tp = tp0 = nd_t = nr = 0
            sub = None if fr == 100 else json.load(open(os.path.join(
                ROOT, "treelite_output", f"thinning_detections_{m}.json")))
            for p_ in PS:
                ref = reference("pell", p_)
                reg = region(ref)
                if fr == 100:
                    d = det[f"pell/{p_}"][LAB[m]]
                else:
                    k = f"pell_{p_}_f{fr}"
                    if k not in sub:
                        continue
                    d = sub[k]
                x, y = np.array(d["x"]), np.array(d["y"])
                ins = reg.contains_points(np.c_[x, y]) if len(x) else np.zeros(0, bool)
                nd = int(ins.sum())
                tp += len(greedy_pairs(x[ins], y[ins], ref["x"], ref["y"], 3.5))
                tp0 += uniform_recall(nd, reg, ref, 3.5, rng)
                nd_t += nd
                nr += len(ref["x"])
            r = dict(recall=round(tp / nr, 3), chance=round(tp0 / nr, 3),
                     excess=round(tp / nr - tp0 / nr, 3), ND=nd_t,
                     precision=round(tp / nd_t, 3) if nd_t else 0.0)
            out["thinning"][LAB[m]][str(fr)] = r
            print(f"{NAME[LAB[m]]:16} {fr:>6}% {r['ND']:>6} {r['recall']:>7.3f} "
                  f"{r['chance']:>7.3f} {r['excess']:>7.3f} {r['precision']:>7.3f}")
    print(f"\n{'model':16} {'drop in recall':>15} {'drop in excess':>15} "
          f"{'detections':>14} {'precision':>18}")
    for m in ("SAT", "FF3D", "t4_p30"):
        a_, b_ = out["thinning"][LAB[m]]["100"], out["thinning"][LAB[m]]["25"]
        print(f"{NAME[LAB[m]]:16} {a_['recall']-b_['recall']:>15.3f} "
              f"{a_['excess']-b_['excess']:>15.3f} "
              f"{a_['ND']:>6} to {b_['ND']:<5} {a_['precision']:>8.3f} to {b_['precision']:<6.3f}")

    with open(os.path.join(ROOT, "chance_floor.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to chance_floor.json")


if __name__ == "__main__":
    main()
