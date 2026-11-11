"""Evaluates the controlled thinning experiment.

The Pellizzano plots thinned by pulse to 60, 40 and 25 % are scored against the SAME
inventory and with the SAME protocol as the rest of the study. Site, sensor, stand and
reference stay fixed; the only thing that changes is density. It is the manipulation the
observational analysis could not do.

Usage: python3 analysis/thinning_evaluation.py [method]
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
from evaluation_core import reference, region, results_path  # noqa: E402
from main_evaluation import greedy_pairs, metrics, spearman  # noqa: E402

FRAC = (100, 60, 40, 25)
NB = 2000


# the 100 % of each method comes from ITS own original run, not from another's
KEY = {"SAT": "SAT", "t4_p30": "TreeLite3D", "FF3D": "FF3D"}


def load_json(met):
    """Detections per plot and fraction, with the 100 % from the unthinned run."""
    d = {}
    all_available = json.load(open(os.path.join(ROOT, "all_detections.json")))["detections"]
    key = KEY.get(met, met)
    for k, v in all_available.items():
        if k.startswith("pell/"):
            d[(k.split("/")[1], 100)] = dict(v[key], points=0)
    sub = json.load(open(os.path.join(ROOT, "treelite_output",
                                      f"thinning_detections_{met}.json")))
    for k, v in sub.items():
        p, f = k[5:].rsplit("_f", 1)
        d[(p, int(f))] = v
    return d


def main():
    met = sys.argv[1] if len(sys.argv) > 1 else "t4_p30"
    det = load_json(met)
    DENS = json.load(open(results_path("thinning_density.json")))["per_plot"]
    ps = [f"p{i}" for i in range(1, 16)]
    missing = [p for p in ps if any((p, fr) not in det for fr in FRAC)]
    if missing:
        print(f"skipped, missing one of the four fractions: {missing}")
        ps = [p for p in ps if p not in missing]
    rows, dens_rec = {}, []
    print(f"{'plot':8} " + " ".join(f"{f:>5}%" for f in FRAC)
          + "    recall by pulse fraction")
    for p in ps:
        ref = reference("pell", p)
        reg = region(ref)
        area = abs(np.sum(reg.vertices[:, 0] * np.roll(reg.vertices[:, 1], -1)
                          - np.roll(reg.vertices[:, 0], -1) * reg.vertices[:, 1]) / 2)
        row = f"{p:8} "
        for f in FRAC:
            d = det.get((p, f))
            if d is None:
                row += "    -- "; continue
            x, y = np.array(d.get("x", [])), np.array(d.get("y", []))
            inside = reg.contains_points(np.c_[x, y]) if len(x) else np.zeros(0, bool)
            pr = greedy_pairs(x[inside], y[inside], ref["x"], ref["y"], 3.5)
            TP, ND, NR = len(pr), int(inside.sum()), len(ref["x"])
            r = TP / NR if NR else 0
            rows.setdefault(f, []).append(dict(TP=TP, ND=ND, NR=NR))
            dens_rec.append((DENS[p][str(f)], r))
            row += f" {r:>5.3f}"
        print(row)

    print(f"\n{'fraction':10} {'mean pts/m2':>13} {'recall':>14} "
          f"{'95% CI':>16} {'precision':>10}")
    output = {}
    for f in FRAC:
        if f not in rows:
            continue
        v = rows[f]
        TP, ND, NR = (sum(x[c] for x in v) for c in ("TP", "ND", "NR"))
        r, pc, f1 = metrics(TP, NR, ND)
        rng = np.random.default_rng(0); b = []
        for _ in range(NB):
            i = rng.integers(0, len(v), len(v))
            a = sum(v[j]["TP"] for j in i); c = sum(v[j]["NR"] for j in i)
            b.append(a / c if c else 0)
        ci = [round(float(np.percentile(b, q)), 3) for q in (2.5, 97.5)]
        output[f] = dict(recall=round(r, 3), precision=round(pc, 3), f1=round(f1, 3),
                      ci=ci, TP=TP, ND=ND, NR=NR)
        print(f"{str(f)+' %':10} {'':>13} {r:>14.3f} {str(ci):>16} {pc:>10.3f}")

    d = np.array([x for x, _ in dens_rec]); rr = np.array([y for _, y in dens_rec])
    rho, pv = spearman(d, rr, nperm=20000)
    print(f"\nrecall versus density, inside the SAME site and sensor:")
    print(f"   {len(d)} observations, rho {rho:+.3f}, p {pv:.4f}")
    output["correlation"] = dict(n=len(d), rho=rho, p=pv)
    json.dump(output, open(os.path.join(ROOT, f"thinning_{met}.json"), "w"), indent=2)
    print(f"\nsaved to thinning_{met}.json")


if __name__ == "__main__":
    main()
