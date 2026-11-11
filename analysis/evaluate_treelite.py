"""Evaluates TreeLite3D with the same protocol as the other methods.

The detections come from run_treelite.py::detections, which computes them in the
container with the SAME rule we use for SegmentAnyTree and ForestFormer3D: one
detection per instance with at least MIN_PTS points, placed at the apex, with the area of
the convex hull in plan view. Here it is only filtered by region and matched.

    python3 analysis/evaluate_treelite.py            # sweep
    python3 analysis/evaluate_treelite.py t5         # one setting
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
from evaluate_ff3d import greedy  # noqa: E402

OUTPUT = os.path.join(ROOT, "treelite_output")
D = 3.5
NB = 2000


def evaluate(label, verbose=False):
    """Recall, precision and F1 per site for one setting of the clustering."""
    det = json.load(open(os.path.join(OUTPUT, f"detections_{label}.json")))
    rows, per_plot = {}, {}
    for site, name, plots in SITES:
        for p in plots:
            key, k = f"{site}_{p}", f"{site}/{p}"
            d = det.get(key)
            if d is not None and str(d.get("error", "")).startswith("0 labels"):
                d = dict(x=[], y=[], h=[], area=[], n=[], assigned=0,
                         points=int(str(d["error"]).split()[-2]))
            if d is None or "error" in d:
                raise SystemExit(f"{label} {key}: {d}")
            x, y = np.array(d["x"]), np.array(d["y"])
            ref = reference(site, p)
            inside = region(ref).contains_points(np.c_[x, y]) if len(x) \
                else np.zeros(0, bool)
            tp = greedy(x[inside], y[inside], ref["x"], ref["y"], D)
            rows.setdefault(name, []).append((tp, int(inside.sum()), len(ref["x"])))
            per_plot[k] = dict(instances=len(x), inside=int(inside.sum()), TP=tp,
                                  NR=len(ref["x"]), assigned=d["assigned"],
                                  points=d["points"],
                                  x=[round(float(v), 2) for v in x[inside]],
                                  y=[round(float(v), 2) for v in y[inside]],
                                  h=[round(float(v), 2) for v in np.array(d["h"])[inside]],
                                  area=[round(float(v), 2) for v in np.array(d["area"])[inside]])
            if verbose:
                print(f"{k:14} {d['points']:>8} {d['assigned']:>9} "
                      f"{len(x):>6} {int(inside.sum()):>7} {tp:>5} {len(ref['x']):>5}")

    summary = {}
    for name, fs in list(rows.items()) + [("Both", [f for v in rows.values() for f in v])]:
        tp = sum(f[0] for f in fs); nd = sum(f[1] for f in fs); nr = sum(f[2] for f in fs)
        r, pr = tp / nr, (tp / nd if nd else 0.0)
        f1 = 2 * pr * r / (pr + r) if pr + r else 0.0
        rng = np.random.default_rng(0); v = []
        for _ in range(NB):
            idx = rng.integers(0, len(fs), len(fs))
            a = sum(fs[i][0] for i in idx); b = sum(fs[i][2] for i in idx)
            v.append(a / b if b else 0)
        summary[name] = dict(
            TP=tp, ND=nd, NR=nr, recall=round(r, 3), precision=round(pr, 3),
            f1=round(f1, 3),
            recall_ci=[round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)])
    return summary, per_plot, rows


def main():
    if len(sys.argv) > 1:
        et = sys.argv[1]
        print(f"{'plot':14} {'points':>8} {'assigned':>9} {'inst':>6} "
              f"{'inside':>7} {'TP':>5} {'NR':>5}")
        summary, per_plot, _ = evaluate(et, verbose=True)
        print(f"\n{'site':12} {'recall':>14} {'CI 95 %':>16} "
              f"{'precision':>10} {'F1':>7}")
        for n, v in summary.items():
            print(f"{n:12} {v['recall']:>14.3f} {str(v['recall_ci']):>16} "
                  f"{v['precision']:>10.3f} {v['f1']:>7.3f}")
        json.dump({"tuning": et, "summary": summary, "per_plot": per_plot},
                  open(os.path.join(ROOT, "treelite_evaluation.json"), "w"),
                  indent=2, ensure_ascii=False)
        print("\nsaved to treelite_evaluation.json")
        return

    ets = sorted((os.path.basename(f)[len("detections_"):-5]
                  for f in glob.glob(os.path.join(OUTPUT, "detections_*.json"))),
                 key=lambda s: float(s[1:].split("_")[0]) if s[1:].split("_")[0].replace(".", "").isdigit() else 1e9)
    print(f"{'setting':10} {'radius':>7} | " + " | ".join(
        f"{n:^26}" for n in ("Chamrousse", "Pellizzano", "Both")))
    print(f"{'':10} {'':>7} | " + " | ".join(
        "  rec   prec     F1   ND" for _ in range(3)))
    for et in ets:
        summary, _, _ = evaluate(et)
        radius = float(et[1:].split("_")[0]) * 0.05
        row = f"{et:10} {radius:>6.2f}m |"
        for n in ("Chamrousse", "Pellizzano", "Both"):
            v = summary[n]
            row += f" {v['recall']:>5.3f} {v['precision']:>6.3f} {v['f1']:>6.3f} {v['ND']:>4} |"
        print(row)


if __name__ == "__main__":
    main()
