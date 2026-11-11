"""Multiple-comparison control, and a thinning test that respects the plot as the unit.

Two things the study needed. First, the correlations it reports are several tests over the
same 22 plots, so the set has to be declared and adjusted; families are declared here by the
question each test answers, and include the tests that were computed and not printed, which
is what makes the correction meaningful. Benjamini-Hochberg rather than Holm: the four gap
statistics share SegmentAnyTree's recall, so they are positively dependent, which is the
condition where BH controls the false discovery rate exactly.

Second, the thinning correlation treated 15 plots seen at four thinning levels as 60
independent observations, which they are not. The paired drop per plot with an exact
sign-flip test over the 15 plots is the test the design licenses, and it comes out stronger.

Usage: python3 analysis/multiplicity.py
"""
import itertools
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
from main_evaluation import greedy_pairs  # noqa: E402

MET = ("SAT", "AMS3D", "FF3D", "TreeLite3D", "WS")
LABEL = {"SAT": "SAT", "FF3D": "FF3D", "t4_p30": "TreeLite3D"}
PLOTS = [f"p{i}" for i in range(1, 16)]


def benjamini_hochberg(pvals):
    """Adjusted values, monotone, in the order the tests were given."""
    m = len(pvals)
    order = np.argsort(pvals)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, pvals[i] * m / (m - rank + 1))
        q[i] = prev
    return q


def plot_recall(method, fraction, detections):
    out = {}
    for p in PLOTS:
        if fraction == 100:
            d = detections[f"pell/{p}"][LABEL[method]]
        else:
            sub = json.load(open(os.path.join(
                ROOT, "treelite_output", f"thinning_detections_{method}.json")))
            k = f"pell_{p}_f{fraction}"
            if k not in sub:
                continue
            d = sub[k]
        ref = reference("pell", p)
        reg = region(ref)
        x, y = np.array(d["x"]), np.array(d["y"])
        inside = reg.contains_points(np.c_[x, y]) if len(x) else np.zeros(0, bool)
        out[p] = len(greedy_pairs(x[inside], y[inside], ref["x"], ref["y"], 3.5)) / len(ref["x"])
    return out


def sign_flip(d):
    """Exact: 2^15 = 32768 sign assignments, so no sampling error in the p."""
    n, obs = len(d), float(np.mean(d))
    hits = sum(1 for s in itertools.product([1, -1], repeat=n)
               if abs(np.dot(s, d) / n) >= abs(obs) - 1e-12)
    return obs, hits / 2 ** n


def main():
    A = json.load(open(os.path.join(ROOT, "all_detections.json")))
    det = A["detections"]
    out = {}

    # ---- the thinning, tested over plots rather than over plot-fraction pairs
    print("Thinning, paired over the 15 plots")
    print(f"{'model':16} {'mean drop':>10} {'95% CI':>18} {'plots down':>11} {'exact p':>10}")
    out["thinning_paired"] = {}
    for m in ("SAT", "FF3D", "t4_p30"):
        a, b = plot_recall(m, 100, det), plot_recall(m, 25, det)
        keys = [p for p in PLOTS if p in a and p in b]
        d = np.array([a[p] - b[p] for p in keys])
        obs, pv = sign_flip(d)
        rng = np.random.default_rng(0)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out["thinning_paired"][LABEL[m]] = dict(
            mean_drop=round(obs, 3), ci=[round(float(lo), 3), round(float(hi), 3)],
            plots_down=int((d > 0).sum()), n_plots=len(d), p_exact=pv)
        print(f"{LABEL[m]:16} {obs:>10.3f} [{lo:>6.3f},{hi:>6.3f}] {int((d>0).sum()):>6}/{len(d):<4} {pv:>10.5f}")

    # ---- the declared families, adjusted
    M = json.load(open(results_path("main_evaluation.json")))
    fam = {
        "F1_gap_vs_density": {m: M["density_vs_recall"][m]["p_diff"]
                              for m in MET if m != "SAT"},
        "F2_recall_vs_density": {m: M["density_vs_recall"][m]["p"] for m in MET},
        "F3_thinning": {LABEL[m]: out["thinning_paired"][LABEL[m]]["p_exact"]
                        for m in ("SAT", "FF3D", "t4_p30")},
    }
    out["families"] = {}
    for name, tests in fam.items():
        keys = list(tests)
        q = benjamini_hochberg([tests[k] for k in keys])
        out["families"][name] = {k: dict(p=tests[k], q=round(float(qq), 4),
                                         survives=bool(qq < 0.05))
                                 for k, qq in zip(keys, q)}
        print(f"\n{name}, {len(keys)} tests, Benjamini-Hochberg at 5%")
        for k, qq in zip(keys, q):
            print(f"   {k:14} p {tests[k]:>8.4f}   q {qq:>7.4f}   "
                  f"{'survives' if qq < 0.05 else 'does not survive'}")

    with open(os.path.join(ROOT, "multiplicity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to multiplicity.json")


if __name__ == "__main__":
    main()
