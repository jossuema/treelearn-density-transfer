"""Second round of robustness checks.

  1. Non inferiority against the most demanding oracle: AMS3D with the best of its 25
     settings chosen with OUR F1, not with the one from Tusa's protocol.
  2. The main table redone with optimal matching (maximum cardinality Hungarian), to
     see whether the order between methods changes.

Reuses the functions of review_checks.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_checks as R  # noqa: E402  (runs review_checks.py; its prints come out first)

print("\n" + "=" * 90)
print("1. SegmentAnyTree against the most demanding oracle, margin", R.DELTA)
name = "oracle, our F1"
for s in ("cham", "pell", "global"):
    rows = []
    for k, (tp, nd, nr) in R.sat_rows(s):
        sit, p = k.split("/")
        atp, and_, _ = R.sel[name][(sit, p)]
        rows.append((tp, nd, nr, atp, and_))
    tot = np.sum(np.array(rows, float), 0)
    for met, fn in (("recall", lambda t: t[0] / t[2] - t[3] / t[2]),
                    ("f1", lambda t: R.f1(t[0], t[1], t[2]) - R.f1(t[3], t[4], t[2]))):
        q = R.boot(rows, fn, q=(5, 50, 95))
        print(f"  {s:6} {met:6} dif {fn(tot):+.3f}  p5 {q[0]:+.3f}  p95 {q[2]:+.3f}  "
              f"not inferior {'YES' if q[0] > -R.DELTA else 'no'}  "
              f"equivalent {'YES' if q[0] > -R.DELTA and q[2] < R.DELTA else 'no'}")

print("\n2. Main table with optimal matching")
print(f"  {'site':12} {'method':6} {'greedy rec/prec/F1':>22} {'Hungarian rec/prec/F1':>22}")
for s, ps in R.SITES.items():
    for m in ("SAT", "AMS3D", "WS"):
        g = [0, 0, 0]; h = [0, 0, 0]
        for p in ps:
            a = R.TREES[f"{s}/{p}"]; dd = a["det"][m]
            nr = len(a["inv"]["x"]); nd = len(dd["x"])
            g[0] += R.greedy(dd["x"], dd["y"], a["inv"]["x"], a["inv"]["y"]); g[1] += nd; g[2] += nr
            h[0] += R.optimal(dd["x"], dd["y"], a["inv"]["x"], a["inv"]["y"]); h[1] += nd; h[2] += nr
        fg = f"{g[0]/g[2]:.3f}/{g[0]/g[1]:.3f}/{R.f1(*g):.3f}"
        fh = f"{h[0]/h[2]:.3f}/{h[0]/h[1]:.3f}/{R.f1(*h):.3f}"
        print(f"  {s:12} {m:6} {fg:>22} {fh:>22}")
