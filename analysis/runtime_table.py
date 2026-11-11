"""Builds the runtime comparison of the three methods.

What makes the three comparable is not the time of one pass, but the time it costs to
reach the result that gets published. SegmentAnyTree runs a single time, with its
public checkpoint and without tuning anything. AMS3D publishes the best of 25 settings
per plot, and the watershed the best of 16, so to reproduce their figure you have to
run them 25 and 16 times per plot. That multiplier is part of the cost and is reported
separately, not hidden.

The CPU measurements come from measure_runtimes.py, the 22 plots in a single run and on the same
machine, using process CPU time. It matters that it is a single run: measuring them in
separate batches, the ones that coincided with other tasks came out between 21 and 48
per cent slower, because under load the processor lowers its frequency and that affects
wall clock and CPU time alike. The GPU ones come from sat_run_result.json, measured on
Modal on a T4. The hardware asymmetry is real and is stated.

Usage: python3 analysis/runtime_table.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import results_path  # noqa: E402

T = json.load(open(results_path("measure_runtimes.json")))
SAT = json.load(open(os.path.join(ROOT, "sat_run_result.json")))

# settings per plot that each method needs to give its published figure
CONFIGS = {"SAT": 1, "AMS3D": 25, "WS": 16}

# SegmentAnyTree input points: it ran on full_las, which includes the ground, and not on
# the .las cropped to 1 m that the other two use. That has to be said.
sat = {}
for r in SAT:
    key = r["file"].replace(".las", "").replace("_", "/", 1)
    res = list(r["summary"].values())[0] if r.get("summary") else {}
    sat[key] = {"s": r["seconds"], "points": res.get("n_points")}

P = T["plots"]
keys = sorted(P, key=lambda k: P[k]["points"])

print(f"CPU machine: {T['machine']['cpu']}, {T['machine']['cores']} cores, "
      f"{T['machine']['system']}")
print("GPU: NVIDIA T4 on Modal, according to sat_run_result.json\n")

print(f"{'plot':14} {'pts CPU':>9} {'pts GPU':>9} {'WS s':>7} {'AMS3D s':>9} {'SAT s':>8}")
for k in keys:
    d = P[k]
    a = d.get("ams_cpu")
    s = sat.get(k, {})
    print(f"{k:14} {d['points']:>9} {str(s.get('points','')):>9} {d.get('ws_cpu',0):>7.2f} "
          f"{(f'{a:.1f}' if a else '-'):>9} {(f'{s.get(chr(115),0):.1f}' if s else '-'):>8}")

# --------------------------------------------------------------- throughput
print("\nTHROUGHPUT, points per second, median over the measured plots")
rate = {}
ws = [P[k]["points"] / P[k]["ws_cpu"] for k in keys if P[k].get("ws_cpu", 0) > 0]
rate["WS"] = float(np.median(ws))
am = [P[k]["points"] / P[k]["ams_cpu"] for k in keys if P[k].get("ams_cpu")]
rate["AMS3D"] = float(np.median(am)) if am else float("nan")
st = [v["points"] / v["s"] for v in sat.values() if v.get("points")]
rate["SAT"] = float(np.median(st))
for m in ("WS", "AMS3D", "SAT"):
    print(f"  {m:6} {rate[m]:>12,.0f} pts/s"
          + ("   (GPU T4)" if m == "SAT" else "   (CPU, 1 process)")
          + (f"   n={len(am)} plots" if m == "AMS3D" else ""))

# --------------------------------------------------------------- total cost
print("\nCOST OF ONE PASS over the 22 plots, extrapolating with the measured throughput")
tot_cpu = sum(P[k]["points"] for k in keys)
tot_gpu = sum(v["points"] for v in sat.values() if v.get("points"))
print(f"  total points: {tot_cpu:,} in the cropped .las, {tot_gpu:,} in the full_las of SAT")
one_pass = {}
one_pass["WS"] = sum(P[k]["ws_cpu"] for k in keys)
one_pass["AMS3D"] = sum(P[k]["ams_cpu"] for k in keys)
one_pass["SAT"] = sum(v["s"] for v in sat.values())
for m in ("WS", "AMS3D", "SAT"):
    med = "measured" if m != "AMS3D" else f"{len(am)} measured, {22-len(am)} extrapolated"
    print(f"  {m:6} {one_pass[m]:>10.1f} s = {one_pass[m]/60:>7.1f} min   ({med})")

print("\nCOST OF REPRODUCING THE PUBLISHED FIGURE, with the sweep each method needs")
for m in ("WS", "AMS3D", "SAT"):
    t = one_pass[m] * CONFIGS[m]
    print(f"  {m:6} {CONFIGS[m]:>3} config/plot  {t:>10.1f} s = {t/3600:>7.2f} h")

print("\nRatios versus SegmentAnyTree, one pass:")
for m in ("WS", "AMS3D"):
    print(f"  {m:6} one pass {one_pass['SAT']/one_pass[m]:>8.1f} times faster than SAT; "
          f"with its sweep, {one_pass['SAT']/(one_pass[m]*CONFIGS[m]):>6.2f} times")

json.dump({"machine": T["machine"], "throughput": rate, "one_pass_s": one_pass,
           "configs": CONFIGS, "points_cpu": tot_cpu, "points_gpu": tot_gpu,
           "ams_measurements": len(am)},
          open(os.path.join(ROOT, "runtime_summary.json"), "w"), indent=2, ensure_ascii=False)
print("\nsaved to runtime_summary.json")
