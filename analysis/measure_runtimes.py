"""Measures the runtime of AMS3D and of the watershed, plot by plot.

The method, and it has to be respected for the numbers to mean anything:
  - Everything SEQUENTIAL on the same machine. Timing two methods at once skews
    them through CPU contention.
  - What is timed is the PROCESS CPU TIME, not the wall clock. The machine is a work
    machine with other applications on top of it: the same computation measured by wall
    clock at two different moments varies by more than 50 %, while CPU time does not
    count the time the process spends off the CPU. Both are saved.
  - What is timed is the setting each method ended up using in the study, the best of
    its sweep for that plot.
  - Each measurement takes the median of several repetitions when the method is fast.
    AMS3D takes minutes, so there it is a single pass.
  - The JSON is written after each plot, so it is possible to stop and keep what has
    been measured up to that point.

The SegmentAnyTree time is NOT measured here: it is in sat_run_result.json, measured
on Modal on a T4 GPU. The hardware asymmetry is real and has to be said out loud, not
hidden.

Usage: python3 analysis/measure_runtimes.py [--ams N]
  --ams N  how many plots to time with AMS3D, from smallest to largest in size (0 = none)
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import (reference, det_watershed, SITES,
                             BASE as DATA, results_path)  # noqa: E402
# The mean shift port is not redistributed: it lives next to the datasets.
sys.path.insert(0, os.path.join(DATA, "ams_port"))

OUTPUT = os.path.join(ROOT, "measure_runtimes.json")


def machine_info():
    """What someone else needs in order to put these numbers in context."""
    d = {"system": platform.platform(), "python": platform.python_version()}
    try:
        d["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        d["cores"] = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True))
    except Exception:
        d["cpu"] = platform.processor() or "unknown"
        d["cores"] = os.cpu_count()
    return d


def configs_ams():
    out = {}
    for s, f in (("cham", "e1_best_per_plot_cham.csv"), ("pell", "e1_best_per_plot_pel.csv")):
        p = os.path.join(DATA, "ams_port", "out", f)
        for r in csv.DictReader(open(p)):
            # the name carries m1 and m2 multiplied by 1000
            parts = r["cfg"].split("_")
            m1 = int(parts[parts.index("m1") + 1]) / 1000.0
            m2 = int(parts[parts.index("m2") + 1]) / 1000.0
            out[(s, r["plot"])] = (m1, m2)
    return out


def configs_ws():
    wb = json.load(open(results_path("watershed_sweep.json")))
    out = {}
    for name, d in wb.items():
        s = "cham" if name == "Chamrousse" else "pell"
        for pl, c in d.get("configs", {}).items():
            out[(s, pl)] = (float(c["hmin"]), float(c["mindist"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ams", type=int, default=6)
    args = ap.parse_args()

    cfa, cfw = configs_ams(), configs_ws()
    res = {"machine": machine_info(), "plots": {}}
    if os.path.exists(OUTPUT):
        try:
            res = json.load(open(OUTPUT))
            res["machine"] = machine_info()
        except Exception:
            pass
    res.setdefault("plots", {})

    # 1. inventory of sizes, and the watershed, which is cheap
    print("Watershed, median of 3 passes per plot")
    print(f"{'plot':14} {'points':>9} {'ws_s':>8}")
    order = []
    for site, name, plots in SITES:
        for p in plots:
            key = f"{site}/{p}"
            ref = reference(site, p)
            import laspy
            n = int(len(laspy.read(ref["las"]).x))
            order.append((n, site, p, key))
            hm, md = cfw.get((site, p), (2.0, 2.0))
            ts, cs = [], []
            for _ in range(5):
                t0, p0 = time.perf_counter(), time.process_time()
                det_watershed(ref, hmin=hm, mindist=md)
                ts.append(time.perf_counter() - t0)
                cs.append(time.process_time() - p0)
            d = res["plots"].setdefault(key, {})
            d.update(points=n, ws_s=round(float(np.median(ts)), 3),
                     ws_cpu=round(float(np.median(cs)), 4), ws_cfg=[hm, md])
            print(f"{key:14} {n:>9} {np.median(ts):>8.2f}", flush=True)
            json.dump(res, open(OUTPUT, "w"), indent=2, ensure_ascii=False)

    # 2. AMS3D, from the smallest plot to the largest, a single pass
    order.sort()
    target = order[:args.ams] if args.ams >= 0 else order
    print(f"\nAMS3D, a single pass, {len(target)} plots from smallest to largest")
    print(f"{'plot':14} {'points':>9} {'wall_s':>9} {'cpu_s':>9} {'pts/cpu_s':>9}")
    from ams_engine import run_e1
    for n, site, p, key in target:
        if res["plots"].get(key, {}).get("ams_cpu"):
            print(f"{key:14} already measured, skipped", flush=True)
            continue
        m1, m2 = cfa[(site, p)]
        las = reference(site, p)["las"]
        t0, p0 = time.perf_counter(), time.process_time()
        run_e1(las, m1, m2)
        dt, dc = time.perf_counter() - t0, time.process_time() - p0
        d = res["plots"].setdefault(key, {})
        d.update(ams_s=round(dt, 2), ams_cpu=round(dc, 2), ams_cfg=[m1, m2])
        print(f"{key:14} {n:>9} {dt:>9.1f} {dc:>9.1f} {n/dc:>9.0f}", flush=True)
        json.dump(res, open(OUTPUT, "w"), indent=2, ensure_ascii=False)

    json.dump(res, open(OUTPUT, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved to {OUTPUT}")


if __name__ == "__main__":
    main()
