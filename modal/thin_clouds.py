"""Thins the Pellizzano plots by removing whole PULSES, not scattered points.

Why by pulse: a sparser ALS emits fewer pulses, but each pulse still returns all of its
returns. Dropping points at random would break the multiple-return structure, which is
exactly one of the features the models consume (number_of_returns and return_number).
The LAS files carry gps_time, so they can be grouped.

Why nested: the pulses are permuted ONCE with a fixed seed and each level takes a
prefix. That way the 25 % is a subset of the 40 % and that one of the 60 %, and the
differences between levels do not carry the noise of three separate draws.

Pellizzano runs from 91 to 173 pts/m2; at 40 % it lands at 36 to 69, the range of Chamrousse.
That allows comparing at equal density without changing site, sensor or stand.

Usage: python3 modal/thin_clouds.py
"""
import os

import laspy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
# TREELEARN_DATA points at the directory holding the datasets, which are not
# redistributed with the code.
DATA = os.environ.get("TREELEARN_DATA") or BASE
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(DATA)
ORIGIN = os.path.join(DATA, "trento_dataset", "las")
OUT = os.path.join(ROOT, "thinning")
FRACTIONS = (0.60, 0.40, 0.25)
PLOTS = [f"p{i}" for i in range(1, 16)]


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'plot':9} {'pulses':>8} {'points':>8} " +
          " ".join(f"{int(f*100):>13}%" for f in FRACTIONS))
    for p in PLOTS:
        L = laspy.read(os.path.join(ORIGIN, f"{p}.las"))
        t = np.asarray(L.gps_time)
        pulses, inverse = np.unique(t, return_inverse=True)
        # a single permutation: the levels are nested prefixes
        order = np.random.default_rng(0).permutation(len(pulses))
        row = f"{p:9} {len(pulses):>8} {len(L.points):>8} "
        for f in FRACTIONS:
            k = int(round(len(pulses) * f))
            keep = np.zeros(len(pulses), bool)
            keep[order[:k]] = True
            m = keep[inverse]
            output = laspy.LasData(L.header)
            output.points = L.points[m].copy()
            name = f"pell_{p}_f{int(f*100):02d}.las"
            output.write(os.path.join(OUT, name))
            row += f" {int(m.sum()):>13}"
        print(row)
    n = len([f for f in os.listdir(OUT) if f.endswith(".las")])
    mb = sum(os.path.getsize(os.path.join(OUT, f))
             for f in os.listdir(OUT)) / 1e6
    print(f"\n{n} files, {mb:.0f} MB in {OUT}")


if __name__ == "__main__":
    main()
