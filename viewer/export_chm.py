"""Module A of the viewer: terrain geometry of the 22 plots.

For each plot it produces:
  - A CHM (canopy height model) with 0.5 m cells, grid of the maximum z,
    gaussian smoothing sigma=1.0. Same procedure as det_watershed in evaluation_core.py.
  - That CHM turned into an 8 bit grayscale PNG, embedded as a data URI.
  - The world rectangle that covers exactly that image.
  - The polygon of the inventoried region (convex hull dilated by 1.5 m).
  - Basic statistics of the plot.

Output: viewer/data/geo.json

Usage: python3 viewer/export_chm.py
"""
import base64
import io
import json
import os
import sys
import time

import laspy
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# The analysis module is the one in charge: from there come the data paths, the list
# of plots and the region() function.
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import reference, region, SITES, BUF  # noqa: E402
OUTPUT = os.path.join(HERE, "data", "geo.json")

CELL = 0.5      # meters per CHM cell
SIGMA = 1.0      # gaussian smoothing, same as in det_watershed
MAX_PNG = 120 * 1024   # limit per plot: 120 KB


def build_chm(px, py, pz, cell=CELL, sigma=SIGMA):
    """Grid of the maximum z per cell, smoothed. Returns (chm, x0, y0).

    Row i of the array grows with y (row 0 is the lowest y), same as in
    det_watershed. The flip for the screen happens later, when painting the PNG.
    """
    x0, y0 = float(px.min()), float(py.min())
    nx = int((px.max() - x0) / cell) + 1
    ny = int((py.max() - y0) / cell) + 1
    chm = np.zeros((ny, nx), np.float32)
    j = np.clip(((px - x0) / cell).astype(int), 0, nx - 1)
    i = np.clip(((py - y0) / cell).astype(int), 0, ny - 1)
    np.maximum.at(chm, (i, j), pz)
    return gaussian_filter(chm, sigma), x0, y0


def chm_a_png(chm, ceiling):
    """CHM to 8 bit gray PNG, normalized between 0 and ceiling, row 0 on top.

    Returns (data_uri, width, height). If the PNG goes over 120 KB, it halves the
    image and retries until it fits.
    """
    ceiling = max(float(ceiling), 1e-6)
    v = np.clip(chm / ceiling, 0.0, 1.0)
    # In the CHM row i grows with y; on screen the row grows downwards,
    # so it has to be flipped vertically.
    v = np.flipud(v)
    img = Image.fromarray((v * 255.0 + 0.5).astype(np.uint8), mode="L")

    while True:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        raw = buf.getvalue()
        if len(raw) <= MAX_PNG or min(img.size) <= 8:
            break
        img = img.resize((max(1, img.width // 2), max(1, img.height // 2)),
                         Image.BILINEAR)
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    return uri, img.width, img.height


def polygon_area(v):
    """Area of the closed polygon by the shoelace formula."""
    x, y = v[:, 0], v[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def process(site, plot):
    ref = reference(site, plot)

    L = laspy.read(ref["las"])
    px, py, pz = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    n_points = int(px.size)          # all the returns in the file

    m = pz >= 0                      # only positive heights, as in det_watershed
    px, py, pz = px[m], py[m], pz[m]

    chm, x0, y0 = build_chm(px, py, pz)
    ny, nx = chm.shape
    x1, y1 = x0 + nx * CELL, y0 + ny * CELL

    h_max = float(pz.max())
    h_p99 = float(np.percentile(pz, 99))
    uri, w_png, h_png = chm_a_png(chm, h_p99)

    tile_area = float(nx * CELL * ny * CELL)
    verts = np.asarray(region(ref, buf=1.5).vertices, float)
    region_area = polygon_area(verts)

    return {
        "chm_png": uri,
        "chm_extent": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "chm_max_h": round(h_p99, 2),
        "region": [[round(float(a), 2), round(float(b), 2)] for a, b in verts],
        "n_points": n_points,
        "area_tile_m2": round(tile_area, 2),
        "area_region_m2": round(region_area, 2),
        "pct_region": round(100.0 * region_area / tile_area, 2) if tile_area else 0.0,
        "density_pts_m2": round(n_points / tile_area, 2) if tile_area else 0.0,
        "h_max": round(h_max, 2),
        "h_p99": round(h_p99, 2),
    }, (nx, ny, w_png, h_png, len(uri))


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    output = {}
    t0 = time.time()
    for site, name, plots in SITES:
        for p in plots:
            key = f"{site}/{p}"
            t = time.time()
            output[key], info = process(site, p)
            nx, ny, w, h, nb64 = info
            print(f"{key:12} grid {nx:4}x{ny:<4} png {w:4}x{h:<4} "
                  f"{nb64/1024:7.1f} KB b64  pct_region "
                  f"{output[key]['pct_region']:5.2f}  ({time.time()-t:.1f}s)",
                  flush=True)

    with open(OUTPUT, "w") as f:
        json.dump(output, f, ensure_ascii=False)

    nbytes = os.path.getsize(OUTPUT)
    print(f"\n{len(output)} plots in {OUTPUT}")
    print(f"size {nbytes} bytes = {nbytes/1024/1024:.2f} MB  ({time.time()-t0:.1f}s total)")

    # Checks: nothing empty and the region percentages inside their range.
    empty = [k for k, v in output.items() if not v["chm_png"]]
    print("empty chm_png:", empty if empty else "none")
    for pref, lo, hi in (("cham", 31, 46), ("pell", 23, 40)):
        vals = [(k, v["pct_region"]) for k, v in output.items() if k.startswith(pref)]
        outside = [(k, p) for k, p in vals if not (lo <= p <= hi)]
        arr = [p for _, p in vals]
        print(f"{pref}: pct_region min {min(arr):.2f} max {max(arr):.2f} "
              f"mean {sum(arr)/len(arr):.2f}  outside [{lo},{hi}]: "
              f"{outside if outside else 'none'}")


if __name__ == "__main__":
    main()
