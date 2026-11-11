"""Builds everything the viewer needs, in the right order.

It matters because the steps are not independent: patch_exports.py rewrites fields that
export_chm.py and export_trees.py had left in another shape, and pack_json_to_js.py
has to go last. If they are run loose and in another order, the .js files do not
match the .json files.

Usage: python3 build_viewer_data.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

STEPS = [
    ("export_chm.py", "canopy, region and per plot statistics"),
    ("export_trees.py", "inventory, detections, pairs and metrics"),
    ("export_clouds.py", "thinned point clouds"),
    ("add_instance_counts.py", "real instance count, before the thinning"),
    ("patch_exports.py", "coverage over the point box and the detections outside"),
    ("export_crowns.py", "crown hulls of SegmentAnyTree and four radius polygons"),
    ("export_species.py", "species of every tree of the inventory"),
    ("pack_json_to_js.py", "wrap the JSON in .js so it can be opened with file://"),
]


def main():
    for file, que in STEPS:
        path = os.path.join(HERE, file)
        if not os.path.exists(path):
            print(f"MISSING {file}, skipped")
            continue
        print(f"\n=== {file}: {que}")
        r = subprocess.run([sys.executable, path], cwd=BASE)
        if r.returncode:
            print(f"\n{file} failed with code {r.returncode}. Stopping here.")
            return r.returncode
    print("\ndone. Open index.html, or serve it with:")
    print(f"  python3 -m http.server 8777 --directory '{HERE}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
