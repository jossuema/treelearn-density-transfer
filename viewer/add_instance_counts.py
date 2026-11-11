"""Adds to clouds.json the real number of instances of the original file.

Thinning down to 9000 points loses whole instances: the ones that had few points can
end up with none. The 3D panel announced the count of the thinned cloud as if it were
the total, which invites confusion with the detections the tree panel reports, which
are also filtered by a minimum number of points and by the inventoried region. Both
numbers are kept here so it can be said clearly.
"""
import json
import os

import laspy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(os.path.dirname(HERE))
SAT_OUT = os.path.join(ROOT, "sat_output_all")

cloud_data = json.load(open(os.path.join(HERE, "data", "clouds.json")))
for key in cloud_data:
    site, plot = key.split("/")
    laz = os.path.join(SAT_OUT, f"{site}_{plot}_out.laz")
    inst = np.asarray(laspy.read(laz).PredInstance).astype(np.int64)
    total = int(len(np.unique(inst[inst > 0])))
    cloud_data[key]["n_inst_total"] = total
    cloud_data[key]["n_inst_dropped"] = max(0, total - cloud_data[key]["n_inst"])
    print(f"{key:14s} thinned {cloud_data[key]['n_inst']:4d} of {total:4d} instances "
          f"({100*cloud_data[key]['n_inst']/total:.0f} %)")

json.dump(cloud_data, open(os.path.join(HERE, "data", "clouds.json"), "w"), separators=(",", ":"))
print("\nclouds.json updated")
