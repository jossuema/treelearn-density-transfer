"""SegmentAnyTree zero-shot on our ALS plots, in Modal.

Inference with this image works, but the last step (exporting to LAS) always fails with
`ValueError: Cannot convert non-finite values (NA or inf) to integer`. It is issue #31 of
the repository. Exact cause: in `nibio_inference/merge_pt_ss_is.py`, line 100, the authors
left COMMENTED OUT the line that filled in the points with no instance assigned, and then
`pandas_to_las.py:118-122` tries to convert that column to integer.

Patch applied here: fill the non-finite values with 0 right before the conversions. For
`PredInstance` the 0 already means "no instance", so it is what the commented line meant
to do and it does not alter any prediction.

Environment detail: the image ships with python3.8 with torch and MinkowskiEngine in /usr/bin,
but add_python installs Modal's 3.11 in /usr/local/bin and puts it first in the PATH. The
.sh scripts call plain `python3`, so /usr/bin has to be forced to the front.

Usage:
    modal profile activate <your-profile>
    modal volume put treeseg-data <file.las> /input/<name>.las
    modal run modal/run_sat.py --files "cham_las1b.las,pell_p4.las"
    modal volume get treeseg-data /output ./sat_output
"""
import json
import os
import shutil
import subprocess
import time

import modal

IMG = "donaldmaen/segment-any-tree:latest"
WD = "/home/nibio/mutable-outside-world"
VOL = "/vol"
P2L = f"{WD}/nibio_inference/pandas_to_las.py"

PATCH = (
    r"sed -i '/^    las_file = laspy.LasData(las_header)$/a\    import numpy as _npfix\n"
    r"    df = df.replace([_npfix.inf, -_npfix.inf], _npfix.nan).fillna(0)' " + P2L
)

image = (
    modal.Image.from_registry(IMG, add_python="3.11")
    .entrypoint([])
    .run_commands(PATCH, f"grep -n _npfix {P2L}")   # the grep fails if the patch did not go in
)
vol = modal.Volume.from_name("treeseg-data", create_if_missing=True)
app = modal.App("sat-run", image=image, volumes={VOL: vol})

# /usr/bin first: that is where the python3.8 with torch and MinkowskiEngine is.
ENV = {**os.environ,
       "PATH": "/usr/bin:/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/sbin",
       "PYTHONPATH": WD}

COUNT = r'''
import glob, json, sys
import numpy as np, laspy
out = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.la[sz]")):
    L = laspy.read(f)
    dims = [d.name for d in L.point_format.dimensions]
    info = {"n_points": int(len(L.points)), "fields": dims}
    for cand in ("PredInstance", "preds_instance_segmentation", "instance", "treeID"):
        if cand in dims:
            v = np.asarray(getattr(L, cand))
            ids, counts = np.unique(v[v > 0], return_counts=True)
            info["instance_field"] = cand
            info["n_instances"] = int(len(ids))
            info["pts_per_instance_median"] = int(np.median(counts)) if len(ids) else 0
            info["pct_points_without_instance"] = round(100.0 * float((v <= 0).mean()), 1)
            break
    for cand in ("PredSemantic", "preds_semantic_segmentation"):
        if cand in dims:
            s = np.asarray(getattr(L, cand))
            u, c = np.unique(s, return_counts=True)
            info["semantic_field"] = {int(a): int(b) for a, b in zip(u, c)}
            break
    out[f.split("/")[-1]] = info
print("CNT" + json.dumps(out) + "CNT")
'''


# max_containers=6: the Starter plan allows 10 GPUs at a time; we leave some margin.
@app.function(gpu="T4", timeout=3 * 3600, memory=24576, cpu=4.0, max_containers=6)
def segment(name: str) -> dict:
    vol.reload()
    src = next((p for p in (f"{VOL}/input_all/{name}", f"{VOL}/input/{name}")
                if os.path.exists(p)), None)
    if src is None:
        return {"file": name, "error": f"{name} is not in /input_all nor in /input"}

    tmp_in, tmp_out = "/tmp/in", "/tmp/out"
    for d in (tmp_in, tmp_out):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    shutil.copy(src, tmp_in)

    t0 = time.time()
    p = subprocess.run(["bash", "run_inference.sh", tmp_in, tmp_out, "true"],
                       cwd=WD, env=ENV, capture_output=True, text=True)
    res = {"file": name, "seconds": round(time.time() - t0, 1),
           "returncode": p.returncode}

    base = name.rsplit(".", 1)[0]
    dest = f"{VOL}/output/sat/{base}"
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)

    # Final result (merged LAS/LAZ)
    final = os.path.join(tmp_out, "final_results")
    res["outputs"] = sorted(os.listdir(final)) if os.path.isdir(final) else []
    if res["outputs"]:
        for f in os.listdir(final):
            shutil.copy(os.path.join(final, f), dest)
        c = subprocess.run(["/usr/bin/python3", "-c", COUNT, final],
                           capture_output=True, text=True, env=ENV)
        if "CNT" in c.stdout:
            res["summary"] = json.loads(c.stdout.split("CNT")[1])
        else:
            res["count_error"] = (c.stdout + c.stderr)[-1500:]

    # Backup: the raw predictions in PLY, in case the export to LAS fails again
    for f in os.listdir(tmp_out):
        if f.endswith(".ply") and ("instance" in f or "semantic" in f):
            shutil.copy(os.path.join(tmp_out, f), dest)
            res.setdefault("ply_fallback", []).append(f)

    vol.commit()
    if p.returncode != 0:
        res["stderr_final"] = p.stderr[-2500:]
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res


@app.local_entrypoint()
def main(files: str = "cham_las1b.las,pell_p4.las"):
    names = [a.strip() for a in files.split(",") if a.strip()]
    out = []
    for n, r in zip(names, segment.map(names, return_exceptions=True)):
        out.append({"file": n, "error": str(r)} if isinstance(r, Exception) else r)
    print("\n" + "=" * 78)
    for r in out:
        if r.get("summary"):
            for f, i in r["summary"].items():
                print(f"  OK  {r['file']:18s} {i.get('n_instances','?'):>4} instances  "
                      f"{i.get('n_points','?'):>8} pts  "
                      f"{i.get('pct_points_without_instance','?')}% without instance  "
                      f"{r['seconds']} s")
        else:
            print(f"  FAIL  {r['file']}  rc={r.get('returncode')} {r.get('error','')}")
    print("=" * 78)
    json.dump(out, open("sat_run_result.json", "w"), indent=2, ensure_ascii=False)
    print("detail in sat_run_result.json")
