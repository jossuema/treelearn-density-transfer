"""TreeLite3D inference (Rizaldy et al., RSE 2026) on our 22 plots.

Official Pointcept 1.5.0 image, which is what TreeLite3D is built on.
The repository is cloned at build time and two touch-ups of naming and
saving are applied to it (see patch_treelite.py). The weights are the ones
published by the authors.

    MODAL_PROFILE=<your-profile> modal run modal/run_treelite.py::probe
    MODAL_PROFILE=<your-profile> modal run --detach modal/run_treelite.py

The tester saves in exp/trees/<exp>/result/ one .npy per plot with the instance label
of each point, in the order of the input points. When it finishes everything is copied
to the outputs volume.
"""
import os
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/workspace/TreeLite3D"
EXP = "TREE-UNIFIED_trees_insseg-pointgroup-v1m1-0-spunet-base"
CFG = "TREE-UNIFIED-insseg-pointgroup-v1m1-0-spunet-base"
PY = os.environ.get("TL_PY", "/opt/conda/bin/python")
GPU = os.environ.get("TL_GPU", "T4")

weights = modal.Volume.from_name("treelite-weights")
data = modal.Volume.from_name("treelite-data")
outputs = modal.Volume.from_name("treelite-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "pointcept/pointcept:v1.5.0-pytorch1.11.0-cuda11.3-cudnn8-devel",
        add_python="3.11",
    )
    # the image puts /opt/conda/bin (python 3.8) first in the PATH and the Modal client
    # needs 3.10 or newer. The real work is launched with the absolute path of the conda
    # interpreter, which is the one that has torch, so reordering does not get in the way.
    .env({"PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/conda/bin"})
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        f"git clone --depth 1 https://github.com/aldinorizaldy/TreeLite3D.git {REPO}",
        # the configuration file of the repository is literally called "... copy.py",
        # and the -c flag of test.sh looks for it without that suffix
        f"cd {REPO}/configs/trees && mv '{CFG} copy.py' '{CFG}.py'",
    )
    .add_local_file(os.path.join(HERE, "patch_treelite.py"), "/patch_file.py", copy=True)
    .run_commands("python3 /patch_file.py")
)
app = modal.App("treelite", image=image)
VOLS = {"/weights": weights, "/data": data, "/outputs": outputs}


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 30)
def probe():
    """Checks interpreter, GPU and the compiled extensions before spending hours."""
    subprocess.run("nvidia-smi | head -12", shell=True)
    script = (
        "import torch, numpy, sys; "
        "print('python', sys.version.split()[0], 'torch', torch.__version__, "
        "'numpy', numpy.__version__); "
        "print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); "
        "import pointops, pointgroup_ops, spconv; print('extensions ok'); "
        "d = torch.load('/data/test/cham_1b.pth'); "
        "print('pth', {k: (v.shape, str(v.dtype)) for k, v in d.items()})"
    )
    r = subprocess.run([PY, "-c", script], cwd=REPO, capture_output=True, text=True)
    print(r.stdout or "", r.stderr[-3000:] if r.returncode else "")
    return r.returncode


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 60 * 5, cpu=4, memory=32768)
def infer(names=None, sample=False, folder="test",
            options="", label=""):
    import glob
    import shutil
    import time

    # the weights and the data live in volumes; the repository expects to find them inside
    model = f"{REPO}/exp/trees/{EXP}/model"
    os.makedirs(model, exist_ok=True)
    if not os.path.exists(f"{model}/model_best.pth"):
        os.symlink("/weights/weights/model_best.pth", f"{model}/model_best.pth")
    test_dir = f"{REPO}/data/trees_UNIFIED/test"
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir)
    all_files = sorted(os.path.basename(p) for p in glob.glob(f"/data/{folder}/*.pth"))
    chosen = [n for n in all_files if names is None or n[:-4] in names]
    for n in chosen:
        os.symlink(f"/data/{folder}/{n}", f"{test_dir}/{n}")
    if sample:  # the example plot of the authors, as a check on the setup
        for f in sorted(glob.glob("/weights/sample/*.pth")):
            os.symlink(f, f"{test_dir}/{os.path.basename(f)}")
            chosen.append(os.path.basename(f))
    print(f"{len(chosen)} plots: {', '.join(n[:-4] for n in chosen)}")

    t0 = time.time()
    # the same as scripts/test.sh does, but direct: this way configuration keys can be
    # added on the fly, which test.sh does not allow because it builds the --options alone
    exd = f"exp/trees/{EXP}"
    cmd = (f"PYTHONPATH=./ {PY} -u tools/test.py --config-file configs/trees/{CFG}.py"
           f" --num-gpus 1 --options save_path={exd}"
           f" weight={exd}/model/model_best.pth {options}")
    print(f"--- {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-8000:])
    if r.returncode:
        print("--- error\n" + r.stderr[-8000:])
    print(f"--- code {r.returncode}, {time.time() - t0:.0f} s")

    # it is copied no matter what: the tester saves inside the loop, so there can be
    # results even if the final evaluation (which uses a fake ground truth) blows up
    res = f"{REPO}/exp/trees/{EXP}/result"
    dest = f"/outputs/{label or folder}"
    os.makedirs(dest, exist_ok=True)
    copied = []
    for f in sorted(glob.glob(f"{res}/*.npy")):
        shutil.copy(f, f"{dest}/{os.path.basename(f)}")
        copied.append(os.path.basename(f))
    for f in sorted(glob.glob(f"{REPO}/exp/trees/{EXP}/*.log")):
        shutil.copy(f, f"{dest}/{os.path.basename(f)}")
    outputs.commit()
    print(f"--- {len(copied)} files copied")
    return r.returncode, copied


@app.function(volumes=VOLS, timeout=60 * 30, cpu=4, memory=32768)
def statistics():
    """Compares the example plot of the authors with one of ours.

    It matters because the input features are coordinates plus the three color channels,
    and NormalizeColor divides them by 127.5 without looking at where they come from. If
    the scale of the intensity is not alike, the model sees something else.
    """
    script = """
import numpy as np, torch, glob, os
def summary(f):
    d = torch.load(f)
    c, col = d["coord"], d["color"]
    ext = c.max(0) - c.min(0)
    dens = len(c) / (ext[0] * ext[1])
    print(f"{os.path.basename(f)[:34]:36} {len(c):>9} points  {dens:>8.1f} pts/m2  "
          f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} m")
    for j, name in enumerate(["intensity", "n returns", "return"]):
        q = np.percentile(col[:, j], [0, 25, 50, 75, 100])
        print(f"      {name:12} " + "  ".join(f"{v:>10.1f}" for v in q) +
              f"   after NormalizeColor: {q[2]/127.5-1:.2f}")
print(f"{'file':36} {'points':>9}          {'density':>8}")
print("      channel       min        p25        p50        p75        max")
for f in sorted(glob.glob("/weights/sample/*.pth")): summary(f)
for f in ["/data/test/cham_1b.pth", "/data/test/pell_p5.pth"]: summary(f)
"""
    import subprocess
    r = subprocess.run([PY, "-c", script], capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:] if r.returncode else "")


@app.function(volumes=VOLS, timeout=60 * 20, cpu=2, memory=16384)
def summary(folder="test"):
    """How many instances came out per plot, without downloading the files."""
    script = """
import glob, os
import numpy as np
print(f"{'plot':26} {'points':>9} {'assigned':>10} {'%':>7} {'instances':>11}")
for f in sorted(glob.glob("/outputs/FOLDER/assigned_*.npy")):
    n = os.path.basename(f)[len("assigned_"):-4]
    a = np.load(f).reshape(-1)
    e = np.load(f"/outputs/FOLDER/instance_pred_{n}.npy").reshape(-1)
    ni = len(np.unique(e[a])) if len(e) == len(a) and a.sum() else 0
    print(f"{n:26} {len(a):>9} {int(a.sum()):>10} {100*a.mean():>6.1f}% {ni:>11}")
"""
    import subprocess
    r = subprocess.run([PY, "-c", script.replace("FOLDER", folder)],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:] if r.returncode else "")


@app.function(volumes=VOLS, timeout=60 * 30, cpu=4, memory=32768)
def detections(folder="test", data_dir="test"):
    """Turns the labels into detections inside the container and returns the JSON.

    It is the same as evaluate_treelite.py does locally, but here it avoids downloading
    hundreds of megabytes per variant: only apex, height and area of each instance come
    back. The coordinates are rebuilt from the .pth plus the origin that the converter
    saved, which is exactly what the model received.
    """
    script = """
import glob, json, os
import numpy as np
import torch
MIN_PTS = 10
def area_hull(xy):
    # convex hull by monotone chain and area by the shoelace theorem; same
    # definition as ConvexHull(xy).volume, which in 2D is the area
    if len(xy) < 3:
        return 0.5
    pts = sorted(set(map(tuple, np.round(xy, 6))))
    if len(pts) < 3:
        return 0.5
    def half(pts):
        r = []
        for q in pts:
            while len(r) >= 2 and (r[-1][0]-r[-2][0])*(q[1]-r[-2][1]) - \
                                  (r[-1][1]-r[-2][1])*(q[0]-r[-2][0]) <= 0:
                r.pop()
            r.append(q)
        return r
    c = half(pts)[:-1] + half(pts[::-1])[:-1]
    if len(c) < 3:
        return 0.5
    a = sum(c[i][0]*c[(i+1) % len(c)][1] - c[(i+1) % len(c)][0]*c[i][1]
            for i in range(len(c))) / 2
    return max(abs(a), 0.5)

output = {}
for f in sorted(glob.glob("/outputs/FOLDER/assigned_*.npy")):
    n = os.path.basename(f)[len("assigned_"):-4]
    if not os.path.exists("/data/DATA/%s.pth" % n):
        continue
    asig = np.load(f).reshape(-1)
    label = np.load("/outputs/FOLDER/instance_pred_%s.npy" % n).reshape(-1)
    d = torch.load("/data/DATA/%s.pth" % n)
    org = np.load("/data/DATA/%s_origin.npy" % n)
    xyz = np.asarray(d["coord"]) + org
    if len(label) == 0 and asig.sum() == 0:
        # with no proposal at all the model returns an empty array, not a failure
        output[n] = dict(x=[], y=[], h=[], area=[], n=[],
                         points=len(xyz), assigned=0)
        continue
    if not (len(label) == len(asig) == len(xyz)):
        output[n] = dict(error="%d labels, %d marks, %d points" %
                         (len(label), len(asig), len(xyz)))
        continue
    X, Y, H, A, N = [], [], [], [], []
    for i in np.unique(label[asig]):
        m = asig & (label == i)
        if m.sum() < MIN_PTS:
            continue
        k = int(np.argmax(xyz[m, 2]))
        X.append(float(xyz[m][k, 0])); Y.append(float(xyz[m][k, 1]))
        H.append(float(xyz[m, 2].max())); A.append(area_hull(xyz[m][:, :2]))
        N.append(int(m.sum()))
    output[n] = dict(x=X, y=Y, h=H, area=A, n=N,
                     points=len(xyz), assigned=int(asig.sum()))
print("JSON" + json.dumps(output))
"""
    import subprocess
    r = subprocess.run(
        [PY, "-c", script.replace("FOLDER", folder).replace("DATA", data_dir)],
        capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        return {}
    import json
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")][0]
    return json.loads(line[4:])


@app.function(volumes=VOLS, timeout=60 * 20, cpu=2, memory=32768)
def check_order(folder="test", data_dir="test"):
    """That the labels come back in the order of the input points.

    The tester saves origin_coord, which are the coordinates the model saw after
    CenterShift. If the order is the same as the one in the .pth, centering both clouds
    makes them match point by point. If it were not, every detection would be garbage.
    """
    script = """
import glob, os
import numpy as np
import torch
for f in sorted(glob.glob("/outputs/FOLDER/orig_coord_*.npy")):
    n = os.path.basename(f)[len("orig_coord_"):-4]
    if not os.path.exists("/data/DATA/%s.pth" % n):
        continue
    oc = np.load(f)
    c = np.asarray(torch.load("/data/DATA/%s.pth" % n)["coord"])
    if len(oc) != len(c):
        print("%-14s DIFFERENT NUMBER OF POINTS %d vs %d" % (n, len(oc), len(c)))
        continue
    dv = np.abs((oc - oc.mean(0)) - (c - c.mean(0))).max(1)
    print("%-14s %8d points  type %s  p50 %.2e  p99.99 %.2e  max %.2e  "
          "more than 1 cm: %d" % (n, len(c), oc.dtype, np.percentile(dv, 50),
                               np.percentile(dv, 99.99), dv.max(), int((dv > 0.01).sum())))
"""
    import subprocess
    r = subprocess.run(
        [PY, "-c", script.replace("FOLDER", folder).replace("DATA", data_dir)],
        capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:] if r.returncode else "")


@app.local_entrypoint()
def main(plots: str = "", sample: bool = False, folder: str = "test",
         options: str = "", label: str = ""):
    code, copied = infer.remote(
        plots.split(",") if plots else None, sample, folder, options, label)
    print(f"\ncode {code}, {len(copied)} files")
    for c in copied[:8]:
        print("   ", c)


@app.local_entrypoint()
def sweep(thresholds: str = "1,2,3,4,5,6,8,10", folder: str = "test",
            extra: str = "", suffix: str = ""):
    """Sweeps the clustering radius and saves the detections of each setting.

    The effective radius is cluster_thresh * voxel_size, with voxel_size 0.05 m: the
    value 1 they come with by default is 5 cm, meant for clouds of thousands of points
    per square meter. They are all launched at once and each one leaves its JSON.
    """
    import json

    output = os.path.join(os.path.dirname(os.path.dirname(HERE)), "treelite_output")
    os.makedirs(output, exist_ok=True)
    vals = [v.strip() for v in thresholds.split(",") if v.strip()]
    launched = {}
    for v in vals:
        et = f"t{v}{suffix}"
        op = f"model.cluster_thresh={v}" + (f" {extra}" if extra else "")
        launched[v] = (et, infer.spawn(None, False, folder, op, et))
    for v, (et, h) in launched.items():
        code, copied = h.get()
        print(f"threshold {v:>3} ({float(v) * 0.05:.2f} m): code {code}, "
              f"{len(copied)} files")
    for v, (et, _) in launched.items():
        det = detections.remote(et, folder)
        with open(os.path.join(output, f"detections_{et}.json"), "w") as f:
            json.dump(det, f)
        n = sum(len(d.get("x", [])) for d in det.values())
        print(f"threshold {v:>3}: {n} instances in {len(det)} plots")


@app.local_entrypoint()
def extract(folder: str = "test", data: str = "test", output: str = ""):
    """Downloads the detections of a results folder into a local JSON."""
    import json

    det = detections.remote(folder, data)
    root = os.path.dirname(os.path.dirname(HERE))
    path = os.path.join(root, "treelite_output",
                        output or f"detections_{folder}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(det, f)
    n = sum(len(d.get("x", [])) for d in det.values())
    print(f"{len(det)} plots, {n} instances -> {path}")
