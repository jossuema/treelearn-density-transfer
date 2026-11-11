"""Smoke test: does SegmentAnyTree's CUDA 11.1 image run on Modal?

The question is specific: Modal's servers run driver 580.95.05 with CUDA 13.0 and their
documentation says that major versions 11.x "may not" be compatible. The SegmentAnyTree
image is compiled with CUDA 11.1 and torch 1.9.0+cu111.

This test answers four things, in order of importance:
  1. does torch 1.9.0+cu111 see the GPU and run a real operation on it?
  2. does MinkowskiEngine (a compiled extension) import and do a sparse convolution?
  3. is the checkpoint inside the image the real file (665 MB) or the Git-LFS pointer (134 B)?
  4. which exact versions are inside?

The image used is donaldmaen/segment-any-tree:latest (14 Jan 2025), the one a user reports
as working in issue #31 of the repository; the author's most recent one has two open
pandas bugs.

GPU: T4, the cheapest (0.59 USD/h) and within the list of architectures the image was
compiled for (6.0;7.0;7.5;8.0;8.6 -> T4 is sm_75).

Usage:
    modal profile activate <your-profile>
    modal run modal/smoke_sat.py
"""
import json
import subprocess

import modal

APP = "sat-smoke"
IMG = "donaldmaen/segment-any-tree:latest"
CKPT = "/home/nibio/mutable-outside-world/model_file/PointGroup-PAPER.pt"

# add_python: the image ships with python3.8 and Modal's runtime needs 3.10 or newer.
# entrypoint([]): the image has ENTRYPOINT ["bash","run_oracle_pipeline.sh"], clear it.
image = modal.Image.from_registry(IMG, add_python="3.11").entrypoint([])

app = modal.App(APP, image=image)

# The probe runs with the image's interpreter (python3.8), not with Modal's.
PROBE = r'''
import json, os, sys
out = {"python": sys.version.split()[0], "exe": sys.executable}

try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda_build"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))
        # THE real test: a real operation on the GPU, not just is_available()
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        out["gpu_matmul_ok"] = bool(c == c)   # False if it is NaN
except Exception as e:
    out["torch_error"] = "%s: %s" % (type(e).__name__, e)

try:
    import MinkowskiEngine as ME
    out["minkowski"] = ME.__version__
    import torch
    coords = torch.IntTensor([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,1,1]])
    feats = torch.rand(4, 3)
    x = ME.SparseTensor(features=feats.cuda(), coordinates=coords.cuda())
    conv = ME.MinkowskiConvolution(3, 8, kernel_size=3, dimension=3).cuda()
    y = conv(x)
    out["minkowski_conv_ok"] = tuple(y.F.shape)[1] == 8
except Exception as e:
    out["minkowski_error"] = "%s: %s" % (type(e).__name__, e)

ck = "CKPT_PATH"
out["ckpt_exists"] = os.path.exists(ck)
if out["ckpt_exists"]:
    n = os.path.getsize(ck)
    out["ckpt_bytes"] = n
    # 665,666,007 = the real file; ~134 = Git-LFS pointer
    out["ckpt_is_real"] = n > 10_000_000
    with open(ck, "rb") as f:
        head = f.read(4)
    out["ckpt_magic"] = head.hex()          # 504b0304 = torch.save zip
    if not out["ckpt_is_real"]:
        with open(ck) as f:
            out["ckpt_head_text"] = f.read(200)

for mod in ("pandas", "numpy", "laspy", "hydra", "torch_points_kernels"):
    try:
        m = __import__(mod)
        out[mod] = getattr(m, "__version__", "?")
    except Exception as e:
        out[mod] = "ERROR %s" % type(e).__name__

print("PROBE_JSON_START" + json.dumps(out) + "PROBE_JSON_END")
'''.replace("CKPT_PATH", CKPT)


@app.function(gpu="T4", timeout=1800)
def smoke() -> dict:
    res = {}

    # 1. Which interpreters the image has and which one has torch
    cands = subprocess.run(
        "ls -1 /usr/bin/python3* /usr/local/bin/python3* 2>/dev/null | sort -u",
        shell=True, capture_output=True, text=True).stdout.split()
    res["interpretes"] = cands

    interp = None
    for c in cands:
        ok = subprocess.run([c, "-c", "import torch"], capture_output=True)
        if ok.returncode == 0:
            interp = c
            break
    res["interpreter_with_torch"] = interp

    # 2. nvidia-smi as seen from inside the container
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True)
    res["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()[:300]

    if interp is None:
        res["error"] = "no interpreter in the image imports torch"
        print(json.dumps(res, indent=2))
        return res

    # 3. The real probe
    p = subprocess.run([interp, "-c", PROBE], capture_output=True, text=True,
                       cwd="/home/nibio/mutable-outside-world")
    raw = p.stdout
    if "PROBE_JSON_START" in raw:
        res["probe"] = json.loads(raw.split("PROBE_JSON_START")[1].split("PROBE_JSON_END")[0])
    else:
        res["probe_stdout"] = raw[-3000:]
    res["probe_returncode"] = p.returncode
    if p.returncode != 0 or "probe" not in res:
        res["probe_stderr"] = p.stderr[-3000:]

    print(json.dumps(res, indent=2))
    return res


@app.local_entrypoint()
def main():
    r = smoke.remote()
    print("\n" + "=" * 70)
    pr = r.get("probe", {})
    verdict = []
    verdict.append(("torch sees the GPU", pr.get("cuda_available")))
    verdict.append(("real operation on the GPU", pr.get("gpu_matmul_ok")))
    verdict.append(("MinkowskiEngine imports", "minkowski" in pr))
    verdict.append(("sparse conv on the GPU", pr.get("minkowski_conv_ok")))
    verdict.append(("real checkpoint (not an LFS pointer)", pr.get("ckpt_is_real")))
    for k, v in verdict:
        print(("  OK   " if v else "  FAILS") + "  " + k)
    print("=" * 70)
    with open("smoke_sat_result.json", "w") as f:
        json.dump(r, f, indent=2)
    print("full result in smoke_sat_result.json")
