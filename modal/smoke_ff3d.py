"""ForestFormer3D smoke test on Modal.

The risky part here is building the image, not the inference: the official Dockerfile
compiles MinkowskiEngine, torch-scatter and the ScanNet segmenter from source, on top
of a PyTorch 1.13.1 base with CUDA 11.6. If that goes through, the rest is
mechanical.

Four things are checked, in this order:
  1. that the image builds,
  2. that the GPU responds and torch sees it,
  3. that MinkowskiEngine and the modules the model uses import,
  4. that laspy reads a .las, because the pipeline takes LAS and LAZ directly.

Usage: MODAL_PROFILE=<your-profile> modal run modal/smoke_ff3d.py
"""
import os
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "ff3d_repo")
CTX = os.path.join(REPO, "ff3d_forestsens")


def fetch_repo():
    """The Dockerfile does COPY . /workspace and its comment says the build
    context must be the ff3d_forestsens folder. So the whole repository is
    needed, not just the Dockerfile."""
    if not os.path.isdir(CTX):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/bxiang233/FF3D_inference.git", REPO], check=True)
    return os.path.join(CTX, "Dockerfile")


# The base image ships with python 3.10 with everything installed. Modal needs its own python for
# the client, so the real work is launched as a subprocess with the image's one, just
# like we did with SegmentAnyTree.
# context_dir is essential: without it Modal takes the current working directory as the
# context and the Dockerfile's COPY . /workspace copies the wrong thing.
image = (modal.Image.from_dockerfile(fetch_repo(), context_dir=CTX, add_python="3.11")
         .entrypoint([]))

app = modal.App("ff3d-smoke", image=image)

PROBE = r'''
import subprocess, sys
print("image python:", sys.version.split()[0], flush=True)
import torch
print("torch:", torch.__version__, "| torch's cuda:", torch.version.cuda, flush=True)
print("gpu available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0), flush=True)
    a = torch.randn(2048, 2048, device="cuda")
    print("matmul on gpu ok:", float((a @ a).sum()) == float((a @ a).sum()), flush=True)
for m in ("MinkowskiEngine", "mmengine", "mmdet3d", "spconv", "torch_scatter",
          "torch_points_kernels", "laspy"):
    try:
        mod = __import__(m)
        print(f"  {m:22} ok  {getattr(mod, '__version__', '')}", flush=True)
    except Exception as e:
        print(f"  {m:22} FAILS  {type(e).__name__}: {e}", flush=True)
'''


# The A100 requires a payment method even with free credit. The T4 does not, and this
# account already uses it with SegmentAnyTree. MinkowskiEngine was compiled for 6.0 to
# 8.6, so the T4 (7.5) fits. If memory runs short, the README says to lower "chunk"
# in the config.
GPU = os.environ.get("FF3D_GPU", "T4")


@app.function(gpu=GPU, timeout=60 * 60)
def probe():
    py = "/opt/conda/bin/python"
    if not os.path.exists(py):
        py = subprocess.run(["bash", "-lc", "command -v python3"], capture_output=True,
                            text=True).stdout.strip()
    print("=== nvidia-smi", flush=True)
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
    print(f"=== probe with {py}", flush=True)
    r = subprocess.run([py, "-c", PROBE], capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode:
        print("STDERR:", r.stderr[-3000:], flush=True)
    return r.returncode


@app.local_entrypoint()
def main():
    code = probe.remote()
    print("probe exit code:", code)
