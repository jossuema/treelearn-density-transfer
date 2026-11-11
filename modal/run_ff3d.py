"""Steps 2 and 3: runs ForestFormer3D through its official entrypoint on our plots.

Layout that the pipeline expects, read from tools/inference_bluepoint_forestsens.sh
and from run_oracle_pipeline.sh:
    input    /workspace/data/ForAINetV2/test_data   (takes .las and .laz)
    weights  /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth
    output   /workspace/work_dirs/output

Modal will not let you mount a volume over a path that already has content, and
work_dirs carries markers from the repository. So the weights are mounted at /weights
and the path is passed by MODEL_PATH, a variable that inference_bluepoint_forestsens.sh
respects. The inputs arrive in their own volume and are copied inside, because the
pipeline deletes its input folder when it finishes.

Usage:
  MODAL_PROFILE=<your-profile> modal run modal/run_ff3d.py            # one plot
  MODAL_PROFILE=<your-profile> modal run modal/run_ff3d.py --all-available  # all of them
"""
import os
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = os.path.join(HERE, "ff3d_repo", "ff3d_forestsens")

image = (modal.Image.from_dockerfile(os.path.join(CTX, "Dockerfile"),
                                     context_dir=CTX, add_python="3.11")
         .entrypoint([]))

weights = modal.Volume.from_name("ff3d-weights")
inputs = modal.Volume.from_name("ff3d-inputs")
outputs = modal.Volume.from_name("ff3d-outputs")

app = modal.App("ff3d-run", image=image)
GPU = os.environ.get("FF3D_GPU", "T4")


@app.function(gpu=GPU, timeout=6 * 3600, memory=16384,
              volumes={"/weights": weights, "/inputs": inputs, "/outputs": outputs})
def run(names):
    """Processes several plots IN THE SAME container.

    It is done this way on purpose: the first one pays for setting up the environment,
    above all the recompilation of torch_points_kernels that the entrypoint does on the
    fly, and the following ones give the real marginal cost."""
    if isinstance(names, str):
        names = [names]
    import glob
    import shutil
    import time

    IN = "/workspace/data/ForAINetV2/test_data"
    OUT = "/workspace/work_dirs/output"
    results = []
    for idx, name in enumerate(names):
      os.makedirs(IN, exist_ok=True)
      os.makedirs(OUT, exist_ok=True)
      for f in glob.glob(os.path.join(IN, "*")):
        os.remove(f) if os.path.isfile(f) else shutil.rmtree(f, ignore_errors=True)
      base = os.path.basename(name)
      shutil.copy(f"/inputs/{name}", os.path.join(IN, base))
      print(f"[{idx+1}/{len(names)}] input: {name}, "
            f"{os.path.getsize(os.path.join(IN, base))/1e6:.1f} MB", flush=True)

      env = {**os.environ,
             "MODEL_PATH": "/weights/clean_forestformer/epoch_3000_fix.pth",
             "IN_BUCKET": IN, "OUT_BUCKET": OUT}
      t0 = time.time()
      r = subprocess.run(["bash", "-lc", "bash /workspace/entrypoint_ff3d.sh"],
                         capture_output=True, text=True, cwd="/workspace", env=env)
      dt = time.time() - t0
      print(f"--- {name}: {dt:.0f} s, code {r.returncode}", flush=True)
      if idx == 0:
        print("HEADER:", r.stdout[:4000], flush=True)
      if r.returncode:
        print("STDERR:", r.stderr[-4000:], flush=True)

      produced = []
      for root, _, files in os.walk(OUT):
        for a in files:
            pth = os.path.join(root, a)
            produced.append((os.path.relpath(pth, OUT), os.path.getsize(pth)))
            shutil.copy(pth, f"/outputs/{base.replace('.las','')}__{a}")
      outputs.commit()
      results.append({"plot": name, "seconds": round(dt, 1),
                         "code": r.returncode, "files": produced})
      print("output:", produced, flush=True)
    return results


@app.local_entrypoint()
def main(all_available: bool = False, names: str = "", batches_n: int = 4):
    import json
    if names:
        name_list = [n.strip() for n in names.split(",") if n.strip()]
        print(f"{len(name_list)} clouds: {name_list[:3]} ...")
        batches = [name_list[i::batches_n] for i in range(batches_n)]
        res = [x for t in run.map(batches) for x in t]
    elif all_available:
        names = sorted(f.path.lstrip("/") for f in inputs.listdir("/")
                         if f.path.endswith(".las"))
        print(f"{len(names)} plots: {names}")
        # they are split into batches so that each container amortizes its own setup
        batches = [names[i::4] for i in range(4)]
        res = [x for t in run.map(batches) for x in t]
    else:
        res = run.remote(["cham_1b.las", "pell_p1.las"])
    print(json.dumps(res, indent=2, ensure_ascii=False))
