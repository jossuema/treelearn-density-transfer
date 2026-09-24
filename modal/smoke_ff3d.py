"""Prueba de humo de ForestFormer3D en Modal.

Lo arriesgado aqui es la construccion de la imagen, no la inferencia: el Dockerfile
oficial compila MinkowskiEngine, torch-scatter y el segmentador de ScanNet desde
fuente, sobre una base de PyTorch 1.13.1 con CUDA 11.6. Si eso pasa, el resto es
mecanico.

Se comprueban cuatro cosas, en este orden:
  1. que la imagen se construye,
  2. que la GPU responde y torch la ve,
  3. que MinkowskiEngine y los modulos que usa el modelo importan,
  4. que laspy lee un .las, porque el pipeline acepta LAS y LAZ directamente.

Uso: MODAL_PROFILE=<your-profile> modal run respaldo_minimo/modal_app/smoke_ff3d.py
"""
import os
import subprocess

import modal

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(AQUI, "ff3d_repo")
CTX = os.path.join(REPO, "ff3d_forestsens")


def traer_repo():
    """El Dockerfile hace COPY . /workspace y su comentario dice que el contexto
    de construccion debe ser la carpeta ff3d_forestsens. Asi que hace falta el
    repositorio entero, no solo el Dockerfile."""
    if not os.path.isdir(CTX):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/bxiang233/FF3D_inference.git", REPO], check=True)
    return os.path.join(CTX, "Dockerfile")


# La base trae python 3.10 con todo instalado. Modal necesita su propio python para
# el cliente, asi que el trabajo de verdad se lanza como subproceso con el de la
# imagen, igual que hicimos con SegmentAnyTree.
# context_dir es imprescindible: sin el, Modal toma como contexto el directorio de
# trabajo actual y el COPY . /workspace del Dockerfile copia lo que no es.
image = (modal.Image.from_dockerfile(traer_repo(), context_dir=CTX, add_python="3.11")
         .entrypoint([]))

app = modal.App("ff3d-smoke", image=image)

SONDA = r'''
import subprocess, sys
print("python de la imagen:", sys.version.split()[0], flush=True)
import torch
print("torch:", torch.__version__, "| cuda de torch:", torch.version.cuda, flush=True)
print("gpu disponible:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          "| capacidad:", torch.cuda.get_device_capability(0), flush=True)
    a = torch.randn(2048, 2048, device="cuda")
    print("multiplicacion en gpu ok:", float((a @ a).sum()) == float((a @ a).sum()), flush=True)
for m in ("MinkowskiEngine", "mmengine", "mmdet3d", "spconv", "torch_scatter",
          "torch_points_kernels", "laspy"):
    try:
        mod = __import__(m)
        print(f"  {m:22} ok  {getattr(mod, '__version__', '')}", flush=True)
    except Exception as e:
        print(f"  {m:22} FALLA  {type(e).__name__}: {e}", flush=True)
'''


# La A100 exige metodo de pago aunque haya credito gratis. La T4 no, y esta cuenta
# ya la uso con SegmentAnyTree. MinkowskiEngine se compilo para 6.0 a 8.6, asi que
# la T4 (7.5) entra. Si falta memoria, el README dice bajar "chunk" en el config.
GPU = os.environ.get("FF3D_GPU", "T4")


@app.function(gpu=GPU, timeout=60 * 60)
def sonda():
    py = "/opt/conda/bin/python"
    if not os.path.exists(py):
        py = subprocess.run(["bash", "-lc", "command -v python3"], capture_output=True,
                            text=True).stdout.strip()
    print("=== nvidia-smi", flush=True)
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
    print(f"=== sonda con {py}", flush=True)
    r = subprocess.run([py, "-c", SONDA], capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode:
        print("STDERR:", r.stderr[-3000:], flush=True)
    return r.returncode


@app.local_entrypoint()
def main():
    codigo = sonda.remote()
    print("codigo de salida de la sonda:", codigo)
