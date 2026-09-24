"""Prueba de humo: ¿corre la imagen CUDA 11.1 de SegmentAnyTree en Modal?

La duda es concreta: los servidores de Modal llevan driver 580.95.05 con CUDA 13.0 y su
documentación dice que las versiones mayores 11.x "puede que no" sean compatibles. La imagen
de SegmentAnyTree está compilada con CUDA 11.1 y torch 1.9.0+cu111.

Esta prueba responde cuatro cosas, en orden de importancia:
  1. ¿torch 1.9.0+cu111 ve la GPU y ejecuta una operación real en ella?
  2. ¿MinkowskiEngine (extensión compilada) importa y hace una convolución dispersa?
  3. ¿El checkpoint dentro de la imagen es el archivo real (665 MB) o el puntero de Git-LFS (134 B)?
  4. ¿Qué versiones exactas hay dentro?

Se usa la imagen donaldmaen/segment-any-tree:latest (14 ene 2025), que es la que un usuario
reporta como funcional en el issue #31 del repositorio; la más reciente del autor tiene dos
fallos abiertos de pandas.

GPU: T4, la más barata (0.59 USD/h) y dentro de la lista de arquitecturas con la que se
compiló la imagen (6.0;7.0;7.5;8.0;8.6 -> T4 es sm_75).

Uso:
    modal profile activate <your-profile>
    modal run respaldo_minimo/modal_app/smoke_sat.py
"""
import json
import subprocess

import modal

APP = "sat-smoke"
IMG = "donaldmaen/segment-any-tree:latest"
CKPT = "/home/nibio/mutable-outside-world/model_file/PointGroup-PAPER.pt"

# add_python: la imagen trae python3.8 y el runtime de Modal necesita 3.10 o mas.
# entrypoint([]): la imagen tiene ENTRYPOINT ["bash","run_oracle_pipeline.sh"], hay que limpiarlo.
image = modal.Image.from_registry(IMG, add_python="3.11").entrypoint([])

app = modal.App(APP, image=image)

# El probe corre con el interprete de la imagen (python3.8), no con el de Modal.
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
        # LA prueba de verdad: una operacion real en GPU, no solo is_available()
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        out["gpu_matmul_ok"] = bool(c == c)   # False si es NaN
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
    # 665,666,007 = archivo real; ~134 = puntero de Git-LFS
    out["ckpt_is_real"] = n > 10_000_000
    with open(ck, "rb") as f:
        head = f.read(4)
    out["ckpt_magic"] = head.hex()          # 504b0304 = zip de torch.save
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

    # 1. Que interpretes hay en la imagen y cual tiene torch
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
    res["interprete_con_torch"] = interp

    # 2. nvidia-smi visto desde dentro del contenedor
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True)
    res["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()[:300]

    if interp is None:
        res["error"] = "ningun interprete de la imagen importa torch"
        print(json.dumps(res, indent=2))
        return res

    # 3. El probe de verdad
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
    veredicto = []
    veredicto.append(("torch ve la GPU", pr.get("cuda_available")))
    veredicto.append(("operacion real en GPU", pr.get("gpu_matmul_ok")))
    veredicto.append(("MinkowskiEngine importa", "minkowski" in pr))
    veredicto.append(("conv dispersa en GPU", pr.get("minkowski_conv_ok")))
    veredicto.append(("checkpoint real (no puntero LFS)", pr.get("ckpt_is_real")))
    for k, v in veredicto:
        print(("  OK   " if v else "  FALLA") + "  " + k)
    print("=" * 70)
    with open("smoke_sat_result.json", "w") as f:
        json.dump(r, f, indent=2)
    print("resultado completo en smoke_sat_result.json")
