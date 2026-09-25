"""Paso 2 y 3: corre ForestFormer3D por su entrypoint oficial sobre nuestras parcelas.

Disposicion que espera el pipeline, leida de tools/inference_bluepoint_forestsens.sh
y de run_oracle_pipeline.sh:
    entrada  /workspace/data/ForAINetV2/test_data   (acepta .las y .laz)
    pesos    /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth
    salida   /workspace/work_dirs/output

Modal no deja montar un volumen sobre una ruta que ya tiene contenido, y work_dirs
trae marcadores del repositorio. Asi que los pesos se montan en /pesos y se le pasa
la ruta por MODEL_PATH, variable que inference_bluepoint_forestsens.sh respeta. Las
entradas llegan en su propio volumen y se copian dentro, porque el pipeline borra su
carpeta de entrada al acabar.

Uso:
  MODAL_PROFILE=treelearn modal run respaldo_minimo/modal_app/run_ff3d.py            # una parcela
  MODAL_PROFILE=treelearn modal run respaldo_minimo/modal_app/run_ff3d.py --todas    # las que haya
"""
import os
import subprocess

import modal

AQUI = os.path.dirname(os.path.abspath(__file__))
CTX = os.path.join(AQUI, "ff3d_repo", "ff3d_forestsens")

image = (modal.Image.from_dockerfile(os.path.join(CTX, "Dockerfile"),
                                     context_dir=CTX, add_python="3.11")
         .entrypoint([]))

pesos = modal.Volume.from_name("ff3d-pesos")
entradas = modal.Volume.from_name("ff3d-entradas")
salidas = modal.Volume.from_name("ff3d-salidas")

app = modal.App("ff3d-run", image=image)
GPU = os.environ.get("FF3D_GPU", "T4")


@app.function(gpu=GPU, timeout=6 * 3600, memory=16384,
              volumes={"/pesos": pesos, "/entradas": entradas, "/salidas": salidas})
def correr(nombres):
    """Procesa varias parcelas EN EL MISMO contenedor.

    Se hace asi a proposito: la primera paga el montaje del entorno, sobre todo la
    recompilacion de torch_points_kernels que el entrypoint hace en caliente, y las
    siguientes dan el coste marginal de verdad."""
    if isinstance(nombres, str):
        nombres = [nombres]
    import glob
    import shutil
    import time

    IN = "/workspace/data/ForAINetV2/test_data"
    OUT = "/workspace/work_dirs/output"
    resultados = []
    for idx, nombre in enumerate(nombres):
      os.makedirs(IN, exist_ok=True)
      os.makedirs(OUT, exist_ok=True)
      for f in glob.glob(os.path.join(IN, "*")):
        os.remove(f) if os.path.isfile(f) else shutil.rmtree(f, ignore_errors=True)
      base = os.path.basename(nombre)
      shutil.copy(f"/entradas/{nombre}", os.path.join(IN, base))
      print(f"[{idx+1}/{len(nombres)}] entrada: {nombre}, "
            f"{os.path.getsize(os.path.join(IN, base))/1e6:.1f} MB", flush=True)

      env = {**os.environ,
             "MODEL_PATH": "/pesos/clean_forestformer/epoch_3000_fix.pth",
             "IN_BUCKET": IN, "OUT_BUCKET": OUT}
      t0 = time.time()
      r = subprocess.run(["bash", "-lc", "bash /workspace/entrypoint_ff3d.sh"],
                         capture_output=True, text=True, cwd="/workspace", env=env)
      dt = time.time() - t0
      print(f"--- {nombre}: {dt:.0f} s, codigo {r.returncode}", flush=True)
      if idx == 0:
        print("CABECERA:", r.stdout[:4000], flush=True)
      if r.returncode:
        print("STDERR:", r.stderr[-4000:], flush=True)

      producidos = []
      for raiz, _, archivos in os.walk(OUT):
        for a in archivos:
            pth = os.path.join(raiz, a)
            producidos.append((os.path.relpath(pth, OUT), os.path.getsize(pth)))
            shutil.copy(pth, f"/salidas/{base.replace('.las','')}__{a}")
      salidas.commit()
      resultados.append({"parcela": nombre, "segundos": round(dt, 1),
                         "codigo": r.returncode, "archivos": producidos})
      print("salida:", producidos, flush=True)
    return resultados


@app.local_entrypoint()
def main(todas: bool = False, nombres: str = "", tandas_n: int = 4):
    import json
    if nombres:
        lista = [n.strip() for n in nombres.split(",") if n.strip()]
        print(f"{len(lista)} nubes: {lista[:3]} ...")
        tandas = [lista[i::tandas_n] for i in range(tandas_n)]
        res = [x for t in correr.map(tandas) for x in t]
    elif todas:
        nombres = sorted(f.path.lstrip("/") for f in entradas.listdir("/")
                         if f.path.endswith(".las"))
        print(f"{len(nombres)} parcelas: {nombres}")
        # se reparten en tandas para que cada contenedor amortice su montaje
        tandas = [nombres[i::4] for i in range(4)]
        res = [x for t in correr.map(tandas) for x in t]
    else:
        res = correr.remote(["cham_1b.las", "pell_p1.las"])
    print(json.dumps(res, indent=2, ensure_ascii=False))
