"""Inferencia de TreeLite3D (Rizaldy et al., RSE 2026) sobre nuestras 22 parcelas.

Imagen oficial de Pointcept 1.5.0, que es sobre lo que esta construido TreeLite3D.
El repositorio se clona al construir y se le aplican dos retoques de nombrado y
guardado (ver parche_treelite.py). Los pesos son los publicados por los autores.

    MODAL_PROFILE=treelearn modal run respaldo_minimo/modal_app/run_treelite.py::sonda
    MODAL_PROFILE=treelearn modal run --detach respaldo_minimo/modal_app/run_treelite.py

El probador guarda en exp/trees/<exp>/result/ un .npy por parcela con la etiqueta de
instancia de cada punto, en el orden de los puntos de entrada. Al terminar se copia
todo al volumen de salidas.
"""
import os
import subprocess

import modal

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = "/workspace/TreeLite3D"
EXP = "TREE-UNIFIED_trees_insseg-pointgroup-v1m1-0-spunet-base"
CFG = "TREE-UNIFIED-insseg-pointgroup-v1m1-0-spunet-base"
PY = os.environ.get("TL_PY", "/opt/conda/bin/python")
GPU = os.environ.get("TL_GPU", "T4")

pesos = modal.Volume.from_name("treelite-pesos")
datos = modal.Volume.from_name("treelite-datos")
salidas = modal.Volume.from_name("treelite-salidas", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "pointcept/pointcept:v1.5.0-pytorch1.11.0-cuda11.3-cudnn8-devel",
        add_python="3.11",
    )
    # la imagen pone /opt/conda/bin (python 3.8) delante en el PATH y el cliente de
    # Modal necesita 3.10 o mas. El trabajo de verdad se lanza con la ruta absoluta
    # del interprete de conda, que es el que tiene torch, asi que reordenar no estorba.
    .env({"PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/conda/bin"})
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        f"git clone --depth 1 https://github.com/aldinorizaldy/TreeLite3D.git {REPO}",
        # el fichero de configuracion del repositorio se llama literalmente "... copy.py",
        # y el guion -c de test.sh lo busca sin ese sufijo
        f"cd {REPO}/configs/trees && mv '{CFG} copy.py' '{CFG}.py'",
    )
    .add_local_file(os.path.join(AQUI, "parche_treelite.py"), "/parche.py", copy=True)
    .run_commands("python3 /parche.py")
)
app = modal.App("treelite", image=image)
VOLS = {"/pesos": pesos, "/datos": datos, "/salidas": salidas}


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 30)
def sonda():
    """Comprueba interprete, GPU y las extensiones compiladas antes de gastar horas."""
    subprocess.run("nvidia-smi | head -12", shell=True)
    guion = (
        "import torch, numpy, sys; "
        "print('python', sys.version.split()[0], 'torch', torch.__version__, "
        "'numpy', numpy.__version__); "
        "print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); "
        "import pointops, pointgroup_ops, spconv; print('extensiones ok'); "
        "d = torch.load('/datos/test/cham_1b.pth'); "
        "print('pth', {k: (v.shape, str(v.dtype)) for k, v in d.items()})"
    )
    r = subprocess.run([PY, "-c", guion], cwd=REPO, capture_output=True, text=True)
    print(r.stdout or "", r.stderr[-3000:] if r.returncode else "")
    return r.returncode


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 60 * 5, cpu=4, memory=32768)
def inferir(nombres=None, muestra=False, carpeta="test",
            opciones="", etiqueta=""):
    import glob
    import shutil
    import time

    # los pesos y los datos viven en volumenes; el repositorio espera encontrarlos dentro
    modelo = f"{REPO}/exp/trees/{EXP}/model"
    os.makedirs(modelo, exist_ok=True)
    if not os.path.exists(f"{modelo}/model_best.pth"):
        os.symlink("/pesos/pesos/model_best.pth", f"{modelo}/model_best.pth")
    prueba = f"{REPO}/data/trees_UNIFIED/test"
    shutil.rmtree(prueba, ignore_errors=True)
    os.makedirs(prueba)
    todos = sorted(os.path.basename(p) for p in glob.glob(f"/datos/{carpeta}/*.pth"))
    elegidos = [n for n in todos if nombres is None or n[:-4] in nombres]
    for n in elegidos:
        os.symlink(f"/datos/{carpeta}/{n}", f"{prueba}/{n}")
    if muestra:  # la parcela de ejemplo de los autores, como control del montaje
        for f in sorted(glob.glob("/pesos/muestra/*.pth")):
            os.symlink(f, f"{prueba}/{os.path.basename(f)}")
            elegidos.append(os.path.basename(f))
    print(f"{len(elegidos)} parcelas: {', '.join(n[:-4] for n in elegidos)}")

    t0 = time.time()
    # lo mismo que hace scripts/test.sh, pero directo: asi se pueden anadir claves de
    # configuracion al vuelo, que test.sh no deja porque construye el --options el solo
    exd = f"exp/trees/{EXP}"
    cmd = (f"PYTHONPATH=./ {PY} -u tools/test.py --config-file configs/trees/{CFG}.py"
           f" --num-gpus 1 --options save_path={exd}"
           f" weight={exd}/model/model_best.pth {opciones}")
    print(f"--- {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-8000:])
    if r.returncode:
        print("--- error\n" + r.stderr[-8000:])
    print(f"--- codigo {r.returncode}, {time.time() - t0:.0f} s")

    # se copia pase lo que pase: el probador guarda dentro del bucle, asi que puede
    # haber resultados aunque la evaluacion final (que usa una verdad ficticia) reviente
    res = f"{REPO}/exp/trees/{EXP}/result"
    destino = f"/salidas/{etiqueta or carpeta}"
    os.makedirs(destino, exist_ok=True)
    copiados = []
    for f in sorted(glob.glob(f"{res}/*.npy")):
        shutil.copy(f, f"{destino}/{os.path.basename(f)}")
        copiados.append(os.path.basename(f))
    for f in sorted(glob.glob(f"{REPO}/exp/trees/{EXP}/*.log")):
        shutil.copy(f, f"{destino}/{os.path.basename(f)}")
    salidas.commit()
    print(f"--- {len(copiados)} ficheros copiados")
    return r.returncode, copiados


@app.function(volumes=VOLS, timeout=60 * 30, cpu=4, memory=32768)
def estadisticas():
    """Compara la parcela de ejemplo de los autores con una nuestra.

    Importa porque las caracteristicas de entrada son coordenadas mas los tres canales
    de color, y NormalizeColor las divide por 127.5 sin mirar de donde vienen. Si la
    escala de la intensidad no se parece, el modelo ve otra cosa.
    """
    guion = """
import numpy as np, torch, glob, os
def resumen(f):
    d = torch.load(f)
    c, col = d["coord"], d["color"]
    ext = c.max(0) - c.min(0)
    dens = len(c) / (ext[0] * ext[1])
    print(f"{os.path.basename(f)[:34]:36} {len(c):>9} puntos  {dens:>8.1f} pts/m2  "
          f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} m")
    for j, nom in enumerate(["intensidad", "n retornos", "retorno"]):
        q = np.percentile(col[:, j], [0, 25, 50, 75, 100])
        print(f"      {nom:12} " + "  ".join(f"{v:>10.1f}" for v in q) +
              f"   tras NormalizeColor: {q[2]/127.5-1:.2f}")
print(f"{'fichero':36} {'puntos':>9}          {'densidad':>8}")
print("      canal         min        p25        p50        p75        max")
for f in sorted(glob.glob("/pesos/muestra/*.pth")): resumen(f)
for f in ["/datos/test/cham_1b.pth", "/datos/test/pell_p5.pth"]: resumen(f)
"""
    import subprocess
    r = subprocess.run([PY, "-c", guion], capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:] if r.returncode else "")


@app.function(volumes=VOLS, timeout=60 * 20, cpu=2, memory=16384)
def resumen(carpeta="test"):
    """Cuantas instancias salieron por parcela, sin bajarse los ficheros."""
    guion = """
import glob, os
import numpy as np
print(f"{'parcela':26} {'puntos':>9} {'asignados':>10} {'%':>7} {'instancias':>11}")
for f in sorted(glob.glob("/salidas/CARPETA/asignado_*.npy")):
    n = os.path.basename(f)[len("asignado_"):-4]
    a = np.load(f).reshape(-1)
    e = np.load(f"/salidas/CARPETA/instance_pred_{n}.npy").reshape(-1)
    ni = len(np.unique(e[a])) if len(e) == len(a) and a.sum() else 0
    print(f"{n:26} {len(a):>9} {int(a.sum()):>10} {100*a.mean():>6.1f}% {ni:>11}")
"""
    import subprocess
    r = subprocess.run([PY, "-c", guion.replace("CARPETA", carpeta)],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:] if r.returncode else "")


@app.function(volumes=VOLS, timeout=60 * 30, cpu=4, memory=32768)
def detecciones(carpeta="test", datos_dir="test"):
    """Convierte las etiquetas en detecciones dentro del contenedor y devuelve el JSON.

    Es lo mismo que hace evaluar_treelite.py en local, pero aqui evita bajarse cientos
    de megas por variante: solo vuelven apice, altura y area de cada instancia. Las
    coordenadas se reconstruyen del .pth mas el origen que guardo el conversor, que es
    exactamente lo que recibio el modelo.
    """
    guion = """
import glob, json, os
import numpy as np
import torch
MIN_PTS = 10
def area_hull(xy):
    # casco convexo por cadena monotona y area por el teorema del zapatero; misma
    # definicion que ConvexHull(xy).volume, que en 2D es el area
    if len(xy) < 3:
        return 0.5
    pts = sorted(set(map(tuple, np.round(xy, 6))))
    if len(pts) < 3:
        return 0.5
    def media(pts):
        r = []
        for q in pts:
            while len(r) >= 2 and (r[-1][0]-r[-2][0])*(q[1]-r[-2][1]) - \
                                  (r[-1][1]-r[-2][1])*(q[0]-r[-2][0]) <= 0:
                r.pop()
            r.append(q)
        return r
    c = media(pts)[:-1] + media(pts[::-1])[:-1]
    if len(c) < 3:
        return 0.5
    a = sum(c[i][0]*c[(i+1) % len(c)][1] - c[(i+1) % len(c)][0]*c[i][1]
            for i in range(len(c))) / 2
    return max(abs(a), 0.5)

salida = {}
for f in sorted(glob.glob("/salidas/CARPETA/asignado_*.npy")):
    n = os.path.basename(f)[len("asignado_"):-4]
    if not os.path.exists("/datos/DATOS/%s.pth" % n):
        continue
    asig = np.load(f).reshape(-1)
    etq = np.load("/salidas/CARPETA/instance_pred_%s.npy" % n).reshape(-1)
    d = torch.load("/datos/DATOS/%s.pth" % n)
    org = np.load("/datos/DATOS/%s_origen.npy" % n)
    xyz = np.asarray(d["coord"]) + org
    if len(etq) == 0 and asig.sum() == 0:
        # sin ninguna propuesta el modelo devuelve una matriz vacia, no un fallo
        salida[n] = dict(x=[], y=[], h=[], area=[], n=[],
                         puntos=len(xyz), asignados=0)
        continue
    if not (len(etq) == len(asig) == len(xyz)):
        salida[n] = dict(error="%d etiquetas, %d marcas, %d puntos" %
                         (len(etq), len(asig), len(xyz)))
        continue
    X, Y, H, A, N = [], [], [], [], []
    for i in np.unique(etq[asig]):
        m = asig & (etq == i)
        if m.sum() < MIN_PTS:
            continue
        k = int(np.argmax(xyz[m, 2]))
        X.append(float(xyz[m][k, 0])); Y.append(float(xyz[m][k, 1]))
        H.append(float(xyz[m, 2].max())); A.append(area_hull(xyz[m][:, :2]))
        N.append(int(m.sum()))
    salida[n] = dict(x=X, y=Y, h=H, area=A, n=N,
                     puntos=len(xyz), asignados=int(asig.sum()))
print("JSON" + json.dumps(salida))
"""
    import subprocess
    r = subprocess.run(
        [PY, "-c", guion.replace("CARPETA", carpeta).replace("DATOS", datos_dir)],
        capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        return {}
    import json
    linea = [l for l in r.stdout.splitlines() if l.startswith("JSON")][0]
    return json.loads(linea[4:])


@app.function(volumes=VOLS, timeout=60 * 20, cpu=2, memory=32768)
def comprobar_orden(carpeta="test", datos_dir="test"):
    """Que las etiquetas vuelvan en el orden de los puntos de entrada.

    El probador guarda origin_coord, que son las coordenadas que vio el modelo tras
    CenterShift. Si el orden es el mismo que el del .pth, centrar ambas nubes las hace
    coincidir punto a punto. Si no lo fuera, todas las detecciones serian basura.
    """
    guion = """
import glob, os
import numpy as np
import torch
for f in sorted(glob.glob("/salidas/CARPETA/orig_coord_*.npy")):
    n = os.path.basename(f)[len("orig_coord_"):-4]
    if not os.path.exists("/datos/DATOS/%s.pth" % n):
        continue
    oc = np.load(f)
    c = np.asarray(torch.load("/datos/DATOS/%s.pth" % n)["coord"])
    if len(oc) != len(c):
        print("%-14s DISTINTO NUMERO DE PUNTOS %d vs %d" % (n, len(oc), len(c)))
        continue
    dv = np.abs((oc - oc.mean(0)) - (c - c.mean(0))).max(1)
    print("%-14s %8d puntos  tipo %s  p50 %.2e  p99.99 %.2e  max %.2e  "
          "mas de 1 cm: %d" % (n, len(c), oc.dtype, np.percentile(dv, 50),
                               np.percentile(dv, 99.99), dv.max(), int((dv > 0.01).sum())))
"""
    import subprocess
    r = subprocess.run(
        [PY, "-c", guion.replace("CARPETA", carpeta).replace("DATOS", datos_dir)],
        capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:] if r.returncode else "")


@app.local_entrypoint()
def main(parcelas: str = "", muestra: bool = False, carpeta: str = "test",
         opciones: str = "", etiqueta: str = ""):
    codigo, copiados = inferir.remote(
        parcelas.split(",") if parcelas else None, muestra, carpeta, opciones, etiqueta)
    print(f"\ncodigo {codigo}, {len(copiados)} ficheros")
    for c in copiados[:8]:
        print("   ", c)


@app.local_entrypoint()
def barrido(umbrales: str = "1,2,3,4,5,6,8,10", carpeta: str = "test",
            extra: str = "", sufijo: str = ""):
    """Barre el radio de agrupamiento y guarda las detecciones de cada ajuste.

    El radio efectivo es cluster_thresh * voxel_size, con voxel_size 0,05 m: el valor
    1 que traen por defecto son 5 cm, pensados para nubes de miles de puntos por metro
    cuadrado. Se lanzan todos a la vez y cada uno deja su JSON.
    """
    import json

    salida = os.path.join(os.path.dirname(os.path.dirname(AQUI)), "salida_treelite")
    os.makedirs(salida, exist_ok=True)
    vals = [v.strip() for v in umbrales.split(",") if v.strip()]
    lanzados = {}
    for v in vals:
        et = f"t{v}{sufijo}"
        op = f"model.cluster_thresh={v}" + (f" {extra}" if extra else "")
        lanzados[v] = (et, inferir.spawn(None, False, carpeta, op, et))
    for v, (et, h) in lanzados.items():
        codigo, copiados = h.get()
        print(f"umbral {v:>3} ({float(v) * 0.05:.2f} m): codigo {codigo}, "
              f"{len(copiados)} ficheros")
    for v, (et, _) in lanzados.items():
        det = detecciones.remote(et, carpeta)
        with open(os.path.join(salida, f"detecciones_{et}.json"), "w") as f:
            json.dump(det, f)
        n = sum(len(d.get("x", [])) for d in det.values())
        print(f"umbral {v:>3}: {n} instancias en {len(det)} parcelas")
