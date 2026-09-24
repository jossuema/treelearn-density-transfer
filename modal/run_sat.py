"""SegmentAnyTree zero-shot sobre nuestras parcelas ALS, en Modal.

La inferencia de la imagen funciona, pero el último paso (exportar a LAS) falla siempre con
`ValueError: Cannot convert non-finite values (NA or inf) to integer`. Es el issue #31 del
repositorio. Causa exacta: en `nibio_inference/merge_pt_ss_is.py`, línea 100, los autores
dejaron COMENTADA la línea que rellenaba los puntos sin instancia asignada, y después
`pandas_to_las.py:118-122` intenta convertir esa columna a entero.

Parche aplicado aquí: rellenar los no finitos con 0 justo antes de las conversiones. Para
`PredInstance` el 0 ya significa "sin instancia", así que es lo que la línea comentada
pretendía y no altera ninguna predicción.

Detalle de entorno: la imagen trae python3.8 con torch y MinkowskiEngine en /usr/bin, pero
add_python instala el 3.11 de Modal en /usr/local/bin y lo pone primero en el PATH. Los
scripts .sh llaman a `python3` a secas, así que hay que forzar /usr/bin al principio.

Uso:
    modal profile activate <your-profile>
    modal volume put treeseg-data <archivo.las> /input/<nombre>.las
    modal run respaldo_minimo/modal_app/run_sat.py --archivos "cham_las1b.las,pell_p4.las"
    modal volume get treeseg-data /output ./salida_sat
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

PARCHE = (
    r"sed -i '/^    las_file = laspy.LasData(las_header)$/a\    import numpy as _npfix\n"
    r"    df = df.replace([_npfix.inf, -_npfix.inf], _npfix.nan).fillna(0)' " + P2L
)

image = (
    modal.Image.from_registry(IMG, add_python="3.11")
    .entrypoint([])
    .run_commands(PARCHE, f"grep -n _npfix {P2L}")   # el grep falla si el parche no entró
)
vol = modal.Volume.from_name("treeseg-data", create_if_missing=True)
app = modal.App("sat-run", image=image, volumes={VOL: vol})

# /usr/bin primero: ahí está el python3.8 con torch y MinkowskiEngine.
ENV = {**os.environ,
       "PATH": "/usr/bin:/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/sbin",
       "PYTHONPATH": WD}

CONTAR = r'''
import glob, json, sys
import numpy as np, laspy
out = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.la[sz]")):
    L = laspy.read(f)
    dims = [d.name for d in L.point_format.dimensions]
    info = {"n_puntos": int(len(L.points)), "campos": dims}
    for cand in ("PredInstance", "preds_instance_segmentation", "instance", "treeID"):
        if cand in dims:
            v = np.asarray(getattr(L, cand))
            ids, cuentas = np.unique(v[v > 0], return_counts=True)
            info["campo_instancia"] = cand
            info["n_instancias"] = int(len(ids))
            info["pts_por_instancia_mediana"] = int(np.median(cuentas)) if len(ids) else 0
            info["pct_puntos_sin_instancia"] = round(100.0 * float((v <= 0).mean()), 1)
            break
    for cand in ("PredSemantic", "preds_semantic_segmentation"):
        if cand in dims:
            s = np.asarray(getattr(L, cand))
            u, c = np.unique(s, return_counts=True)
            info["semantica"] = {int(a): int(b) for a, b in zip(u, c)}
            break
    out[f.split("/")[-1]] = info
print("CNT" + json.dumps(out) + "CNT")
'''


# max_containers=6: el plan Starter permite 10 GPU a la vez; dejamos margen.
@app.function(gpu="T4", timeout=3 * 3600, memory=24576, cpu=4.0, max_containers=6)
def segmentar(nombre: str) -> dict:
    vol.reload()
    src = next((p for p in (f"{VOL}/input_all/{nombre}", f"{VOL}/input/{nombre}")
                if os.path.exists(p)), None)
    if src is None:
        return {"archivo": nombre, "error": f"no existe {nombre} en /input_all ni /input"}

    tmp_in, tmp_out = "/tmp/in", "/tmp/out"
    for d in (tmp_in, tmp_out):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    shutil.copy(src, tmp_in)

    t0 = time.time()
    p = subprocess.run(["bash", "run_inference.sh", tmp_in, tmp_out, "true"],
                       cwd=WD, env=ENV, capture_output=True, text=True)
    res = {"archivo": nombre, "segundos": round(time.time() - t0, 1),
           "returncode": p.returncode}

    base = nombre.rsplit(".", 1)[0]
    dest = f"{VOL}/output/sat/{base}"
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)

    # Resultado final (LAS/LAZ fusionado)
    final = os.path.join(tmp_out, "final_results")
    res["salidas"] = sorted(os.listdir(final)) if os.path.isdir(final) else []
    if res["salidas"]:
        for f in os.listdir(final):
            shutil.copy(os.path.join(final, f), dest)
        c = subprocess.run(["/usr/bin/python3", "-c", CONTAR, final],
                           capture_output=True, text=True, env=ENV)
        if "CNT" in c.stdout:
            res["resumen"] = json.loads(c.stdout.split("CNT")[1])
        else:
            res["conteo_error"] = (c.stdout + c.stderr)[-1500:]

    # Respaldo: las predicciones crudas en PLY, por si el export a LAS vuelve a fallar
    for f in os.listdir(tmp_out):
        if f.endswith(".ply") and ("instance" in f or "semantic" in f):
            shutil.copy(os.path.join(tmp_out, f), dest)
            res.setdefault("ply_respaldo", []).append(f)

    vol.commit()
    if p.returncode != 0:
        res["stderr_final"] = p.stderr[-2500:]
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res


@app.local_entrypoint()
def main(archivos: str = "cham_las1b.las,pell_p4.las"):
    nombres = [a.strip() for a in archivos.split(",") if a.strip()]
    out = []
    for n, r in zip(nombres, segmentar.map(nombres, return_exceptions=True)):
        out.append({"archivo": n, "error": str(r)} if isinstance(r, Exception) else r)
    print("\n" + "=" * 78)
    for r in out:
        if r.get("resumen"):
            for f, i in r["resumen"].items():
                print(f"  OK  {r['archivo']:18s} {i.get('n_instancias','?'):>4} instancias  "
                      f"{i.get('n_puntos','?'):>8} pts  "
                      f"{i.get('pct_puntos_sin_instancia','?')}% sin instancia  "
                      f"{r['segundos']} s")
        else:
            print(f"  FALLA  {r['archivo']}  rc={r.get('returncode')} {r.get('error','')}")
    print("=" * 78)
    json.dump(out, open("sat_run_result.json", "w"), indent=2, ensure_ascii=False)
    print("detalle en sat_run_result.json")
