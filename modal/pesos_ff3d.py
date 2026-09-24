"""Paso 1: deja los pesos de ForestFormer3D en un volumen de Modal.

Se bajan dentro de Modal y no desde el portatil: son 198 MB y asi no dependen de la
conexion local ni hay que volver a subirlos en cada ejecucion.

El pipeline espera el checkpoint en
  /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth
segun tools/inference_bluepoint_forestsens.sh, asi que el volumen se montara en
/workspace/work_dirs y el contenido del zip tiene que quedar con esa forma.

Uso: MODAL_PROFILE=treelearn modal run respaldo_minimo/modal_app/pesos_ff3d.py
"""
import modal

URL = "https://zenodo.org/records/16742708/files/clean_forestformer.zip?download=1"
MD5 = "553d67379331966509076f3fbb409e57"

vol = modal.Volume.from_name("ff3d-pesos", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("curl", "unzip")
app = modal.App("ff3d-pesos", image=image)


@app.function(volumes={"/pesos": vol}, timeout=30 * 60)
def bajar():
    import hashlib
    import os
    import subprocess

    if os.path.exists("/pesos/clean_forestformer"):
        print("ya estaba, contenido actual:")
    else:
        print("descargando de Zenodo...", flush=True)
        subprocess.run(["curl", "-sL", "-o", "/tmp/w.zip", URL], check=True)
        h = hashlib.md5(open("/tmp/w.zip", "rb").read()).hexdigest()
        print(f"md5 descargado {h}  esperado {MD5}  {'coincide' if h == MD5 else 'NO COINCIDE'}")
        if h != MD5:
            raise SystemExit("la descarga no cuadra con el md5 publicado")
        subprocess.run(["unzip", "-q", "-o", "/tmp/w.zip", "-d", "/pesos"], check=True)
        vol.commit()

    for raiz, _, archivos in os.walk("/pesos"):
        for a in sorted(archivos):
            p = os.path.join(raiz, a)
            print(f"  {os.path.getsize(p)/1e6:8.1f} MB  {p}")
    return True


@app.local_entrypoint()
def main():
    bajar.remote()
