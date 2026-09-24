"""Paso 1 de TreeLite3D: pesos y datos de muestra a un volumen de Modal.

Los pesos y la muestra estan en dos carpetas de Google Drive, que es fragil: sin suma
de verificacion publicada y con limites de descarga. Por eso se copian a un volumen
propio en cuanto se bajan, como ya avisaba nuestro documento de modelos.

La muestra hace falta ademas por otra razon: la entrada del modelo son ficheros .pth y
el README solo dice "prepare your data similar to the sample data", asi que hay que
abrir uno para deducir el formato exacto.

Uso: MODAL_PROFILE=treelearn modal run respaldo_minimo/modal_app/pesos_treelite.py
"""
import modal

PESOS = "12IYq9gx-p1BQntIBxKyNLWlb47LIeM3A"
MUESTRA = "1O1MdVK4Gr579nCj6p3TmEg4XyD-dIKWu"

vol = modal.Volume.from_name("treelite-pesos", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("gdown", "torch", "numpy")
app = modal.App("treelite-pesos", image=image)


@app.function(volumes={"/vol": vol}, timeout=45 * 60)
def bajar():
    import os
    import subprocess

    for carpeta, destino in ((PESOS, "/vol/pesos"), (MUESTRA, "/vol/muestra")):
        if os.path.isdir(destino) and os.listdir(destino):
            print(f"{destino} ya tiene contenido")
            continue
        os.makedirs(destino, exist_ok=True)
        print(f"bajando {carpeta} en {destino}...", flush=True)
        r = subprocess.run(["gdown", "--folder", "-O", destino,
                            f"https://drive.google.com/drive/folders/{carpeta}"],
                           capture_output=True, text=True)
        print(r.stdout[-2500:], flush=True)
        if r.returncode:
            print("STDERR:", r.stderr[-2500:], flush=True)
    vol.commit()

    print("\n=== contenido del volumen")
    for raiz, _, archivos in os.walk("/vol"):
        for a in sorted(archivos):
            p = os.path.join(raiz, a)
            print(f"  {os.path.getsize(p)/1e6:9.2f} MB  {p}")

    # formato de la muestra: es lo que hay que reproducir con nuestras parcelas
    import glob
    import torch
    for p in sorted(glob.glob("/vol/muestra/**/*.pth", recursive=True))[:2]:
        # weights_only=False porque el .pth lleva arrays de numpy. Ejecuta pickle, asi
        # que solo vale para artefactos publicados por los autores, como es el caso.
        d = torch.load(p, map_location="cpu", weights_only=False)
        print(f"\n=== {p}")
        print("  tipo:", type(d).__name__)
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    print(f"  {k:16} {type(v).__name__} forma {tuple(v.shape)} "
                          f"tipo {v.dtype} min {float(v.min()):.2f} max {float(v.max()):.2f}")
                except Exception:
                    print(f"  {k:16} {type(v).__name__} {str(v)[:60]}")
    return True


@app.local_entrypoint()
def main():
    bajar.remote()
