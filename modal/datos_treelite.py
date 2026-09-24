"""Convierte nuestras 22 parcelas al formato .pth que espera TreeLite3D.

El formato se dedujo de su muestra, porque el README solo dice "prepare your data
similar to the sample data":
    coord        (N,3) float64   coordenadas locales, minimo en cero
    color        (N,3) float64   intensidad, numero de retornos, numero de retorno
    semantic_gt  (N,1) int64
    instance_gt  (N,1) int64

La configuracion declara in_channels=6 y feat_keys=("coord","color"), asi que esos tres
canales SI se usan. Se pasan los valores crudos del LAS sin reescalar: es la lectura
literal de la muestra y la unica que no mete una decision nuestra. En su muestra la
intensidad llega a 963288 con un preprocesado que mezcla convenciones; eso no se imita.

Como no hay anotacion por punto, semantic_gt e instance_gt van a cero. Solo sirven para
las metricas internas del modelo, no para predecir.

Uso: MODAL_PROFILE=treelearn modal run respaldo_minimo/modal_app/datos_treelite.py
"""
import modal

entradas = modal.Volume.from_name("ff3d-entradas")
datos = modal.Volume.from_name("treelite-datos", create_if_missing=True)
# numpy<2 a proposito: numpy 2 serializa los arrays bajo numpy._core y la imagen
# de Pointcept (torch 1.11) lleva numpy 1.x, que no sabe leer ese pickle.
image = modal.Image.debian_slim().pip_install("numpy<2", "torch", "laspy", "lazrs")
app = modal.App("treelite-datos", image=image)


# Cuartiles de la intensidad en la parcela de ejemplo de los autores. Su intensidad
# va en escala de 16 bits y la nuestra no, y NormalizeColor divide por 127.5 a secas:
# el modelo entreno viendo ~200 en ese canal y con los valores crudos ve ~0.
P25_AUT, P50_AUT, P75_AUT = 21549.0, 25654.0, 27150.0


@app.function(volumes={"/entradas": entradas, "/datos": datos}, timeout=60 * 60)
def convertir(destino="test", escala=None):
    """escala: None deja la intensidad cruda; "cuartiles" la lleva afinmente a los
    cuartiles de los autores; "constante" la sustituye por su mediana, que borra la
    informacion del canal pero lo deja en el rango que el modelo espera."""
    import glob
    import os

    import laspy
    import numpy as np
    import torch

    os.makedirs(f"/datos/{destino}", exist_ok=True)
    print(f"{'parcela':14} {'puntos':>9} {'intensidad':>18} {'extension xy':>14}")
    for f in sorted(glob.glob("/entradas/*.las")):
        nombre = os.path.basename(f).replace(".las", "")
        L = laspy.read(f)
        xyz = np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)]).astype(np.float64)
        # coordenadas locales con minimo en cero, igual que la muestra
        origen = xyz.min(0)
        coord = xyz - origen
        inten = np.asarray(L.intensity).astype(np.float64)
        if escala == "cuartiles":
            q25, q75 = np.percentile(inten, [25, 75])
            inten = P25_AUT + (inten - q25) * (P75_AUT - P25_AUT) / max(q75 - q25, 1e-9)
        elif escala == "constante":
            inten = np.full(len(inten), P50_AUT)
        color = np.column_stack([
            inten,
            np.asarray(L.number_of_returns),
            np.asarray(L.return_number)]).astype(np.float64)
        n = len(coord)
        d = {"coord": coord, "color": color,
             "semantic_gt": np.zeros((n, 1), np.int64),
             "instance_gt": np.zeros((n, 1), np.int64)}
        torch.save(d, f"/datos/{destino}/{nombre}.pth")
        # el origen hace falta para devolver las detecciones a su sistema
        np.save(f"/datos/{destino}/{nombre}_origen.npy", origen)
        print(f"{nombre:14} {n:>9} {int(color[:,0].min()):>7} a {int(color[:,0].max()):<7} "
              f"{coord[:,0].max():>6.1f} x {coord[:,1].max():>5.1f} m")
    datos.commit()
    return sorted(os.path.basename(p) for p in glob.glob(f"/datos/{destino}/*.pth"))


@app.local_entrypoint()
def main(destino: str = "test", escala: str = ""):
    r = convertir.remote(destino, escala or None)
    print(f"\n{len(r)} ficheros .pth generados en {destino}")
