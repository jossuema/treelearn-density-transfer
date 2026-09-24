"""Pipeline TreeSeg PointNet++ en Modal (cliente 1.4.x).

BORRADOR. Escrito el 3 de septiembre de 2026 sin ejecutarlo contra Modal. Probar primero:

    modal profile activate <your-profile>
    modal run modal_app/treeseg_modal.py::smoke

Diseño (ver docs/PLAN_MODAL.md):
  - La imagen lleva el código (model3d/, dataset/build_forinstance.py, dataset/sim_stats.py) y la
    carpeta dataset/consolidated_v6/ (22 .npz alpinos, index.json y todos los splits_*.json; 40 MB;
    datos con licencia, el workspace es privado). Llegan a /root/treeseg con la MISMA disposición
    que en esta carpeta.
  - Al arrancar cada función, ese árbol se copia a /root/work (escribible con seguridad) y ahí se
    ejecutan los scripts, así las rutas relativas a ROOT siguen funcionando.
  - Volume persistente "treeseg-data" montado en /vol:
        /vol/forinstance_raw/    descarga de Zenodo descomprimida (una sola vez)
        /vol/forinstance/        .npz a densidad ALS simulada + splits.json
        /vol/runs/<nombre>/      checkpoints y JSON de cada corrida
    /root/work/model3d/checkpoints se enlaza a /vol/runs/<nombre>, porque train3d.py y
    eval_transfer.py escriben siempre ahí y el disco del contenedor se pierde al terminar.

Épocas: `finetune` afina 80 (modelo principal); `seeds`, `loso` y `norad` afinan 40, como las
corridas secundarias del paper. Se cambian con --ft-epochs y --sec-epochs.

Uso (desde la raíz de TreeSeg-GRSL, con el perfil personal activo):
    modal run --detach modal_app/treeseg_modal.py::main --stage ingest
    modal run --detach modal_app/treeseg_modal.py::main --stage pretrain --epochs 120
    modal run --detach modal_app/treeseg_modal.py::main --stage finetune
    modal run --detach modal_app/treeseg_modal.py::main --stage seeds --seeds 5
    modal run --detach modal_app/treeseg_modal.py::main --stage loso
    modal run --detach modal_app/treeseg_modal.py::main --stage norad
    modal volume get treeseg-data /runs ./results/modal_runs
Progreso: `modal app list` y `modal app logs <ID ap-...> -f`.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = "treeseg-pointnet2"
VOL_NAME = "treeseg-data"
VOL = "/vol"                      # punto de montaje del Volume en el contenedor
CODE = "/root/treeseg"            # árbol que llega con la imagen (solo lectura por si acaso)
WORK = "/root/work"               # copia escribible; ROOT de los scripts en el contenedor
ZENODO_ZIP = "https://zenodo.org/records/8287792/files/FORinstance_dataset.zip?download=1"

LOCAL_ROOT = Path(__file__).resolve().parent.parent   # .../TreeSeg-GRSL

# ---------------------------------------------------------------------------
# Imagen. Python fijado (debian_slim sin argumento copiaría el 3.10 del Mac).
# torch en su propia capa: las ruedas de PyPI traen su CUDA; el contenedor no tiene toolkit.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "curl")
    .uv_pip_install("torch==2.13.0")
    .uv_pip_install("numpy<2.2", "scipy", "scikit-image", "matplotlib", "laspy[lazrs]", "pandas", "tqdm")
    # copy=False: los archivos se envían al arrancar el contenedor, editar código no reconstruye la imagen.
    .add_local_dir(str(LOCAL_ROOT / "model3d"), remote_path=f"{CODE}/model3d",
                   ignore=["__pycache__", "checkpoints", "*.pt", "*.pyc"])
    .add_local_file(str(LOCAL_ROOT / "dataset" / "build_forinstance.py"), f"{CODE}/dataset/build_forinstance.py")
    .add_local_file(str(LOCAL_ROOT / "dataset" / "sim_stats.py"), f"{CODE}/dataset/sim_stats.py")
    .add_local_dir(str(LOCAL_ROOT / "dataset" / "consolidated_v6"), remote_path=f"{CODE}/dataset/consolidated_v6")
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image, volumes={VOL: vol})

ALPINE = f"{WORK}/dataset/consolidated_v6"          # npz + splits (copia de trabajo)

# dataset3d.SRC_DIRS antepone TREESEG_DIRS (separado por ':').
ENV = {
    "TREESEG_DIRS": f"{VOL}/forinstance:{ALPINE}",
    "PYTHONUNBUFFERED": "1",
    "MPLBACKEND": "Agg",
}

SPLIT_FILES = {
    "ft": "splits_ft.json",
    **{f"ft_seed{s}": f"splits_ft_seed{s}.json" for s in range(1, 6)},
    "loso_train-cham": "splits_loso_train-cham.json",
    "loso_train-pell": "splits_loso_train-pell.json",
}


def _run(cmd, cwd=WORK):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True, env={**os.environ, **ENV})


def _prepare(run_name: str) -> Path:
    """Copia el código a WORK (escribible) y enlaza model3d/checkpoints -> /vol/runs/<run_name>."""
    if not Path(WORK).exists():
        shutil.copytree(CODE, WORK, symlinks=True)
    out = Path(f"{VOL}/runs/{run_name}")
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(f"{WORK}/model3d/checkpoints")
    if ck.is_symlink():
        ck.unlink()
    elif ck.exists():
        shutil.rmtree(ck)
    ck.symlink_to(out, target_is_directory=True)
    return out


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------
# 0. Humo: torch, CUDA, el Volume, y si el árbol de código se puede escribir
# ---------------------------------------------------------------------------
@app.function(gpu="T4", timeout=600)
def smoke() -> dict:
    import torch

    def writable(d: str) -> bool:
        try:
            p = Path(d) / ".write_test"
            p.write_text("ok")
            p.unlink()
            return True
        except Exception as e:      # noqa: BLE001
            print("no escribible:", d, e)
            return False

    _prepare("smoke")
    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "code_tree": sorted(os.listdir(CODE)),
        "alpine_npz": len(list(Path(ALPINE).glob("*.npz"))),
        "splits_present": sorted(p.name for p in Path(ALPINE).glob("splits*.json")),
        "code_dir_writable": writable(f"{CODE}/model3d"),
        "work_dir_writable": writable(f"{WORK}/model3d"),
        "checkpoints_symlink": os.path.realpath(f"{WORK}/model3d/checkpoints"),
        "volume_root": sorted(os.listdir(VOL)),
    }
    print(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# 1. Ingesta: FOR-instance de Zenodo -> Volume -> npz a densidad ALS simulada
# ---------------------------------------------------------------------------
@app.function(cpu=4.0, memory=16384, timeout=3 * 3600)
def ingest(vox: float = 0.25, occlusion: float = 12.0) -> str:
    _prepare("ingest")
    raw = Path(f"{VOL}/forinstance_raw")
    if not raw.exists():
        zip_path = "/tmp/FORinstance_dataset.zip"
        _run(["curl", "-L", "--retry", "5", "-o", zip_path, ZENODO_ZIP], cwd="/tmp")
        _run(["unzip", "-q", zip_path, "-d", "/tmp/forinstance_unz"], cwd="/tmp")
        shutil.copytree("/tmp/forinstance_unz", raw)
        vol.commit()
    out = Path(f"{VOL}/forinstance")
    _run(["python", "dataset/build_forinstance.py", raw, "--out", out,
          "--vox", vox, "--occlusion", occlusion, "--no-intensity"])
    _run(["python", "model3d/make_splits_forinstance.py", out])       # escribe out/splits.json
    vol.commit()
    return str(out)


# ---------------------------------------------------------------------------
# 2. Preentrenamiento en FOR-instance simulado (parámetros del paper)
# ---------------------------------------------------------------------------
@app.function(gpu="A100-40GB", cpu=8.0, memory=32768, timeout=8 * 3600, retries=0)
def pretrain(run_name: str = "pretrain", epochs: int = 120, bs: int = 24,
             npoints: int = 16384, samples: int = 12, seed: int = 0) -> str:
    out = _prepare(run_name)
    _run(["python", "model3d/train3d.py", "--split", "transfer",
          "--splits", f"{VOL}/forinstance/splits.json",
          "--epochs", epochs, "--bs", bs, "--npoints", npoints, "--samples", samples,
          "--sample", "fps", "--amp", "1", "--seed", seed, "--workers", "8"])
    vol.commit()
    return str(out / "p2_transfer_best.pt")


# ---------------------------------------------------------------------------
# 3. Fine-tuning + evaluación controlada (una corrida por configuración)
#    cfg = {run_name, init, splits (clave de SPLIT_FILES), seed, epochs, extra (flags de train3d)}
# ---------------------------------------------------------------------------
@app.function(gpu="L4", cpu=4.0, memory=16384, timeout=2 * 3600, max_containers=5)
def finetune_eval(cfg: dict) -> dict:
    vol.reload()                                   # ver el checkpoint de preentrenamiento
    out = _prepare(cfg["run_name"])
    splits = f"{ALPINE}/{SPLIT_FILES[cfg.get('splits', 'ft')]}"
    shutil.copy(splits, out / Path(splits).name)   # constancia del split usado
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    _run(["python", "model3d/train3d.py", "--split", "finetune",
          "--init", cfg["init"], "--splits", splits,
          "--epochs", cfg.get("epochs", 40), "--bs", cfg.get("bs", 16), "--samples", 16,
          "--lr", cfg.get("lr", 3e-4), "--npoints", 16384, "--sample", "fps", "--amp", "1",
          "--seed", cfg.get("seed", 0), "--workers", "4", *cfg.get("extra", [])])
    ckpt = out / "p2_finetune_best.pt"
    # Evaluación con el protocolo de Tusa: DL + baseline watershed, 3 distancias, bootstrap por parcela.
    _run(["python", "model3d/eval_experiments.py", "--ckpt", ckpt, "--holdout", splits,
          "--both", "--dmatch", "2.5,3.5,5.0", "--boot", 2000, "--strata-dm", "3.5",
          "--out", out / "experiments_eval.json"])
    # Tabla geométrica por sitio (escribe checkpoints/transfer_eval.json, que ya vive en out/).
    _run(["python", "model3d/eval_transfer.py", "--ckpt", ckpt, "--holdout", splits, "--npoints", 16384])
    vol.commit()
    res = _read_json(out / "experiments_eval.json")
    summary = {}
    for tag, sites in res.items():
        for site, r in sites.items():
            dm = r.get("dmatch", {}).get("3.5", {})
            summary[f"{tag} | {site}"] = {k: dm.get(k) for k in ("recall", "precision", "f1", "recall_ci")}
    return {"run_name": cfg["run_name"], "splits": cfg.get("splits", "ft"),
            "seed": cfg.get("seed", 0), "epochs": cfg.get("epochs", 40), "dm3.5": summary}


# ---------------------------------------------------------------------------
# Punto de entrada local: corre en el Mac; cada .remote()/.map() corre en Modal.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(stage: str = "smoke", epochs: int = 120, ft_epochs: int = 80, sec_epochs: int = 40,
         seeds: int = 5, pretrain_run: str = "pretrain"):
    Path("results/modal_runs").mkdir(parents=True, exist_ok=True)
    init = f"{VOL}/runs/{pretrain_run}/p2_transfer_best.pt"

    def save(name: str, obj) -> None:
        Path(f"results/modal_runs/{name}.json").write_text(json.dumps(obj, indent=2))
        print(json.dumps(obj, indent=2))

    def run_many(cfgs: list) -> list:
        return [r if not isinstance(r, Exception) else {"error": str(r)}
                for r in finetune_eval.map(cfgs, return_exceptions=True)]

    if stage == "smoke":
        print(smoke.remote())
    elif stage == "ingest":
        print("convertido ->", ingest.remote())
    elif stage == "pretrain":
        print("preentrenado ->", pretrain.remote(run_name=pretrain_run, epochs=epochs))
    elif stage == "finetune":
        save("finetune_main", finetune_eval.remote(
            {"run_name": "finetune_main", "init": init, "splits": "ft", "seed": 0, "epochs": ft_epochs}))
    elif stage == "seeds":
        save("seeds", run_many([{"run_name": f"ft_seed{s}", "init": init, "splits": f"ft_seed{s}",
                                 "seed": s, "epochs": sec_epochs} for s in range(1, seeds + 1)]))
    elif stage == "loso":
        save("loso", run_many([{"run_name": k, "init": init, "splits": k, "seed": 0, "epochs": sec_epochs}
                               for k in ("loso_train-cham", "loso_train-pell")]))
    elif stage == "norad":
        save("abl_norad", finetune_eval.remote(
            {"run_name": "abl_norad", "init": init, "splits": "ft", "seed": 0, "epochs": sec_epochs,
             "extra": ["--rad_w", "0"]}))
    else:
        raise SystemExit(f"stage desconocido: {stage} (smoke | ingest | pretrain | finetune | seeds | loso | norad)")
    print("Checkpoints y JSON completos: modal volume get treeseg-data /runs ./results/modal_runs")
