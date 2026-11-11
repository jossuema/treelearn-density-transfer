"""Step 1: store the ForestFormer3D weights on a Modal volume.

They are downloaded inside Modal and not from the laptop: they are 198 MB and this way
they do not depend on the local connection and do not have to be uploaded again on every run.

The pipeline expects the checkpoint at
  /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth
according to tools/inference_bluepoint_forestsens.sh, so the volume will be mounted at
/workspace/work_dirs and the contents of the zip have to end up with that shape.

Usage: MODAL_PROFILE=<your-profile> modal run modal/download_ff3d_weights.py
"""
import modal

URL = "https://zenodo.org/records/16742708/files/clean_forestformer.zip?download=1"
MD5 = "553d67379331966509076f3fbb409e57"

vol = modal.Volume.from_name("ff3d-weights", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("curl", "unzip")
app = modal.App("ff3d-weights", image=image)


@app.function(volumes={"/weights": vol}, timeout=30 * 60)
def download():
    import hashlib
    import os
    import subprocess

    if os.path.exists("/weights/clean_forestformer"):
        print("already there, current contents:")
    else:
        print("downloading from Zenodo...", flush=True)
        subprocess.run(["curl", "-sL", "-o", "/tmp/w.zip", URL], check=True)
        h = hashlib.md5(open("/tmp/w.zip", "rb").read()).hexdigest()
        print(f"downloaded md5 {h}  expected {MD5}  {'matches' if h == MD5 else 'MISMATCH'}")
        if h != MD5:
            raise SystemExit("the download does not match the published md5")
        subprocess.run(["unzip", "-q", "-o", "/tmp/w.zip", "-d", "/weights"], check=True)
        vol.commit()

    for root, _, files in os.walk("/weights"):
        for a in sorted(files):
            p = os.path.join(root, a)
            print(f"  {os.path.getsize(p)/1e6:8.1f} MB  {p}")
    return True


@app.local_entrypoint()
def main():
    download.remote()
