"""TreeLite3D step 1: weights and sample data to a Modal volume.

The weights and the sample sit in two Google Drive folders, which is fragile: no published
checksum and with download limits. That is why they are copied to a volume of our own as
soon as they are downloaded, as our model notes already warned.

The sample is needed for one more reason: the model input is .pth files and the README
only says "prepare your data similar to the sample data", so one has to open one to work
out the exact format.

Usage: MODAL_PROFILE=<your-profile> modal run modal/download_treelite_weights.py
"""
import modal

WEIGHTS = "12IYq9gx-p1BQntIBxKyNLWlb47LIeM3A"
SAMPLE = "1O1MdVK4Gr579nCj6p3TmEg4XyD-dIKWu"

vol = modal.Volume.from_name("treelite-weights", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("gdown", "torch", "numpy")
app = modal.App("treelite-weights", image=image)


@app.function(volumes={"/vol": vol}, timeout=45 * 60)
def download():
    import os
    import subprocess

    for folder, dest in ((WEIGHTS, "/vol/weights"), (SAMPLE, "/vol/sample")):
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"{dest} already has content")
            continue
        os.makedirs(dest, exist_ok=True)
        print(f"downloading {folder} into {dest}...", flush=True)
        r = subprocess.run(["gdown", "--folder", "-O", dest,
                            f"https://drive.google.com/drive/folders/{folder}"],
                           capture_output=True, text=True)
        print(r.stdout[-2500:], flush=True)
        if r.returncode:
            print("STDERR:", r.stderr[-2500:], flush=True)
    vol.commit()

    print("\n=== volume contents")
    for root, _, files in os.walk("/vol"):
        for a in sorted(files):
            p = os.path.join(root, a)
            print(f"  {os.path.getsize(p)/1e6:9.2f} MB  {p}")

    # sample format: it is what we have to reproduce with our plots
    import glob
    import torch
    for p in sorted(glob.glob("/vol/sample/**/*.pth", recursive=True))[:2]:
        # weights_only=False because the .pth carries numpy arrays. It runs pickle, so it
        # only holds for artifacts published by the authors, which is the case here.
        d = torch.load(p, map_location="cpu", weights_only=False)
        print(f"\n=== {p}")
        print("  type:", type(d).__name__)
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    print(f"  {k:16} {type(v).__name__} shape {tuple(v.shape)} "
                          f"type {v.dtype} min {float(v.min()):.2f} max {float(v.max()):.2f}")
                except Exception:
                    print(f"  {k:16} {type(v).__name__} {str(v)[:60]}")
    return True


@app.local_entrypoint()
def main():
    download.remote()
