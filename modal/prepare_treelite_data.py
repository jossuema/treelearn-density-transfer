"""Converts our 22 plots to the .pth format that TreeLite3D expects.

The format was deduced from their sample, because the README only says "prepare your data
similar to the sample data":
    coord        (N,3) float64   local coordinates, minimum at zero
    color        (N,3) float64   intensity, number of returns, return number
    semantic_gt  (N,1) int64
    instance_gt  (N,1) int64

The config declares in_channels=6 and feat_keys=("coord","color"), so those three
channels ARE used. The raw LAS values are passed without rescaling: that is the literal
reading of the sample and the only one that does not introduce a choice of our own. In their
sample the intensity reaches 963288 with a preprocessing that mixes conventions; that is
not imitated.

Since there is no per-point annotation, semantic_gt and instance_gt go to zero. They only
serve the model's internal metrics, not prediction.

Usage: MODAL_PROFILE=<your-profile> modal run modal/prepare_treelite_data.py
"""
import modal

inputs = modal.Volume.from_name("ff3d-inputs")
data = modal.Volume.from_name("treelite-data", create_if_missing=True)
# numpy<2 on purpose: numpy 2 serializes arrays under numpy._core and the Pointcept
# image (torch 1.11) ships numpy 1.x, which cannot read that pickle.
image = modal.Image.debian_slim().pip_install("numpy<2", "torch", "laspy", "lazrs")
app = modal.App("treelite-data", image=image)


# Intensity quartiles in the authors' example plot. Their intensity is on a 16-bit
# scale and ours is not, and NormalizeColor just divides by 127.5: the model trained
# seeing ~200 in that channel and with the raw values it sees ~0.
P25_AUT, P50_AUT, P75_AUT = 21549.0, 25654.0, 27150.0


@app.function(volumes={"/inputs": inputs, "/data": data}, timeout=60 * 60)
def convert(dest="test", scale=None, origin=""):
    """scale: None leaves the intensity raw; "quartiles" maps it affinely onto the
    authors' quartiles; "constant" replaces it with their median, which erases the
    information in the channel but leaves it in the range the model expects."""
    import glob
    import os

    import laspy
    import numpy as np
    import torch

    os.makedirs(f"/data/{dest}", exist_ok=True)
    print(f"{'plot':14} {'points':>9} {'intensity':>18} {'xy extent':>14}")
    pattern = f"/inputs/{origin}/*.las" if origin else "/inputs/*.las"
    for f in sorted(glob.glob(pattern)):
        name = os.path.basename(f).replace(".las", "")
        L = laspy.read(f)
        xyz = np.column_stack([np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)]).astype(np.float64)
        # local coordinates with the minimum at zero, same as the sample
        origin = xyz.min(0)
        coord = xyz - origin
        inten = np.asarray(L.intensity).astype(np.float64)
        if scale == "quartiles":
            q25, q75 = np.percentile(inten, [25, 75])
            inten = P25_AUT + (inten - q25) * (P75_AUT - P25_AUT) / max(q75 - q25, 1e-9)
        elif scale == "constant":
            inten = np.full(len(inten), P50_AUT)
        color = np.column_stack([
            inten,
            np.asarray(L.number_of_returns),
            np.asarray(L.return_number)]).astype(np.float64)
        n = len(coord)
        d = {"coord": coord, "color": color,
             "semantic_gt": np.zeros((n, 1), np.int64),
             "instance_gt": np.zeros((n, 1), np.int64)}
        torch.save(d, f"/data/{dest}/{name}.pth")
        # the origin is needed to bring the detections back to their own system
        np.save(f"/data/{dest}/{name}_origin.npy", origin)
        print(f"{name:14} {n:>9} {int(color[:,0].min()):>7} to {int(color[:,0].max()):<7} "
              f"{coord[:,0].max():>6.1f} x {coord[:,1].max():>5.1f} m")
    data.commit()
    return sorted(os.path.basename(p) for p in glob.glob(f"/data/{dest}/*.pth"))


@app.local_entrypoint()
def main(dest: str = "test", scale: str = "", origin: str = ""):
    r = convert.remote(dest, scale or None, origin)
    print(f"\n{len(r)} .pth files generated in {dest}")
