"""Fully-constrained linear spectral unmixing of both dates onto a shared endmember library.

Solves, per pixel, for fractions f of GV / NPV_SOIL / CHAR / SHADE:

    minimize || A f - r ||^2   subject to   f >= 0  and  sum(f) = 1

over the 355 cross-date-QA'd bands. Non-negativity comes from NNLS; the sum-to-one
constraint is imposed by augmenting the system with a heavily weighted row of ones, the
standard FCLS formulation (Heinz & Chang 2001).

Both dates are unmixed against the SAME library, which is what makes the two sets of
fraction maps directly differenceable. The SHADE endmember carries the illumination term,
so the 34-vs-73 degree sun elevation difference lands in the shade fraction rather than
contaminating GV / CHAR / soil.

Outputs per date: fraction maps, per-pixel RMSE of the reconstruction, and shade-normalized
fractions (each material fraction divided by 1 - shade), which is the quantity to compare
across dates because it removes the illumination term entirely.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import rasterio
from affine import Affine
from scipy.optimize import nnls

sys.path.insert(0, "scripts")
from invariant_check import PATHS, overlap_windows  # noqa: E402
from tanager import FIELDS, SR, Scene  # noqa: E402

SUM_TO_ONE_WEIGHT = 30.0  # weight on the sum-to-one row; large => constraint near-exact
TILE_ROWS = 64


def fcls(A, R):
    """Fully constrained least squares. A is (bands, k), R is (bands, n) -> (k, n) + rmse."""
    k = A.shape[1]
    Aa = np.vstack([A, np.full((1, k), SUM_TO_ONE_WEIGHT)])
    out = np.full((k, R.shape[1]), np.nan, dtype="f4")
    rmse = np.full(R.shape[1], np.nan, dtype="f4")
    ok = np.isfinite(R).all(axis=0)
    for j in np.flatnonzero(ok):
        b = np.append(R[:, j], SUM_TO_ONE_WEIGHT)
        f, _ = nnls(Aa, b)
        out[:, j] = f
        rmse[j] = np.sqrt(np.mean((A @ f - R[:, j]) ** 2))
    return out, rmse


def unmix_scene(scene, r0, c0, nrow, ncol, usable, A):
    k = A.shape[1]
    frac = np.full((k, nrow, ncol), np.nan, dtype="f4")
    rmse = np.full((nrow, ncol), np.nan, dtype="f4")
    valid = scene.valid_mask()[r0 : r0 + nrow, c0 : c0 + ncol]
    dset = scene.handle[SR]
    band_idx = np.flatnonzero(usable)

    t0 = time.time()
    for t in range(0, nrow, TILE_ROWS):
        t1 = min(t + TILE_ROWS, nrow)
        cube = dset[:, r0 + t : r0 + t1, c0 : c0 + ncol][band_idx].astype("f4")
        cube[cube == -9999.0] = np.nan
        h = t1 - t
        R = cube.reshape(len(band_idx), h * ncol)
        R[:, ~valid[t:t1].ravel()] = np.nan
        f, e = fcls(A, R)
        frac[:, t:t1, :] = f.reshape(k, h, ncol)
        rmse[t:t1, :] = e.reshape(h, ncol)
        done = t1 / nrow
        print(f"    {100 * done:5.1f}%  ({time.time() - t0:5.0f}s elapsed)", end="\r")
    print()
    return frac, rmse


def write_tif(path, arr, origin, names=None):
    arr = np.atleast_3d(arr.T).T if arr.ndim == 2 else arr
    transform = Affine.translation(origin[0], origin[1]) * Affine.scale(30, -30)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype="float32", crs="EPSG:32611", transform=transform,
        compress="deflate", nodata=np.nan,
    ) as dst:
        dst.write(arr.astype("f4"))
        if names:
            for i, n in enumerate(names, start=1):
                dst.set_band_description(i, n)


def main():
    em = np.load("data/endmembers.npz", allow_pickle=True)
    A = em["library"].T.astype("f8")  # (bands, k)
    names = [str(n) for n in em["names"]]
    usable = em["usable"]
    print(f"library {names}, design matrix {A.shape}")

    scenes = {k: Scene.open(p) for k, p in PATHS.items()}
    wins, nrow, ncol, origin = overlap_windows(scenes)
    print(f"overlap {nrow} x {ncol} px\n")

    results = {}
    for k, s in scenes.items():
        print(f"unmixing {k}")
        r0, c0 = wins[k]
        frac, rmse = unmix_scene(s, r0, c0, nrow, ncol, usable, A)
        results[k] = (frac, rmse)

        shade = frac[names.index("SHADE")]
        illum = np.clip(1.0 - shade, 1e-3, None)
        norm = frac / illum  # shade-normalized: the cross-date comparable quantity

        write_tif(f"data/frac_{k}.tif", frac, origin, names)
        write_tif(f"data/fracnorm_{k}.tif", norm, origin, names)
        write_tif(f"data/rmse_{k}.tif", rmse[None], origin, ["rmse"])

        ok = np.isfinite(rmse)
        print(f"  valid px {ok.sum():,}  median RMSE {np.nanmedian(rmse):.4f}  "
              f"p95 {np.nanpercentile(rmse, 95):.4f}")
        for i, n in enumerate(names):
            print(f"    mean {n:9s} {np.nanmean(frac[i]):.3f}   "
                  f"shade-norm {np.nanmean(norm[i][ok]):.3f}")
        print()

    np.savez_compressed(
        "data/fractions.npz",
        names=np.array(names), origin=np.array(origin),
        **{f"frac_{k}": v[0] for k, v in results.items()},
        **{f"rmse_{k}": v[1] for k, v in results.items()},
    )
    print("wrote data/fractions.npz and per-date GeoTIFFs")
    for s in scenes.values():
        s.handle.close()


if __name__ == "__main__":
    main()
