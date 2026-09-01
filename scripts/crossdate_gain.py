"""Characterize the Jan->Jul radiometric relationship per band, and derive a correction.

The invariant-target check showed deep water agreeing to ~0.001 reflectance while bright
targets differ by ~+0.025. That is the signature of a multiplicative (gain) difference
with a near-zero additive offset -- consistent with residual BRDF / sub-pixel shadowing at
34 deg sun elevation vs 73 deg, rather than a calibration offset.

This fits, per band, an ordinary least squares relation

    R_jul = gain * R_jan + offset

over pseudo-invariant pixels: valid in both dates, non-vegetated in both dates (NDVI <
0.15, so seasonal greenness cannot drive the fit), and **outside every fire perimeter**
(so the burn scars we actually want to measure are excluded from the calibration).

The resulting per-band gain/offset is written to data/crossdate_gain.npz for downstream
use, and is itself a diagnostic: a smooth gain curve near a constant means the two dates
are spectrally consistent and safely comparable; band-to-band jumps would mean they are not.
"""

from __future__ import annotations

import json
import sys

import matplotlib

# Only force the headless backend when run as a script; importing this module
# from a notebook must not clobber the inline backend.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from affine import Affine
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

sys.path.insert(0, "scripts")
from invariant_check import PATHS, PROBE, overlap_windows, probe_bands  # noqa: E402
from tanager import Scene  # noqa: E402

N_FIT = 4000
RNG = np.random.default_rng(1)
UTM = Transformer.from_crs(4326, 32611, always_xy=True).transform


def fire_mask(nrow, ncol, origin):
    """Rasterize all fire perimeters onto the overlap grid."""
    perims = json.load(open("data/fire_perimeters_2024_2025.geojson"))
    geoms = [shp_transform(UTM, shape(f["geometry"]).buffer(0)) for f in perims["features"]]
    transform = Affine.translation(origin[0], origin[1]) * Affine.scale(30, -30)
    return rasterio.features.rasterize(
        geoms, out_shape=(nrow, ncol), transform=transform, dtype="uint8"
    ).astype(bool)


def main():
    scenes = {k: Scene.open(p) for k, p in PATHS.items()}
    wins, nrow, ncol, origin = overlap_windows(scenes)

    refl, valid = {}, {}
    for k, s in scenes.items():
        r0, c0 = wins[k]
        refl[k] = probe_bands(s, r0, c0, nrow, ncol)
        valid[k] = s.valid_mask()[r0 : r0 + nrow, c0 : c0 + ncol]

    burned = fire_mask(nrow, ncol, origin)
    print(f"fire perimeters cover {burned.sum()} px ({burned.sum() * 900 / 1e6:.1f} km2) of overlap")

    sel = valid["jan23"] & valid["jul26"] & ~burned
    for k in refl:
        ndvi = (refl[k]["nir"] - refl[k]["red"]) / (refl[k]["nir"] + refl[k]["red"] + 1e-9)
        sel &= np.isfinite(refl[k]["nir"]) & (ndvi < 0.15)
    print(f"PIF pixels for the fit: {sel.sum()}")

    idx = np.flatnonzero(sel.ravel())
    take = RNG.choice(idx, size=min(N_FIT, idx.size), replace=False)
    rr, cc = np.unravel_index(take, sel.shape)

    spec = {}
    for k, s in scenes.items():
        r0, c0 = wins[k]
        spec[k] = s.spectra(rr + r0, cc + c0)

    wl = scenes["jan23"].wavelengths
    good = scenes["jan23"].good & scenes["jul26"].good
    nb = len(wl)
    gain = np.full(nb, np.nan)
    offset = np.full(nb, np.nan)
    r2 = np.full(nb, np.nan)

    for b in range(nb):
        if not good[b]:
            continue
        x, y = spec["jan23"][:, b], spec["jul26"][:, b]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 200:
            continue
        x, y = x[m], y[m]
        g, o = np.polyfit(x, y, 1)
        gain[b], offset[b] = g, o
        pred = g * x + o
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2[b] = 1 - ss_res / ss_tot

    np.savez(
        "data/crossdate_gain.npz",
        wavelengths=wl, gain=gain, offset=offset, r2=r2, good=good, n_pixels=len(take)
    )

    def summarize(lo, hi, label):
        m = good & (wl > lo) & (wl < hi)
        print(
            f"  {label:14s} gain {np.nanmedian(gain[m]):6.3f}   "
            f"offset {np.nanmedian(offset[m]):+7.4f}   R2 {np.nanmedian(r2[m]):5.3f}"
        )

    print(f"\nper-band OLS  R_jul = gain * R_jan + offset   (n={len(take)} PIF px)")
    summarize(450, 700, "VIS")
    summarize(700, 1300, "NIR")
    summarize(1500, 1780, "SWIR1")
    summarize(2000, 2400, "SWIR2")
    summarize(376, 2500, "all good bands")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (arr, name, ref) in zip(
        axes, [(gain, "gain", 1.0), (offset, "offset", 0.0), (r2, "R$^2$", None)]
    ):
        ax.plot(wl, np.where(good, arr, np.nan), color="#111", lw=1.2)
        if ref is not None:
            ax.axhline(ref, color="r", ls="--", lw=0.9)
        ax.set_xlabel("wavelength (nm)")
        ax.set_title(name)
        ax.grid(alpha=0.25)
    axes[2].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("data/quicklook/crossdate_gain.png", dpi=115)
    print("\nwrote data/crossdate_gain.npz and data/quicklook/crossdate_gain.png")

    for s in scenes.values():
        s.handle.close()


if __name__ == "__main__":
    main()
