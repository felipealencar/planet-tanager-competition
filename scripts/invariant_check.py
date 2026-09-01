"""Radiometric comparability check between the Jan and Jul Tanager acquisitions.

The January scene is flagged `quality_category: test`, uses a different collection mode,
and was acquired at 34 deg sun elevation vs 73 deg for July. Before building any two-date
analysis on this pair we need to know whether their surface reflectances are on the same
scale, or whether the difference between dates is dominated by calibration/illumination
rather than by surface change.

Method: sample pseudo-invariant features (PIFs) -- surfaces that genuinely should not
change between January and July -- and compare their 426-band spectra across dates.

  deep water   : dark, spectrally flat, no vegetation, effectively flat-lying (so it
                 carries almost no topographic-shading signal). Isolates calibration.
  bright PIF   : bright, low-NDVI in BOTH dates (roofs, roads, quarries, beach sand).
                 Isolates calibration at the bright end of the dynamic range.

Both classes are sampled only where both dates flag the pixel valid, and only inside the
scene overlap. Outputs a spectral comparison figure and per-band difference statistics.
"""

from __future__ import annotations

import sys

import matplotlib

# Only force the headless backend when run as a script; importing this module
# from a notebook must not clobber the inline backend.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from tanager import Scene  # noqa: E402

PATHS = {
    "jan23": "data/hdf5/20250123_185518_92_4001_ortho_sr.h5",
    "jul26": "data/hdf5/20250726_192422_87_4001_ortho_sr.h5",
}
PROBE = {"blue": 480, "green": 560, "red": 660, "nir": 860, "swir1": 1650, "swir2": 2200}
N_SAMPLE = 150
RNG = np.random.default_rng(0)


def overlap_windows(scenes):
    """Pixel offsets of the common extent in each scene. Grids are 30 m aligned."""
    lefts, tops, rights, bottoms = [], [], [], []
    for s in scenes.values():
        lefts.append(s.transform.c)
        tops.append(s.transform.f)
        rights.append(s.transform.c + s.shape[1] * 30)
        bottoms.append(s.transform.f - s.shape[0] * 30)
    x0, y0 = max(lefts), min(tops)
    x1, y1 = min(rights), max(bottoms)
    nrow, ncol = int((y0 - y1) / 30), int((x1 - x0) / 30)
    wins = {}
    for k, s in scenes.items():
        r0 = int(round((s.transform.f - y0) / 30))
        c0 = int(round((x0 - s.transform.c) / 30))
        assert r0 >= 0 and c0 >= 0, "grids not aligned as expected"
        wins[k] = (r0, c0)
    return wins, nrow, ncol, (x0, y0)


def probe_bands(scene, r0, c0, nrow, ncol):
    out = {}
    for name, wl in PROBE.items():
        out[name] = scene.band(wl)[r0 : r0 + nrow, c0 : c0 + ncol]
    return out


def main():
    scenes = {k: Scene.open(p) for k, p in PATHS.items()}
    wins, nrow, ncol, origin = overlap_windows(scenes)
    print(f"overlap grid {nrow} x {ncol} px  ({nrow * ncol * 900 / 1e6:.0f} km2)")
    print(f"windows (row0, col0): {wins}\n")

    refl, valid = {}, {}
    for k, s in scenes.items():
        r0, c0 = wins[k]
        refl[k] = probe_bands(s, r0, c0, nrow, ncol)
        valid[k] = s.valid_mask()[r0 : r0 + nrow, c0 : c0 + ncol]

    both = valid["jan23"] & valid["jul26"]
    for k in refl:
        both &= np.isfinite(refl[k]["nir"]) & np.isfinite(refl[k]["swir2"])
    print(f"valid in both dates: {both.sum()} px ({100 * both.mean():.1f}% of overlap)")

    ndvi = {
        k: (refl[k]["nir"] - refl[k]["red"]) / (refl[k]["nir"] + refl[k]["red"] + 1e-9)
        for k in refl
    }
    bright = {k: np.nanmean([refl[k][b] for b in PROBE], axis=0) for k in refl}

    water = both.copy()
    for k in refl:
        water &= (refl[k]["nir"] < 0.02) & (refl[k]["swir1"] < 0.015) & (refl[k]["blue"] > 0.01)

    pif = both.copy()
    for k in refl:
        pif &= (ndvi[k] < 0.15) & (bright[k] > 0.18) & (refl[k]["swir2"] > 0.12)

    print(f"deep-water candidates: {water.sum()} px")
    print(f"bright-PIF candidates: {pif.sum()} px\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    stats = {}
    for row, (label, mask) in enumerate([("deep water", water), ("bright PIF", pif)]):
        idx = np.flatnonzero(mask.ravel())
        if idx.size == 0:
            print(f"!! no {label} pixels found")
            continue
        take = RNG.choice(idx, size=min(N_SAMPLE, idx.size), replace=False)
        rr, cc = np.unravel_index(take, mask.shape)
        spec = {}
        for k, s in scenes.items():
            r0, c0 = wins[k]
            spec[k] = np.nanmedian(s.spectra(rr + r0, cc + c0), axis=0)

        wl = scenes["jan23"].wavelengths
        good = scenes["jan23"].good & scenes["jul26"].good
        ax = axes[row, 0]
        for k, c in [("jan23", "#3b82f6"), ("jul26", "#f97316")]:
            y = np.where(good, spec[k], np.nan)
            ax.plot(wl, y, color=c, lw=1.3, label=k)
        ax.set_title(f"{label}  (median of {len(take)} px)")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("surface reflectance")
        ax.legend()
        ax.grid(alpha=0.25)

        diff = np.where(good, spec["jul26"] - spec["jan23"], np.nan)
        ax = axes[row, 1]
        ax.plot(wl, diff, color="#111", lw=1.2)
        ax.axhline(0, color="r", lw=0.8, ls="--")
        ax.set_title(f"{label}: jul26 - jan23")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("d reflectance")
        ax.grid(alpha=0.25)

        stats[label] = {
            "n": int(len(take)),
            "median_abs_diff": float(np.nanmedian(np.abs(diff))),
            "mean_diff": float(np.nanmean(diff)),
            "vnir_mean_diff": float(np.nanmean(diff[(wl > 450) & (wl < 900)])),
            "swir_mean_diff": float(np.nanmean(diff[(wl > 1500) & (wl < 2400)])),
            "jan_median_refl": float(np.nanmean(np.where(good, spec["jan23"], np.nan))),
            "jul_median_refl": float(np.nanmean(np.where(good, spec["jul26"], np.nan))),
        }

    plt.tight_layout()
    plt.savefig("data/quicklook/invariant_check.png", dpi=115)

    print(f"{'target':12s} {'n':>4s} {'jan mean':>9s} {'jul mean':>9s} "
          f"{'md|diff|':>9s} {'VNIR bias':>10s} {'SWIR bias':>10s}")
    for label, s in stats.items():
        print(
            f"{label:12s} {s['n']:4d} {s['jan_median_refl']:9.4f} {s['jul_median_refl']:9.4f} "
            f"{s['median_abs_diff']:9.4f} {s['vnir_mean_diff']:10.4f} {s['swir_mean_diff']:10.4f}"
        )
    print("\nwrote data/quicklook/invariant_check.png")

    for s in scenes.values():
        s.handle.close()


if __name__ == "__main__":
    main()
