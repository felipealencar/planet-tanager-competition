"""Build the post-fire endmember library, with the library size chosen from the data.

Two earlier designs failed, and both failures shaped this one (documented in FINDINGS.md):

  1. Blind VCA over the whole overlap returned deep ocean and deep topographic shadow as
     simplex vertices -- the true extremes of the scene are not fire materials.
  2. Region-constrained VCA plus hand-specified absorption features still failed, because
     the assumed continuum shoulders were wrong: the true SWIR2 continuum peaks near
     2200 nm, so every "absorption depth" computed against fixed 1990/2450 shoulders came
     out negative and no candidate ever matched NPV or SOIL.

So rather than assuming a five-material library and forcing candidates into it, this
measures which materials are actually separable at Tanager's SNR, and keeps only those.
Class medians (n = 10k-40k pixels each) are used as endmembers instead of single VCA
vertices -- a median over thousands of pixels is far less noisy than one extreme pixel,
at the cost of being slightly less "pure".

Separability is measured by spectral angle, which is brightness-invariant, so it isolates
material differences from illumination differences. Measured on this pair:

    GV vs CHAR         35.0 deg    strongly separable
    GV vs NPV/soil     26.0 deg    strongly separable
    CHAR vs NPV/soil   10.8 deg    separable
    CHAR vs bright-burn 5.4 deg    NOT separable -> same material, different brightness
    bright-burn vs ash  2.8 deg    NOT separable -> no distinct ash endmember exists

The last two lines are the result that sets the library. Dark and bright burned areas
share a spectral shape and differ only in brightness, which the photometric shade
endmember already accounts for; and no spectrally distinct white ash is detectable 16
days after the Palisades fire at 30 m. Forcing ASH into the library would make the
unmixing ill-conditioned for no gain. Final library: GV, CHAR, NPV_SOIL, SHADE.

The SHADE zero vector is what makes fractions comparable across dates: it absorbs the
spectrally flat ~5% gain measured in crossdate_gain.py, so the 34-vs-73 degree sun
elevation difference does not propagate into the fraction maps.
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
from crossdate_gain import fire_mask  # noqa: E402
from invariant_check import PATHS, overlap_windows, probe_bands  # noqa: E402
from tanager import Scene  # noqa: E402

N_MEDIAN = 2000
MIN_ANGLE_DEG = 8.0  # below this, two endmembers are treated as the same material
RNG = np.random.default_rng(11)
COLORS = {"GV": "#22c55e", "NPV_SOIL": "#b45309", "CHAR": "#1f2937", "SHADE": "#000000"}


def spectral_angle_deg(a, b):
    return float(
        np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
    )


def class_masks(scene, r0, c0, nrow, ncol, burned):
    """Rule-based candidate masks. Deliberately conservative -- purity beats pixel count."""
    R = probe_bands(scene, r0, c0, nrow, ncol)
    valid = scene.valid_mask()[r0 : r0 + nrow, c0 : c0 + ncol]
    bright = np.nanmean([R[b] for b in ("green", "red", "nir", "swir1")], axis=0)
    ndvi = (R["nir"] - R["red"]) / (R["nir"] + R["red"] + 1e-9)
    water = (R["nir"] < 0.03) & (R["swir1"] < 0.03)
    valid &= np.isfinite(R["nir"]) & np.isfinite(R["swir2"]) & (bright > 0.04) & ~water
    return {
        "GV": valid & ~burned & (ndvi > 0.60),
        "NPV_SOIL": valid & ~burned & (ndvi < 0.22) & (bright > 0.15),
        "CHAR": valid & burned & (bright < 0.10) & (ndvi < 0.25),
    }


def median_spectrum(scene, wins_k, mask, usable):
    idx = np.flatnonzero(mask.ravel())
    if idx.size < 200:
        return None, int(idx.size)
    take = RNG.choice(idx, size=min(N_MEDIAN, idx.size), replace=False)
    rr, cc = np.unravel_index(take, mask.shape)
    r0, c0 = wins_k
    spec = scene.spectra(rr + r0, cc + c0)[:, usable]
    return np.nanmedian(spec, axis=0), int(idx.size)


def main():
    usable = np.load("data/usable_bands.npy")
    scenes = {k: Scene.open(p) for k, p in PATHS.items()}
    wins, nrow, ncol, origin = overlap_windows(scenes)
    wl = scenes["jan23"].wavelengths[usable]
    burned = fire_mask(nrow, ncol, origin)

    # Library comes from the January scene: it is the only date with fresh char, and a
    # single-date library is internally consistent. GV differs by only 4.2 deg between
    # dates, so using the January GV for both is safe.
    src = "jan23"
    masks = class_masks(scenes[src], *wins[src], nrow, ncol, burned)

    library, names, counts = [], [], []
    for cls in ["GV", "NPV_SOIL", "CHAR"]:
        spec, n = median_spectrum(scenes[src], wins[src], masks[cls], usable)
        if spec is None:
            print(f"!! {cls}: only {n} candidate px, dropped")
            continue
        angles = [spectral_angle_deg(spec, prev) for prev in library]
        if angles and min(angles) < MIN_ANGLE_DEG:
            print(f"!! {cls}: {min(angles):.1f} deg from {names[int(np.argmin(angles))]}, dropped")
            continue
        library.append(spec)
        names.append(cls)
        counts.append(n)
        print(f"  {cls:9s} median of {min(N_MEDIAN, n)} px (pool {n})")

    library.append(np.zeros(int(usable.sum())))
    names.append("SHADE")
    counts.append(0)
    library = np.vstack(library)

    print("\npairwise spectral angle (deg), material endmembers:")
    mat = names[:-1]
    print(" " * 11 + "".join(f"{n:>10s}" for n in mat))
    for i, n in enumerate(mat):
        print(f"  {n:9s}" + "".join(f"{spectral_angle_deg(library[i], library[j]):10.1f}"
                                    for j in range(len(mat))))

    cond = np.linalg.cond(library[:-1].T)
    print(f"\ncondition number of the material design matrix: {cond:.1f}")

    np.savez("data/endmembers.npz", library=library, names=np.array(names),
             counts=np.array(counts), wavelengths=wl, usable=usable, source=src)

    fig, ax = plt.subplots(figsize=(11, 6))
    for row, name, n in zip(library, names, counts):
        if name == "SHADE":
            continue
        ax.plot(wl, row, lw=1.5, color=COLORS.get(name), label=f"{name}  (n={n:,})")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("surface reflectance")
    ax.set_title(f"Tanager-1 post-fire endmember library — class medians, {src}")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("data/quicklook/endmembers.png", dpi=115)
    print(f"\nlibrary: {names}")
    print("wrote data/endmembers.npz and data/quicklook/endmembers.png")

    for s in scenes.values():
        s.handle.close()


if __name__ == "__main__":
    main()
