"""Post-fire analysis: char loading in January, and vegetation recovery by July.

Everything here uses shade-normalized fractions, f / (1 - f_shade), which removes the
photometric illumination term and is therefore the quantity that can legitimately be
compared between a 34-degree-sun January scene and a 73-degree-sun July scene.

Produces:
  - per-fire statistics against an unburned control drawn from the same overlap
  - a char-fraction map for January and a GV-recovery map for January -> July
  - distribution plots of char and GV by fire and by date
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

UTM = Transformer.from_crs(4326, 32611, always_xy=True).transform
FIRES = ["PALISADES", "Franklin", "KENNETH"]
IGNITION = {"PALISADES": "2025-01-07", "Franklin": "2024-12-10", "KENNETH": "2025-01-09"}


def per_fire_masks(nrow, ncol, origin):
    perims = json.load(open("data/fire_perimeters_2024_2025.geojson"))
    transform = Affine.translation(origin[0], origin[1]) * Affine.scale(30, -30)
    masks = {}
    for name in FIRES:
        geoms = [
            shp_transform(UTM, shape(f["geometry"]).buffer(0))
            for f in perims["features"]
            if f["properties"]["poly_IncidentName"] == name
        ]
        if not geoms:
            continue
        masks[name] = rasterio.features.rasterize(
            geoms, out_shape=(nrow, ncol), transform=transform, dtype="uint8"
        ).astype(bool)
    any_fire = np.zeros((nrow, ncol), bool)
    for m in masks.values():
        any_fire |= m
    return masks, any_fire


def main():
    d = np.load("data/fractions.npz")
    names = [str(n) for n in d["names"]]
    origin = d["origin"]
    iGV, iCH, iSH = names.index("GV"), names.index("CHAR"), names.index("SHADE")

    norm, valid = {}, {}
    for k in ("jan23", "jul26"):
        frac = d[f"frac_{k}"]
        illum = np.clip(1.0 - frac[iSH], 1e-3, None)
        norm[k] = frac / illum
        valid[k] = np.isfinite(d[f"rmse_{k}"]) & (frac[iSH] < 0.9)
    both = valid["jan23"] & valid["jul26"]
    nrow, ncol = both.shape

    masks, any_fire = per_fire_masks(nrow, ncol, origin)
    # Control: unburned, valid in both dates. This absorbs the seasonal (wet->dry)
    # signal, so recovery inside scars is measured relative to what unburned chaparral
    # did over the same six months rather than against zero.
    control = both & ~any_fire
    print(f"control (unburned, valid both dates): {control.sum():,} px "
          f"({control.sum() * 900 / 1e6:.1f} km2)\n")

    def stat(mask, k, idx):
        v = norm[k][idx][mask]
        v = v[np.isfinite(v)]
        return np.median(v) if v.size else np.nan

    rows = []
    for label, mask in [("CONTROL (unburned)", control)] + [
        (f, masks[f] & both) for f in FIRES if f in masks
    ]:
        n = int(mask.sum())
        if n < 50:
            continue
        char_jan = stat(mask, "jan23", iCH)
        char_jul = stat(mask, "jul26", iCH)
        gv_jan = stat(mask, "jan23", iGV)
        gv_jul = stat(mask, "jul26", iGV)
        ctl_dgv = None
        rows.append((label, n, char_jan, char_jul, gv_jan, gv_jul, gv_jul - gv_jan))

    ctl = rows[0]
    print(f"{'region':22s} {'px':>7s} {'km2':>6s} {'char Jan':>9s} {'char Jul':>9s} "
          f"{'GV Jan':>7s} {'GV Jul':>7s} {'dGV':>7s} {'dGV vs ctl':>11s}")
    for label, n, cj, cl, gj, gl, dg in rows:
        rel = dg - ctl[6]
        print(f"{label:22s} {n:7,d} {n * 900 / 1e6:6.1f} {cj:9.3f} {cl:9.3f} "
              f"{gj:7.3f} {gl:7.3f} {dg:+7.3f} {rel:+11.3f}")

    # ---- figures ----------------------------------------------------------
    char_jan = np.where(both, norm["jan23"][iCH], np.nan)
    dgv = np.where(both, norm["jul26"][iGV] - norm["jan23"][iGV], np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    im = axes[0].imshow(char_jan, cmap="inferno", vmin=0, vmax=0.8)
    axes[0].set_title("Char fraction, 23 Jan 2025 (shade-normalized)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, label="char fraction")

    im = axes[1].imshow(dgv, cmap="RdYlGn", vmin=-0.4, vmax=0.4)
    axes[1].set_title("$\\Delta$GV, Jan $\\rightarrow$ Jul 2025 (shade-normalized)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, label="$\\Delta$ GV fraction")

    transform = Affine.translation(origin[0], origin[1]) * Affine.scale(30, -30)
    for ax in axes:
        for name, m in masks.items():
            ax.contour(m.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("data/quicklook/char_and_recovery.png", dpi=115)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, mask, color in [
        ("control", control, "#64748b"),
        ("Palisades", masks.get("PALISADES", control) & both, "#ef4444"),
        ("Franklin", masks.get("Franklin", control) & both, "#eab308"),
    ]:
        v = norm["jan23"][iCH][mask]
        axes[0].hist(v[np.isfinite(v)], bins=60, range=(0, 1), density=True,
                     histtype="step", lw=1.6, color=color, label=label)
        v = dgv[mask]
        axes[1].hist(v[np.isfinite(v)], bins=60, range=(-0.6, 0.6), density=True,
                     histtype="step", lw=1.6, color=color, label=label)
    axes[0].set_xlabel("char fraction, Jan 23")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("$\\Delta$GV, Jan $\\rightarrow$ Jul")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    plt.tight_layout()
    plt.savefig("data/quicklook/distributions.png", dpi=115)
    print("\nwrote data/quicklook/char_and_recovery.png and distributions.png")


if __name__ == "__main__":
    main()
