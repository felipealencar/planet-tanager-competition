"""Pre-fire vegetation type and fire history on the Tanager overlap grid.

The recovery numbers in analyze.py compare each fire against a single pooled unburned
control. That control is not valid, because the fires did not burn the same vegetation:
Palisades is chaparral-dominated and Franklin is coastal-scrub-dominated, and those two
communities have very different seasonal trajectories between January and July. Any
"Franklin recovers faster than Palisades" statement drawn from a pooled control is
therefore confounded with vegetation type. This module supplies the stratifier that
removes the confound, plus the fire history that turns out to reframe the whole scene.

Two public sources, neither requiring authentication:

  1. **LANDFIRE LF2023 Existing Vegetation Type** (USGS, 30 m). Chosen over the NPS
     Santa Monica Mountains alliance map because it is served as a live ImageServer and
     because LANDFIRE reruns its disturbance logic every release -- LF2023 already
     reflects Woolsey (2018) and every earlier fire, whereas the NPS polygon map is a
     mid-2000s snapshot. LF2023 postdates all prior fires and predates all four
     2024-25 fires, so it is a genuine pre-fire map for this project.
  2. **CAL FIRE historic fire perimeters** (FRAP, 1980-2023), which supply a per-pixel
     count of how many times each pixel had already burned before the 2024-25 fires.

**The grid is free, again.** LANDFIRE is natively 30 m and the ImageServer will export
directly into EPSG:32611 on a caller-specified extent, so requesting the exact Tanager
overlap window returns a 713 x 743 array on the identical origin -- same "no resampling
before analysis" property the two Tanager dates already had. Nearest-neighbour
interpolation is forced, because these are class codes and any averaging would invent
categories that do not exist.

The headline result from the fire history is that **81.7% of the area that burned in
2024-25 had already burned at least once since 1980, and 32.8% three or more times**,
which is the context every recovery number in this project should be read against. Only
11.7 km² (18.3%) was burning for the first time.

An earlier version of this module reported that *no* pixel was burning for the first
time. That was an artifact of the query: it counted the 2024-25 fires themselves in the
"prior" tally, so every burned pixel trivially had a prior fire. The count is cut at 2023
via the `through` argument to prior_fire_count, and the qualitative point survives while
the absolute claim does not.

Outputs:
  data/vegtype.npz                     EVT codes, prior-fire count, grid origin
  data/lf23_evt_lut.csv                LANDFIRE class-code lookup table
  data/fire_history_1980_2023.geojson  CAL FIRE perimeters intersecting the AOI
  data/quicklook/vegtype.png
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import sys

import matplotlib

# Only force the headless backend when run as a script; importing this module
# from a notebook must not clobber the inline backend.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import requests
from affine import Affine
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

sys.path.insert(0, "scripts")

EVT_SERVICE = (
    "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2023/"
    "LF2023_EVT_CONUS/ImageServer"
)
EVT_LUT_URL = "https://landfire.gov/sites/default/files/CSV/LF2023/LF23_EVT_240.csv"
CALFIRE = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Historic_Fire_Perimeters/FeatureServer/0/query"
)
AOI_WGS84 = "-118.90,33.88,-118.45,34.25"

VEG_NPZ = pathlib.Path("data/vegtype.npz")
LUT_PATH = pathlib.Path("data/lf23_evt_lut.csv")
HIST_PATH = pathlib.Path("data/fire_history_1980_2023.geojson")

UTM = Transformer.from_crs(4326, 32611, always_xy=True).transform

# The classes this project reports on. Everything else (developed, water, agriculture)
# is real but not interpretable as post-fire recovery, so it is excluded from statistics
# rather than silently pooled into a background class.
CHAPARRAL = 7110
COASTAL_SCRUB = 7092
NATURAL = {
    7110: "Dry-Mesic Chaparral",
    7092: "Coastal Scrub",
    7113: "Coast Live Oak Woodland",
    7097: "Mesic Chaparral",
    7014: "Mixed Evergreen Woodland",
    7129: "Native Grassland",
    9301: "Ruderal Grassland (exotic)",
    9337: "Ruderal Scrub (exotic)",
}
EXOTIC = {9301, 9337}


def grid():
    """The Tanager overlap window: origin, shape, and its affine transform."""
    d = np.load("data/fractions.npz")
    origin = d["origin"]
    nrow, ncol = d["frac_jan23"].shape[1:]
    transform = Affine.translation(origin[0], origin[1]) * Affine.scale(30, -30)
    return origin, nrow, ncol, transform


def fetch_lut():
    """LANDFIRE class-code -> class-name table."""
    if not LUT_PATH.exists():
        LUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(EVT_LUT_URL, timeout=180)
        r.raise_for_status()
        LUT_PATH.write_bytes(r.content)
    return {int(row["VALUE"]): row for row in csv.DictReader(LUT_PATH.open())}


def fetch_evt(origin, nrow, ncol):
    """Export LF2023 EVT directly onto the Tanager grid -- no resampling afterwards."""
    xmin, ymax = float(origin[0]), float(origin[1])
    xmax, ymin = xmin + ncol * 30, ymax - nrow * 30
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 32611,
        "imageSR": 32611,
        "size": f"{ncol},{nrow}",
        "format": "tiff",
        "pixelType": "U16",
        # class codes: averaging them would invent categories that do not exist
        "interpolation": "RSP_NearestNeighbor",
        "noDataInterpretation": "esriNoDataMatchAny",
        "f": "json",
    }
    meta = requests.get(f"{EVT_SERVICE}/exportImage", params=params, timeout=300).json()
    if "href" not in meta:
        raise RuntimeError(f"LANDFIRE exportImage failed: {meta}")
    tif = requests.get(meta["href"], timeout=600).content
    with rasterio.open(io.BytesIO(tif)) as src:
        evt = src.read(1)
        assert src.shape == (nrow, ncol), f"grid mismatch: {src.shape} != {(nrow, ncol)}"
        assert abs(src.transform[0] - 30) < 1e-6, "LANDFIRE did not return a 30 m grid"
        assert abs(src.transform[2] - xmin) < 1e-6, "LANDFIRE origin does not align"
    return evt


def fetch_fire_history():
    """Every CAL FIRE perimeter since 1980 intersecting the scene area."""
    if HIST_PATH.exists():
        return json.loads(HIST_PATH.read_text())
    params = {
        "where": "YEAR_ >= 1980",
        "geometry": AOI_WGS84,
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FIRE_NAME,YEAR_,ALARM_DATE,GIS_ACRES",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
        "resultRecordCount": 2000,
    }
    gj = requests.get(CALFIRE, params=params, timeout=300).json()
    HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    HIST_PATH.write_text(json.dumps(gj))
    return gj


def prior_fire_count(gj, nrow, ncol, transform, through=2023):
    """Per-pixel count of fires that burned here BEFORE the 2024-25 fires."""
    count = np.zeros((nrow, ncol), np.int16)
    burned = []
    for feat in gj.get("features", []):
        props, geom = feat["properties"], feat.get("geometry")
        year = props.get("YEAR_")
        if not geom or year is None or int(year) > through:
            continue
        try:
            poly = shp_transform(UTM, shape(geom).buffer(0))
        except Exception:
            continue
        mask = rasterio.features.rasterize(
            [poly], out_shape=(nrow, ncol), transform=transform, dtype="uint8"
        ).astype(bool)
        if not mask.any():
            continue
        count += mask
        burned.append((int(year), props.get("FIRE_NAME"), mask))
    return count, burned


def load():
    """Cached accessor used by analyze/recovery_by_type and the notebook."""
    if not VEG_NPZ.exists():
        main()
    d = np.load(VEG_NPZ)
    return d["evt"], d["prior_fires"], fetch_lut()


def main():
    origin, nrow, ncol, transform = grid()
    lut = fetch_lut()

    print(f"Tanager overlap grid: {nrow} x {ncol} @ 30 m, origin {tuple(origin)}")
    evt = fetch_evt(origin, nrow, ncol)
    print("LANDFIRE LF2023 EVT exported on the identical grid (no resampling)\n")

    vals, counts = np.unique(evt, return_counts=True)
    order = np.argsort(-counts)
    print("EVT composition of the overlap:")
    for i in order[:12]:
        v = int(vals[i])
        name = lut.get(v, {}).get("EVT_NAME", "?")
        print(f"  {counts[i] * 900 / 1e6:7.1f} km2  {100 * counts[i] / evt.size:5.1f}%  {name}")

    gj = fetch_fire_history()
    count, burned = prior_fire_count(gj, nrow, ncol, transform)
    print(f"\nCAL FIRE perimeters 1980-2023 touching the overlap: {len(burned)}")

    from analyze import per_fire_masks

    masks, any_fire = per_fire_masks(nrow, ncol, origin)
    print("\nLargest reburns under the 2024-25 scars:")
    ranked = sorted(
        ((y, n, (m & any_fire).sum() * 900 / 1e6) for y, n, m in burned),
        key=lambda r: -r[2],
    )
    for year, name, km2 in ranked[:8]:
        if km2 < 0.2:
            continue
        print(f"  {year}  {str(name)[:24]:24s} {km2:6.1f} km2")

    print("\nPrior fires (1980-2023) per pixel INSIDE the 2024-25 scars:")
    total = max(int(any_fire.sum()), 1)
    for k in range(0, int(count.max()) + 1):
        px = int(((count == k) & any_fire).sum())
        if px == 0 and k > 0:
            continue
        print(f"  {k} prior: {px * 900 / 1e6:7.1f} km2 ({100 * px / total:5.1f}%)")
    first_time = int(((count == 0) & any_fire).sum())
    print(
        f"\n  -> {first_time * 900 / 1e6:.1f} km2 of the 2024-25 burn area was burning "
        f"for the first time since 1980."
    )

    np.savez_compressed(
        VEG_NPZ, evt=evt, prior_fires=count, origin=origin
    )
    print(f"\nwrote {VEG_NPZ}")

    plot(evt, count, masks, lut)


def plot(evt, count, masks, lut):
    codes = [c for c in NATURAL if (evt == c).any()]
    colors = ["#16a34a", "#84cc16", "#166534", "#22c55e",
              "#0e7490", "#fbbf24", "#f97316", "#dc2626"]
    lookup = {c: i for i, c in enumerate(codes)}
    disp = np.full(evt.shape, np.nan)
    for c, i in lookup.items():
        disp[evt == c] = i

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(disp, cmap=ListedColormap(colors[: len(codes)]),
                   vmin=-0.5, vmax=len(codes) - 0.5, interpolation="nearest")
    axes[0].set_title("LANDFIRE LF2023 pre-fire vegetation type")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=colors[i],
                          label=NATURAL[c]) for c, i in lookup.items()]
    axes[0].legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)

    im = axes[1].imshow(np.where(count > 0, count, np.nan), cmap="inferno",
                        vmin=0, vmax=4, interpolation="nearest")
    axes[1].set_title("Fires per pixel, 1980-2023 (before the 2024-25 fires)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, label="prior fire count")

    for ax in axes:
        for mask in masks.values():
            ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    pathlib.Path("data/quicklook").mkdir(parents=True, exist_ok=True)
    plt.savefig("data/quicklook/vegtype.png", dpi=115)
    print("wrote data/quicklook/vegtype.png")


if __name__ == "__main__":
    main()
