"""Sentinel-2 pre-fire baseline and independent dNBR validation of the Tanager char map.

The Tanager fire collection has no pre-fire acquisition over the Santa Monica Mountains --
its earliest scene here is already 16 days post-Palisades. Sentinel-2 supplies the missing
"before", and does double duty:

  1. **Pre-burn baseline.** A true pre-fire image, so the story is not limited to
     post-fire states.
  2. **Independent validation.** Sentinel-2 dNBR is computed from a different sensor,
     different processing chain and different spectral sampling than Tanager. Comparing it
     to the Tanager char fraction is therefore a genuinely external check, unlike anything
     derived from the Tanager cubes themselves.

Scenes (MGRS 11SLT, from the AWS Earth Search STAC API -- public, no authentication):

  pre   2024-11-13   27 days before Franklin ignited, 55 before Palisades  cloud 0.0%
  post  2025-01-12    5 days after Palisades ignited, 11 before Tanager    cloud 0.0%
  feb   2025-02-21   fully post-containment, unambiguous final scar        cloud 5.0%
  late  2025-08-05   10 days after the Tanager July acquisition            cloud 1.0%

**Scene selection matters more than it looks.** The obvious picks -- the scenes closest in
date to the two Tanager acquisitions -- are partial swaths covering only the western third
of the AOI (74% nodata, footprints stopping at -118.70 while the AOI runs to -118.58).
Bounding-box intersection in a STAC query does not imply coverage. Scenes here are
therefore filtered on `s2:nodata_pixel_percentage < 5` AND a footprint that fully contains
the AOI, which costs some temporal proximity but buys complete, gap-free maps.

Two post-fire dates are carried deliberately: `post` (Jan 12) is closest to peak scar
contrast but falls while Palisades was still burning on its eastern flank, so `feb`
(Feb 21, after full containment on Jan 31) is the one used for validation.

NBR uses B8A (865 nm, 20 m) and B12 (2190 nm, 20 m); both native 20 m, so no band is
upsampled relative to the other. Everything is resampled once onto the exact Tanager
30 m grid so all products are pixel-aligned.

Licensing note: Copernicus Sentinel data is free and open, and the competition FAQ
explicitly permits Sentinel-2 alongside Tanager.
"""

from __future__ import annotations

import sys

import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds

sys.path.insert(0, "scripts")

SEARCH = "https://earth-search.aws.element84.com/v1/search"
BBOX = [-118.79, 33.99, -118.57, 34.20]
SCENES = {
    "pre": "S2A_11SLT_20241113_0_L2A",
    "post": "S2A_11SLT_20250112_0_L2A",
    "feb": "S2C_11SLT_20250221_0_L2A",
    "late": "S2B_11SLT_20250805_0_L2A",
}
MAX_NODATA_PCT = 5.0
# SCL classes to discard: nodata, saturated, shadow, cloud med/high prob, thin cirrus.
SCL_BAD = {0, 1, 3, 8, 9, 10}


def search(start, end):
    """All scenes intersecting the AOI in a window, with coverage diagnostics."""
    r = requests.post(
        SEARCH,
        json={
            "collections": ["sentinel-2-l2a"],
            "bbox": BBOX,
            "datetime": f"{start}/{end}",
            "limit": 100,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["features"]


def covers_aoi(item):
    """True only if the scene footprint fully contains the AOI and has little nodata.

    A STAC bbox *intersection* is not coverage: Sentinel-2 partial swaths over this AOI
    routinely return 74% nodata while still matching a bbox query.
    """
    b = item["bbox"]
    contains = b[0] <= BBOX[0] and b[1] <= BBOX[1] and b[2] >= BBOX[2] and b[3] >= BBOX[3]
    nodata = item["properties"].get("s2:nodata_pixel_percentage", 100.0)
    return contains and nodata < MAX_NODATA_PCT


def find_scene(scene_id):
    """Fetch one pinned scene by id, and assert it really covers the AOI."""
    r = requests.get(
        f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{scene_id}",
        timeout=60,
    )
    r.raise_for_status()
    item = r.json()
    if not covers_aoi(item):
        raise RuntimeError(f"{scene_id} does not fully cover the AOI")
    return item


def read_on_grid(href, dst_transform, dst_shape, bounds, resampling):
    """Windowed read of a COG, reprojected onto the Tanager 30 m grid."""
    with rasterio.open(href) as src:
        win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=win)
        src_transform = src.window_transform(win)
        out = np.zeros(dst_shape, dtype=data.dtype)
        reproject(
            data, out,
            src_transform=src_transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=src.crs,
            resampling=resampling,
        )
    return out


def scale_reflectance(dn, item):
    """DN -> surface reflectance, honouring the post-baseline-04.00 BOA offset."""
    x = dn.astype("f4")
    x[x == 0] = np.nan
    if not item["properties"].get("earthsearch:boa_offset_applied", False):
        x = x - 1000.0
    return x / 10000.0


def main():
    origin = np.load("data/fractions.npz")["origin"]
    nrow, ncol = 713, 743
    transform = Affine.translation(origin[0], origin[1]) * Affine.scale(30, -30)
    bounds = (origin[0], origin[1] - nrow * 30, origin[0] + ncol * 30, origin[1])
    print(f"target grid {nrow} x {ncol} @ 30 m, EPSG:32611")
    print(f"bounds {bounds}\n")

    nbr, meta = {}, {}
    for label, scene_id in SCENES.items():
        item = find_scene(scene_id)
        pid = item["id"]
        date = item["properties"]["datetime"][:10]
        cloud = item["properties"]["eo:cloud_cover"]
        print(f"{label:5s} {pid}  {date}  cloud {cloud:.1f}%")

        nir = read_on_grid(item["assets"]["nir08"]["href"], transform, (nrow, ncol),
                           bounds, Resampling.bilinear)
        swir = read_on_grid(item["assets"]["swir22"]["href"], transform, (nrow, ncol),
                            bounds, Resampling.bilinear)
        scl = read_on_grid(item["assets"]["scl"]["href"], transform, (nrow, ncol),
                           bounds, Resampling.nearest)

        nir = scale_reflectance(nir, item)
        swir = scale_reflectance(swir, item)
        # Guard the ratio: where NIR+SWIR approaches zero (deep water, deep shadow,
        # residual nodata) NBR is numerically meaningless and blows up.
        denom = nir + swir
        bad = np.isin(scl, list(SCL_BAD)) | ~(denom > 0.01) | (nir < 0) | (swir < 0)
        n = (nir - swir) / np.where(denom > 0.01, denom, np.nan)
        n[bad] = np.nan
        nbr[label] = n
        meta[label] = {"id": pid, "date": date, "cloud": cloud}
        print(f"      valid {np.isfinite(n).mean() * 100:5.1f}%   "
              f"median NBR {np.nanmedian(n):+.3f}")

    dnbr = {
        "dnbr_post": nbr["pre"] - nbr["post"],  # peak scar contrast, still burning east
        "dnbr_feb": nbr["pre"] - nbr["feb"],    # post-containment -> used for validation
        "dnbr_late": nbr["pre"] - nbr["late"],  # residual severity after ~9 months
    }

    layers = [(f"nbr_{k}", nbr[k]) for k in SCENES] + list(dnbr.items())
    with rasterio.open(
        "data/sentinel2_nbr.tif", "w", driver="GTiff", height=nrow, width=ncol,
        count=len(layers), dtype="float32", crs="EPSG:32611", transform=transform,
        compress="deflate", nodata=np.nan,
    ) as dst:
        for i, (name, arr) in enumerate(layers, start=1):
            dst.write(arr.astype("f4"), i)
            dst.set_band_description(i, name)

    np.savez_compressed(
        "data/sentinel2.npz",
        origin=origin, meta=np.array([str(meta)], dtype=object),
        **{f"nbr_{k}": v for k, v in nbr.items()}, **dnbr,
    )
    print()
    for name, arr in dnbr.items():
        print(f"{name:11s} median {np.nanmedian(arr):+.3f}  "
              f"p99 {np.nanpercentile(arr, 99):+.3f}  max {np.nanmax(arr):+.3f}")
    print("wrote data/sentinel2_nbr.tif and data/sentinel2.npz")


if __name__ == "__main__":
    main()
