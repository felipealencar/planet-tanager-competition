"""Validate the Tanager char fraction against AVIRIS-3 — char fraction vs char fraction.

NASA/JPL overflew these fires with AVIRIS-3, an airborne imaging spectrometer, on three
dates in January 2025. Critically, one of them is **2025-01-23** — the same day as the
Tanager January acquisition, roughly 1.3 hours later. A near-simultaneous overpass by an
independent spectrometer is the strongest validation available for this project:

    burn severity products   correlate our char FRACTION against a severity PROXY
    AVIRIS-3                 compares char fraction against CHAR FRACTION, same day,
                             independent instrument, independent processing chain

Method, chosen so the comparison is apples-to-apples:

  1. Discover flight lines via CMR (open, no auth). Keep lines intersecting both the
     Tanager overlap and a burn perimeter, preferring the 2025-01-23 same-day flights.
  2. **Stream** windowed reads over HTTP range requests. The L2A ortho files are NetCDF-4
     at ~2 GB per line and 61 lines qualify (~123 GB) — far more than fits on disk, so
     nothing is downloaded whole.
  3. Resample the EXISTING Tanager endmember library onto the AVIRIS wavelength grid.
     Same endmembers, same physics, different spectral sampling — so disagreement reflects
     instrument and scene, not a different definition of "char".
  4. Unmix with FCLS, identically to the Tanager pipeline.
  5. Aggregate AVIRIS char fraction to the 30 m Tanager grid and compare per pixel.

    python scripts/aviris3.py --discover     # no credentials needed
    python scripts/aviris3.py --inspect      # dump one file's structure
    python scripts/aviris3.py                # full comparison
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from affine import Affine
from shapely.geometry import Polygon, box, shape
from shapely.ops import transform as shp_transform

sys.path.insert(0, "scripts")

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
AVIRIS_L2A_COLLECTION = "C3369603199-ORNL_CLOUD"
AOI_BBOX = "-118.85,33.95,-118.50,34.25"
TEMPORAL = "2025-01-01T00:00:00Z,2025-03-01T00:00:00Z"
FIRES = ("PALISADES", "Franklin", "KENNETH")

GRID_BOUNDS = (332760, 3763260, 355050, 3784650)  # Tanager overlap, EPSG:32611
GRID_SHAPE = (713, 743)
GRID_TRANSFORM = Affine.translation(GRID_BOUNDS[0], GRID_BOUNDS[3]) * Affine.scale(30, -30)
SUM_TO_ONE_WEIGHT = 30.0
TANAGER_DATE = "2025-01-23"


# --------------------------------------------------------------------- discovery
def _fetch_all_granules():
    entries, page = [], 1
    while True:
        r = requests.get(
            CMR_GRANULES,
            params={
                "collection_concept_id": AVIRIS_L2A_COLLECTION,
                "bounding_box": AOI_BBOX, "temporal": TEMPORAL,
                "page_size": 500, "page_num": page,
            },
            timeout=120,
        )
        r.raise_for_status()
        batch = r.json()["feed"]["entry"]
        entries += batch
        if len(batch) < 500:
            break
        page += 1
    return entries


def _polygon(entry):
    coords = [float(v) for v in entry["polygons"][0][0].split()]
    return Polygon([(coords[i + 1], coords[i]) for i in range(0, len(coords), 2)])


def _reflectance_href(entry):
    """The orthorectified L2A reflectance product is a NetCDF-4 file, not a GeoTIFF."""
    for link in entry.get("links", []):
        href = link.get("href", "")
        if href.endswith("_RFL_ORT.nc") and "/protected/" in href:
            return href
    return None


def discover():
    from pyproj import Transformer

    to_wgs = Transformer.from_crs(32611, 4326, always_xy=True).transform
    overlap = shp_transform(to_wgs, box(*GRID_BOUNDS))
    perims = json.load(open("data/fire_perimeters_2024_2025.geojson"))
    scars = {
        f["properties"]["poly_IncidentName"]: shape(f["geometry"]).buffer(0)
        for f in perims["features"]
        if f["properties"]["poly_IncidentName"] in FIRES
    }

    entries = _fetch_all_granules()
    selected = []
    for e in entries:
        poly = _polygon(e)
        inter = poly.intersection(overlap)
        if inter.is_empty:
            continue
        covered = {
            n: inter.intersection(s).area / s.area * 100
            for n, s in scars.items()
            if inter.intersects(s)
        }
        covered = {n: v for n, v in covered.items() if v > 0.5}
        if not covered:
            continue
        href = _reflectance_href(e)
        if href is None:
            continue
        selected.append({
            "id": e["title"], "date": e["time_start"][:10], "time": e["time_start"][11:19],
            "href": href, "size_mb": float(e.get("granule_size", 0)), "scars": covered,
        })

    # Prefer the same-day flights, then by how much burn scar each line covers.
    selected.sort(key=lambda g: (g["date"] != TANAGER_DATE, -sum(g["scars"].values())))
    return entries, selected


# --------------------------------------------------------------------- remote IO
_BYTES_READ = [0]


class _CountingFile:
    """Wraps a file object to tally bytes actually pulled over the network.

    Storage cost of streaming is ~0, but *bandwidth* cost depends entirely on how the
    NetCDF is chunked: a contiguous or band-chunked variable forces a spatial-window read
    to drag most of the 2 GB file across the wire. Measuring it turns that unknown into a
    number, so the decision to keep streaming (or fall back to downloading a few lines)
    is made on evidence.
    """

    def __init__(self, inner):
        self._inner = inner

    def read(self, *a, **kw):
        data = self._inner.read(*a, **kw)
        _BYTES_READ[0] += len(data)
        return data

    def readinto(self, buf):
        # h5py reads through readinto(), not read(); counting only read() reported 0 MB.
        n = self._inner.readinto(buf)
        _BYTES_READ[0] += n or 0
        return n

    def __getattr__(self, name):
        return getattr(self._inner, name)


def bytes_read_mb():
    return _BYTES_READ[0] / 1e6


def reset_bytes():
    _BYTES_READ[0] = 0


def open_remote(href):
    """Open a protected NetCDF-4 granule over HTTP range requests (no full download).

    Uses h5py rather than xarray because the reflectance cube lives in a NetCDF *group*
    (`/reflectance/reflectance`), while the spatial coordinates and CRS live in the root
    group — xarray's open_dataset reads only one group at a time.

    Verified layout for AVIRIS-3 L2A ORT:
        /easting, /northing              1-D coords, 3.1 m spacing, EPSG:32611
        /transverse_mercator             CRS (crs_wkt attr) + GeoTransform
        /reflectance/reflectance         (wavelength, northing, easting) f4,
                                         chunks (10, 256, 256), gzip
        /reflectance/wavelength, /fwhm   284 bands, 389.8-2493.5 nm
    """
    import fsspec
    import h5py

    from earthdata_auth import auth_headers

    fs = fsspec.filesystem("http", headers=auth_headers(), block_size=4 * 1024 * 1024)
    return h5py.File(_CountingFile(fs.open(href)), "r")


def describe_h5(f):
    print(f"  root vars: {list(f.keys())}")
    d = f["reflectance/reflectance"]
    wl = f["reflectance/wavelength"][:]
    print(f"  reflectance {d.shape} chunks={d.chunks} {d.compression}")
    print(f"  wavelengths {wl[0]:.1f}-{wl[-1]:.1f} nm, n={len(wl)}")
    print(f"  easting {f['easting'][0]:.0f}..{f['easting'][-1]:.0f}, "
          f"northing {f['northing'][0]:.0f}..{f['northing'][-1]:.0f}")


def select_bands(n_bands, chunk=10, stride=1):
    """Pick bands in whole chunk-groups so skipping actually saves network traffic.

    The cube is chunked 10 bands deep. Taking every Nth *band* still touches every chunk
    and saves nothing; taking every Nth *chunk* halves (or thirds) the bytes pulled while
    keeping full spectral span. At ~7.4 nm sampling, dropping to ~15 nm loses nothing for
    four broad endmembers.
    """
    groups = range(0, (n_bands + chunk - 1) // chunk, stride)
    idx = np.concatenate([np.arange(g * chunk, min((g + 1) * chunk, n_bands)) for g in groups])
    return idx


def fcls(A, R):
    from scipy.optimize import nnls

    k = A.shape[1]
    Aa = np.vstack([A, np.full((1, k), SUM_TO_ONE_WEIGHT)])
    out = np.full((k, R.shape[1]), np.nan, dtype="f4")
    ok = np.isfinite(R).all(axis=0) & (np.nanmax(R, axis=0) > 0.001)
    for j in np.flatnonzero(ok):
        f, _ = nnls(Aa, np.append(R[:, j], SUM_TO_ONE_WEIGHT))
        out[:, j] = f
    return out


def resample_library(lib, wl_src, wl_dst):
    out = np.zeros((lib.shape[0], len(wl_dst)))
    for i, row in enumerate(lib):
        good = np.isfinite(row)
        out[i] = np.interp(wl_dst, wl_src[good], row[good], left=np.nan, right=np.nan)
    return out


def process_line(href, lib, wl_tanager, names, verbose=False, band_stride=2):
    """Stream one flight line, unmix it, return char fraction on the Tanager 30 m grid."""
    import rasterio.crs
    from rasterio.warp import Resampling, reproject

    with open_remote(href) as f:
        if verbose:
            describe_h5(f)

        east = f["easting"][:].astype("f8")
        north = f["northing"][:].astype("f8")
        wl_all = f["reflectance/wavelength"][:].astype("f8")
        crs = rasterio.crs.CRS.from_wkt(f["transverse_mercator"].attrs["crs_wkt"].decode())

        # AVIRIS-3 ORT is already EPSG:32611, the Tanager CRS, so the window is a
        # straight coordinate comparison -- no reprojection before unmixing.
        minx, miny, maxx, maxy = GRID_BOUNDS
        xs = np.flatnonzero((east >= minx) & (east <= maxx))
        ys = np.flatnonzero((north >= miny) & (north <= maxy))
        if xs.size < 4 or ys.size < 4:
            return None, None, "no spatial overlap with the Tanager grid"

        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        dset = f["reflectance/reflectance"]
        bands = select_bands(dset.shape[0], chunk=(dset.chunks or (10,))[0], stride=band_stride)
        wl = wl_all[bands]
        cube = dset[bands, y0:y1, x0:x1].astype("f4")

        dx = float(east[1] - east[0])
        dy = float(north[1] - north[0])
        window_transform = Affine.translation(
            float(east[x0]) - dx / 2, float(north[y0]) - dy / 2
        ) * Affine.scale(dx, dy)

    cube[cube <= -0.05] = np.nan
    cube[cube > 1.5] = np.nan
    nb, h, w = cube.shape

    # Aggregate REFLECTANCE to 30 m first, then unmix -- rather than unmixing all 1.75M
    # native 3.1 m pixels and averaging the fractions afterwards. Two reasons:
    #
    #   Correctness: this reproduces what Tanager physically does. Tanager integrates
    #   radiance over a 30 m footprint and we unmix that integrated spectrum. Doing the
    #   same to AVIRIS isolates instrument and calibration differences from scale effects,
    #   instead of confounding the two. (Unmixing fine then averaging answers a different
    #   question: "what does AVIRIS see at 3 m, smoothed".)
    #
    #   Cost: ~19k NNLS solves per line instead of ~1.75M, roughly 90x fewer.
    coarse = np.full((nb,) + GRID_SHAPE, np.nan, dtype="f4")
    reproject(
        cube, coarse,
        src_transform=window_transform, src_crs=crs,
        dst_transform=GRID_TRANSFORM, dst_crs="EPSG:32611",
        src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average,
    )

    A = resample_library(lib, wl_tanager, wl).T  # (bands, k)
    keep = np.isfinite(A).all(axis=1)
    A, coarse = A[keep], coarse[keep]

    flat = coarse.reshape(A.shape[0], -1)
    populated = np.isfinite(flat).all(axis=0)
    frac = np.full((len(names), flat.shape[1]), np.nan, dtype="f4")
    if populated.any():
        frac[:, populated] = fcls(A, flat[:, populated])
    frac = frac.reshape((len(names),) + GRID_SHAPE)

    char = frac[names.index("CHAR")] / np.clip(1.0 - frac[names.index("SHADE")], 1e-3, None)
    return (char, len(wl),
            f"{h}x{w} px @ {dx:.1f} m -> {int(populated.sum()):,} cells @ 30 m, "
            f"{len(wl)}/{len(wl_all)} bands")


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild figures/stats from data/aviris3_char.npz")
    ap.add_argument("--max-lines", type=int, default=8)
    ap.add_argument("--band-stride", type=int, default=2,
                    help="read every Nth 10-band chunk (2 halves bytes pulled)")
    args = ap.parse_args(argv)

    if args.replot:
        cached = np.load("data/aviris3_char.npz", allow_pickle=True)
        return report(cached["char"], [str(x) for x in cached["lines"]])

    entries, selected = discover()
    same_day = [g for g in selected if g["date"] == TANAGER_DATE]
    print(f"{len(entries)} AVIRIS-3 L2A granules over the AOI")
    print(f"{len(selected)} intersect the Tanager overlap AND a burn scar")
    print(f"{len(same_day)} of those are same-day with the Tanager scene ({TANAGER_DATE})\n")
    for g in selected[: args.max_lines]:
        scars = ", ".join(f"{n} {v:.0f}%" for n, v in g["scars"].items())
        star = " <- same day" if g["date"] == TANAGER_DATE else ""
        print(f"  {g['id'][:40]:42s} {g['date']} {g['time']} {g['size_mb']:6.0f} MB  "
              f"[{scars}]{star}")
    if args.discover:
        return 0
    if not selected:
        print("nothing to process")
        return 1

    em = np.load("data/endmembers.npz", allow_pickle=True)
    lib, names, wl_tan = em["library"], [str(n) for n in em["names"]], em["wavelengths"]

    if args.inspect:
        print(f"\ninspecting {selected[0]['id']}")
        with open_remote(selected[0]["href"]) as f:
            describe_h5(f)
        return 0

    print()
    import time

    stack, used = [], []
    for i, g in enumerate(selected[: args.max_lines]):
        reset_bytes()
        t0 = time.time()
        try:
            arr, nb, note = process_line(g["href"], lib, wl_tan, names,
                                         verbose=(i == 0), band_stride=args.band_stride)
        except Exception as exc:
            print(f"  !! {g['id'][:40]:42s} {type(exc).__name__}: {exc}")
            continue
        pulled, elapsed = bytes_read_mb(), time.time() - t0
        efficiency = 100 * pulled / max(g["size_mb"], 1)
        if arr is None or not np.isfinite(arr).any():
            print(f"  -- {g['id'][:40]:42s} {note}")
            continue
        stack.append(arr)
        used.append(g)
        print(f"  ok {g['id'][:40]:42s} {note}, {np.isfinite(arr).sum():,} grid px | "
              f"pulled {pulled:.0f} MB of {g['size_mb']:.0f} ({efficiency:.0f}%) in {elapsed:.0f}s")

    if not stack:
        print("\nno flight lines yielded data")
        return 1

    aviris = np.nanmean(np.dstack(stack), axis=2)
    np.savez_compressed(
        "data/aviris3_char.npz", char=aviris,
        lines=np.array([g["id"] for g in used]), dates=np.array([g["date"] for g in used]),
    )

    return report(aviris, [g["id"] for g in used])


def report(aviris, lines):
    """Compare AVIRIS-3 char fractions with Tanager's, and plot cropped to the data."""
    from analyze import per_fire_masks

    fr = np.load("data/fractions.npz")
    nm = [str(n) for n in fr["names"]]
    frac = fr["frac_jan23"]
    tan = frac[nm.index("CHAR")] / np.clip(1.0 - frac[nm.index("SHADE")], 1e-3, None)

    ok = np.isfinite(aviris) & np.isfinite(tan) & np.isfinite(fr["rmse_jan23"])
    ok &= frac[nm.index("SHADE")] < 0.9
    a, t = aviris[ok], tan[ok]
    r = float(np.corrcoef(a, t)[0, 1])

    masks, any_fire = per_fire_masks(*GRID_SHAPE, fr["origin"])
    burned = ok & any_fire
    unburned = ok & ~any_fire

    print(f"\nAVIRIS-3 vs Tanager char fraction - {ok.sum():,} px "
          f"({ok.sum() * 900 / 1e6:.1f} km2), {len(lines)} same-day flight lines")
    print(f"  Pearson r         = {r:+.3f}")
    print(f"  mean bias (T - A) = {np.mean(t - a):+.3f}")
    print(f"  RMSE              = {np.sqrt(np.mean((t - a) ** 2)):.3f}")
    if burned.sum() > 100:
        print(f"  inside perimeters   r = {np.corrcoef(aviris[burned], tan[burned])[0, 1]:+.3f}"
              f"  ({burned.sum():,} px, AVIRIS median {np.median(aviris[burned]):.3f}, "
              f"Tanager {np.median(tan[burned]):.3f})")
    if unburned.sum() > 100:
        print(f"  outside perimeters  AVIRIS median {np.median(aviris[unburned]):.3f}, "
              f"Tanager {np.median(tan[unburned]):.3f}  ({unburned.sum():,} px)")

    rows, cols = np.where(np.isfinite(aviris))
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    crop = lambda arr: arr[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, arr, title in [
        (axes[0], crop(np.where(ok, aviris, np.nan)),
         "AVIRIS-3 char fraction\nairborne ~3 m, 23 Jan 20:2x UTC"),
        (axes[1], crop(np.where(ok, tan, np.nan)),
         "Tanager-1 char fraction\nspaceborne 30 m, 23 Jan 18:55 UTC"),
    ]:
        im = ax.imshow(arr, cmap="inferno", vmin=0, vmax=0.8)
        ax.set(title=title, xticks=[], yticks=[])
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for m in masks.values():
            ax.contour(crop(m).astype(float), levels=[0.5], colors="cyan", linewidths=0.8)

    axes[2].hexbin(a, t, gridsize=45, bins="log", cmap="viridis",
                   extent=(0, 1, 0, 1), mincnt=1)
    axes[2].plot([0, 1], [0, 1], "w--", lw=1.2)
    axes[2].set(xlabel="AVIRIS-3 char fraction (airborne)",
                ylabel="Tanager-1 char fraction (spaceborne)",
                title=f"same-day char vs char\nr = {r:+.3f}, n = {ok.sum():,}")
    axes[2].grid(False)
    plt.tight_layout()
    plt.savefig("data/quicklook/validation_aviris3.png", dpi=115)
    print("wrote data/quicklook/validation_aviris3.png")
    return 0



if __name__ == "__main__":
    sys.exit(main())
