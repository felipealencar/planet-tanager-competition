"""Survey of every candidate field-calibrated burn severity product for these fires.

The project's remaining validation gap was this: our severity check compares the Tanager
char fraction against our OWN Sentinel-2 dNBR, which shares an author with the thing it
validates. Closing that properly needs a product built by someone else.

This script checks each candidate and reports why it does or does not work. It is kept in
the repo because the negative result is itself worth recording -- the gap is not an
oversight, and someone extending this project should not have to rediscover the same five
dead ends.

Verdict (see FINDINGS.md for the full write-up):

  MTBS                      NO   archive stops at 2024; MTBS lags ~18 months
  USFS BAER SBS 2025        NO   mosaic exists but is entirely nodata over this AOI --
                                 Palisades/Franklin/Kenneth were state-responsibility
                                 fires under California's WERT program, not federal BAER
  NASA JPL S1+S2 severity   NO   footprint is centred on the Eaton fire; only ~1 km2 of
                                 non-zero severity falls in our overlap vs a ~30 km2 scar
  NASA JPL Sentinel-2 dNBR  NO   covers the AOI, but its pre-fire date (2 Jan 2025) is
                                 AFTER the Franklin fire, so Franklin reads as negative
                                 dNBR; disagrees with an independent dNBR at r = -0.09
  AVIRIS-3 (JPL)            YES  but gated behind NASA Earthdata Login

AVIRIS-3 is the right answer and would be a genuinely superior validation to any soil burn
severity product: it overflew these fires on 2025-01-11 with an airborne imaging
spectrometer and NASA published a relative char-and-ash product from it. That measures the
same physical material our char endmember measures, rather than a severity proxy. It needs
credentials this pipeline does not have; instructions are at the bottom of this file.

Run: python scripts/external_validation_survey.py
"""

from __future__ import annotations

import sys

import numpy as np
import rasterio
import requests

sys.path.insert(0, "scripts")

AOI_UTM = (332760, 3763260, 355050, 3784650)  # Tanager overlap, EPSG:32611
AOI_WGS = (-118.79, 33.99, -118.577, 34.19)
SHAPE = (713, 743)

MTBS_SERVICE = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer"
BAER_SERVICE = (
    "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/"
    "USFS_EDW_BAER_SoilBurnSeverityClassification/ImageServer"
)
NASA_BASE = "https://gis.earthdata.nasa.gov/gis05/rest/services/DISASTERS_202501_FIRE_CA"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
AVIRIS_L2A_COLLECTION = "C3369603199-ORNL_CLOUD"


def export_image(service, pixel_type="F32", interp="RSP_BilinearInterpolation", **extra):
    """Export an ArcGIS ImageServer raster onto the Tanager 30 m grid."""
    params = {
        "bbox": ",".join(str(v) for v in AOI_UTM),
        "bboxSR": 32611, "imageSR": 32611,
        "size": f"{SHAPE[1]},{SHAPE[0]}",
        "format": "tiff", "pixelType": pixel_type,
        "interpolation": interp, "f": "image", **extra,
    }
    r = requests.get(f"{service}/exportImage", params=params, timeout=180)
    r.raise_for_status()
    path = "/tmp/_survey.tif"
    with open(path, "wb") as f:
        f.write(r.content)
    with rasterio.open(path) as src:
        return src.read(1)


def check_mtbs():
    r = requests.get(f"{MTBS_SERVICE}?f=json", timeout=60)
    years = sorted(
        {int(x["name"][:4]) for x in r.json()["layers"] if x["name"][:4].isdigit()}
    )
    latest = max(years)
    return latest >= 2025, f"burned-area layers cover {min(years)}-{latest}; needs 2025"


def check_baer():
    r = requests.get(
        f"{BAER_SERVICE}/query",
        params={
            "where": "1=1", "geometry": ",".join(str(v) for v in AOI_WGS),
            "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "outFields": "name,beginyear",
            "returnGeometry": "false", "f": "json",
        },
        timeout=90,
    )
    rasters = {f["attributes"]["name"]: f["attributes"]["beginyear"]
               for f in r.json().get("features", [])}
    target = [n for n, y in rasters.items() if y == 2025]
    if not target:
        return False, "no 2025 raster in the mosaic"
    oid = requests.get(
        f"{BAER_SERVICE}/query",
        params={"where": f"name='{target[0]}'", "outFields": "objectid",
                "returnGeometry": "false", "f": "json"}, timeout=60,
    ).json()["features"][0]["attributes"]["objectid"]
    arr = export_image(
        BAER_SERVICE, pixel_type="U8", interp="RSP_NearestNeighbor", noData=255,
        mosaicRule=f'{{"mosaicMethod":"esriMosaicLockRaster","lockRasterIds":[{oid}]}}',
    )
    valid = int((arr != 255).sum())
    return valid > 1000, f"{target[0]} present but {valid} valid px over the AOI"


def check_nasa_s1s2():
    arr = export_image(
        f"{NASA_BASE}/2501_s1_s2_burnseverity/ImageServer",
        pixel_type="U8", interp="RSP_NearestNeighbor", noData=255,
    ).astype("f4")
    covered = arr != 255
    burned = covered & (arr > 0)
    km2 = burned.sum() * 900 / 1e6
    return km2 > 15, f"{km2:.1f} km2 of non-zero severity in the overlap (scar is ~30 km2)"


def check_nasa_dnbr():
    """Covers the AOI, so compare it against our own dNBR and the known fire history."""
    from analyze import per_fire_masks

    arr = export_image(f"{NASA_BASE}/2501_sentinel2_dnbr/ImageServer")
    mine = np.load("data/sentinel2.npz")["dnbr_post"]
    masks, _ = per_fire_masks(*SHAPE, np.load("data/fractions.npz")["origin"])
    ok = np.isfinite(arr) & np.isfinite(mine)
    r = float(np.corrcoef(arr[ok], mine[ok])[0, 1])
    frank = float(np.median(arr[masks["Franklin"]]))
    pal = float(np.median(arr[masks["PALISADES"]]))
    good = r > 0.5 and frank > 0.1
    return good, (f"Palisades median {pal:+.3f}, Franklin median {frank:+.3f} "
                  f"(negative: their pre-fire date postdates Franklin), r={r:+.3f} vs ours")


def check_aviris():
    r = requests.get(
        CMR_GRANULES,
        params={
            "collection_concept_id": AVIRIS_L2A_COLLECTION,
            "bounding_box": "-118.85,33.95,-118.50,34.25",
            "temporal": "2025-01-01T00:00:00Z,2025-03-01T00:00:00Z",
            "page_size": 100,
        },
        timeout=90,
    )
    entries = r.json()["feed"]["entry"]
    dates = sorted({e["time_start"][:10] for e in entries})
    href = next(
        (l["href"] for l in entries[0]["links"] if l.get("rel", "").endswith("/data#")), ""
    )
    auth = requests.get(href, timeout=60, allow_redirects=True)
    open_access = auth.status_code == 200 and b"Access denied" not in auth.content[:200]
    return open_access, (f"{len(entries)}+ flight lines on {', '.join(dates)}; "
                         f"download {'open' if open_access else 'requires Earthdata Login'}")


CHECKS = [
    ("MTBS burn severity", check_mtbs),
    ("USFS BAER SBS 2025", check_baer),
    ("NASA JPL S1+S2 severity", check_nasa_s1s2),
    ("NASA JPL Sentinel-2 dNBR", check_nasa_dnbr),
    ("AVIRIS-3 L2A (JPL)", check_aviris),
]


def main():
    print("Survey of external burn severity references for the Jan 2025 Santa Monica fires\n")
    print(f"{'source':26s} {'usable':>7s}  detail")
    print("-" * 100)
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:  # a dead service is itself a finding
            ok, detail = False, f"query failed: {type(exc).__name__}: {exc}"
        print(f"{name:26s} {'YES' if ok else 'NO':>7s}  {detail}")
    print("\nConclusion: no open, unauthenticated, field-calibrated severity product covers")
    print("these fires. AVIRIS-3 would be the strongest possible reference and does cover")
    print("them, but needs NASA Earthdata credentials -- see the notes in this file.")


# ---------------------------------------------------------------------------
# Getting AVIRIS-3, if you have NASA Earthdata credentials
#
#   AVIRIS-3 overflew these fires on 2025-01-11 -- four days after Palisades ignited and
#   twelve days before the Tanager January scene -- at metre-scale GSD. NASA published a
#   relative char-and-ash product from those flights. That is a stronger validation than
#   any soil burn severity map, because it measures the same material the char endmember
#   measures with an independent imaging spectrometer, rather than a severity proxy.
#
#   1. Register at https://urs.earthdata.nasa.gov and accept the ORNL DAAC EULA.
#   2. Add to ~/.netrc:
#          machine urs.earthdata.nasa.gov login <user> password <pass>
#   3. List granules (this part needs no auth):
#          python scripts/external_validation_survey.py     # prints the count
#   4. Download each *_RFL_ORT.tif with:  curl -n -L -c /tmp/c -b /tmp/c <href>
#   5. Unmix them with the SAME library:  scripts/unmix.py already takes an arbitrary cube.
#
#   Comparing char fraction to char fraction across two independent imaging spectrometers
#   would close this gap far more convincingly than a severity-class correlation.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
