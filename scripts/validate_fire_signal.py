"""Step 1 of the project plan: does a fire signal actually exist in the Jan/Jul overlap?

Rather than inferring burn scars from imagery alone (which is seasonally confounded --
January is wet-season green, July is dry-season brown), this queries the authoritative
NIFC/WFIGS interagency fire perimeter archive for every fire that burned in the scene
area, and intersects those perimeters with the two Tanager footprints.

Outputs:
  data/fire_perimeters_2024_2025.geojson
  data/quicklook/perimeters_overlay.png
"""

import datetime
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, shape
from shapely.ops import transform

WFIGS = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)
UTM = Transformer.from_crs(4326, 32611, always_xy=True).transform

# Footprints from the STAC item geometries (WGS84).
JUL26 = Polygon(
    [
        (-118.7910628466147, 34.21932506980487),
        (-118.84336011674677, 34.0512100596783),
        (-118.61020764125392, 33.99952075931865),
        (-118.57294146689644, 34.17803482712942),
    ]
)
JAN23 = Polygon(
    [
        (-118.74674636594759, 34.19042451838735),
        (-118.81000229560739, 33.95259148244884),
        (-118.56582754732204, 33.90531288319584),
        (-118.5024818852516, 34.14556791007762),
    ]
)

QUICKLOOK = {
    "jan23": ("20250123_185518_92_4001", "JAN 23 2025 - 16 days post-Palisades"),
    "jul26": ("20250726_192422_87_4001", "JUL 26 2025 - ~6 months later"),
}
COLORS = {
    "PALISADES": "#ff3b30",
    "Franklin": "#ffd60a",
    "KENNETH": "#00e5ff",
    "BROAD": "#c77dff",
}
PERIM_PATH = pathlib.Path("data/fire_perimeters_2024_2025.geojson")


def km2(geom_wgs84):
    return transform(UTM, geom_wgs84).area / 1e6


def fetch_perimeters():
    """All fires discovered between Nov 2024 and Aug 2025 intersecting the scene area."""
    if PERIM_PATH.exists():
        return json.loads(PERIM_PATH.read_text())
    params = {
        "where": (
            "attr_FireDiscoveryDateTime > TIMESTAMP '2024-11-01 00:00:00' AND "
            "attr_FireDiscoveryDateTime < TIMESTAMP '2025-08-05 00:00:00'"
        ),
        "geometry": "-118.90,33.88,-118.45,34.25",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "poly_IncidentName,attr_FireDiscoveryDateTime,attr_IncidentSize",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
    }
    r = requests.get(WFIGS, params=params, timeout=120)
    r.raise_for_status()
    PERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERIM_PATH.write_bytes(r.content)
    return r.json()


def report(perims):
    overlap = JUL26.intersection(JAN23)
    print(
        f"jul26 {km2(JUL26):7.1f} km2 | jan23 {km2(JAN23):7.1f} km2 "
        f"| overlap {km2(overlap):7.1f} km2\n"
    )
    print(
        f"{'fire':14s} {'discovered':12s} {'area km2':>9s} "
        f"{'in jul26':>9s} {'in jan23':>9s} {'in BOTH':>9s}"
    )
    total = 0.0
    rows = []
    for f in perims["features"]:
        geom = shape(f["geometry"]).buffer(0)
        area = km2(geom)
        if area < 0.05:
            continue
        ts = f["properties"]["attr_FireDiscoveryDateTime"]
        disc = (
            datetime.datetime.fromtimestamp(ts / 1000, datetime.UTC).date() if ts else None
        )
        rows.append(
            (
                f["properties"]["poly_IncidentName"],
                disc,
                area,
                km2(geom.intersection(JUL26)),
                km2(geom.intersection(JAN23)),
                km2(geom.intersection(overlap)),
            )
        )
    for name, disc, area, in_jul, in_jan, both in sorted(rows, key=lambda r: -r[5]):
        print(
            f"{name[:14]:14s} {str(disc):12s} {area:9.1f} "
            f"{in_jul:9.1f} {in_jan:9.1f} {both:9.1f}"
        )
        total += both
    print(
        f"\nBurned area inside the same-sensor overlap: {total:.1f} km2 "
        f"({100 * total / km2(overlap):.1f}% of overlap)"
    )
    return overlap


def plot(perims, overlap):
    """Side-by-side quicklooks with perimeters, to confirm scars line up with the archive."""
    bbox = [332760, 3763260, 355050, 3784650]  # UTM 11N intersection of the two rasters
    overlap_utm = transform(UTM, overlap)
    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    for ax, (item_id, title) in zip(axes, QUICKLOOK.values()):
        with rasterio.open(f"data/quicklook/{item_id}_visual.tif") as src:
            window = from_bounds(*bbox, src.transform)
            rgb = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        ax.imshow(
            np.transpose(rgb, (1, 2, 0)),
            extent=[bbox[0], bbox[2], bbox[1], bbox[3]],
        )
        ax.plot(*overlap_utm.exterior.xy, color="white", lw=1.6, ls="--", label="Tanager overlap")
        seen = set()
        for f in perims["features"]:
            name = f["properties"]["poly_IncidentName"]
            if name not in COLORS:
                continue
            geom = transform(UTM, shape(f["geometry"]).buffer(0))
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for part in parts:
                ax.plot(
                    *part.exterior.xy,
                    color=COLORS[name],
                    lw=1.9,
                    label=name if name not in seen else None,
                )
                seen.add(name)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(loc="lower right", fontsize=9, framealpha=0.85)
    plt.tight_layout()
    plt.savefig("data/quicklook/perimeters_overlay.png", dpi=115)
    print("\nwrote data/quicklook/perimeters_overlay.png")


if __name__ == "__main__":
    perims = fetch_perimeters()
    plot(perims, report(perims))
