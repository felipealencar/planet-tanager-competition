"""Fetch Tanager-1 STAC item JSONs and summarize assets / acquisition properties.

Usage: python scripts/fetch_metadata.py
Writes data/stac/<item_id>.json and prints a summary table.
"""

import json
import pathlib

import requests

STAC_BASE = "https://www.planet.com/data/stac/tanager-core-imagery/fire"
ITEMS = {
    "jul26": "20250726_192422_87_4001",
    "jan23": "20250123_185518_92_4001",
}
OUT = pathlib.Path("data/stac")

PROPS = [
    "datetime",
    "cloud_percent",
    "light_haze_percent",
    "quality_category",
    "collection_mode",
    "gsd",
    "view:off_nadir",
    "view:sun_elevation",
    "view:sun_azimuth",
    "location_description",
]


def fetch(item_id):
    path = OUT / f"{item_id}.json"
    if not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{STAC_BASE}/{item_id}/{item_id}.json", timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return json.loads(path.read_text())


def main():
    for label, item_id in ITEMS.items():
        item = fetch(item_id)
        print("=" * 70)
        print(f"{label}  {item_id}")
        for k in PROPS:
            print(f"  {k:22s} {item['properties'].get(k)}")
        print("  assets:")
        for key, asset in sorted(item["assets"].items()):
            head = requests.head(asset["href"], allow_redirects=True, timeout=60)
            size = int(head.headers.get("content-length", 0)) / 1e6
            nbands = len(asset.get("eo:bands") or asset.get("bands") or [])
            print(f"    {key:22s} {size:9.1f} MB  bands={nbands}")


if __name__ == "__main__":
    main()
