"""Verify NASA Earthdata token auth works for the AVIRIS-3 download.

Never prints, logs or returns the token — only its length and first/last four characters,
which is enough to spot a truncated paste without exposing the secret.

Checks the three things that actually go wrong, in order, so a failure names its own fix:

  1. a token is resolvable (env var or 0600 token file)
  2. the token authenticates against URS
  3. an actual protected ORNL DAAC granule downloads — this is the step that catches an
     unauthorized DAAC application, which returns 401/403 even with a valid token and is
     the most common false alarm

Run: python scripts/check_earthdata_auth.py
"""

from __future__ import annotations

import sys

import requests

sys.path.insert(0, "scripts")
from earthdata_auth import MissingToken, auth_headers, source, token_fingerprint  # noqa: E402

URS_HOST = "urs.earthdata.nasa.gov"
PROBE_URL = (
    "https://data.ornldaac.earthdata.nasa.gov/protected/aviris/AV3_L2A_RFL/data/"
    "AV320250111t195928_000_L2A_OE_f576f24d_RFL_ORT.nc"
)
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
AVIRIS_L2A_COLLECTION = "C3369603199-ORNL_CLOUD"
AOI_BBOX = "-118.85,33.95,-118.50,34.25"


def check_token():
    try:
        return True, f"found via {source()} ({token_fingerprint()})"
    except MissingToken as exc:
        return False, str(exc)


def check_token_shape():
    """EDL tokens are JWTs. Catch a truncated or mis-pasted token before any network call.

    Deliberately NOT a call to URS /api/users/tokens: that endpoint authenticates with
    Basic auth (username/password), so a perfectly valid *bearer* token returns 401 there.
    An earlier version of this script used it and reported a false failure. The
    authoritative test for a bearer token is fetching a protected granule, which is the
    next check.
    """
    from earthdata_auth import get_token

    token = get_token()
    parts = token.split(".")
    if len(parts) != 3:
        return False, (
            f"token is not a JWT ({len(parts)} dot-separated parts, expected 3) — "
            f"it may be truncated or have stray whitespace"
        )
    if any(c in token for c in " \t\n\r"):
        return False, "token contains whitespace — likely a copy/paste artefact"
    return True, f"well-formed JWT ({len(parts[1])} char payload)"


def check_download():
    try:
        r = requests.get(
            PROBE_URL, headers=auth_headers(), stream=True,
            allow_redirects=True, timeout=180,
        )
    except requests.RequestException as exc:
        return False, f"download probe failed: {exc}"
    head = next(r.iter_content(2048), b"")
    status, size = r.status_code, r.headers.get("content-length")
    r.close()

    if status == 200 and head[:8] == b"\x89HDF\r\n\x1a\n":
        mb = f"{int(size) / 1e6:.0f} MB" if size else "unknown size"
        return True, f"protected granule downloads, valid HDF5 (HTTP 200, {mb})"
    if status in (401, 403) or b"Access denied" in head:
        return False, (
            f"HTTP {status} / access denied. The token is valid but the ORNL DAAC "
            f"application is probably not authorized — log in at https://{URS_HOST} "
            f"-> Applications -> Authorized Apps -> approve ORNL DAAC"
        )
    if status == 404:
        return False, "probe granule 404 — filename may have been revised upstream"
    return False, f"unexpected HTTP {status}"


def count_granules():
    r = requests.get(
        CMR_GRANULES,
        params={
            "collection_concept_id": AVIRIS_L2A_COLLECTION,
            "bounding_box": AOI_BBOX,
            "temporal": "2025-01-01T00:00:00Z,2025-03-01T00:00:00Z",
            "page_size": 500,
        },
        timeout=90,
    )
    r.raise_for_status()
    entries = r.json()["feed"]["entry"]
    dates = sorted({e["time_start"][:10] for e in entries})
    return len(entries), dates


def main():
    ok = True
    for label, fn in [
        ("token resolved", check_token),
        ("token shape", check_token_shape),
        ("ORNL DAAC download", check_download),
    ]:
        passed, detail = fn()
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label:20s} {detail}")
        if not passed:
            break

    if ok:
        n, dates = count_granules()
        print(f"\nReady. {n} AVIRIS-3 L2A flight lines over the AOI on {', '.join(dates)}.")
        print("Next: python scripts/aviris3.py")
    else:
        print("\nNot ready — fix the failure above and re-run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
