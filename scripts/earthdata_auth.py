"""Resolve NASA Earthdata bearer-token credentials for AVIRIS-3 downloads.

Token auth is preferred over ~/.netrc here: the token is scoped and expires (~60 days),
whereas .netrc means writing a long-lived account password to disk in cleartext.

Resolution order, first hit wins:

  1. $EARTHDATA_TOKEN                      -- ephemeral, best for CI
  2. ~/.earthdata_token                    -- persisted across shells, must be 0600
  3. ./.earthdata_token                    -- repo-local, gitignored, must be 0600

Nothing in this module prints, logs or returns the token itself; callers get headers.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

TOKEN_FILES = [Path.home() / ".earthdata_token", Path(".earthdata_token")]
URS = "https://urs.earthdata.nasa.gov"


class MissingToken(RuntimeError):
    pass


def _read_token_file(path: Path) -> str | None:
    if not path.exists():
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise MissingToken(
            f"{path} has permissions {oct(mode)} — a token file must not be "
            f"group/world readable. Fix with: chmod 600 {path}"
        )
    token = path.read_text().strip()
    return token or None


def get_token() -> str:
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if token:
        return token
    for path in TOKEN_FILES:
        token = _read_token_file(path)
        if token:
            return token
    raise MissingToken(
        "No Earthdata token found. Generate one at "
        f"{URS} -> Generate Token, then persist it with:\n"
        "    printf '%s' \"$EARTHDATA_TOKEN\" > ~/.earthdata_token && "
        "chmod 600 ~/.earthdata_token"
    )


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {get_token()}"}


def token_fingerprint() -> str:
    """Describe the token without revealing it, for diagnostics."""
    token = get_token()
    return f"{len(token)} chars, starts {token[:4]}..., ends ...{token[-4:]}"


def source() -> str:
    if os.environ.get("EARTHDATA_TOKEN", "").strip():
        return "$EARTHDATA_TOKEN"
    for path in TOKEN_FILES:
        try:
            if _read_token_file(path):
                return str(path)
        except MissingToken:
            continue
    return "none"
