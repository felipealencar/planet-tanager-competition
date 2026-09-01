"""Minimal reader for Tanager-1 ortho HDF-EOS5 surface reflectance cubes.

The cube is HDF-EOS5, not plain HDF5: the georeferencing lives in a text blob at
`HDFEOS INFORMATION/StructMetadata.0` rather than in normal attributes, and the spectral
axis (wavelengths, FWHM, per-band quality flags) hangs off the `surface_reflectance`
dataset. This module pulls all of that into an affine transform + numpy arrays so the
rest of the pipeline can ignore the container format.

Bands are read lazily by wavelength -- the cubes are ~1 GB each and chunked (14, 52, 53),
so pulling a handful of wavelengths is cheap but touching all 426 is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import h5py
import numpy as np
from affine import Affine

GRID = "HDFEOS/GRIDS/HYP"
FIELDS = f"{GRID}/Data Fields"
SR = f"{FIELDS}/surface_reflectance"
UNC = f"{FIELDS}/surface_reflectance_uncertainty"
FILL = -9999.0

ANCILLARY = [
    "aerosol_optical_depth",
    "beta_cirrus_mask",
    "beta_cloud_mask",
    "column_water_vapour",
    "nodata_pixels",
    "sensor_azimuth",
    "sensor_to_ground_path_length",
    "sensor_zenith",
    "sun_azimuth",
    "sun_zenith",
    "time",
]


def _parse_struct_metadata(blob: str) -> dict:
    """Pull grid corners and shape out of the HDF-EOS StructMetadata text blob."""

    def grab(key):
        m = re.search(rf"{key}=\(([-\d.]+),([-\d.]+)\)", blob)
        return (float(m.group(1)), float(m.group(2)))

    def grab_int(key):
        return int(re.search(rf"{key}=(\d+)", blob).group(1))

    return {
        "ul": grab("UpperLeftPointMtrs"),
        "lr": grab("LowerRightMtrs"),
        "xdim": grab_int("XDim"),
        "ydim": grab_int("YDim"),
        "zone": grab_int("ZoneCode"),
    }


@dataclass
class Scene:
    """A Tanager ortho SR cube. Open with `Scene.open(path)`; use as a context manager."""

    path: str
    handle: h5py.File
    wavelengths: np.ndarray  # nm, shape (426,)
    fwhm: np.ndarray  # nm, shape (426,)
    good: np.ndarray  # bool, shape (426,) -- provider per-band quality flag
    transform: Affine
    epsg: int
    shape: tuple  # (rows, cols)

    @classmethod
    def open(cls, path: str) -> "Scene":
        f = h5py.File(path, "r")
        sr = f[SR]
        meta = _parse_struct_metadata(f["HDFEOS INFORMATION/StructMetadata.0"][()].decode())
        (ulx, uly), (lrx, lry) = meta["ul"], meta["lr"]
        rows, cols = meta["ydim"], meta["xdim"]
        transform = Affine.translation(ulx, uly) * Affine.scale(
            (lrx - ulx) / cols, (lry - uly) / rows
        )
        return cls(
            path=path,
            handle=f,
            wavelengths=np.asarray(sr.attrs["wavelengths"], dtype="f8"),
            fwhm=np.asarray(sr.attrs["fwhm"], dtype="f8"),
            good=np.asarray(sr.attrs["good_wavelengths"]).astype(bool),
            transform=transform,
            epsg=int(f[GRID].attrs["epsg_code"]),
            shape=(rows, cols),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()

    # -- spectral access -------------------------------------------------

    def band_index(self, wavelength_nm: float) -> int:
        """Index of the band whose center is nearest `wavelength_nm`."""
        return int(np.abs(self.wavelengths - wavelength_nm).argmin())

    def band(self, wavelength_nm: float, uncertainty: bool = False) -> np.ndarray:
        """One band as a masked float32 array, fill values set to NaN."""
        i = self.band_index(wavelength_nm)
        arr = self.handle[UNC if uncertainty else SR][i].astype("f4")
        arr[arr == FILL] = np.nan
        return arr

    def spectra(self, rows, cols, uncertainty: bool = False) -> np.ndarray:
        """Full 426-band spectra at the given pixel coordinates -> (n_pixels, 426).

        Reads the whole spectral axis for scattered pixels, so keep `rows`/`cols` small
        (endmember picking, invariant-target checks) rather than passing a whole scene.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)
        dset = self.handle[UNC if uncertainty else SR]
        out = np.empty((len(rows), len(self.wavelengths)), dtype="f4")
        for n, (r, c) in enumerate(zip(rows, cols)):
            out[n] = dset[:, r, c]
        out[out == FILL] = np.nan
        return out

    # -- ancillary -------------------------------------------------------

    def ancillary(self, name: str) -> np.ndarray:
        if name not in ANCILLARY:
            raise KeyError(f"{name!r} not in {ANCILLARY}")
        arr = self.handle[f"{FIELDS}/{name}"][()]
        if arr.dtype.kind == "f":
            arr = arr.astype("f4")
            arr[arr == FILL] = np.nan
        return arr

    def valid_mask(self) -> np.ndarray:
        """True where the pixel is usable: has data, not cloud, not cirrus."""
        nodata = self.ancillary("nodata_pixels")
        cloud = self.ancillary("beta_cloud_mask")
        cirrus = self.ancillary("beta_cirrus_mask")
        return (nodata == 0) & (cloud == 0) & (cirrus == 0)

    def rowcol(self, x: float, y: float) -> tuple:
        """Map projected (x, y) in the scene CRS to integer (row, col)."""
        col, row = ~self.transform * (x, y)
        return int(row), int(col)

    def __repr__(self):
        return (
            f"<Scene {self.path.split('/')[-1]} {self.shape} EPSG:{self.epsg} "
            f"{self.good.sum()}/{len(self.good)} good bands>"
        )
