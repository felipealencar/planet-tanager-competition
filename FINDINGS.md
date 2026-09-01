# Step 1 findings — fire signal validation

Resolves open question #1 in the project brief. Reproduce with
`python scripts/validate_fire_signal.py`.

## Headline: the pre/post framing is inverted

**No fire burned in this area between the January and July scenes.** Every fire in the
overlap zone predates the January acquisition. Queried against the authoritative
NIFC/WFIGS interagency perimeter archive (all fires discovered Nov 2024 – Aug 2025
intersecting the scene area):

| Fire | Discovered | Perimeter | In jul26 | In jan23 | In **both** |
|---|---|---:|---:|---:|---:|
| Palisades | 2025-01-07 | 96.1 km² | 29.8 | 84.9 | **29.8** |
| Franklin | 2024-12-10 | 17.0 km² | 17.0 | 17.0 | **17.0** |
| Kenneth | 2025-01-09 | 4.0 km² | 4.0 | 2.8 | **2.8** |
| Broad | 2024-11-06 | 0.2 km² | 0.2 | 0.2 | **0.2** |

Footprints: jul26 421.4 km², jan23 627.6 km², overlap 289.5 km².
**49.7 km² of burn scar sits inside the same-sensor overlap (17.2% of it).**

Perimeters were overlaid on both quicklooks (`data/quicklook/perimeters_overlay.png`) and
line up visually with the scar boundaries in the imagery, in both dates — an independent
confirmation that the archive and the imagery agree.

So the July 26 scene shows **no new fire**. That negative result matters: it means the
naive "pre-fire vs post-fire dNBR" plan is not available, but something arguably better is:

## What the pair actually is

A **same-sensor, 426-band post-fire recovery pair** over three scars of different age and
severity:

- **Jan 23, 2025** = 16 days after Palisades ignited, 6 weeks after Franklin. Fresh char,
  before the first significant rains redistribute it. (White ash was expected here too,
  but turned out not to be spectrally separable — see Step 3.)
- **Jul 26, 2025** = ~6 months of recovery later, end of the first dry season — resprouting
  chaparral, ash washed/blown out, exposed mineral soil.

Three scars with different ignition dates inside one overlap gives a built-in age gradient
(Franklin ~6 wks older than Palisades at the January acquisition) rather than a single
before/after contrast.

## Acquisition caveats that constrain quantitative comparison

Pulled from the STAC properties; these are real and need to be handled, not ignored:

| | jan23 | jul26 |
|---|---|---|
| `quality_category` | **test** | standard |
| `collection_mode` | standard_sensitivity | **maximum_sensitivity** |
| `gsd` | 39 m | 35.17 m |
| `view:sun_elevation` | **34.1°** | **73.0°** |
| `view:off_nadir` | 20° | 17° |
| `cloud_percent` | 0 (haze 2) | 1 (haze 0) |

1. **Sun elevation 34° vs 73°** is the big one. In terrain as steep as the Santa Monica
   Mountains, a 39° difference in solar elevation produces severe differential
   topographic shading and BRDF effects between the two dates. A raw band-difference or
   dNBR between these scenes will partly measure illumination, not surface change.
   Mitigation: topographic correction against a DEM, and/or restrict quantitative claims
   to ratio/normalized indices and to slopes of similar aspect, and/or work in a
   shade-normalized unmixing framework where a photometric shade endmember absorbs it.
2. The January scene is flagged `quality_category: test` and uses a different collection
   mode — radiometric comparability between the two is not guaranteed and should be
   checked empirically (e.g. over invariant targets: ocean, bare rock, paved surfaces).
3. **Both `ortho_beta_udm` COG assets are 0 bytes.** However, the masks are not actually
   missing — see below, they live inside the HDF5 cube. Ignore the standalone UDM asset.

## The HDF5 cube carries more than the COG assets suggest

The `ortho_sr_hdf5` asset lists 863 bands = 426 surface reflectance + 426 per-band
reflectance **uncertainty** + 11 ancillary layers. The ancillary set is the useful surprise:

```
aerosol_optical_depth   beta_cirrus_mask   beta_cloud_mask   column_water_vapour
nodata_pixels           sensor_azimuth     sensor_zenith     sun_azimuth
sun_zenith              sensor_to_ground_path_length         time
```

Two consequences for the caveats above:

- **Per-pixel sun and sensor geometry ship with the data.** The 34°-vs-73° illumination
  problem can be attacked with actual per-pixel `sun_zenith`/`sun_azimuth` rather than a
  single scene-level solar angle, which makes topographic/BRDF handling considerably more
  tractable than the STAC scene properties implied.
- **Cloud, cirrus and nodata masks exist per pixel** inside the cube, so caveat 3 is
  resolved — masking does not have to be derived from scratch.
- Per-band **uncertainty** enables genuinely weighted unmixing and lets any recovery
  signal be reported with error bars rather than as a bare fraction — directly useful for
  the "Scientific Integrity" criterion.

Spectral coverage confirmed identical across both dates: 426 bands, 376.4–2499.0 nm,
~4.99 nm mean sampling. Grids: jan23 1063×957, jul26 822×839, both EPSG:32611.

368 of 426 bands carry `good_wavelengths = 1`; the 58 flagged bands are exactly the
1342–1438 nm and 1783–1967 nm atmospheric water-vapour absorption regions, as expected.

## Co-registration: free

Both ortho products are on the **same 30 m EPSG:32611 grid**, and the upper-left corners
differ by exactly 96 and 109 pixels (2880 m, 3270 m). The overlap is therefore an integer
pixel-offset crop in each scene — no resampling, no reprojection, no warping artifacts
introduced before analysis. Overlap window: 713 × 743 px, jan23 at (row 0, col 0),
jul26 at (row 109, col 96). 57.5% of that window is valid in both dates after applying
the cloud/cirrus/nodata masks.

## Step 2 findings — the two dates are radiometrically comparable

Run `python scripts/invariant_check.py` then `python scripts/crossdate_gain.py`.

**Pseudo-invariant targets** (surfaces that genuinely should not change Jan→Jul):

| target | n | jan mean R | jul mean R | median abs diff | VNIR bias | SWIR bias |
|---|---:|---:|---:|---:|---:|---:|
| deep water | 150 | 0.0094 | 0.0101 | 0.0011 | +0.0040 | −0.0005 |
| bright PIF | 150 | 0.2319 | 0.2515 | 0.0231 | +0.0252 | +0.0198 |

Near-zero difference at the dark end, ~+0.025 at reflectance 0.23: that is a
**multiplicative** difference, not an additive one. Confirmed by fitting
`R_jul = gain · R_jan + offset` per band over 4,000 pseudo-invariant pixels (valid in both
dates, NDVI < 0.15 in both dates, and **excluded from all four fire perimeters** so the
scars being measured cannot contaminate the calibration):

| region | gain | offset | R² |
|---|---:|---:|---:|
| VIS 450–700 nm | 1.048 | +0.0036 | 0.879 |
| NIR 700–1300 nm | 1.071 | +0.0013 | 0.873 |
| SWIR1 1500–1780 nm | 1.043 | +0.0001 | 0.863 |
| SWIR2 2000–2400 nm | 1.055 | −0.0007 | 0.850 |
| **all good bands** | **1.049** | **+0.0002** | **0.861** |

**This is the key enabling result.** The July scene is brighter than January by a
**spectrally flat ~5% gain** (p5–p95 across usable bands: 0.951–1.090) with essentially
zero additive offset and R² ~0.86. Three consequences:

1. The `quality_category: test` flag on the January scene does **not** manifest as
   spectral distortion. The gain curve is smooth with no band-to-band jumps, so the two
   dates are spectrally consistent and safe to compare.
2. A flat multiplicative gain is a **brightness** term, which is exactly what a photometric
   shade endmember absorbs by construction in a linear unmixing. The chosen
   unmixing-based framing therefore neutralizes the 34°-vs-73° sun-elevation problem
   rather than merely tolerating it — this is the technical justification for preferring
   fraction maps over dNBR here, and it is now measured rather than assumed.
3. The residual +0.0036 additive offset confined to the VIS is consistent with a small
   difference in atmospheric path radiance between the two dates. It is small enough to
   ignore for unmixing, but it argues against relying on any blue-band-driven index.

**Usable band set after cross-date QA** (provider `good_wavelengths`, plus R² > 0.7 and
0.85 < gain < 1.20): **355 of 426 bands**, saved to `data/usable_bands.npy`.

| range | bands |
|---|---:|
| 401.3 – 1337.4 nm | 188 |
| 1452.6 – 1777.6 nm | 66 |
| 1972.2 – 2444.5 nm | 96 |
| 2454.4 – 2474.2 nm | 5 |

Only the extreme detector edges (< 400 nm, > 2475 nm) drop out beyond the water-vapour
regions — so the SWIR char/ash diagnostic region around 2000–2400 nm survives QA fully
intact on both dates, which is the part the science depends on.

## Assets

Direct, unauthenticated GCS URLs (no Planet account needed):

- `ortho_sr_hdf5` — jan23 1340 MB, jul26 1093 MB (surface reflectance, the analysis product)
- `ortho_radiance_hdf5`, `basic_*` variants also available
- `ortho_visual` — ~1.5 MB RGB COG, EPSG:32611, 30 m grid (used for the quicklooks here)
- `geolocation_array`, `thumbnail`

Both scenes are already on a common CRS/grid (EPSG:32611), so co-registration for the
ortho products should be straightforward.

---

# Step 3 findings — endmember library

Run `python scripts/endmembers.py`.

## Two failed designs, and what they taught

Recorded because they shaped the final method and belong in the write-up:

1. **Blind VCA over the whole overlap failed.** The extreme vertices of the full scene
   simplex are deep ocean and deep topographic shadow, not fire materials. Two of the
   eight returned vertices were near-duplicate dry-vegetation spectra, and no soil vertex
   was found at all.
2. **Region-constrained VCA with hand-specified absorption features also failed**, for a
   subtler reason: the assumed continuum shoulders were wrong. The true SWIR2 continuum in
   these spectra peaks near 2200 nm, so every "absorption depth" measured against fixed
   1990/2450 nm shoulders came out *negative*, and the NPV and SOIL labels could never
   match anything.

The fix was to stop assuming a library and measure which materials are actually separable.

## Separability decides the library size

Spectral angle is brightness-invariant, so it separates material differences from
illumination differences. Measured between class medians (n = 10k–40k px each):

| | GV | NPV/soil | CHAR | bright-burn | ash-cand |
|---|---:|---:|---:|---:|---:|
| **GV** | 0.0 | 26.0 | 35.0 | 34.7 | 37.0 |
| **NPV/soil** | 26.0 | 0.0 | 10.8 | 10.4 | 12.9 |
| **CHAR** | 35.0 | 10.8 | 0.0 | 5.4 | 5.3 |
| **bright-burn** | 34.7 | 10.4 | 5.4 | 0.0 | **2.8** |

Two results follow:

- **GV January vs GV July = 4.2°.** The same material recovers the same spectral shape on
  both dates — an independent confirmation of cross-date consistency, obtained without
  reference to the PIF regression.
- **No spectrally distinct white-ash endmember exists in this pair.** The ash candidate
  sits 2.8° from bright burned substrate, and bright burned substrate sits 5.4° from char.
  Dark and bright burned areas share a spectral *shape* and differ in *brightness* — which
  the photometric shade endmember already accounts for. This is a genuine negative result:
  16 days after ignition, at 30 m, after the Santa Ana wind event, no separable white-ash
  signal survives. Forcing ASH into the library would only make the system ill-conditioned.

**Final library: GV, NPV_SOIL, CHAR, SHADE.** Condition number of the material design
matrix: 25.8 (well-conditioned). Endmembers are class *medians* rather than single VCA
vertices — far less noisy, at the cost of slightly less spectral purity.

# Step 4 findings — unmixing and recovery

Run `python scripts/unmix.py` then `python scripts/analyze.py`. FCLS (Heinz & Chang 2001),
355 bands, ~15 s per date for the full 713 × 743 overlap.

| date | valid px | median RMSE | GV | NPV/soil | CHAR | SHADE |
|---|---:|---:|---:|---:|---:|---:|
| jan23 | 393,872 | 0.0167 | 0.261 | 0.381 | 0.085 | 0.274 |
| jul26 | 417,253 | 0.0383 | 0.210 | 0.715 | 0.005 | 0.072 |

The shade fraction tracks the sun elevation exactly as designed (0.274 at 34° vs 0.072 at
73°), which is the mechanism that keeps it out of the material fractions. July RMSE is
higher because the library is January-derived and July vegetation is drought-stressed.

## Shade-normalized results by fire

Fractions divided by (1 − shade); control is unburned land inside the same overlap
(199.3 km²), which absorbs the seasonal wet→dry signal so recovery is measured against
what unburned chaparral did over the same six months.

| region | km² | char Jan | char Jul | GV Jan | GV Jul | ΔGV | ΔGV vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| control (unburned) | 199.3 | 0.000 | 0.000 | 0.499 | 0.222 | −0.277 | — |
| Palisades | 28.8 | 0.274 | 0.000 | 0.000 | 0.088 | +0.088 | **+0.365** |
| Franklin | 16.2 | 0.180 | 0.000 | 0.000 | 0.209 | +0.209 | **+0.486** |
| Kenneth | 2.8 | 0.473 | 0.000 | 0.000 | 0.000 | +0.000 | **+0.277** |

Findings:

- **Char fraction is 0.000 in unburned land on both dates.** The per-pixel unmixing has no
  spatial information, so this null is not geometrically enforced (see Step 10 for the
  circularity check this claim originally failed). The char endmember simply does not
  activate outside the perimeters.
- **Char is completely gone by July** (0.000 in every scar) — six months and one wet
  season fully remove the charred-material signal.
- **January char ranks inversely with scar age**: Kenneth 0.473 (14 days old) > Palisades
  0.274 (16 days) > Franklin 0.180 (44 days). Franklin had six extra weeks of weathering
  before the January acquisition.
- **Recovery ranks with time since fire**: Franklin (oldest, coastal) recovers most at
  +0.486 above control, Palisades +0.365, Kenneth +0.277 with zero absolute GV gain.
- Unburned control *loses* GV (−0.277) over the same window — the seasonal signal that
  would have been mistaken for "no recovery" without a control.

## Blind burn detection skill

Char fraction scored against the NIFC perimeters, treating them as ground truth:

```
AUC = 0.768        (n_burn = 52,953 px, n_unburn = 221,689 px)
threshold 0.20 ->  precision 0.638  recall 0.548  F1 0.589
threshold 0.30 ->  precision 0.791  recall 0.459  F1 0.581
```

Interpret with care: **a fire perimeter is not a burn mask.** Perimeters enclose unburned
islands and lightly burned patches, typically 10–30% of the enclosed area, so recall is
structurally capped well below 1 and AUC understates the true skill. Precision 0.79 at
threshold 0.30 is the more meaningful number: where the model claims heavy char, it is
almost always inside a real perimeter.

---

# Step 5 findings — Sentinel-2 pre-burn baseline and independent validation

Run `python scripts/sentinel2.py` then `python scripts/validate.py`.

Tanager has no pre-fire acquisition over this area — its earliest scene is already 16 days
post-Palisades. Sentinel-2 L2A (via AWS Earth Search, public, no authentication) supplies
the missing "before" and doubles as the project's only fully external validation.

## Scene selection was the hard part

The first attempt picked the scenes *closest in date* to the Tanager acquisitions
(2024-12-06 and 2025-01-20). Both turned out to be **partial swaths with ~74% nodata**,
covering only the western third of the AOI — their footprints stop at −118.70 while the AOI
runs to −118.58. Only 14–17% of the grid had data, and dNBR p99 came out at +0.15, far too
low for a severe burn.

**A STAC bbox intersection is not coverage.** Scenes are now filtered on footprint
containment of the AOI *and* `s2:nodata_pixel_percentage < 5`, which costs temporal
proximity but yields complete maps:

| role | scene | date | note | cloud |
|---|---|---|---|---:|
| pre | S2A_11SLT_20241113 | 2024-11-13 | 27 d before Franklin, 55 before Palisades | 0.0% |
| post | S2A_11SLT_20250112 | 2025-01-12 | 5 d after Palisades ignited | 0.0% |
| feb | S2C_11SLT_20250221 | 2025-02-21 | after full containment (Jan 31) | 5.0% |
| late | S2B_11SLT_20250805 | 2025-08-05 | 10 d after the Tanager July scene | 1.0% |

A second bug: NBR blew up to 7×10⁷ where NIR+SWIR ≈ 0 (deep water, deep shadow). The ratio
is now guarded on `NIR + SWIR > 0.01`.

Resulting dNBR is in the expected physical range: median +0.08, p99 +0.92, max +1.21.

## Independent agreement

Tanager char fraction (426-band unmixing, 23 Jan) vs Sentinel-2 dNBR (2-band index,
13 Nov → 12 Jan), over 306,393 comparable pixels (275.8 km²). Different satellite,
different optics, different processing chain, different algorithm; nothing in the unmixing
was informed by Sentinel-2. (The perimeters did locate the endmember training pixels — see
*The claim, as it was written* below.)

```
Pearson  r = +0.659
Spearman r = +0.345
```

| dNBR severity class (USGS/MTBS) | km² | char p50 | char p90 | GV p50 |
|---|---:|---:|---:|---:|
| unburned (< 0.10) | 159.1 | 0.000 | 0.138 | 0.548 |
| low (0.10–0.27) | 59.5 | 0.000 | 0.238 | 0.293 |
| moderate-low (0.27–0.44) | 12.4 | 0.146 | 0.627 | 0.000 |
| moderate-high (0.44–0.66) | 20.6 | 0.194 | 0.716 | 0.000 |
| high (> 0.66) | 24.2 | 0.488 | 0.964 | 0.000 |

**Char median increases monotonically across all five severity classes**, and GV decreases
monotonically. Two honest caveats:

- Spearman (+0.345) is much lower than Pearson (+0.659) because FCLS produces many *exact*
  zeros; the char distribution is zero-inflated and rank correlation is penalised by ties.
  Report both.
- Agreement is not the whole story, and the hexbin shows why: char fraction continues to
  resolve structure *within* the high-dNBR class, where dNBR saturates. The 426-band
  product is not merely reproducing the index — it keeps discriminating after the index
  stops.

# Deliverable

`Tanager_Fire_Recovery.ipynb` — 49 cells, structured as Q1 / Q2 / Q3 plus a rejected-claims
section, executes clean end to end in seconds
from the cached artifacts in `data/`. Regenerate with `python scripts/build_notebook.py`
(the notebook is built from source so it stays reviewable as a `.py` file) and execute with
`jupyter nbconvert --to notebook --execute --inplace`.

Note for anyone importing these modules: `matplotlib.use("Agg")` is guarded behind
`if __name__ == "__main__"` in every plotting script. Without that guard, importing
`analyze` from a notebook silently kills the inline backend and every subsequent figure
vanishes without an error.

---

# Step 6 — attempting to close the validation gap (negative result)

Run `python scripts/external_validation_survey.py`.

The open gap was that severity validation compared the Tanager char fraction against *our
own* Sentinel-2 dNBR — same author as the thing being validated. Closing it needs a
product built by someone else. Five candidates were checked programmatically:

| source | usable | why |
|---|---|---|
| MTBS burn severity | **no** | archive covers 1984–2024; MTBS lags ~18 months, so no 2025 |
| USFS BAER SBS 2025 | **no** | `BAER_SBS_2025_v2` exists but returns **0 valid px** over the AOI |
| NASA JPL S1+S2 severity | **no** | 0.9 km² non-zero severity in the overlap vs a ~30 km² scar |
| NASA JPL Sentinel-2 dNBR | **no** | pre-fire date postdates Franklin; r = −0.09 vs an independent dNBR |
| **AVIRIS-3 (JPL)** | **yes**, but gated | 100+ flight lines on 2025-01-11 and 2025-01-16; needs Earthdata Login |

Details worth keeping:

- **BAER has no coverage because these were the wrong kind of fire.** Palisades, Franklin
  and Kenneth were state-responsibility incidents handled under California's WERT
  (Watershed Emergency Response Team) program, not federal Burned Area Emergency Response.
  The federal SBS mosaic simply has nothing here. CAL FIRE's own RdNBR and CBI severity
  services stop at 2024.
- **The NASA S1+S2 product is centred on the Eaton fire**, ~40 km east. Its published
  extent nominally clips our AOI, but the actual data footprint does not cover Palisades.
- **The NASA Sentinel-2 dNBR covers the AOI but cannot be used as a reference here**,
  because its pre-fire scene is 2 Jan 2025 — *after* the Franklin fire of 10 Dec 2024.
  Franklin consequently reads as **negative** dNBR (median −0.194): their "before" image
  already contains the Franklin scar, so the differencing measures early regrowth. This is
  correct behaviour for their purpose and wrong for ours, and it is a good illustration of
  why a validation reference has to be checked against the fire history rather than
  trusted because it is authoritative.

## What would actually close it

**AVIRIS-3.** NASA overflew these fires on **11 January 2025** — four days after Palisades
ignited, twelve days before the Tanager January scene — with an airborne imaging
spectrometer, and published a **relative char-and-ash product** from those flights.

That is a materially better reference than any soil burn severity map would have been:
it measures *the same physical material* the char endmember measures, with an independent
imaging spectrometer, rather than correlating a fraction against a severity proxy. The
comparison would be char fraction against char fraction.

It is blocked only by credentials. The ArcGIS image services NASA published for the char/ash
and dNBR products are stopped server-side, and the underlying L2A reflectance granules on
ORNL DAAC sit behind NASA Earthdata Login (HTTP Basic). Granule *search* is open — 100+
flight lines are discoverable over the AOI via CMR — but download is not.

Step-by-step instructions for completing this with credentials are at the bottom of
`scripts/external_validation_survey.py`. The work is small: download the flight lines,
unmix them with the existing library via `scripts/unmix.py`, and compare fractions directly.

**Current validation status, stated plainly:** the char fraction is validated by (a) being
exactly 0.000 in unburned land without being told where the fires were, (b) ranking
inversely with scar age across three fires, and (c) Pearson r = +0.659 with a Sentinel-2
dNBR computed in this project, monotonic across all five USGS severity classes. It is
**not** yet validated against an externally authored severity product, and no such product
is openly available for these fires.

---

# Step 7 — the validation gap is closed: AVIRIS-3, same day

Run `python scripts/check_earthdata_auth.py` then `python scripts/aviris3.py`.

The gap identified in Step 6 is now closed, and with a better reference than the soil burn
severity product originally sought.

## Why this is the strongest possible check

NASA/JPL overflew these fires with **AVIRIS-3**, an airborne imaging spectrometer, on
**2025-01-23 — the same day as the Tanager January acquisition**. Tanager imaged at
18:55 UTC; the flight lines used here were acquired 20:13–20:23 UTC, a gap of ~1.4 hours.

Every other candidate compares our char *fraction* against a severity *proxy*. This
compares **char fraction against char fraction**:

| | Tanager-1 | AVIRIS-3 |
|---|---|---|
| platform | spaceborne | airborne |
| GSD | 30 m | ~3 m |
| bands | 426 (376–2499 nm) | 284 (390–2493 nm) |
| acquired | 23 Jan 18:55 UTC | 23 Jan 20:13–20:23 UTC |

Same endmember library, same FCLS, same 30 m output grid. Disagreement can only come from
the instruments or the processing, not from a different definition of "char" or from real
surface change over 1.4 hours.

## Method notes that mattered

- **The files are NetCDF-4, not GeoTIFF**, with the cube in a *group*
  (`/reflectance/reflectance`, 284 × 1300 × 1349, chunked `(10, 256, 256)`, gzip) and the
  coordinates and CRS in the root group. An earlier filter looking for `.tif` matched
  nothing, which is why Step 6 reported AVIRIS as unavailable in the first pass.
- **Already EPSG:32611**, the Tanager CRS — no reprojection needed before unmixing.
- **Streamed, not downloaded.** 53 lines qualify at ~2 GB each (~123 GB) against 8.5 GB of
  free disk. Reads go over HTTP range requests via fsspec + h5py, so disk cost is ~0.
  Because the cube is spatially chunked, windowed reads work; band selection takes whole
  10-band chunk groups (`stride=2`, 144 of 284 bands) so skipping actually saves traffic.
  ~11–16 min per line end to end.
- **Aggregate reflectance to 30 m first, then unmix** — not unmix at 3 m and average the
  fractions. This reproduces what Tanager physically does (integrate radiance over a 30 m
  footprint, then unmix), so the comparison isolates instrument differences from scale
  effects. It is also ~90× cheaper: ~19k NNLS solves per line instead of ~1.75M.

## Result

4 same-day flight lines, 56,067 co-located 30 m pixels (50.5 km²):

```
Pearson r          = +0.785
mean bias (T - A)  = -0.060
RMSE               =  0.202
inside perimeters    r = +0.702   (23,203 px)
                     AVIRIS median 0.468   Tanager median 0.238
outside perimeters   AVIRIS median 0.000   Tanager median 0.000   (32,864 px)
```

**Two independent imaging spectrometers both return exactly 0.000 char in unburned land.**
The Tanager null was previously supported only by its own unmixing; it is now confirmed by
a different instrument on the same day. The maps are visually near-identical — scar
boundaries and fine structure inside the perimeters both match
(`data/quicklook/validation_aviris3.png`).

## The honest caveat: pattern is validated, absolute scale is not

Inside the perimeters, **Tanager reads roughly half the AVIRIS char fraction**
(median 0.238 vs 0.468) even though the two agree strongly on spatial pattern (r = +0.70).
The comparison is at matched 30 m resolution, so this is not a scale artifact.

The most likely cause is **endmember purity**. The CHAR endmember is a *class median* over
dark burned Tanager pixels — which is itself already a mixture, not pure char. In linear
unmixing the fraction scale is set by how pure the endmember is: an impure endmember
inflates fractions when applied to data containing purer material, which is exactly what
AVIRIS at 3 m provides. This is a known property of spectral mixture analysis, not a bug,
but it does bound the claim.

**So:** Tanager char fraction is validated as a *relative* measure — where it says more
char, there is more char, confirmed by an independent same-day airborne spectrometer — but
its *absolute* values should not be read as literal areal char cover. Every conclusion in
this project (char is 0 in unburned land, char ranks inversely with scar age, char is gone
by July, recovery ranks with time since fire) depends only on the relative measure.

---

# Step 8 — recovery stratified by vegetation type and burn history

Run `python scripts/vegtype.py` then `python scripts/recovery_by_type.py`.

Step 4 measured each fire against a single pooled unburned control. That control was not
valid, and this step establishes why, fixes it, and reports what survives the fix.

## The stratifier is free, the same way co-registration was

**LANDFIRE LF2023 Existing Vegetation Type** (USGS, 30 m, public, no authentication) is
served as a live ImageServer that exports directly into a caller-specified CRS and extent.
Requesting the exact Tanager overlap window returns **713 × 743 on origin (332760,
3784650) at 30 m** — pixel-identical to the fraction cubes, no resampling, no warping.
Nearest-neighbour interpolation is forced: these are class codes, and averaging them would
invent categories that do not exist.

LF2023 was chosen over the NPS Santa Monica Mountains alliance-level map for two reasons.
It is served as an API rather than an IRMA download, and LANDFIRE reruns its disturbance
logic every release — so LF2023 already reflects Woolsey (2018) and every earlier fire,
whereas the NPS polygon map is a mid-2000s snapshot. LF2023 postdates all prior fires and
predates all four 2024-25 fires, so it is a genuine pre-fire map here.

| EVT class | overlap km² | burned km² |
|---|---:|---:|
| Southern California Dry-Mesic Chaparral | 108.0 | 17.0 |
| Southern California Coastal Scrub | 63.4 | 10.7 |
| CA Coastal Live Oak Woodland and Savanna | 18.3 | 3.3 |
| California Mesic Chaparral | 15.6 | 1.3 |
| California Ruderal Grassland and Meadow (exotic) | 15.1 | 1.4 |

## The pooled control was invalid: the fires burned opposite vegetation

| fire | Dry-Mesic Chaparral | Coastal Scrub |
|---|---:|---:|
| Palisades | **13.4 km²** | 3.1 km² |
| Franklin | 3.5 km² | **6.7 km²** |
| Kenneth | 0.1 km² | 0.9 km² |

Palisades is chaparral-dominated and Franklin is coastal-scrub-dominated — close to
inverted. And the two communities have very different *unburned* seasonal trajectories
over the same six months:

| unburned control | n px | ΔGV Jan→Jul |
|---|---:|---:|
| Dry-Mesic Chaparral | 60,112 | **−0.164** |
| Coastal Scrub | 36,012 | **−0.066** |
| pooled (what Step 4 used) | 251,906 | −0.162 |

The pooled control is essentially the chaparral control, because chaparral dominates the
unburned area. Applying it to Franklin — which is mostly coastal scrub — subtracted a
baseline roughly 0.10 too negative and inflated Franklin's apparent recovery.

## The conclusion survives the fix

Every burned stratum against an unburned control of the **same** EVT class:

| fire | vegetation type | km² | ΔGV | vs same-type control | char Jan |
|---|---|---:|---:|---:|---:|
| Palisades | Dry-Mesic Chaparral | 13.4 | +0.059 | **+0.223** | 0.286 |
| Palisades | Coastal Scrub | 3.1 | +0.026 | **+0.092** | 0.105 |
| Franklin | Dry-Mesic Chaparral | 3.5 | +0.218 | **+0.382** | 0.180 |
| Franklin | Coastal Scrub | 6.7 | +0.144 | **+0.210** | 0.081 |
| Kenneth | Coastal Scrub | 0.9 | +0.000 | +0.066 | 0.471 |

**Franklin still beats Palisades within each vegetation type separately** (+0.382 vs
+0.223 in chaparral; +0.210 vs +0.092 in coastal scrub). The Step 4 claim that recovery
ranks with time since fire is therefore no longer confounded with what was growing there
— it is controlled for it. The magnitude shrinks (Franklin's pooled-control advantage of
+0.486 becomes +0.382 in chaparral), which is the correction the pooled baseline was
hiding.

Two results fall out for free:

- **Chaparral out-recovers coastal scrub relative to control in both fires.** Consistent
  with resprouter dominance and with release from competition after canopy removal.
- **January char tracks fuel type**: 0.286 in chaparral vs 0.105 in coastal scrub within
  Palisades. Heavier fuel leaves more char, which is another independent check that the
  char endmember behaves physically.

## A shade caveat that bounds which strata are quotable

Shade normalization divides by (1 − f_shade) and is stable only where the shade fraction
is small. At 34° sun elevation in January that holds for chaparral and coastal scrub
(median shade 0.000) but **not** for the canyon and north-facing woodland classes:

| stratum | median shade | p90 shade | quotable |
|---|---:|---:|---|
| Dry-Mesic Chaparral | 0.000 | 0.000 | yes |
| Coastal Scrub | 0.000 | 0.000 | yes |
| Coast Live Oak Woodland | 0.076 | **0.58–0.65** | **no** |
| Mixed Evergreen Woodland | 0.000 | **0.58–0.65** | **no** |

Those classes carry a genuinely high *raw* char fraction (~0.50 before normalization, so
the burn signal is real), but their normalized values are inflated by the shade tail —
Coast Live Oak reads char 0.68 normalized against 0.50 raw.

**Open water is the same artifact at its limit.** Over the 88,635 ocean pixels the January
raw char fraction is **0.053** — the unmixing is behaving correctly and finds essentially
no char — but the shade fraction is **0.938**, so dividing by (1 − f_shade) inflates the
normalized value to **0.856**. No published number in this project is affected, because
water is not in the reported class set, but it demonstrates the failure mode the
`shade_reliable` flag exists to catch: shade-normalized fractions are meaningless as the
shade fraction approaches 1.

Because of that, the fraction layers are masked for display on the same rule the statistics
use: `write_maps` in `recovery_by_type.py` drops open water (LANDFIRE EVT 7292) and anything
whose shade fraction reaches 0.90 on either date, ~28 km² in total. An earlier version left
those pixels painted and explained the resulting bright ocean in a caveat — which meant the
map and the statistics were following two different rules, and the text had to talk a reader
out of what the figure showed. The categorical layers (vegetation type, burn history) are
not shade-normalized quantities and are shown everywhere. Every stratum in
`data/recovery_by_type.json` records `shade_med`, `shade_p90` and a `shade_reliable` flag,
and all headline comparisons are restricted to chaparral and coastal scrub.

## Burn history: this is a reburn landscape

CAL FIRE FRAP historic perimeters, 1980–2023, rasterized to a per-pixel prior-fire count.
132 perimeters touch the overlap.

| prior fires (1980–2023) | km² inside the 2024-25 scars | % |
|---:|---:|---:|
| 0 | 11.7 | 18.3 |
| 1 | 26.9 | 42.2 |
| 2 | 4.3 | 6.8 |
| 3 | 8.9 | 14.0 |
| 4 | 9.9 | 15.6 |
| 5 | 2.0 | 3.2 |

**81.7% of the 2024-25 burn area had burned at least once before since 1980, and 32.8%
three or more times.** Largest reburns: Old Topanga 1993 (43.1 km²), Piuma 1985
(18.9 km²), Canyon 2007 (15.4 km²), Calabasas 1996 (15.0 km²), Woolsey 2018 (9.3 km²).

An earlier version of this count reported 0.0 km² burning for the first time. That was
wrong: the query included the 2024-25 fires themselves in the "prior" tally, so every
pixel trivially had a prior fire. The count is now cut at 2023.

## Burn history predicts what grows there now

Measured on land that did **not** burn in 2024-25, so this is the legacy of past fire
rather than a property of the current scars:

| prior fires | n px | chaparral | coastal scrub | exotic/ruderal |
|---:|---:|---:|---:|---:|
| 0 | 101,157 | 9.7% | 4.1% | 1.0% |
| 1 | 59,678 | 35.2% | 8.9% | 2.3% |
| 2 | 49,078 | 40.5% | 21.9% | 8.5% |
| 3 | 27,497 | 25.7% | 28.5% | 11.4% |
| 4 | 13,719 | 17.0% | **54.6%** | 8.0% |
| 5 | 749 | 6.3% | **57.9%** | 10.9% |

Chaparral → coastal sage scrub → exotics, the documented type-conversion sequence,
recovered from two independent public datasets neither of which was built for this
purpose. (The 0-prior-fire row is dominated by developed and water pixels, which is why
its natural-class percentages are low; the trend to read is across rows 1–5.)

## Burn history vs recovery rate — real, but not what it looks like

Within Dry-Mesic Chaparral burned in 2024-25:

| prior fires | n px | ΔGV | GV Jul | NPV Jul |
|---:|---:|---:|---:|---:|
| 0 | 315 | +0.042 | 0.179 | 0.808 |
| 1 | 12,514 | +0.059 | 0.073 | 0.926 |
| 2 | 1,021 | +0.079 | 0.101 | 0.895 |
| 3 | 2,200 | +0.097 | 0.128 | 0.872 |
| 4 | 2,627 | **+0.220** | 0.227 | 0.769 |

Monotonic across all five levels — and across the four that pass the shade check, +0.059
to +0.220. The 0-prior stratum (315 px, p90 shade 0.332) fails `shade_reliable`, so the
trend is quoted from 1 prior fire upward; it does not depend on its own weakest anchor.

**This must not be read as "reburned chaparral recovers better."** Sites that have burned repeatedly are largely already type-converted, so what
greens up fast there is herbaceous cover, not shrubs. This is the single clearest argument
in the project for why the GV/NPV split matters: a broadband greenness index would score
the most degraded, most type-converted ground as the *best* recovering, and would be
exactly wrong. The strata are also badly unbalanced (315 px vs 12,514).

## What is not resolved

- **LANDFIRE EVT is a modeled product**, not a field survey. It is the stratifier for
  every number above and that dependency should be stated wherever those numbers are.
  The NPS Santa Monica Mountains alliance map would be the independent check; no live
  service for it was found, only an IRMA download.
- Recovery is measured at a single six-month interval. Whether the ranking holds through
  the second wet season is not answerable from this pair.

---

# Step 9 — severity control, and a rejected claim of our own

Run `python scripts/recovery_by_type.py`.

## The severity control

Step 8 established that the vegetation and fire-age effects survive a per-type control. But
vegetation type and burn severity are tangled — chaparral both burned hotter (char 0.286 vs
0.105) and recovered differently — so neither effect is attributable to vegetation until it
survives inside a **fixed severity band**. Severity here is Tanager's own January char
fraction, not an external product, and comparing within a band also cancels any January
floor effect because both sides start from the same place.

**Does the vegetation effect survive?** ΔGV vs same-type control, shade < 0.30:

| Jan char bin | chaparral | coastal scrub | difference |
|---|---:|---:|---:|
| 0.00–0.05 | +0.137 | +0.052 | +0.085 |
| 0.05–0.15 | +0.188 | +0.151 | +0.037 |
| 0.15–0.30 | +0.237 | +0.176 | +0.061 |
| 0.30–0.50 | +0.273 | +0.198 | +0.075 |
| 0.50–1.00 | +0.320 | +0.231 | +0.089 |

**Does the fire-age effect survive?** Chaparral only:

| Jan char bin | Palisades | Franklin | difference |
|---|---:|---:|---:|
| 0.00–0.05 | +0.137 | +0.283 | +0.146 |
| 0.05–0.15 | +0.150 | +0.335 | +0.185 |
| 0.15–0.30 | +0.195 | +0.376 | +0.181 |
| 0.30–0.50 | +0.242 | +0.402 | +0.160 |
| 0.50–1.00 | +0.305 | +0.446 | +0.141 |

**Both survive at every severity level.** Neither result is an artifact of one fire having
burned harder than the other.

**Secondary, with its caveat.** Absolute July greenness *rises* with January severity in
burned chaparral (GV 0.000 → 0.072 → 0.117 → 0.147 → 0.196). This is not the usual floor
effect: January GV is 0.000 in *every* bin, so the trend lives in the July state rather than
the baseline. Consistent with more complete canopy removal producing more vigorous
resprouting, and with high-severity sites carrying more fuel because they were more
productive. But FCLS is zero-inflated, so the low bins sit on a floor of exact zeros;
`gv_jul_zero_frac` records how much of each bin that is.

## Rejected: that the 426 bands invert the broadband answer

This was expected to be the project's headline. It does not hold, and it is recorded here in
full because the project's other claims are only worth what this list makes them worth.

**The claim.** A 2-band greenness index should reach the *opposite* conclusion about recovery,
because it cannot distinguish resprouting shrubs from a flush of non-native annual grass, and
the most repeatedly-burned ground is the most type-converted.

**The test.** NDVI computed from Tanager's own bands (665 nm / 842 nm) on both dates — same
instrument, same pixels, same dates as the unmixing, so the comparison isolates *method* from
*sensor*. Three tests:

**1. Do ΔNDVI and ΔGV rank the burn-history gradient differently?** Partly.

| prior fires | ΔNDVI | ΔGV |
|---:|---:|---:|
| 0 | 0.214 | 0.042 |
| 1 | 0.195 | 0.059 |
| 2 | 0.168 | 0.079 |
| 3 | 0.156 | 0.097 |
| 4 | 0.174 | 0.220 |

Opposite directions, r = +0.54. But ΔNDVI is **not monotonic**, and both are difference
measures sharing a January baseline that reburned sites inflate with wet-season grass — a
baseline artifact, not evidence that the index misleads.

**2. At a *fixed* NDVI, does the unmixing reveal composition the index cannot see?** This is
the decisive test, and it came out **null**:

| July NDVI bin | GV (0–1 prior fires) | GV (3+ prior) |
|---|---:|---:|
| 0.30–0.35 | 0.036 | 0.060 |
| 0.35–0.40 | 0.159 | 0.195 |
| 0.40–0.45 | 0.278 | 0.300 |

At a given NDVI, burn history barely changes composition. The claim is not supported.

**3. Is the low-NDVI "zero green" reading real?** Largely not — it is clipping. FCLS returns
**69% exact zeros** in the NDVI 0.15–0.25 bin and 36% at 0.25–0.30. Zero-inflation was already
flagged in Step 5 as the reason Spearman lags Pearson; it contaminates exactly this comparison.

**What survives.** At the same NDVI, burned July ground reads lower GV than unburned January
ground (0.041 vs 0.191 at NDVI 0.30–0.35). That is suggestive, but the two sides differ in
date *and* in sun elevation (34° vs 73°), so it is confounded by the very effect the shade
endmember exists to handle. A defensible version needs same-date burned-vs-unburned strata and
explicit handling of the zero floor. That is a genuine piece of research, not a figure, and it
is not claimed in this submission.

---

# Step 10 — a circularity check the headline claim originally failed

Reproduce with `scripts/endmembers.py` and the perimeter-blind variant described below.

## The claim, as it was written

Every deliverable stated some version of:

> "Char fraction is 0.000 in unburned land. **The unmixing was never told where the fires
> were.**"

That sentence was **false**, and it took an outside question — why do the bright pixels in
the char map align so precisely with the perimeters? — to surface it.

## What the code actually does

`endmembers.py` imports `fire_mask` and uses it to select every training class:

```python
"GV":       valid & ~burned & (ndvi > 0.60),
"NPV_SOIL": valid & ~burned & (ndvi < 0.22) & (bright > 0.15),
"CHAR":     valid &  burned & (bright < 0.10) & (ndvi < 0.25),
```

So the char endmember is by construction *the median spectrum of dark, low-NDVI pixels
inside the NIFC perimeters*, and both vegetation endmembers are drawn from outside them.
The perimeters were used, and the claim did not distinguish library construction from
inference.

## What survives, and why

Two things are true and they need separating:

1. **Inference is genuinely blind.** FCLS runs per pixel over 355 reflectance values with
   no coordinates, no neighbourhood and no mask. Nothing in the unmixing can enforce a
   spatial boundary.
2. **Library construction was not blind.** The perimeters located the training pixels.

The question is whether (2) manufactures the result in (1). Two measurements say it does not.

**The boundary is a gradient, not a step.** Median char against signed distance to the
perimeter line, over pixels with shade < 0.9:

| distance (30 m px) | median char | p90 char |
|---|---:|---:|
| 20–30 outside | 0.000 | 0.121 |
| 10–20 outside | 0.000 | 0.143 |
| 5–10 outside | 0.000 | 0.236 |
| 2–5 outside | 0.000 | **0.479** |
| 0–2 inside | 0.057 | 0.507 |
| 2–5 inside | 0.174 | 0.679 |
| 5–10 inside | 0.258 | 0.777 |
| 20–60 inside | 0.296 | 0.875 |

Char rises over roughly 150–300 m rather than switching at the line, and it **does activate
outside the perimeters** — p90 reaches 0.479 just outside. A geometrically gated result
could not do that. The visual crispness in the char map is the drawn perimeter stroke plus a
genuinely rapid transition; fire lines here follow roads and ridgelines.

**The endmember barely depends on the perimeter selection.** Rebuilding CHAR with the
identical brightness and NDVI rules but *no* perimeter constraint — every pixel in the
overlap eligible — gives a candidate set of 11,516 px (against 9,893), of which **14.1%
(1,623 px) lie outside the perimeters**. The resulting spectrum sits **1.80°** from the
published one in spectral angle. For scale, this project ruled out a separate white-ash
endmember at 2.8°: the perimeter-aware and perimeter-blind char spectra are closer together
than two materials have to be before this project calls them the same material.

So for CHAR the perimeters were a convenience for finding dark pixels, not the source of the
signal.

**The same test on the other two classes gives a mixed answer, and the unflattering half
belongs here.** Both vegetation classes use the perimeters as an *exclusion* (`~burned`):

| endmember | n perimeter-aware | n blind | angle between |
|---|---:|---:|---:|
| GV | 25,741 | 25,822 | **0.25°** |
| CHAR | 9,893 | 11,516 | **1.80°** |
| NPV_SOIL | 16,081 | 40,555 | **7.53°** |

For GV the exclusion is redundant, as expected — burned ground cannot pass `ndvi > 0.60`
anyway. For **NPV_SOIL it is not**. Dropping it lets bright burned substrate
(`bright > 0.15`, `ndvi < 0.22` inside the scars) into the soil class, which more than
doubles the candidate set and moves the spectrum by 7.53° — above the 2.8° this project uses
as its own separability floor.

That is a real dependency and it should be stated rather than buried: **the library's
soil-versus-char contrast is partly maintained by excluding known-burned area from the soil
class.** It is a defensible choice — a soil endmember contaminated with burned substrate
would be a worse endmember — but it is a choice the perimeters made, not the data. Note the
direction of the effect: a char-contaminated NPV endmember would absorb char-like signal and
push char fractions *down*, so it cannot manufacture the zero-outside null, which is what the
headline claim rests on.

## What was changed

The claim now reads, in all four deliverables: the null is **not geometrically enforced**,
because the per-pixel unmixing carries no spatial information — with the perimeter role in
library construction stated plainly rather than implied away.

The independent checks were never affected by this: the Sentinel-2 dNBR agreement
(r = +0.659, monotonic across all five USGS severity classes) uses no endmember at all.
