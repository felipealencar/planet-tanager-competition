# Six Months After the Fire

**Hyperspectral post-fire recovery mapping with Tanager-1 — Santa Monica Mountains, 2025**

Submission for the Planet Tanager Open Data Competition.

Two Tanager-1 hyperspectral scenes (426 bands, 376–2499 nm, 30 m) overlap the Santa Monica
Mountains: **23 January 2025**, sixteen days after the Palisades Fire ignited, and
**26 July 2025**, six months into recovery. This project unmixes both onto a shared
endmember library and measures char loading and vegetation recovery — validated against an
independent spectrometer flown the same day.

**→ Start with the interactive explorer, [`ui/index.html`](ui/index.html)** — the whole
submission in one page: layer map, pixel inspector, and the limits that constrain every
claim. Then [`Tanager_Fire_Recovery.ipynb`](Tanager_Fire_Recovery.ipynb) for the analysis
end to end (49 cells, runs in seconds from cached artifacts), and [`FINDINGS.md`](FINDINGS.md)
for the full technical record, including everything that failed.

The analysis artifacts the notebook reads (`data/*.npz`, `data/*.tif`) are not tracked in
git — they are rebuilt by running the scripts in the order given under *Pipeline* below. The
outputs saved in the notebook are the record of the run they came from, so it reads without
executing anything.

## What this submission contributes

The call invites a use case, a technical assessment, or a promising research direction,
delivered as a case study, as code, or as a technical analysis. This submission delivers in
all three forms:

| contribution | form | backed by |
|---|---|---|
| **A use case with no existing alternative** — post-fire severity and recovery over 477 km² covering the Palisades, Franklin and Kenneth fires | Lightning case study | No authored severity product exists for these fires; four candidates checked and rejected |
| **Tanager benchmarked against another imaging spectrometer** — same-day airborne AVIRIS-3 | Technical analysis · sensor comparison | r = +0.785 over 56,067 matched 30 m pixels; both return 0.000 char in unburned land; Tanager reads ~0.51× the airborne fraction inside perimeters |
| **A published method applied, and its limits measured** — FCLS unmixing (Heinz & Chang 2001) with library size chosen by separability test | Technical analysis · published method | White-ash endmember ruled out at 2.8°; five claims tested and rejected |
| **A characterisation of the data itself** — the `test`-flagged January scene shown spectrally sound | Technical analysis · product quality | Flat 1.049 gain, zero offset, R² 0.86; 355 of 426 bands survive cross-date QA with 2000–2400 nm intact |
| **A promising research direction** — the specific acquisitions that would extend this work | Research direction | See *Known limits* and the closing section of the explorer |

As code: 18 documented scripts, including an HDF-EOS5 reader for Tanager ortho reflectance
cubes and an AVIRIS-3 comparison that streams 2 GB flight lines over HTTP range requests
instead of downloading 123 GB. The notebook and the interactive explorer are both generated
from the analysis artifacts, so no figure or statistic can drift from the data.

## Three questions

> **Q1 — Can a spectrometer measure what fire left behind as a *material*, rather than
> inferring it from an index?**

Every operational burn-severity product — dNBR, RdNBR, MTBS, BAER — is a two-band ratio
calibrated against field severity. A 426-band cube can instead ask *how much charred
material is in this pixel*.

**Answer: yes.** The char endmember returns **exactly 0.000 in unburned land on both
dates**. That null is not enforced geometrically — the per-pixel unmixing has no spatial
information, and char does activate outside the perimeters where dark burn-like material
exists. The perimeters *are* used to locate training pixels for the library, but rebuilding
the char endmember with no perimeter constraint moves it by only **1.8°** in spectral angle
(below the 2.8° at which two materials were judged inseparable), so the spectrum is not an
artifact of that selection. It ranks inversely with scar age
(Kenneth 0.473 at 14 days > Palisades 0.274 at 16 > Franklin 0.180 at 44), tracks fuel type
(0.286 chaparral vs 0.105 coastal scrub), and keeps resolving structure *inside* the
high-dNBR class where the index saturates. The library is four members, not five, because a
separability test established that **no white-ash endmember exists here** (2.8° from bright
burned substrate).

> **Q2 — How far can you trust a fraction measured from orbit?**

Unmixing is easy to do and easy to fool. NASA/JPL flew **AVIRIS-3** over these fires on
23 January 2025 — the same day as the Tanager scene, 1.4 hours later — so the check is
**char fraction against char fraction** between two imaging spectrometers, not a fraction
against a severity proxy.

**Answer: as a relative measure, and only that.** r = +0.785 over 56,067 co-located 30 m
pixels, and **both instruments independently return 0.000 char in unburned land**. But
inside the perimeters Tanager reads roughly *half* the airborne fraction, most likely
endmember purity — so absolute values are not areal char cover. Every conclusion in this
project depends only on the relative measure.

> **Q3 — Does what burned determine what comes back?**

**Answer: yes — and our own first answer was wrong.** The two large fires burned
near-opposite vegetation (Palisades 13.4 km² chaparral / 3.1 coastal scrub; Franklin 3.5 /
6.7), and the two communities have very different *unburned* seasonal baselines (−0.164 vs
−0.066). The pooled control was therefore invalid. With a per-vegetation-type control the
age ranking survives at corrected magnitude, and it survives again inside every fixed
severity band.

## Headline results

| result | value |
|---|---|
| Char fraction in unburned land (both dates) | **0.000** — the endmember never falsely activates |
| Agreement with same-day airborne AVIRIS-3 | **r = +0.785**, char-fraction vs char-fraction |
| Agreement with independent Sentinel-2 dNBR | **r = +0.659**, monotonic across all 5 USGS classes |
| Recovery, controlled for vegetation type | Franklin **+0.382** vs Palisades **+0.223** (chaparral) |
| …and controlled for severity as well | Franklin leads in **all five** char bins (+0.141 to +0.185) |
| Char remaining in July | **0.000** in every scar |
| Cross-date radiometric consistency | flat **1.049 gain**, zero offset, R² 0.86 |
| Burn area that had burned before since 1980 | **81.7%**, and 32.8% three or more times |

## What made this non-obvious

Five claims were killed by measurement, and each one shaped the method. The full list is in
Part V of the notebook; three that mattered most:

1. **The pre/post framing was wrong.** No fire burned between the two scenes — every fire
   predates the January acquisition. The pair is a *recovery series*. Checked against the
   NIFC/WFIGS archive rather than inferred from imagery, which is seasonally confounded.
2. **The two dates were acquired at 34° and 73° sun elevation.** A naive difference would
   measure topographic shadow. Solved not by topographic correction but by unmixing with a
   photometric **shade** endmember — justified by measuring that the inter-date difference
   is a spectrally *flat* 5% gain, i.e. a pure brightness term. **This is why there is no
   index-based version of this project.**
3. **We could not show that hyperspectral inverts the broadband answer.** We expected to
   demonstrate that a 2-band greenness index reaches the opposite conclusion about recovery,
   computed NDVI from Tanager's own bands to test it, and the decisive test came out null.
   It is reported as a rejected claim rather than quietly dropped.

## Where Tanager is, and is not, load-bearing

LANDFIRE supplies the vegetation strata and CAL FIRE the burn history. Every *measured*
quantity in Q3 is Tanager's: the fractions, the char severity used as a control variable,
and the composition of the recovery. Most of all — there is **no pre-fire Tanager scene**,
so the only reason a 34°-sun January image is comparable to a 73°-sun July image at all is
the photometric shade endmember. A broadband index across those two dates would be measuring
topographic shadow.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/validate_fire_signal.py   # fire perimeters vs footprints
.venv/bin/python scripts/invariant_check.py        # cross-date radiometry
.venv/bin/python scripts/crossdate_gain.py         # per-band gain/offset + band QA
.venv/bin/python scripts/endmembers.py             # endmember library
.venv/bin/python scripts/unmix.py                  # FCLS unmixing, both dates
.venv/bin/python scripts/analyze.py                # recovery statistics
.venv/bin/python scripts/sentinel2.py              # pre-burn baseline + dNBR
.venv/bin/python scripts/validate.py               # independent validation
.venv/bin/python scripts/external_validation_survey.py  # survey of external references
.venv/bin/python scripts/check_earthdata_auth.py   # NASA Earthdata token check
.venv/bin/python scripts/aviris3.py                # same-day AVIRIS-3 validation
.venv/bin/python scripts/vegtype.py                # LANDFIRE veg type + CAL FIRE burn history
.venv/bin/python scripts/recovery_by_type.py       # stratified + severity-controlled recovery
.venv/bin/python scripts/build_notebook.py         # regenerate the notebook
.venv/bin/python scripts/build_ui.py               # regenerate the interactive explorer
```

The two Tanager cubes (~2.4 GB) must be downloaded into `data/hdf5/` before `unmix.py`;
direct asset URLs are listed in `FINDINGS.md`.

## Layout

| path | what |
|---|---|
| `Tanager_Fire_Recovery.ipynb` | the narrative deliverable, structured as Q1 / Q2 / Q3 |
| `FINDINGS.md` | full technical record, including what failed |
| `ui/index.html` | interactive explorer (layer map, pixel inspector, caveats) |
| `scripts/tanager.py` | HDF-EOS5 reader for Tanager ortho SR cubes |
| `scripts/*.py` | one pipeline stage each, runnable standalone |
| `scripts/vegtype.py` | LANDFIRE pre-fire vegetation type + CAL FIRE burn history |
| `scripts/recovery_by_type.py` | stratified recovery, per-type and per-severity controls |
| `data/` | cached artifacts (cubes and GeoTIFFs are gitignored) |

## Known limits

- Char fraction is validated as a **relative** measure; absolute values are not areal cover.
- **LANDFIRE EVT is a modeled product** and is the stratifier for every Q3 number. The NPS
  Santa Monica Mountains alliance map would be the independent check; no live service found.
- Strata whose January shade fraction has a heavy tail are flagged `shade_reliable: false`
  and excluded from all claims — this covers the canyon woodland classes.
- **Species composition is not resolvable at 30 m** in chaparral, where the vegetation grain
  is finer than the pixel. The honest next step is AVIRIS-3 at 3 m.
- Recovery is measured at a single six-month interval.

## Data and licensing

All inputs are public; only AVIRIS-3 requires a (free) login.

- **Tanager-1** — Planet Open STAC catalog, CC-BY-4.0
- **Fire perimeters** — NIFC/WFIGS interagency archive, public domain
- **Historic fire perimeters** — CAL FIRE FRAP, public
- **LANDFIRE LF2023 EVT** — USGS, public domain
- **Sentinel-2 L2A** — Copernicus, free and open, via AWS Earth Search
  (explicitly permitted alongside Tanager by the competition FAQ)
- **AVIRIS-3** — NASA/JPL via ORNL DAAC, requires NASA Earthdata Login
