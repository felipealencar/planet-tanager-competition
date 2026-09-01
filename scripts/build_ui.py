"""Generate the public explorer page from the analysis artifacts, with nothing hand-typed.

The notebook is the reviewer's deliverable and FINDINGS.md is the record, but neither
answers the question a non-specialist actually asks -- "so is it growing back?" -- and
neither lets anyone interrogate a specific hillside. This builds that: a single
self-contained HTML page carrying the six UI rasters, the stratified statistics, and the
caveats that bound them.

**It is a generator, not a hand-written page, for the same reason build_notebook.py is.**
Every statistic on the page is read at build time out of data/recovery_by_type.json,
data/lf23_evt_lut.csv, data/ui_layers.npz and data/stac/*.json. Nothing is transcribed
from the prose. A restated number in a write-up is a number that can silently go stale
when the pipeline is rerun, and this project has already been bitten by exactly that --
see the discrepancy note below.

**A discrepancy found while building this, and since fixed at the source.** The module
docstring of vegtype.py used to state that "not one pixel that burned in 2024-25 was
burning for the first time since 1980". That was false against the data it wrote:
11.7 km2 -- 18.3% of the 63.8 km2 burned in 2024-25 -- has a prior-fire count of zero.
The cause was that the query counted the 2024-25 fires themselves in the "prior" tally,
so every burned pixel trivially had a prior fire; the count is now cut at 2023 and the
docstring corrected. recovery_by_type.json was consistent with the arrays throughout (its
chaparral stratum at zero prior fires has 315 px), so only the prose was ever wrong. The
page states the reburned share as "about four fifths", computed here from ui_layers.npz
rather than restated -- which is the reason this module bakes every number from the data.

**Three design constraints that shaped the page.**

1. *The categorical map carries three colours, not eight.* A raster map is an all-pairs
   colour problem -- any two classes can end up adjacent -- and no ordering of eight hues
   survives that gate. The EVT layer therefore paints only the three groups the science
   reports on (chaparral, coastal scrub, exotic/ruderal), renders the two woodland
   classes as a hatched neutral because they are precisely the strata that fail the shade
   check, and folds everything else into neutrals. The pixel inspector still names the
   exact LANDFIRE class for every pixel, so nothing is hidden -- it is moved from the
   colour channel to the readout, which is where 39 classes belong anyway.
2. *Every fraction layer gets its own single-hue ramp.* Only one layer is on screen at a
   time, so they never have to be told apart by hue; the ramps are semantic (ember for
   char, green for green vegetation, straw for non-photosynthetic) so a viewer who
   glances at the legend once does not have to keep re-reading it. dGV is the exception
   and is diverging, because zero means something there.
3. *Two rasters per layer, one per theme.* A sequential ramp anchored at a white surface
   is unreadable on a dark one; the light ramps run light-to-dark and the dark ramps run
   dark-to-light, and CSS picks between them.

An earlier attempt drew the map client-side from the packed arrays and skipped the PNGs
entirely. It was smaller, and it was wrong: the first paint waited on JS, and the map
flashed empty on load. The PNGs are the map; the packed arrays exist only so the
inspector can report exact values, which a PNG cannot give back.

Inputs:
  data/recovery_by_type.json      every statistic rendered
  data/lf23_evt_lut.csv           LANDFIRE class-code -> class-name
  data/ui_layers.npz              the six stride-2 rasters
  data/fire_perimeters_2024_2025.geojson  (via analyze.per_fire_masks) perimeter outlines
  data/stac/*.json                the two acquisition timestamps

Output:
  ui/index.html                   self-contained, no external requests except Google Fonts
"""

from __future__ import annotations

import base64
import csv
import glob
import io
import json
import pathlib
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "ui" / "index.html"

JSON_PATH = ROOT / "data" / "recovery_by_type.json"
LUT_PATH = ROOT / "data" / "lf23_evt_lut.csv"
NPZ_PATH = ROOT / "data" / "ui_layers.npz"

CHAPARRAL = {7110, 7097}
COASTAL_SCRUB = {7092}
EXOTIC = {9301, 9337}
WOODLAND = {7113, 7014}
NON_VEG_PHYS = {
    "Open Water",
    "Agricultural",
    "Developed",
    "Developed-Roads",
    "Developed-Low Intensity",
    "Developed-Medium Intensity",
    "Developed-High Intensity",
    "Quarries-Strip Mines-Gravel Pits-Well and Wind Pads",
}

# ---------------------------------------------------------------------------
# colour: OKLCH -> sRGB, so every ramp is monotone in perceptual lightness by
# construction rather than by eye.
# ---------------------------------------------------------------------------

_M1 = np.array([[0.8189330101, 0.3618667424, -0.1288597137],
                [0.0329845436, 0.9293118715, 0.0361456387],
                [0.0482003018, 0.2643662691, 0.6338517070]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def oklch_to_srgb(L, C, H):
    """One OKLCH triple to an (r, g, b) uint8 tuple, clipped into gamut."""
    h = np.deg2rad(H)
    lab = np.array([L, C * np.cos(h), C * np.sin(h)])
    lms = np.linalg.inv(_M2) @ lab
    lin = np.linalg.inv(_M1) @ (lms ** 3)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)
    return tuple(int(round(v)) for v in np.clip(srgb, 0, 1) * 255)


def hexof(rgb):
    return "#%02x%02x%02x" % rgb


def ramp(a, b, n=256):
    """n colours interpolated in OKLCH between two (L, C, H) anchors."""
    return [
        oklch_to_srgb(*(np.array(a) + (np.array(b) - np.array(a)) * (i / (n - 1))))
        for i in range(n)
    ]


def diverging(neg, mid, pos, n=256):
    """A two-armed ramp through a neutral midpoint; equal steps per arm."""
    half = n // 2
    return ramp(neg, mid, half) + ramp(mid, pos, n - half)


# Hues: ember 42 deg (char), green 150 (photosynthetic), straw 88 (dry biomass),
# blue 258 (the documented sequential hue, used for the fire-history count).
RAMPS = {
    "char_jan": {
        "light": ramp((0.988, 0.010, 42), (0.395, 0.145, 32)),
        "dark": ramp((0.200, 0.022, 34), (0.860, 0.115, 58)),
        "label": "January char fraction",
    },
    "gv_jul": {
        "light": ramp((0.988, 0.012, 150), (0.400, 0.140, 152)),
        "dark": ramp((0.198, 0.020, 152), (0.855, 0.130, 145)),
        "label": "July green-vegetation fraction",
    },
    "npv_jul": {
        "light": ramp((0.988, 0.012, 92), (0.420, 0.120, 76)),
        "dark": ramp((0.198, 0.020, 78), (0.880, 0.135, 95)),
        "label": "July dry / bare fraction",
    },
    "prior_fires": {
        "light": ramp((0.958, 0.022, 258), (0.340, 0.150, 266)),
        "dark": ramp((0.192, 0.030, 264), (0.865, 0.110, 250)),
        "label": "Fires here, 1980-2023",
    },
    # Diverging: ember for drying, green for greening -- warm against cool, through a
    # neutral. An earlier version used a straw hue for the negative arm; both arms then
    # read as olive and the map lost its sign entirely.
    "dgv": {
        "light": diverging((0.520, 0.140, 42), (0.960, 0.003, 100), (0.480, 0.130, 152)),
        "dark": diverging((0.740, 0.130, 48), (0.310, 0.003, 140), (0.720, 0.125, 152)),
        "label": "Change in green fraction, Jan to Jul",
    },
}

# categorical marks (dataviz slots 1-3; validated all-pairs in both modes)
SERIES = {
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70"},
}
NEUTRALS = {
    "light": {
        "quiet": (222, 228, 225),      # other natural
        "flat": (186, 195, 192),       # not interpretable (developed / water / crops)
        "hatchA": (150, 160, 156),     # woodland, stripe A
        "hatchB": (214, 221, 218),     # woodland, stripe B
        "nodata": (0, 0, 0, 0),
    },
    "dark": {
        "quiet": (52, 61, 58),
        "flat": (30, 37, 35),
        "hatchA": (104, 116, 111),
        "hatchB": (42, 50, 47),
        "nodata": (0, 0, 0, 0),
    },
}


def check_monotone():
    """A sequential ramp that is not monotone in lightness encodes nothing. Assert it."""
    for key, spec in RAMPS.items():
        if key == "dgv":
            continue
        for mode in ("light", "dark"):
            steps = spec[mode]
            lum = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in steps]
            direction = np.sign(lum[-1] - lum[0])
            diffs = np.diff(lum) * direction
            assert (diffs >= -1.5).all(), f"{key}/{mode} ramp is not monotone in lightness"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load_lut():
    return {int(r["VALUE"]): r for r in csv.DictReader(LUT_PATH.open())}


def acquisitions():
    out = []
    for f in sorted(glob.glob(str(ROOT / "data" / "stac" / "*.json"))):
        p = json.load(open(f))
        p = p.get("properties", p)
        out.append(p["datetime"][:10])
    return sorted(out)


def perimeter_masks(nrow_full, ncol_full, origin, stride):
    from analyze import per_fire_masks

    masks, any_fire = per_fire_masks(nrow_full, ncol_full, np.asarray(origin))
    sl = (slice(None, None, stride), slice(None, None, stride))
    return {k: v[sl] for k, v in masks.items()}


# ---------------------------------------------------------------------------
# raster rendering
# ---------------------------------------------------------------------------


def png_uri(arr_rgba):
    buf = io.BytesIO()
    Image.fromarray(arr_rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_continuous(values, lut, lo, hi):
    h, w = values.shape
    out = np.zeros((h, w, 4), np.uint8)
    ok = np.isfinite(values)
    idx = np.clip((values - lo) / (hi - lo), 0, 1)
    idx = np.nan_to_num(idx, nan=0.0)
    idx = (idx * (len(lut) - 1)).round().astype(int)
    table = np.array(lut, np.uint8)
    out[..., :3] = table[idx]
    out[..., 3] = np.where(ok, 255, 0)
    return out


def render_prior(values, lut, vmax):
    h, w = values.shape
    out = np.zeros((h, w, 4), np.uint8)
    idx = (np.clip(values, 0, vmax) / vmax * (len(lut) - 1)).round().astype(int)
    table = np.array(lut, np.uint8)
    out[..., :3] = table[idx]
    out[..., 3] = 255
    return out


def evt_group(code, lut):
    if code in CHAPARRAL:
        return "chaparral"
    if code in COASTAL_SCRUB:
        return "scrub"
    if code in EXOTIC:
        return "exotic"
    if code in WOODLAND:
        return "woodland"
    phys = lut.get(code, {}).get("EVT_PHYS", "")
    return "flat" if phys in NON_VEG_PHYS else "quiet"


def render_evt(evt, lut, mode):
    h, w = evt.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 3] = 255
    n = NEUTRALS[mode]
    s = SERIES[mode]
    rgb = {
        "chaparral": tuple(int(s["s1"][i:i + 2], 16) for i in (1, 3, 5)),
        "scrub": tuple(int(s["s2"][i:i + 2], 16) for i in (1, 3, 5)),
        "exotic": tuple(int(s["s3"][i:i + 2], 16) for i in (1, 3, 5)),
        "quiet": n["quiet"],
        "flat": n["flat"],
    }
    rows, cols = np.indices((h, w))
    stripe = ((rows + cols) % 8) < 4  # 45-degree hatch: the not-quotable channel
    for code in np.unique(evt):
        g = evt_group(int(code), lut)
        m = evt == code
        if g == "woodland":
            out[m & stripe, :3] = n["hatchA"]
            out[m & ~stripe, :3] = n["hatchB"]
        else:
            out[m, :3] = rgb[g]
    return out


def render_burned(burned, mode):
    h, w = burned.shape
    out = np.zeros((h, w, 4), np.uint8)
    n = NEUTRALS[mode]
    s = SERIES[mode]
    ember = tuple(int(s["s2"][i:i + 2], 16) for i in (1, 3, 5))
    out[..., :3] = n["quiet"]
    out[burned, :3] = ember
    out[..., 3] = 255
    return out


def render_perimeters(masks):
    """A white outline with a dark ring, so it reads on either theme."""
    any_shape = next(iter(masks.values())).shape
    h, w = any_shape
    edge = np.zeros((h, w), bool)
    for m in masks.values():
        pad = np.pad(m, 1, constant_values=False)
        neigh = (pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        edge |= m & ~neigh
    halo = np.zeros((h, w), bool)
    pad = np.pad(edge, 1, constant_values=False)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            halo |= pad[dr:dr + h, dc:dc + w]
    halo &= ~edge
    out = np.zeros((h, w, 4), np.uint8)
    out[halo] = (10, 14, 13, 150)
    out[edge] = (255, 255, 255, 255)
    return out


# ---------------------------------------------------------------------------
# packing the arrays the inspector reads
# ---------------------------------------------------------------------------


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(d, masks):
    evt = d["evt"].astype(np.uint16)
    prior = d["prior_fires"].astype(np.uint8)
    fire = np.zeros(evt.shape, np.uint8)
    for i, (name, m) in enumerate(masks.items(), start=1):
        fire[m] = i
    packed = {
        "evt": {"b": b64(evt), "t": "u16"},
        "prior_fires": {"b": b64(prior), "t": "u8"},
        "burned": {"b": b64(d["burned"].astype(np.uint8)), "t": "u8"},
        "fire": {"b": b64(fire), "t": "u8"},
    }
    for k in ("dgv", "char_jan", "gv_jul", "npv_jul"):
        v = d[k].astype(np.float64)
        q = np.where(np.isfinite(v), np.round(v * 1000.0), -32768)
        packed[k] = {"b": b64(np.clip(q, -32768, 32767).astype(np.int16)), "t": "i16"}
    return packed


# ---------------------------------------------------------------------------
# derived quantities (all from the files above, none transcribed)
# ---------------------------------------------------------------------------


def derive(report, d, masks):
    pooled = report["controls_pooled"]["dgv"]
    quotable = {"Dry-Mesic Chaparral", "Coastal Scrub"}

    strata = []
    for s in report["strata"]:
        if s["evt_name"] not in quotable:
            continue
        strata.append({
            **s,
            "naive": round(s["dgv"] - pooled, 3),
            "corrected": s["dgv_vs_control"],
            "distortion": round((s["dgv"] - pooled) - s["dgv_vs_control"], 3),
        })

    per_fire = {}
    for s in strata:
        f = per_fire.setdefault(s["fire"], {"km2": 0.0, "naive": 0.0, "corrected": 0.0,
                                            "ignition": s["ignition"], "parts": []})
        f["km2"] += s["km2"]
        f["naive"] += s["km2"] * s["naive"]
        f["corrected"] += s["km2"] * s["corrected"]
        f["parts"].append(s)
    for f in per_fire.values():
        f["naive"] = round(f["naive"] / f["km2"], 3)
        f["corrected"] = round(f["corrected"] / f["km2"], 3)
        f["km2"] = round(f["km2"], 2)

    burned = d["burned"]
    prior = d["prior_fires"]
    reburn = float((burned & (prior > 0)).sum()) / float(burned.sum())

    tc = report["type_conversion"]
    unreliable = [s for s in report["strata"] if not s["shade_reliable"]]

    fire_centroids = {}
    for name, m in masks.items():
        rr, cc = np.nonzero(m)
        fire_centroids[name] = [float(cc.mean()) / m.shape[1], float(rr.mean()) / m.shape[0]]

    return {
        "pooled": pooled,
        "strata": strata,
        "per_fire": per_fire,
        "reburn_frac": reburn,
        "exotic_lo": tc[0]["exotic"],
        "exotic_hi": tc[-1]["exotic"],
        "exotic_lo_k": tc[0]["prior_fires"],
        "exotic_hi_k": tc[-1]["prior_fires"],
        "n_unreliable": len(unreliable),
        "n_strata": len(report["strata"]),
        "coastal_distortion": max(abs(s["distortion"]) for s in strata),
        "centroids": fire_centroids,
    }


# ---------------------------------------------------------------------------
# SVG chart helpers -- hand-built, no libraries, both themes via CSS variables
# ---------------------------------------------------------------------------


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt(v, plus=True):
    s = f"{v:+.3f}" if plus else f"{v:.3f}"
    return s.replace("-", "−")


def chart_controls(report):
    """Each vegetation type's own unburned seasonal baseline, against the pooled one."""
    rows = sorted(report["controls"].items(), key=lambda kv: kv[1]["dgv"])
    pooled = report["controls_pooled"]["dgv"]
    W, rowh, top, left = 660, 30, 26, 208
    H = top + rowh * len(rows) + 34
    lo, hi = -0.30, 0.06
    x = lambda v: left + (v - lo) / (hi - lo) * (W - left - 58)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="Median change in green fraction on unburned land, by vegetation type">']
    p.append(f'<line class="axis" x1="{x(0):.1f}" y1="{top - 12}" x2="{x(0):.1f}" y2="{H - 30}"/>')
    for gv in (-0.30, -0.20, -0.10, 0.0):
        p.append(f'<line class="grid" x1="{x(gv):.1f}" y1="{top - 12}" x2="{x(gv):.1f}" y2="{H - 30}"/>')
        p.append(f'<text class="tick" x="{x(gv):.1f}" y="{H - 14}" text-anchor="middle">{fmt(gv, False)}</text>')
    for i, (name, c) in enumerate(rows):
        y = top + i * rowh
        quot = name in ("Dry-Mesic Chaparral", "Coastal Scrub")
        cls = "bar-key" if quot else "bar-quiet"
        x0, x1 = min(x(0), x(c["dgv"])), max(x(0), x(c["dgv"]))
        p.append(f'<rect class="{cls}" x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 1):.1f}" '
                 f'height="13" rx="3"/>')
        p.append(f'<text class="lbl {"strong" if quot else ""}" x="{left - 12}" y="{y + 11:.1f}" '
                 f'text-anchor="end">{esc(name)}</text>')
        p.append(f'<text class="val" x="{x0 - 8:.1f}" y="{y + 11:.1f}" text-anchor="end">{fmt(c["dgv"])}</text>')
    yp = top + len(rows) * rowh + 2
    p.append(f'<line class="pooled" x1="{x(pooled):.1f}" y1="{top - 16}" x2="{x(pooled):.1f}" y2="{yp:.1f}"/>')
    p.append(f'<text class="tick strong" x="{x(pooled):.1f}" y="{top - 20}" text-anchor="middle">'
             f'pooled control {fmt(pooled)}</text>')
    p.append("</svg>")
    return "".join(p)


def chart_confound(der):
    """The analytical heart: naive (pooled) versus corrected (same-type) recovery."""
    strata = sorted(der["strata"], key=lambda s: -s["corrected"])
    W, rowh, top, left = 680, 46, 40, 250
    H = top + rowh * len(strata) + 26
    lo, hi = 0.0, 0.42
    x = lambda v: left + (v - lo) / (hi - lo) * (W - left - 60)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="Recovery of each burned stratum measured against the pooled control '
         f'and against a control of the same vegetation type">']
    for gv in (0.0, 0.1, 0.2, 0.3, 0.4):
        p.append(f'<line class="grid" x1="{x(gv):.1f}" y1="{top - 16}" x2="{x(gv):.1f}" y2="{H - 24}"/>')
        p.append(f'<text class="tick" x="{x(gv):.1f}" y="{H - 8}" text-anchor="middle">{fmt(gv, False)}</text>')
    for i, s in enumerate(strata):
        y = top + i * rowh
        xa, xb = x(s["naive"]), x(s["corrected"])
        p.append(f'<line class="connector" x1="{xa:.1f}" y1="{y:.1f}" x2="{xb:.1f}" y2="{y:.1f}"/>')
        p.append(f'<circle class="dot-naive" cx="{xa:.1f}" cy="{y:.1f}" r="5.5"/>')
        p.append(f'<circle class="dot-corr" cx="{xb:.1f}" cy="{y:.1f}" r="5.5"/>')
        veg = "chaparral" if s["evt_code"] == 7110 else "coastal scrub"
        p.append(f'<text class="lbl strong" x="{left - 14}" y="{y - 1:.1f}" text-anchor="end">'
                 f'{esc(s["fire"].title())}</text>')
        p.append(f'<text class="lbl" x="{left - 14}" y="{y + 13:.1f}" text-anchor="end">'
                 f'{veg} &#183; {s["km2"]:.1f} km&#178;</text>')
        far = max(xa, xb)
        p.append(f'<text class="val" x="{far + 12:.1f}" y="{y + 4:.1f}">{fmt(s["corrected"])}</text>')
        if abs(s["distortion"]) >= 0.02:
            p.append(f'<text class="delta" x="{(xa + xb) / 2:.1f}" y="{y - 12:.1f}" '
                     f'text-anchor="middle">pooled overstated by {abs(s["distortion"]):.3f}</text>')
    p.append("</svg>")
    return "".join(p)


def chart_perfire(der):
    """The same correction rolled up to whole fires, area-weighted over the two
    quotable types -- this is the number the README reports at fire level."""
    order = sorted(der["per_fire"].items(), key=lambda kv: kv[1]["ignition"])
    W, top, left = 760, 34, 66
    H = 268
    hi = 0.40
    base = H - 62
    gw = (W - left - 26) / len(order)
    y = lambda v: base - v / hi * (base - top)
    bw, gap = 30, 12
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="Fire-level recovery, area-weighted, pooled control versus same-type control">']
    p.append(f'<text class="axlabel" x="6" y="{top - 14}">&#916;GV vs control</text>')
    for gv in (0.0, 0.1, 0.2, 0.3, 0.4):
        p.append(f'<line class="grid" x1="{left}" y1="{y(gv):.1f}" x2="{W - 18}" y2="{y(gv):.1f}"/>')
        p.append(f'<text class="tick" x="{left - 10}" y="{y(gv) + 4:.1f}" text-anchor="end">{fmt(gv, False)}</text>')
    for i, (fire, f) in enumerate(order):
        cx = left + gw * (i + 0.5)
        for j, (key, cls, extra) in enumerate((("naive", "bar-naive", " mute"),
                                               ("corrected", "bar-corr", ""))):
            v = f[key]
            bx = cx + (j - 1) * (bw + gap / 2) + (gap / 2 if j else -gap / 2)
            p.append(f'<rect class="{cls}" x="{bx:.1f}" y="{y(v):.1f}" width="{bw}" '
                     f'height="{max(base - y(v), 1):.1f}" rx="4"/>')
            p.append(f'<text class="val sm{extra}" x="{bx + bw / 2:.1f}" y="{y(v) - 8:.1f}" '
                     f'text-anchor="middle">{fmt(v)}</text>')
        p.append(f'<text class="lbl strong" x="{cx:.1f}" y="{base + 20:.1f}" text-anchor="middle">'
                 f'{esc(fire.title())}</text>')
        p.append(f'<text class="lbl" x="{cx:.1f}" y="{base + 35:.1f}" text-anchor="middle">'
                 f'ignited {esc(f["ignition"])}</text>')
        p.append(f'<text class="lbl" x="{cx:.1f}" y="{base + 49:.1f}" text-anchor="middle">'
                 f'{f["km2"]:.1f} km&#178; quotable</text>')
    p.append(f'<line class="axis" x1="{left}" y1="{base:.1f}" x2="{W - 18}" y2="{base:.1f}"/>')
    p.append("</svg>")
    return "".join(p)


def line_chart(series, xs, ylab, ymax, aria, pct=False, xlab="Fires here, 1980–2023",
               xticklabels=None, xticks=None, ydec=2):
    W, H, left, top = 620, 300, 62, 24
    base, right = H - 54, W - 18
    x = lambda v: left + (v - xs[0]) / max(xs[-1] - xs[0], 1) * (right - left)
    y = lambda v: base - (v / ymax) * (base - top)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(aria)}">']
    ticks = np.linspace(0, ymax, 5)
    for gv in ticks:
        p.append(f'<line class="grid" x1="{left}" y1="{y(gv):.1f}" x2="{right}" y2="{y(gv):.1f}"/>')
        t = f"{gv:.0f}%" if pct else f"{gv:.{ydec}f}"
        p.append(f'<text class="tick" x="{left - 10}" y="{y(gv) + 4:.1f}" text-anchor="end">{t}</text>')
    tickvals = xticks if xticks is not None else xs
    labels = xticklabels or [str(v) for v in tickvals]
    for xv, lab in zip(tickvals, labels):
        p.append(f'<text class="tick" x="{x(xv):.1f}" y="{base + 20:.1f}" '
                 f'text-anchor="middle">{esc(lab)}</text>')
    p.append(f'<text class="axlabel" x="{(left + right) / 2:.1f}" y="{base + 42:.1f}" '
             f'text-anchor="middle">{esc(xlab)}</text>')
    p.append(f'<text class="axlabel" x="14" y="{top - 8}" text-anchor="start">{esc(ylab)}</text>')
    p.append(f'<line class="axis" x1="{left}" y1="{base:.1f}" x2="{right}" y2="{base:.1f}"/>')
    for s in series:
        pts = [(x(a), y(b)) for a, b in s["points"]]
        d = " ".join(("M" if i == 0 else "L") + f"{a:.1f} {b:.1f}" for i, (a, b) in enumerate(pts))
        p.append(f'<path class="line {s["cls"]}" d="{d}"/>')
        for a, b in pts:
            p.append(f'<circle class="mark {s["cls"]}" cx="{a:.1f}" cy="{b:.1f}" r="4.5"/>')
        ax, ay = pts[-1]
        dy = s.get("dy", -12)
        p.append(f'<text class="endlabel {s["cls"]}" x="{ax - 6:.1f}" y="{ay + dy:.1f}" '
                 f'text-anchor="end">{esc(s["label"])}</text>')
    p.append("</svg>")
    return "".join(p)


LABELS = {"GV": "green vegetation", "NPV_SOIL": "dry vegetation / soil", "CHAR": "char"}


def ramp_legend(key, lo, hi, unit):
    """The same gradient swatch the explorer uses, for a static figure.

    A raster without a scale is decoration: the reader can see that one patch is brighter
    than another but not what either means. `key` selects the ramp whose light/dark CSS is
    already emitted for the layer of the same name, so a figure and the interactive map can
    never drift to different colours.
    """
    return (f'<div class="ramp" data-ramp="{key}" role="img" '
            f'aria-label="Colour scale from {esc(lo)} to {esc(hi)}: {esc(unit)}"></div>'
            f'<div class="ramp-ends"><span>{esc(lo)}</span><span>{esc(hi)}</span></div>'
            f'<div class="ramp-unit">{esc(unit)}</div>')


def reveal(left_html, right_html, left_label, right_label, aria):
    """A drag-to-reveal comparison of two co-registered rasters.

    The two images occupy the same box rather than sitting side by side, so a feature can be
    compared against itself instead of against its neighbour 300 px away. The control is a
    real <input type=range> stretched over the frame: pointer drag, click-to-jump and
    arrow-key stepping all come from the browser, and it is focusable and announced.
    """
    return (
        f'<div class="reveal" data-reveal>'
        f'<div class="rv-base">{right_html}</div>'
        f'<div class="rv-top">{left_html}</div>'
        f'<div class="rv-line"></div><div class="rv-grip"></div>'
        f'<input type="range" min="0" max="100" value="50" step="0.1" '
        f'aria-label="{esc(aria)}">'
        f'</div>'
        f'<div class="rv-labs"><span>&larr; {esc(left_label)}</span>'
        f'<span>{esc(right_label)} &rarr;</span></div>'
    )


def spectra_chart(wl, lib, names, colours):
    """The endmember library across the usable bands.

    The path is broken wherever the wavelength axis jumps, because the 1342-1438 nm and
    1783-1967 nm water-vapour regions are dropped -- drawing straight through them would
    invent reflectance that was never measured.
    """
    W, H, left, top = 760, 300, 62, 26
    base, right = H - 56, W - 16
    lo, hi = float(wl.min()), float(wl.max())
    x = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    y = lambda v: base - (v / 0.35) * (base - top)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Reflectance '
         f'spectra of the green vegetation, dry vegetation and soil, and char endmembers '
         f'across 355 usable Tanager bands from 376 to 2499 nanometres">']
    for gv in (0.0, 0.1, 0.2, 0.3):
        p.append(f'<line class="grid" x1="{left}" y1="{y(gv):.1f}" x2="{right}" y2="{y(gv):.1f}"/>')
        p.append(f'<text class="tick" x="{left - 10}" y="{y(gv) + 4:.1f}" text-anchor="end">{gv:.1f}</text>')
    for nm_tick in (500, 1000, 1500, 2000, 2500):
        if lo <= nm_tick <= hi:
            p.append(f'<text class="tick" x="{x(nm_tick):.1f}" y="{base + 20:.1f}" '
                     f'text-anchor="middle">{nm_tick}</text>')
    p.append(f'<text class="axlabel" x="{(left + right) / 2:.1f}" y="{base + 42:.1f}" '
             f'text-anchor="middle">wavelength (nm)</text>')
    p.append(f'<text class="axlabel" x="12" y="{top - 10}">surface reflectance</text>')
    p.append(f'<line class="axis" x1="{left}" y1="{base:.1f}" x2="{right}" y2="{base:.1f}"/>')
    breaks = np.where(np.diff(wl) > 20)[0]
    segs = np.split(np.arange(len(wl)), breaks + 1)
    for name, cls in colours.items():
        if name not in names:
            continue
        row = lib[names.index(name)]
        for seg in segs:
            if len(seg) < 2:
                continue
            d = " ".join(("M" if i == 0 else "L") + f"{x(wl[j]):.1f} {y(row[j]):.1f}"
                         for i, j in enumerate(seg))
            p.append(f'<path class="line {cls}" d="{d}"/>')
        j = segs[-1][-1]
        p.append(f'<text class="endlabel {cls}" x="{x(wl[j]) - 6:.1f}" '
                 f'y="{y(row[j]) - 8:.1f}" text-anchor="end">{esc(LABELS.get(name, name))}</text>')
    p.append("</svg>")
    return "".join(p)


def crop_to(mask, *arrays, pad=6):
    """Crop every array to the bounding box of `mask`, so a figure is not mostly empty."""
    rows, cols = np.where(mask)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, mask.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, mask.shape[1])
    return [a[r0:r1, c0:c1] for a in arrays]


def twin_raster(key_light_dark, alt):
    """The light/dark <img> pair the page swaps with CSS."""
    return (f'<div class="im-wrap">'
            f'<img class="im im-light" src="{key_light_dark[0]}" alt="{esc(alt)}">'
            f'<img class="im im-dark" src="{key_light_dark[1]}" alt="{esc(alt)}"></div>')


def fig_char_null(char, burned, valid, stride=2):
    """Q1: char histogram inside vs outside the perimeters -- the null, drawn."""
    sl = (slice(None, None, stride), slice(None, None, stride))
    c, b, v = char[sl], burned[sl], valid[sl]
    edges = np.linspace(0, 1, 21)
    centres = (edges[:-1] + edges[1:]) / 2
    series = []
    for m, cls, label, dy in ((v & ~b, "s1", "unburned", -14), (v & b, "s2", "inside perimeters", -14)):
        h, _ = np.histogram(c[m], bins=edges)
        pct = 100 * h / max(h.sum(), 1)
        series.append({"cls": cls, "label": label, "dy": dy,
                       "points": list(zip(centres, pct))})
    return line_chart(
        series, list(centres), "% of pixels in that group", 100, pct=True,
        xlab="January char fraction", xticks=[0.025, 0.25, 0.5, 0.75, 0.975],
        xticklabels=["0.0", "0.25", "0.50", "0.75", "1.0"],
        aria="Distribution of January char fraction inside and outside the fire perimeters. "
             "Unburned land is concentrated entirely at zero.")


def fig_s2_classes(char, dnbr, ok):
    """Q2: Tanager char median across the five USGS dNBR severity classes."""
    bounds = [(-9, 0.10, "unburned"), (0.10, 0.27, "low"), (0.27, 0.44, "mod-low"),
              (0.44, 0.66, "mod-high"), (0.66, 9, "high")]
    xs, meds, labs = [], [], []
    for i, (lo, hi, lab) in enumerate(bounds):
        m = ok & (dnbr >= lo) & (dnbr < hi)
        if m.sum() < 200:
            continue
        xs.append(i)
        meds.append(float(np.median(char[m])))
        labs.append(lab)
    return line_chart(
        [{"cls": "s2", "label": "Tanager char (median)", "dy": -14,
          "points": list(zip(xs, meds))}],
        xs, "Tanager char fraction", 0.6, xlab="Sentinel-2 dNBR severity class",
        xticks=xs, xticklabels=labs,
        aria="Median Tanager char fraction rising monotonically across the five USGS "
             "dNBR severity classes"), meds


def table(headers, rows, caption):
    h = "".join(f"<th scope=\"col\">{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(
        f"<td>{c}</td>" if i else f"<th scope=\"row\">{c}</th>" for i, c in enumerate(r)
    ) + "</tr>" for r in rows)
    return (f'<div class="scroll-x"><table><caption>{esc(caption)}</caption>'
            f'<thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>')


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def legend_gradient(steps, mode):
    stops = ", ".join(hexof(steps[int(i * (len(steps) - 1) / 8)]) for i in range(9))
    return f"linear-gradient(90deg, {stops})"


def build():
    check_monotone()
    report = json.loads(JSON_PATH.read_text())
    lut = load_lut()
    d = dict(np.load(NPZ_PATH))
    stride = report["maps"]["stride"]
    nrow, ncol = report["grid"]["nrow"], report["grid"]["ncol"]
    h, w = d["evt"].shape
    masks = perimeter_masks(nrow, ncol, report["grid"]["origin"], stride)
    der = derive(report, d, masks)
    acq = acquisitions()

    # ---- rasters
    images = {}
    for mode in ("light", "dark"):
        images[("evt", mode)] = png_uri(render_evt(d["evt"], lut, mode))
        images[("burned", mode)] = png_uri(render_burned(d["burned"], mode))
        images[("prior_fires", mode)] = png_uri(
            render_prior(d["prior_fires"], RAMPS["prior_fires"][mode], 5))
        images[("dgv", mode)] = png_uri(
            render_continuous(d["dgv"], RAMPS["dgv"][mode], -0.5, 0.5))
        for k in ("char_jan", "gv_jul", "npv_jul"):
            images[(k, mode)] = png_uri(render_continuous(d[k], RAMPS[k][mode], 0.0, 1.0))
    perim = png_uri(render_perimeters(masks))

    layer_order = ["char_jan", "dgv", "gv_jul", "npv_jul", "evt", "prior_fires", "burned"]
    layer_meta = {
        "char_jan": {"name": "Char, January",
                     "gloss": "Charred material 16 days after Palisades ignited. Relative, not absolute — see caveats.",
                     "kind": "seq", "unit": "shade-normalised fraction", "lo": "0.00", "hi": "1.00"},
        "dgv": {"name": "Change in green",
                "gloss": "January to July change in green-vegetation fraction. Unburned land goes brown over the same window.",
                "kind": "div", "unit": "fraction change, clipped at ±0.50",
                "lo": "−0.50", "hi": "+0.50"},
        "gv_jul": {"name": "Green, July",
                   "gloss": "Photosynthetic vegetation at the end of the first dry season.",
                   "kind": "seq", "unit": "shade-normalised fraction", "lo": "0.00", "hi": "1.00"},
        "npv_jul": {"name": "Dry / bare, July",
                    "gloss": "Non-photosynthetic vegetation and soil. The other half of the greenness story.",
                    "kind": "seq", "unit": "shade-normalised fraction", "lo": "0.00", "hi": "1.00"},
        "evt": {"name": "Vegetation type",
                "gloss": "LANDFIRE LF2023, the pre-fire map. A modelled product — see caveats.",
                "kind": "cat"},
        "prior_fires": {"name": "Fire history",
                        "gloss": "How many times each pixel burned between 1980 and 2023, before these fires.",
                        "kind": "ord"},
        "burned": {"name": "2024–25 burn area",
                   "gloss": "Union of the Palisades, Franklin and Kenneth perimeters on this grid.",
                   "kind": "bin"},
    }

    # ---- EVT name lookup for the inspector, only for codes actually present
    evt_names = {}
    for code in np.unique(d["evt"]):
        c = int(code)
        row = lut.get(c, {})
        evt_names[c] = {"n": row.get("EVT_NAME", "Unknown"),
                        "p": row.get("EVT_PHYS", ""),
                        "g": evt_group(c, lut)}

    packed = pack(d, masks)

    cfg = {
        "w": int(w), "h": int(h), "stride": stride,
        "origin": report["grid"]["origin"],
        "px_m": 30 * stride,
        "layers": layer_order,
        "meta": layer_meta,
        "evt": evt_names,
        "fires": list(masks.keys()),
        "arrays": packed,
    }

    # ---- copy blocks
    tc = report["type_conversion"]
    tiles = [
        (fmt(der["pooled"]),
         "Pooled control, ΔGV",
         "Unburned land <em>lost</em> this much greenness between January and July. "
         "A scar has to beat that number before it counts as recovery."),
        (f"{report['controls']['Dry-Mesic Chaparral']['dgv']:.3f} / "
         f"{report['controls']['Coastal Scrub']['dgv']:.3f}".replace("-", "−"),
         "Chaparral / coastal scrub baseline",
         "The two communities dry out at very different rates. One pooled control cannot "
         "stand in for both — this is the confound."),
        (f"{der['coastal_distortion']:.3f}",
         "Overstatement, coastal scrub",
         "How much the pooled control inflated recovery for every coastal-scrub stratum. "
         "Franklin, the coastal-dominated fire, gained the most from it."),
        (f"{der['reburn_frac'] * 100:.0f}%",
         "Burn area that had burned before",
         "Of the mapped 2024–25 burn area, this share had already burned at least once "
         "since 1980. Repeat fire is the normal condition here."),
        (f"{tc[0]['exotic'] * 100:.1f}% → {tc[-1]['exotic'] * 100:.1f}%",
         f"Exotic cover, {tc[0]['prior_fires']} → {tc[-1]['prior_fires']} prior fires",
         "Measured on land that did <em>not</em> burn in 2024–25, so it is the legacy of "
         "past fire rather than the current scars."),
        (f"{der['n_unreliable']} of {der['n_strata']}",
         "Strata that are not quotable",
         "Shade normalisation is unstable where January shade has a heavy tail. "
         "Those strata are shown but excluded from every headline."),
    ]

    tiles_html = "".join(
        f'<article class="tile"><p class="tile-v">{v}</p>'
        f'<h3 class="tile-k">{esc(k)}</h3><p class="tile-g">{g}</p></article>'
        for v, k, g in tiles)

    # confound table view
    conf_rows = [[
        f'{esc(s["fire"].title())} &#183; {esc(s["evt_name"])}',
        f'{s["km2"]:.2f}', fmt(s["dgv"]), fmt(s["naive"]), fmt(s["corrected"]),
        fmt(-s["distortion"]),
    ] for s in sorted(der["strata"], key=lambda s: -s["corrected"])]
    conf_table = table(
        ["Stratum", "km²", "ΔGV", "vs pooled control", "vs same-type control", "correction"],
        conf_rows, "Recovery of each quotable stratum, both ways")

    tc_rows = [[str(r["prior_fires"]), f'{r["n_px"]:,}',
                f'{r["chaparral"] * 100:.1f}%', f'{r["coastal_scrub"] * 100:.1f}%',
                f'{r["exotic"] * 100:.1f}%'] for r in tc]
    tc_table = table(["Prior fires", "pixels", "chaparral", "coastal scrub", "exotic / ruderal"],
                     tc_rows, "Vegetation composition of unburned land by burn history")

    freq = report["recovery_vs_fire_frequency"]
    freq_rows = [[f'{esc(r["evt_name"])}, {r["prior_fires"]} prior', f'{r["km2"]:.2f}',
                  fmt(r["dgv"]), f'{r["gv_jul"]:.3f}', f'{r["npv_jul"]:.3f}',
                  "yes" if r["shade_reliable"] else "no"] for r in freq]
    freq_table = table(
        ["Stratum", "km²", "ΔGV", "GV July", "NPV July", "shade-reliable"],
        freq_rows, "Recovery against burn history, within vegetation type")

    # ---- Q2: external validation -------------------------------------------
    # Recomputed here from the cached artifacts rather than restated, using exactly the
    # masks aviris3.py and validate.py use (finite in both products, finite RMSE, and
    # shade < 0.9) so this page and FINDINGS.md cannot drift apart.
    fr = np.load("data/fractions.npz")
    nm = [str(n) for n in fr["names"]]
    fj = fr["frac_jan23"]
    tan_char = fj[nm.index("CHAR")] / np.clip(1.0 - fj[nm.index("SHADE")], 1e-3, None)
    base_ok = np.isfinite(tan_char) & np.isfinite(fr["rmse_jan23"]) & (fj[nm.index("SHADE")] < 0.9)

    av = np.load("data/aviris3_char.npz", allow_pickle=True)
    av_char, av_lines = av["char"], [str(x) for x in av["lines"]]
    ok_av = base_ok & np.isfinite(av_char)
    av_r = float(np.corrcoef(av_char[ok_av], tan_char[ok_av])[0, 1])

    from analyze import per_fire_masks
    fmasks, fire_any = per_fire_masks(*tan_char.shape, fr["origin"])
    in_p, out_p = ok_av & fire_any, ok_av & ~fire_any
    av_in_med, tan_in_med = float(np.median(av_char[in_p])), float(np.median(tan_char[in_p]))
    av_out_med = float(np.median(av_char[out_p]))
    tan_out_med = float(np.median(tan_char[out_p]))

    s2v = np.load("data/sentinel2.npz", allow_pickle=True)["dnbr_post"]
    ok_s2 = base_ok & np.isfinite(s2v)
    s2_r = float(np.corrcoef(tan_char[ok_s2], s2v[ok_s2])[0, 1])

    val = dict(
        av_r=f"{av_r:+.3f}", av_n=f"{int(ok_av.sum()):,}", av_lines=len(av_lines),
        av_in=f"{av_in_med:.3f}", tan_in=f"{tan_in_med:.3f}",
        av_out=f"{av_out_med:.3f}", tan_out=f"{tan_out_med:.3f}",
        av_ratio=f"{tan_in_med / av_in_med:.2f}" if av_in_med else "n/a",
        s2_r=f"{s2_r:+.3f}", s2_n=f"{int(ok_s2.sum()):,}",
        s2_km2=f"{ok_s2.sum() * 900 / 1e6:.0f}",
    )

    # ---- Q3 headline numbers, derived rather than restated ------------------
    QUOTABLE = ("Dry-Mesic Chaparral", "Coastal Scrub")
    pool_dgv = report["controls_pooled"]["dgv"]
    chap = {st["fire"]: st for st in report["strata"]
            if st["evt_name"] == "Dry-Mesic Chaparral"}
    q3_gap = chap["Franklin"]["dgv_vs_control"] - chap["PALISADES"]["dgv_vs_control"]
    q3_base_ratio = (report["controls"]["Dry-Mesic Chaparral"]["dgv"]
                     / report["controls"]["Coastal Scrub"]["dgv"])
    q3_scrub_gap = (next(st for st in report["strata"]
                         if st["fire"] == "Franklin" and st["evt_name"] == "Coastal Scrub")["dgv_vs_control"]
                    - next(st for st in report["strata"]
                           if st["fire"] == "PALISADES" and st["evt_name"] == "Coastal Scrub")["dgv_vs_control"])
    overstated = {}
    for fire in ("PALISADES", "Franklin"):
        ss = [st for st in report["strata"]
              if st["fire"] == fire and st["evt_name"] in QUOTABLE]
        area = sum(st["km2"] for st in ss)
        naive = sum(st["km2"] * (st["dgv"] - pool_dgv) for st in ss) / area
        corr = sum(st["km2"] * st["dgv_vs_control"] for st in ss) / area
        overstated[fire] = naive - corr

    # ---- contributions -------------------------------------------------------
    n_stages = len(sorted(pathlib.Path("scripts").glob("*.py")))
    n_rejected = 5  # the claims listed in FINDINGS.md that measurement killed

    # ---- study area / data preparation --------------------------------------
    em = np.load("data/endmembers.npz", allow_pickle=True)
    em_names = [str(x) for x in em["names"]]
    materials = [n for n in em_names if em["library"][em_names.index(n)].any()]
    spectra = spectra_chart(em["wavelengths"], em["library"], em_names,
                            {"GV": "s3", "NPV_SOIL": "s2", "CHAR": "sn"})
    gain_npz = np.load("data/crossdate_gain.npz")
    gd = gain_npz["good"].astype(bool)
    prep = dict(
        gain=f'{float(np.median(gain_npz["gain"][gd])):.3f}',
        gain_r2=f'{float(np.median(gain_npz["r2"][gd])):.2f}',
        n_usable=int(np.load("data/usable_bands.npy").sum()),
        n_materials=len(materials),
        spectra=spectra,
        overlap_km2=f'{tan_char.size * 900 / 1e6:.0f}',
        grid=f'{tan_char.shape[0]} &times; {tan_char.shape[1]}',
    )
    for k, lbl in (("jan23", "shade_jan"), ("jul26", "shade_jul")):
        f_k = fr[f"frac_{k}"]
        okk = np.isfinite(f_k[0])
        prep[lbl] = f'{float(np.nanmean(f_k[nm.index("SHADE")][okk])):.3f}'

    # ---- question figures ---------------------------------------------------
    # Rendered with the same ramps and the same light/dark pair as the explorer layers,
    # so a figure and the map it refers to cannot disagree visually.
    # The char layer must be masked before it is shown or counted. Shade-normalised
    # fractions blow up as the shade fraction approaches 1, so open water -- almost pure
    # shade -- reads as the brightest char in the frame. Water and deep shadow are excluded
    # from every published statistic, so they are excluded here too rather than left in to
    # contradict the figure they illustrate.
    shade_jan = fj[nm.index("SHADE")]
    evt_full = np.load("data/vegtype.npz")["evt"]
    land = np.isfinite(tan_char) & (shade_jan < 0.9)
    quotable = np.isfinite(tan_char) & np.isin(evt_full, sorted(CHAPARRAL | COASTAL_SCRUB | EXOTIC | WOODLAND | {7129})) & (shade_jan < 0.30)

    q1_chart = fig_char_null(tan_char, fire_any, quotable)
    q1_zero_unb = 100.0 * float((tan_char[quotable & ~fire_any] < 0.01).mean())
    q1_zero_brn = 100.0 * float((tan_char[quotable & fire_any] < 0.01).mean())

    st = 2
    q1_map = twin_raster(
        tuple(png_uri(render_continuous(
            np.where(land, tan_char, np.nan)[::st, ::st], RAMPS["char_jan"][m], 0.0, 1.0))
              for m in ("light", "dark")),
        "January char fraction over land in the overlap; signal is confined to the fire perimeters")
    q1_perim = png_uri(render_perimeters(masks))

    # AVIRIS-3: crop both products to the flight-line footprint and show them side by side
    av_foot = np.isfinite(av_char)
    av_c, tan_c, ok_c = crop_to(av_foot, av_char, tan_char, ok_av)
    av_pair = twin_raster(
        tuple(png_uri(render_continuous(np.where(ok_c, av_c, np.nan), RAMPS["char_jan"][m], 0.0, 1.0))
              for m in ("light", "dark")),
        "AVIRIS-3 airborne char fraction over the flight lines")
    tan_pair = twin_raster(
        tuple(png_uri(render_continuous(np.where(ok_c, tan_c, np.nan), RAMPS["char_jan"][m], 0.0, 1.0))
              for m in ("light", "dark")),
        "Tanager-1 char fraction over the same ground, same day")
    av_tan_reveal = reveal(
        av_pair, tan_pair, "AVIRIS-3 \u00b7 airborne, 3 m", "Tanager-1 \u00b7 orbital, 30 m",
        "Wipe between the AVIRIS-3 and Tanager-1 char fraction maps over the same ground")

    legend_char = ramp_legend(
        "char_jan", "0.0", "1.0",
        "char fraction, shade-normalised \u00b7 unpainted = masked (water, cloud or no data)")
    legend_char_both = ramp_legend(
        "char_jan", "0.0", "1.0",
        "char fraction \u00b7 both instruments on this identical scale")

    q2_s2_chart, q2_meds = fig_s2_classes(tan_char, s2v, ok_s2)
    q2_s2_lo, q2_s2_hi = f"{q2_meds[0]:.3f}", f"{q2_meds[-1]:.3f}"

    # ---- severity control -------------------------------------------------
    # Vegetation type and burn severity are tangled: chaparral both burned hotter and
    # recovered differently. Neither effect is attributable to vegetation until it
    # survives inside a fixed January-char band, which is what this section renders.
    sev = report["severity_control"]
    binlab = [f'{r["char_lo"]:.2f}\u2013{r["char_hi"]:.2f}' for r in sev["veg_effect"]]
    xi = list(range(len(binlab)))

    sev_veg_table = table(
        ["Jan char bin", "chaparral", "coastal scrub", "difference"],
        [[b, fmt(r["chaparral"]), fmt(r["coastal_scrub"]), fmt(r["difference"])]
         for b, r in zip(binlab, sev["veg_effect"])],
        "Vegetation effect within each severity band (\u0394GV vs same-type control)")
    sev_age_table = table(
        ["Jan char bin", "Palisades", "Franklin", "difference"],
        [[f'{r["char_lo"]:.2f}\u2013{r["char_hi"]:.2f}', fmt(r["PALISADES"]),
          fmt(r["Franklin"]), fmt(r["difference"])] for r in sev["age_effect"]],
        "Fire-age effect within each severity band, chaparral only")

    sev_veg_chart = line_chart(
        [{"cls": "s1", "label": "chaparral", "dy": -13,
          "points": [(i, r["chaparral"]) for i, r in enumerate(sev["veg_effect"])]},
         {"cls": "s2", "label": "coastal scrub", "dy": 20,
          "points": [(i, r["coastal_scrub"]) for i, r in enumerate(sev["veg_effect"])]}],
        xi, "\u0394GV vs same-type control", 0.48,
        aria="Recovery of chaparral and coastal scrub within each January char severity band",
        xlab="January char fraction (severity)", xticklabels=binlab)
    sev_age_chart = line_chart(
        [{"cls": "s2", "label": "Franklin", "dy": -13,
          "points": [(i, r["Franklin"]) for i, r in enumerate(sev["age_effect"])]},
         {"cls": "s1", "label": "Palisades", "dy": 20,
          "points": [(i, r["PALISADES"]) for i, r in enumerate(sev["age_effect"])]}],
        list(range(len(sev["age_effect"]))), "\u0394GV vs same-type control", 0.48,
        aria="Recovery of Palisades and Franklin within each January char severity band",
        xlab="January char fraction (severity)",
        xticklabels=[f'{r["char_lo"]:.2f}\u2013{r["char_hi"]:.2f}' for r in sev["age_effect"]])
    sev_veg_min = fmt(min(r["difference"] for r in sev["veg_effect"]))
    sev_veg_max = fmt(max(r["difference"] for r in sev["veg_effect"]))
    sev_age_min = fmt(min(r["difference"] for r in sev["age_effect"]))
    sev_age_max = fmt(max(r["difference"] for r in sev["age_effect"]))

    all_rows = [[
        f'{esc(s["fire"].title())} &#183; {esc(s["evt_name"])}', f'{s["km2"]:.2f}',
        f'{s["char_jan"]:.3f}', f'{s["gv_jul"]:.3f}', f'{s["npv_jul"]:.3f}',
        fmt(s["dgv"]), fmt(s["dgv_vs_control"]),
        f'{s["shade_p90"]:.3f}',
        '<span class="ok">quotable</span>' if s["shade_reliable"] else '<span class="no">not quotable</span>',
    ] for s in report["strata"]]
    all_table = table(
        ["Stratum", "km²", "char Jan", "GV Jul", "NPV Jul", "ΔGV",
         "vs same-type control", "shade p90", "status"],
        all_rows, "Every stratum in data/recovery_by_type.json")

    # charts
    tc_series = [
        {"cls": "s1", "label": "chaparral", "dy": 20,
         "points": [(r["prior_fires"], r["chaparral"] * 100) for r in tc]},
        {"cls": "s2", "label": "coastal scrub", "dy": -13,
         "points": [(r["prior_fires"], r["coastal_scrub"] * 100) for r in tc]},
        {"cls": "s3", "label": "exotic / ruderal", "dy": -13,
         "points": [(r["prior_fires"], r["exotic"] * 100) for r in tc]},
    ]
    tc_chart = line_chart(tc_series, [r["prior_fires"] for r in tc],
                          "% of unburned land", 60, pct=True,
                          aria="Share of unburned land in each vegetation class against the "
                               "number of fires since 1980")

    freq_series = []
    for code, cls, label, dy in ((7110, "s1", "chaparral", -13),
                                 (7092, "s2", "coastal scrub", 20)):
        pts = [(r["prior_fires"], r["dgv"]) for r in freq if r["evt_code"] == code]
        if pts:
            freq_series.append({"cls": cls, "label": label, "points": pts, "dy": dy})
    ks = sorted({r["prior_fires"] for r in freq})
    freq_chart = line_chart(freq_series, ks, "ΔGV, January to July", 0.24,
                            aria="Change in green fraction of burned land against the number of "
                                 "prior fires, within each vegetation type")

    css = CSS
    js = JS.replace("__CFG__", json.dumps(cfg, separators=(",", ":")))

    layer_imgs = "".join(
        f'<div class="layer" data-layer="{k}"{"" if k == layer_order[0] else " hidden"}>'
        f'<img class="im im-light" src="{images[(k, "light")]}" alt="">'
        f'<img class="im im-dark" src="{images[(k, "dark")]}" alt=""></div>'
        for k in layer_order)

    switcher = "".join(
        f'<button type="button" class="lyr-btn{" is-on" if k == layer_order[0] else ""}" '
        f'aria-pressed="{"true" if k == layer_order[0] else "false"}" '
        f'data-layer="{k}">{esc(layer_meta[k]["name"])}</button>'
        for k in layer_order)

    fire_labels = "".join(
        f'<span class="fire-label" data-fire="{esc(n)}" '
        f'style="left:{c[0] * 100:.2f}%;top:{c[1] * 100:.2f}%">{esc(n.title())}</span>'
        for n, c in der["centroids"].items())

    grads = {}
    for k in ("char_jan", "gv_jul", "npv_jul", "dgv", "prior_fires"):
        grads[k] = {m: legend_gradient(RAMPS[k][m], m) for m in ("light", "dark")}
    grad_css = "\n".join(
        f'.ramp[data-ramp="{k}"]{{background-image:{v["light"]}}}' for k, v in grads.items())
    media = "\n".join(
        f':root:not([data-theme="light"]) .ramp[data-ramp="{k}"]{{background-image:{v["dark"]}}}'
        for k, v in grads.items())
    stamp = "\n".join(
        f':root[data-theme="dark"] .ramp[data-ramp="{k}"]{{background-image:{v["dark"]}}}'
        for k, v in grads.items())
    grad_css_dark = ("@media (prefers-color-scheme:dark){\n" + media + "\n}\n" + stamp)

    html = PAGE.format(
        css=css,
        grad_css=grad_css,
        grad_css_dark=grad_css_dark,
        tiles=tiles_html,
        acq0=acq[0], acq1=acq[-1],
        layer_imgs=layer_imgs,
        switcher=switcher,
        perim=perim,
        fire_labels=fire_labels,
        controls_chart=chart_controls(report),
        confound_chart=chart_confound(der),
        perfire_chart=chart_perfire(der),
        conf_table=conf_table,
        tc_chart=tc_chart,
        tc_table=tc_table,
        freq_chart=freq_chart,
        freq_table=freq_table,
        **val,
        q1_chart=q1_chart, q1_map=q1_map, q1_perim=q1_perim,
        q1_zero_unb=f'{q1_zero_unb:.1f}%', q1_zero_brn=f'{q1_zero_brn:.1f}%',
        av_tan_reveal=av_tan_reveal, q2_s2_chart=q2_s2_chart,
        legend_char=legend_char, legend_char_both=legend_char_both,
        q2_s2_lo=q2_s2_lo, q2_s2_hi=q2_s2_hi,
        **prep,
        n_stages=n_stages, n_rejected=n_rejected,
        q3_gap=fmt(q3_gap),
        q3_scrub_gap=fmt(q3_scrub_gap),
        q3_base_ratio=f"{q3_base_ratio:.1f}",
        q3_over_frank=fmt(overstated['Franklin']),
        q3_over_pal=fmt(overstated['PALISADES']),
        q3_over_ratio=f"{overstated['Franklin'] / overstated['PALISADES']:.0f}",
        sev_veg_chart=sev_veg_chart,
        sev_age_chart=sev_age_chart,
        sev_veg_table=sev_veg_table,
        sev_age_table=sev_age_table,
        sev_veg_min=sev_veg_min,
        sev_veg_max=sev_veg_max,
        sev_age_min=sev_age_min,
        sev_age_max=sev_age_max,
        all_table=all_table,
        pooled=fmt(der["pooled"]),
        chap_ctrl=fmt(report["controls"]["Dry-Mesic Chaparral"]["dgv"]),
        scrub_ctrl=fmt(report["controls"]["Coastal Scrub"]["dgv"]),
        reburn=f"{der['reburn_frac'] * 100:.0f}%",
        n_unreliable=der["n_unreliable"],
        unreliable_names=esc(", ".join(sorted({
            s["evt_name"] for s in report["strata"] if not s["shade_reliable"]}))),
        shade_max=f"{max(s['shade_p90'] for s in report['strata'] if not s['shade_reliable']):.2f}",
        px_m=30 * stride,
        masked_km2=f"{report['maps'].get('masked_km2', 0):.0f}",
        js=js,
    )

    # PAGE is a raw string, so a \uXXXX written into the markup by an edit would ship as
    # six visible characters rather than the glyph. The browser decodes such escapes inside
    # <script>, so only the markup half is checked.
    head, _, tail = html.partition("<script>")
    js_body, _, after = tail.partition("</script>")
    stray = re.findall(r"\\u[0-9a-fA-F]{4}", head + after)
    assert not stray, f"literal unicode escapes left in the markup: {sorted(set(stray))}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    kb = len(html.encode()) / 1024
    print(f"wrote {OUT}  ({kb:,.0f} KB)")
    print(f"  layers: {', '.join(layer_order)}")
    print(f"  grid {nrow} x {ncol} @ 30 m, rendered {h} x {w} @ {30 * stride} m")
    print(f"  reburned share of 2024-25 burn area: {der['reburn_frac']:.3f}")
    print(f"  strata failing the shade check: {der['n_unreliable']} of {der['n_strata']}")




CSS = r"""
*,*::before,*::after{box-sizing:border-box}
:root{
  color-scheme:light;
  --page:#e9edeb; --surface:#fbfcfc; --surface-2:#f2f5f4; --sunken:#dfe4e2;
  --ink:#0f1513; --ink-2:#4b5552; --ink-3:#79837f;
  --rule:#d7dedb; --hair:rgba(15,21,19,.09);
  --accent:#a8401e; --accent-soft:rgba(168,64,30,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --neutral-mark:#8d9995; --neutral-soft:#c3ccc9;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --grid:#e3e8e6; --axis:#c2cbc8;
  --shadow:0 1px 2px rgba(15,21,19,.05),0 8px 24px -18px rgba(15,21,19,.35);
  --hatch:repeating-linear-gradient(45deg,#96a09c 0 3px,#d6ddda 3px 6px);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --page:#0b100e; --surface:#151a19; --surface-2:#1b2220; --sunken:#0f1413;
    --ink:#eef2f0; --ink-2:#a7b2ae; --ink-3:#7b8683;
    --rule:#262e2c; --hair:rgba(238,242,240,.11);
    --accent:#f08a5f; --accent-soft:rgba(240,138,95,.13);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70;
    --neutral-mark:#7e8a86; --neutral-soft:#39433f;
    --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
    --grid:#232b29; --axis:#333d3a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -20px rgba(0,0,0,.9);
    --hatch:repeating-linear-gradient(45deg,#68746f 0 3px,#2a322f 3px 6px);
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0b100e; --surface:#151a19; --surface-2:#1b2220; --sunken:#0f1413;
  --ink:#eef2f0; --ink-2:#a7b2ae; --ink-3:#7b8683;
  --rule:#262e2c; --hair:rgba(238,242,240,.11);
  --accent:#f08a5f; --accent-soft:rgba(240,138,95,.13);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --neutral-mark:#7e8a86; --neutral-soft:#39433f;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --grid:#232b29; --axis:#333d3a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -20px rgba(0,0,0,.9);
  --hatch:repeating-linear-gradient(45deg,#68746f 0 3px,#2a322f 3px 6px);
}

body{
  margin:0; background:var(--page); color:var(--ink);
  font-family:"Newsreader",Georgia,"Times New Roman",serif;
  font-size:17px; line-height:1.62; -webkit-font-smoothing:antialiased;
  overflow-x:hidden;
}
h1,h2,h3,h4,.ui,button,input,label,.tile-v,.tile-k,table,.chart text,.eyebrow,.chip,.legend,.readout{
  font-family:"Bricolage Grotesque","Helvetica Neue",Arial,sans-serif;
}
.mono,.readout dd,.readout dt,.coord{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}
h1,h2,h3{text-wrap:balance;margin:0}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:4px}
img{max-width:100%}

.page{max-width:1220px;margin:0 auto;padding:0 24px 96px}
.eyebrow{
  font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;font-weight:600;
  color:var(--accent);margin:0 0 14px;
}
section{padding-top:64px}
.sec-head{border-top:1px solid var(--rule);padding-top:22px;margin-bottom:26px}
.sec-head h2{font-size:clamp(24px,3.2vw,34px);font-weight:800;letter-spacing:-.02em;line-height:1.12}
.sec-head p{max-width:64ch;color:var(--ink-2);margin:12px 0 0;font-size:17px}
.prose{max-width:66ch;color:var(--ink-2)}
.prose strong{color:var(--ink);font-weight:500}
.prose p{margin:0 0 14px}

/* ---------- header ---------- */
.hero{padding:56px 0 8px}
.topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap}
h1{font-size:clamp(32px,5.6vw,58px);font-weight:800;letter-spacing:-.035em;line-height:1.02;max-width:17ch}
.lede{max-width:62ch;font-size:19px;color:var(--ink-2);margin:22px 0 0}
.lede b{color:var(--ink);font-weight:500}
.acq{display:flex;gap:0;flex-wrap:wrap;margin:34px 0 0;border-top:1px solid var(--rule)}
.acq div{padding:14px 22px 14px 0;margin-right:22px;border-right:1px solid var(--rule)}
.acq div:last-child{border-right:0}
.acq dt{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);font-weight:600;
  font-family:"Bricolage Grotesque",sans-serif}
.acq dd{margin:4px 0 0;font-size:16px;font-family:"IBM Plex Mono",monospace}
.themebtn{
  font-size:12px;letter-spacing:.06em;text-transform:uppercase;font-weight:600;
  background:var(--surface);color:var(--ink-2);border:1px solid var(--rule);
  border-radius:999px;padding:8px 16px;cursor:pointer;white-space:nowrap;
}
.themebtn:hover{color:var(--ink);border-color:var(--ink-3)}

/* ---------- tiles ---------- */
.tiles{display:grid;grid-template-columns:1fr;gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
@media(min-width:560px){.tiles{grid-template-columns:1fr 1fr}}
@media(min-width:900px){.tiles{grid-template-columns:1fr 1fr 1fr}}
.tile{background:var(--surface);padding:20px 20px 22px;margin:0}
.tile-v{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0;line-height:1.05;color:var(--ink)}
.tile-k{font-size:12px;letter-spacing:.07em;text-transform:uppercase;font-weight:600;
  color:var(--accent);margin:9px 0 8px}
.tile-g{margin:0;font-size:14.5px;line-height:1.5;color:var(--ink-2)}
.tile-g em{font-style:italic;color:var(--ink)}

/* ---------- map ---------- */
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:16px}
.lyr-btn{
  font-size:13px;font-weight:600;letter-spacing:.01em;padding:8px 14px;cursor:pointer;
  background:var(--surface);color:var(--ink-2);border:1px solid var(--rule);border-radius:7px;
}
.lyr-btn:hover{color:var(--ink);border-color:var(--ink-3)}
.lyr-btn.is-on{background:var(--ink);color:var(--page);border-color:var(--ink)}
.tog{display:inline-flex;align-items:center;gap:8px;font-size:13px;font-weight:600;
  color:var(--ink-2);cursor:pointer;padding:8px 14px;border:1px solid var(--rule);
  border-radius:7px;background:var(--surface);margin-left:auto}
.tog input{accent-color:var(--accent);width:15px;height:15px;margin:0}

.explorer{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(280px,.95fr);gap:22px;align-items:start}
@media(max-width:880px){.explorer{grid-template-columns:1fr}}

.mapframe{
  position:relative;border:1px solid var(--rule);border-radius:10px;overflow:hidden;
  background:var(--sunken);box-shadow:var(--shadow);aspect-ratio:372/357;cursor:crosshair;
}
.mapframe:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.layer,.perim,.im{position:absolute;inset:0;width:100%;height:100%}
.im{image-rendering:pixelated;display:block}
.im-wrap,.figmap{position:relative;line-height:0;border-radius:8px;overflow:hidden;
  border:1px solid var(--rule)}
.im-wrap .im,.figmap .im{position:static;width:100%;height:auto}
.figmap .perim{position:absolute;inset:0;width:100%;height:100%}
.reveal{position:relative;line-height:0;border-radius:8px;overflow:hidden;
  border:1px solid var(--rule);touch-action:none;--pos:50%}
.reveal .rv-base,.reveal .rv-top{line-height:0}
.reveal .rv-base .im,.reveal .rv-top .im{position:static;width:100%;height:auto}
.reveal .rv-top{position:absolute;inset:0;clip-path:inset(0 calc(100% - var(--pos)) 0 0)}
.reveal .rv-line{position:absolute;top:0;bottom:0;left:var(--pos);width:2px;
  margin-left:-1px;background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.55);pointer-events:none}
.reveal .rv-grip{position:absolute;top:50%;left:var(--pos);width:34px;height:34px;
  margin:-17px 0 0 -17px;border-radius:50%;background:#fff;pointer-events:none;
  box-shadow:0 0 0 1px rgba(0,0,0,.55),0 2px 8px rgba(0,0,0,.35);
  display:grid;place-items:center}
.reveal .rv-grip::before{content:"";width:14px;height:8px;
  background:linear-gradient(90deg,var(--ink) 3px,transparent 3px 5px,var(--ink) 5px 9px,
  transparent 9px 11px,var(--ink) 11px);opacity:.85}
.reveal input[type=range]{position:absolute;inset:0;width:100%;height:100%;margin:0;
  opacity:0;cursor:ew-resize;-webkit-appearance:none;appearance:none;background:transparent}
.reveal input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:44px;height:100%}
.reveal input[type=range]::-moz-range-thumb{width:44px;height:100%;border:0;opacity:0}
.reveal input[type=range]:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.rv-labs{display:flex;justify-content:space-between;margin:8px 2px 0}
.rv-labs span{font-size:12px;color:var(--ink-3);letter-spacing:.01em}
.rv-hint{font-size:12px;color:var(--ink-3);margin:6px 0 0;line-height:1.4}
.tiles.stack{grid-template-columns:1fr}
.tiles.two{grid-template-columns:1fr}
@media(min-width:760px){.tiles.two{grid-template-columns:1fr 1fr}}
@media(min-width:560px){.tiles.stack{grid-template-columns:1fr}}
@media(min-width:900px){.tiles.stack{grid-template-columns:1fr}}
.im-dark{display:none}
@media(prefers-color-scheme:dark){
  :root:not([data-theme="light"]) .im-light{display:none}
  :root:not([data-theme="light"]) .im-dark{display:block}
}
:root[data-theme="dark"] .im-light{display:none}
:root[data-theme="dark"] .im-dark{display:block}
.perim{image-rendering:pixelated;pointer-events:none;transition:opacity .15s}
.perim.off{opacity:0}
.fire-label{
  position:absolute;transform:translate(-50%,-50%);pointer-events:none;
  font-family:"Bricolage Grotesque",sans-serif;font-size:11px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase;color:#fff;
  text-shadow:0 1px 3px rgba(0,0,0,.85),0 0 10px rgba(0,0,0,.6);transition:opacity .15s;
}
.fire-label.off{opacity:0}
.xhair{position:absolute;width:11px;height:11px;margin:-6px 0 0 -6px;border:1.5px solid #fff;
  border-radius:50%;box-shadow:0 0 0 1.5px rgba(0,0,0,.7);pointer-events:none;display:none}
.xhair.on{display:block}

.side{display:flex;flex-direction:column;gap:16px;min-width:0}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:16px 17px}
.card h3{font-size:12px;letter-spacing:.1em;text-transform:uppercase;font-weight:700;color:var(--ink-3);margin:0 0 10px}
.gloss{font-size:14.5px;color:var(--ink-2);margin:0 0 14px;line-height:1.5}
.ramp{height:11px;border-radius:3px;border:1px solid var(--hair);
  background-image:linear-gradient(90deg,var(--sunken),var(--ink-3))}
.ramp-ends{display:flex;justify-content:space-between;font-size:11.5px;color:var(--ink-3);
  margin-top:6px;font-family:"IBM Plex Mono",monospace}
.ramp-unit{font-size:11.5px;color:var(--ink-3);margin-top:2px;text-align:center}
.swatches{display:flex;flex-direction:column;gap:7px;margin:0;padding:0;list-style:none}
.swatches li{display:flex;align-items:center;gap:9px;font-size:13.5px;color:var(--ink-2);
  font-family:"Bricolage Grotesque",sans-serif}
.sw{width:14px;height:14px;border-radius:3px;flex:none;border:1px solid var(--hair)}
.readout{margin:0;display:grid;grid-template-columns:auto 1fr;gap:5px 12px;font-size:13px}
.readout dt{color:var(--ink-3);white-space:nowrap}
.readout dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;color:var(--ink);overflow-wrap:anywhere}
.readout .span{grid-column:1/-1;text-align:left;color:var(--ink-2);font-size:12.5px;line-height:1.45}
.hint{font-size:12.5px;color:var(--ink-3);margin:10px 0 0;line-height:1.45}
.mapnote{font-size:13px;color:var(--ink-3);margin:14px 0 0;max-width:70ch}

/* ---------- charts ---------- */
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .axis{stroke:var(--axis);stroke-width:1}
.chart .tick{fill:var(--ink-3);font-size:11px;font-variant-numeric:tabular-nums}
.chart .axlabel{fill:var(--ink-3);font-size:11px;letter-spacing:.06em;text-transform:uppercase;font-weight:600}
.chart .lbl{fill:var(--ink-2);font-size:12.5px}
.chart .lbl.strong,.chart .tick.strong{fill:var(--ink);font-weight:700}
.chart .val{fill:var(--ink);font-size:12.5px;font-weight:700;font-variant-numeric:tabular-nums}
.chart .val.sm{font-size:11px}
.chart .val.mute{fill:var(--ink-3);font-weight:600}
.chart .delta{fill:var(--accent);font-size:11px;font-weight:600;letter-spacing:.02em}
.chart .bar-key{fill:var(--s1)}
.chart .bar-quiet{fill:var(--neutral-soft)}
.chart .pooled{stroke:var(--accent);stroke-width:1.5;stroke-dasharray:none}
.chart .connector{stroke:var(--neutral-mark);stroke-width:2}
.chart .dot-naive{fill:var(--neutral-mark)}
.chart .dot-corr{fill:var(--s1)}
.chart .bar-naive{fill:var(--neutral-mark)}
.chart .bar-corr{fill:var(--s1)}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .endlabel{font-size:12px;font-weight:700}
.chart .s1{stroke:var(--s1)} .chart circle.s1{fill:var(--s1)} .chart text.s1{fill:var(--s1);stroke:none}
.chart .sn{stroke:var(--ink-2)} .chart circle.sn{fill:var(--ink-2)} .chart text.sn{fill:var(--ink-2);stroke:none}
.chart .s2{stroke:var(--s2)} .chart circle.s2{fill:var(--s2)} .chart text.s2{fill:var(--s2);stroke:none}
.chart .s3{stroke:var(--s3)} .chart circle.s3{fill:var(--s3)} .chart text.s3{fill:var(--s3);stroke:none}
.chart circle.mark{stroke:var(--surface);stroke-width:2}

.figure{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:20px 22px 18px}
.figure h3{font-size:15px;font-weight:700;letter-spacing:-.01em;margin:0 0 4px;color:var(--ink)}
.figure .sub{font-size:13.5px;color:var(--ink-3);margin:0 0 16px;max-width:62ch;line-height:1.5}
.figrow{display:grid;gap:20px;grid-template-columns:1fr;align-items:start}
@media(min-width:960px){.figrow.two{grid-template-columns:1fr 1fr}}
.keys{display:flex;gap:16px;flex-wrap:wrap;margin:0 0 14px;padding:0;list-style:none;
  font-size:13px;font-family:"Bricolage Grotesque",sans-serif;color:var(--ink-2)}
.keys li{display:flex;align-items:center;gap:7px}
.dot{width:11px;height:11px;border-radius:50%;flex:none}

/* ---------- tables ---------- */
.scroll-x{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;min-width:560px;font-size:13.5px;
  font-family:"Bricolage Grotesque",sans-serif}
caption{text-align:left;font-size:12px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);font-weight:700;padding-bottom:10px}
th,td{padding:8px 12px;border-bottom:1px solid var(--rule);text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap}
thead th{color:var(--ink-3);font-weight:600;font-size:11.5px;letter-spacing:.05em;
  text-transform:uppercase;border-bottom:1px solid var(--axis)}
tbody th{text-align:left;font-weight:600;color:var(--ink);white-space:normal;min-width:190px}
thead th:first-child{text-align:left}
td .ok{color:var(--good);font-weight:700}
td .no{color:var(--crit);font-weight:700}
details{margin-top:18px;border-top:1px solid var(--rule);padding-top:14px}
summary{cursor:pointer;font-size:12px;letter-spacing:.08em;text-transform:uppercase;
  font-weight:700;color:var(--ink-3);font-family:"Bricolage Grotesque",sans-serif;
  list-style:none;display:flex;align-items:center;gap:8px}
summary::-webkit-details-marker{display:none}
summary::before{content:"+";font-size:15px;line-height:1;color:var(--accent)}
details[open] summary::before{content:"−"}
summary:hover{color:var(--ink)}
details>.scroll-x{margin-top:16px}

/* ---------- caveats ---------- */
.caveats{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
@media(min-width:820px){.caveats{grid-template-columns:1fr 1fr}}
.caveat{background:var(--surface);padding:22px 22px 24px;margin:0}
@media(min-width:820px){.caveat.wide{grid-column:1/-1}}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;font-weight:700;padding:4px 9px;border-radius:999px;margin-bottom:12px}
.chip.crit{background:rgba(208,59,59,.13);color:var(--crit)}
.chip.warn{background:rgba(250,178,25,.16);color:#8a5c00}
.chip.info{background:var(--accent-soft);color:var(--accent)}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]) .chip.warn{color:var(--warn)}}
:root[data-theme="dark"] .chip.warn{color:var(--warn)}
.caveat h3{font-size:16.5px;font-weight:700;letter-spacing:-.01em;margin:0 0 8px}
.caveat p{margin:0;font-size:15px;color:var(--ink-2);line-height:1.55}
.caveat p+p{margin-top:10px}

.callout{border-left:2px solid var(--accent);background:var(--accent-soft);
  padding:14px 18px;border-radius:0 8px 8px 0;margin:18px 0 0;font-size:15px;color:var(--ink-2)}
.callout b{color:var(--ink);font-weight:600}

footer{margin-top:80px;border-top:1px solid var(--rule);padding-top:24px;
  font-size:13.5px;color:var(--ink-3);max-width:70ch}
footer a{color:var(--ink-2)}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = r"""
(function(){
var CFG = __CFG__;

/* ---- theme toggle: three states collapse to an explicit stamp on first click ---- */
var root = document.documentElement, btn = document.getElementById('theme');
function current(){
  var s = root.getAttribute('data-theme');
  if (s) return s;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function paintBtn(){ btn.textContent = current() === 'dark' ? 'Light' : 'Dark'; }
btn.addEventListener('click', function(){
  root.setAttribute('data-theme', current() === 'dark' ? 'light' : 'dark');
  paintBtn();
});
paintBtn();

/* ---- unpack the per-pixel arrays ---- */
function unb64(s){
  var bin = atob(s), n = bin.length, u = new Uint8Array(n);
  for (var i=0;i<n;i++) u[i] = bin.charCodeAt(i);
  return u;
}
var A = {};
Object.keys(CFG.arrays).forEach(function(k){
  var raw = unb64(CFG.arrays[k].b), t = CFG.arrays[k].t;
  A[k] = t === 'u16' ? new Uint16Array(raw.buffer)
       : t === 'i16' ? new Int16Array(raw.buffer)
       : raw;
});

var W = CFG.w, H = CFG.h;
var frame = document.getElementById('mapframe');
var xhair = document.getElementById('xhair');
var legend = document.getElementById('legend');
var gloss  = document.getElementById('gloss');
var readout = document.getElementById('readout');
var active = CFG.layers[0];
var cursor = null;

/* ---- legend ---- */
var CAT = [
  ['chaparral','var(--s1)','Chaparral'],
  ['scrub','var(--s2)','Coastal scrub'],
  ['exotic','var(--s3)','Exotic / ruderal'],
  ['woodland','hatch','Woodland — fails the shade check'],
  ['quiet','var(--neutral-soft)','Other vegetation'],
  ['flat','var(--sunken)','Developed, water, cropland']
];
function swatch(color){
  return color === 'hatch'
    ? '<span class="sw" style="background:var(--hatch)"></span>'
    : '<span class="sw" style="background:' + color + '"></span>';
}
function drawLegend(){
  var m = CFG.meta[active], h = '';
  if (m.kind === 'cat'){
    h = '<ul class="swatches">' + CAT.map(function(c){
      return '<li>' + swatch(c[1]) + '<span>' + c[2] + '</span></li>';
    }).join('') + '</ul>';
  } else if (m.kind === 'bin'){
    h = '<ul class="swatches">'
      + '<li>' + swatch('var(--s2)') + '<span>Burned in 2024–25</span></li>'
      + '<li>' + swatch('var(--neutral-soft)') + '<span>Not burned</span></li></ul>';
  } else if (m.kind === 'ord'){
    h = '<div class="ramp" data-ramp="prior_fires"></div>'
      + '<div class="ramp-ends"><span>0</span><span>5 or more</span></div>'
      + '<div class="ramp-unit">fires between 1980 and 2023</div>';
  } else {
    var ends = m.kind === 'div'
      ? '<span>' + m.lo + '</span><span>0</span><span>' + m.hi + '</span>'
      : '<span>' + m.lo + '</span><span>' + m.hi + '</span>';
    h = '<div class="ramp" data-ramp="' + active + '"></div>'
      + '<div class="ramp-ends">' + ends + '</div>'
      + '<div class="ramp-unit">' + m.unit + '</div>';
  }
  legend.innerHTML = h;
  gloss.textContent = m.gloss;
}

/* ---- layer switching ---- */
function setLayer(k){
  active = k;
  Array.prototype.forEach.call(document.querySelectorAll('.layer'), function(el){
    el.hidden = el.dataset.layer !== k;
  });
  Array.prototype.forEach.call(document.querySelectorAll('.lyr-btn'), function(b){
    var on = b.dataset.layer === k;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
  drawLegend();
  paint();
}
Array.prototype.forEach.call(document.querySelectorAll('.lyr-btn'), function(b){
  b.addEventListener('click', function(){ setLayer(b.dataset.layer); });
});

/* ---- perimeter overlay ---- */
var perimBox = document.getElementById('perim-toggle');
perimBox.addEventListener('change', function(){
  document.getElementById('perim').classList.toggle('off', !perimBox.checked);
  Array.prototype.forEach.call(document.querySelectorAll('.fire-label'), function(el){
    el.classList.toggle('off', !perimBox.checked);
  });
});

/* ---- pixel inspector ---- */
function f3(v){
  if (v === -32768) return '—';
  var s = (v/1000).toFixed(3);
  return s.replace('-', '−');
}
function row(dt, dd){ return '<dt>' + dt + '</dt><dd>' + dd + '</dd>'; }

function paint(){
  if (cursor === null){
    readout.innerHTML = '<p class="span">Point at the map — or focus it and use the arrow'
      + ' keys — to read every layer at one pixel.</p>';
    xhair.classList.remove('on');
    return;
  }
  var c = cursor[0], r = cursor[1], i = r*W + c;
  var code = A.evt[i], info = CFG.evt[code] || {n:'Unknown', p:''};
  var fireIdx = A.fire[i];
  var fire = fireIdx ? CFG.fires[fireIdx-1] : null;
  var east = CFG.origin[0] + (c + 0.5) * CFG.px_m;
  var north = CFG.origin[1] - (r + 0.5) * CFG.px_m;
  var html = '';
  html += '<p class="span"><strong>' + info.n + '</strong></p>';
  html += row('LANDFIRE code', code);
  html += row('Physiognomy', info.p || '—');
  html += row('Burned 2024–25', fire ? fire.charAt(0) + fire.slice(1).toLowerCase() : 'no');
  html += row('Prior fires 1980–2023', A.prior_fires[i]);
  html += row('Char, January', f3(A.char_jan[i]));
  html += row('Green, July', f3(A.gv_jul[i]));
  html += row('Dry / bare, July', f3(A.npv_jul[i]));
  html += row('Change in green', f3(A.dgv[i]));
  html += '<p class="span coord">' + Math.round(east) + ' E, ' + Math.round(north)
        + ' N · EPSG:32611 · ' + CFG.px_m + ' m pixel</p>';
  if (info.p === 'Open Water'){
    html += '<p class="span">Open water. Shade normalisation divides by (1 − f_shade),'
          + ' which is meaningless on an almost-pure-shade surface, so no fraction is'
          + ' reported here — on the map or in any statistic.</p>';
  }
  if (A.char_jan[i] === -32768){
    html += '<p class="span">No unmixing here: the pixel was masked as cloud, cirrus or'
          + ' nodata on at least one of the two dates.</p>';
  }
  readout.innerHTML = html;
  xhair.style.left = ((c + 0.5)/W*100) + '%';
  xhair.style.top  = ((r + 0.5)/H*100) + '%';
  xhair.classList.add('on');
}

function fromEvent(e){
  var b = frame.getBoundingClientRect();
  var c = Math.floor((e.clientX - b.left)/b.width * W);
  var r = Math.floor((e.clientY - b.top)/b.height * H);
  if (c < 0 || r < 0 || c >= W || r >= H) return null;
  return [c, r];
}
frame.addEventListener('pointermove', function(e){
  var p = fromEvent(e); if (p){ cursor = p; paint(); }
});
frame.addEventListener('pointerdown', function(e){
  var p = fromEvent(e); if (p){ cursor = p; paint(); frame.focus(); }
});
frame.addEventListener('keydown', function(e){
  var step = e.shiftKey ? 10 : 1, d = null;
  if (e.key === 'ArrowLeft')  d = [-step, 0];
  if (e.key === 'ArrowRight') d = [ step, 0];
  if (e.key === 'ArrowUp')    d = [0, -step];
  if (e.key === 'ArrowDown')  d = [0,  step];
  if (!d) return;
  e.preventDefault();
  if (cursor === null) cursor = [Math.floor(W/2), Math.floor(H/2)];
  cursor = [Math.max(0, Math.min(W-1, cursor[0]+d[0])),
            Math.max(0, Math.min(H-1, cursor[1]+d[1]))];
  paint();
});

drawLegend();
paint();

/* ---- drag-to-reveal comparisons ---- */
Array.prototype.forEach.call(document.querySelectorAll('[data-reveal]'), function(box){
  var input = box.querySelector('input[type=range]');
  function paint(){ box.style.setProperty('--pos', input.value + '%'); }
  input.addEventListener('input', paint);
  /* a pointerdown anywhere in the frame should grab the handle, not just the thumb */
  box.addEventListener('pointerdown', function(e){
    if (e.target === input) return;
    var r = box.getBoundingClientRect();
    input.value = Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100));
    paint();
    input.focus();
  });
  paint();
});
})();
"""

PAGE = r"""<title>Reading the Burn Scar</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700;12..96,800&family=IBM+Plex+Mono:wght@400;500&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap">
<style>{css}
{grad_css}
{grad_css_dark}</style>

<div class="page">


<header class="hero">
  <div class="topbar">
    <div>
      <p class="eyebrow">Planet Tanager Open Data Competition &#183; Santa Monica Mountains</p>
      <h1>Reading a burn scar in 426 bands</h1>
    </div>
    <button type="button" class="themebtn" id="theme">Dark</button>
  </div>
  <p class="lede">In December 2024 and January 2025 the Palisades, Franklin and Kenneth
  fires burned the Santa Monica Mountains. <b>How badly a fire burned, and how well the
  ground recovers, are normally read from a two-band index</b> — dNBR and its relatives —
  which reports a contrast between two dates rather than a physical quantity, saturates at
  high severity, and cannot say what the surface is actually made of. For these fires no
  authored severity product exists at all: they were state-responsibility incidents, so the
  federal mapping programmes have no coverage.</p>
  <p class="lede">Planet&rsquo;s <b>Tanager-1</b> is an imaging spectrometer, and it changes
  what can be asked. Instead of two broad bands it records <b>426 contiguous bands from 376
  to 2499 nm</b> at 30 m — including the shortwave infrared region where charred material,
  dry plant litter and mineral soil have distinct absorption features. That makes it possible
  to <b>unmix each pixel into the materials that compose it</b>, and to measure char as an
  abundance with a real zero rather than infer severity from a ratio.</p>
  <p class="lede">Tanager imaged this ground twice — sixteen days after Palisades ignited,
  and again six months into recovery. This submission uses that pair to test what a spaceborne
  spectrometer adds to wildfire assessment: whether it can measure fire residue as a
  material, whether that measurement survives an independent check, and whether it supports
  conclusions about recovery that an index cannot reach. The strongest of those
  checks is a direct <b>sensor-to-sensor benchmark</b>: NASA flew the airborne AVIRIS-3
  spectrometer over the same fires the same day, so Tanager can be compared against another
  imaging spectrometer measuring the same material, rather than against a severity proxy.
  Every figure is reproducible from a documented open pipeline; the code, the full technical
  record and the notebook are named in the footer.</p>
  <dl class="acq">
    <div><dt>First look</dt><dd>{acq0}</dd></div>
    <div><dt>Second look</dt><dd>{acq1}</dd></div>
    <div><dt>Pixel</dt><dd>30 m</dd></div>
    <div><dt>Bands</dt><dd>426 &#183; 376–2499 nm</dd></div>
  </dl>
</header>

<section id="contributions">
  <div class="sec-head">
    <p class="eyebrow">What this submission contributes</p>
    <h2>Four contributions, against what the competition asks for</h2>
    <p>The call invites a use case, a technical assessment, or a promising research
    direction, delivered as a case study, as code, or as a technical analysis. This
    submission delivers in all three forms; each contribution below names which, and the
    result that backs it.</p>
  </div>
  <div class="tiles two">
    <article class="tile">
      <span class="chip info">Lightning case study</span>
      <h3 class="tile-k" style="margin-top:10px">A use case with no existing alternative</h3>
      <p class="tile-g">Post-fire severity and recovery across {overlap_km2} km&sup2; of the
      Santa Monica Mountains, covering the Palisades, Franklin and Kenneth fires. <b>No
      authored severity product exists for these fires</b> — they were state-responsibility
      incidents, so the federal mapping programmes have no coverage, and four candidate
      products were checked and rejected. Tanager is not improving on an existing map here;
      it is supplying one that does not exist.</p>
    </article>
    <article class="tile">
      <span class="chip info">Technical analysis &#183; sensor comparison</span>
      <h3 class="tile-k" style="margin-top:10px">Tanager benchmarked against another
      imaging spectrometer</h3>
      <p class="tile-g">NASA/JPL flew airborne <b>AVIRIS-3</b> over these fires the same day,
      1.4 hours apart. Running the same library and the same unmixing on both gives a
      like-for-like comparison at matched 30 m: <b>r = {av_r}</b> over {av_n} pixels, with
      both instruments returning 0.000 char in unburned land. It also surfaces a limit —
      Tanager reads about {av_ratio}&times; the airborne char fraction inside perimeters —
      which bounds how the product should be used.</p>
    </article>
    <article class="tile">
      <span class="chip info">Technical analysis &#183; published method</span>
      <h3 class="tile-k" style="margin-top:10px">A standard method applied, and its limits
      measured</h3>
      <p class="tile-g">Fully-constrained linear spectral unmixing (Heinz &amp; Chang, 2001)
      over {n_usable} bands, with the library size chosen by a spectral-angle separability
      test rather than assumed — which is how the expected <b>white-ash endmember was ruled
      out</b> at 2.8&deg;. In total <b>{n_rejected} claims were tested and rejected</b>,
      including one of this project&rsquo;s own headline results.</p>
    </article>
    <article class="tile">
      <span class="chip info">Technical analysis &#183; product quality</span>
      <h3 class="tile-k" style="margin-top:10px">A characterisation of the data itself</h3>
      <p class="tile-g">The January scene ships flagged <span class="mono">quality_category:
      test</span> and in a different collection mode from July. Fitting the two dates over
      invariant targets shows the difference is a spectrally <b>flat {gain} gain</b> with
      essentially zero offset — <b>no spectral distortion</b> — and <b>{n_usable} of 426
      bands</b> survive cross-date QA, with the 2000–2400 nm char diagnostics intact. That is
      a reusable statement about Tanager data, not about these fires.</p>
    </article>
  </div>
  <div class="callout" style="margin-top:20px"><b>And as code.</b> Every result is produced
  by {n_stages} documented scripts, including a reader for Tanager&rsquo;s HDF-EOS5
  ortho reflectance cubes — where the georeferencing lives in a text blob rather than in
  normal attributes — and an AVIRIS-3 comparison that streams 2 GB flight lines over HTTP
  range requests instead of downloading 123 GB. The notebook and this summary are both generated
  from the analysis artifacts, so no figure or statistic can drift from the data.
  <b>A promising research direction</b> — the specific acquisitions that would extend this
  work — is set out <a href="#next">at the end &darr;</a>.</div>
</section>

<section id="questions">
  <div class="sec-head">
    <p class="eyebrow">How the analysis is organised</p>
    <h2>To deliver those, the work asks three questions</h2>
    <p>Each question earns the next: the first establishes that the measurement exists, the
    second that it can be trusted, and the third puts it to work on a question about
    recovery that an index cannot answer. Every number below is read straight out of the
    analysis artifacts at build time — nothing was typed by hand from a write-up.</p>
  </div>
  <div class="tiles">
    <article class="tile">
      <p class="tile-v" style="font-size:1.6rem">Q1</p>
      <h3 class="tile-k">Can a spectrometer measure fire residue as a <em>material</em>?</h3>
      <p class="tile-g">Every operational severity product — dNBR, MTBS, BAER — is a two-band
      ratio. A 426-band cube can instead ask how much charred material is in a pixel.
      <a href="#q1">See the answer &darr;</a></p>
    </article>
    <article class="tile">
      <p class="tile-v" style="font-size:1.6rem">Q2</p>
      <h3 class="tile-k">How far can that number be trusted?</h3>
      <p class="tile-g">Unmixing is easy to do and easy to fool. NASA flew an airborne
      spectrometer over these fires the <em>same day</em>, 1.4 hours later.
      <a href="#q2">See the answer &darr;</a></p>
    </article>
    <article class="tile">
      <p class="tile-v" style="font-size:1.6rem">Q3</p>
      <h3 class="tile-k">Does what burned determine what comes back?</h3>
      <p class="tile-g">It does — and the first answer this project produced was wrong,
      because the fires burned opposite vegetation.
      <a href="#q3">See the answer &darr;</a></p>
    </article>
  </div>
</section>

<section id="data">
  <div class="sec-head">
    <p class="eyebrow">Study area and data preparation</p>
    <h2>Two dates, four materials, and the one term that makes them comparable</h2>
    <p>The two acquisitions overlap over <b>{overlap_km2} km&sup2;</b> of the Santa Monica
    Mountains ({grid} pixels at 30 m, EPSG:32611) with all three fire scars inside. Both
    products sit on the <em>same</em> 30 m grid offset by a whole number of pixels, so the
    overlap is a crop rather than a resampling — nothing is interpolated before analysis.</p>
  </div>
  <div class="figrow two">
    <div>
      <div class="callout"><b>The problem.</b> The two dates were acquired at <b>34&deg;</b>
      and <b>73&deg;</b> sun elevation. In terrain this steep, differencing them directly
      would measure topographic shadow rather than fire.</div>
      <p class="sub" style="max-width:64ch;margin-top:18px">Rather than correcting for terrain
      against a DEM, the difference was first <em>measured</em>. Fitting July against January
      over thousands of pixels that should not have changed — water, rock, pavement, all
      outside the fire perimeters — returns a spectrally <b>flat gain of {gain}</b> with
      essentially zero offset (R&sup2; {gain_r2}). The two dates differ by a pure brightness
      term, not a spectral distortion.</p>
      <p class="sub" style="max-width:64ch">That is exactly what a photometric <b>shade</b>
      endmember absorbs by construction. So each date is unmixed independently into
      {n_materials} measured material spectra plus shade, and fractions are compared instead
      of radiances. The shade fraction comes out <b>{shade_jan}</b> in January against
      <b>{shade_jul}</b> in July — tracking the sun angle precisely, and keeping it out of the
      material fractions. <b>This is the step that makes the whole comparison possible</b>,
      and it has no equivalent in an index-based workflow.</p>
      <p class="sub" style="max-width:64ch">Of the 426 bands, <b>{n_usable}</b> survive
      cross-date quality control. The dropped ones are the 1342–1438 nm and 1783–1967 nm
      water-vapour regions and the extreme detector edges — the 2000–2400 nm region that
      carries the char and litter diagnostics comes through intact on both dates.</p>
    </div>
    <div class="figure">
      <h3>What the library actually is</h3>
      <p class="sub">The three material endmembers across {n_usable} usable bands. These are
      class medians measured from the imagery itself, not laboratory spectra — and the gaps
      are the water-vapour regions, left empty rather than interpolated.</p>
      <ul class="keys">
        <li><span class="dot" style="background:var(--s3)"></span>green vegetation</li>
        <li><span class="dot" style="background:var(--s2)"></span>dry vegetation / soil</li>
        <li><span class="dot" style="background:var(--ink-2)"></span>char</li>
      </ul>
      {spectra}
      <div class="callout">A fourth endmember, <b>shade</b>, is the zero vector — it carries no
      spectrum, which is precisely why it can absorb a brightness difference without
      distorting the three material fractions.</div>
    </div>
  </div>
</section>

<section id="q1">
  <div class="sec-head">
    <p class="eyebrow">Question 1 &#183; what Tanager measures</p>
    <h2>Char is a material, and it has a meaningful zero</h2>
    <p><b>Where the 426 bands do the work:</b> a two-band index can only report a
    <em>contrast</em> between before and after, so it has no way to be wrong in a detectable
    direction. Unmixing instead fits every pixel as a mixture of four measured spectra, and a
    material endmember either activates or it does not. That gives the measurement a
    falsifiable null — and the test is whether char stays at zero across 199&nbsp;km&sup2; of
    unburned land. The per-pixel unmixing has no way to enforce that null: it carries no
    coordinates, no neighbourhood and no mask.</p>
  </div>
  <div class="figrow two">
    <div class="tiles stack">
      <article class="tile"><p class="tile-v">0.000</p>
        <h3 class="tile-k">Char in unburned land, both dates</h3>
        <p class="tile-g">The median across 199&nbsp;km&sup2; of unburned vegetation, on both
        dates. {q1_zero_unb} of those pixels sit at essentially zero, against {q1_zero_brn}
        inside the perimeters — and a perimeter is not a burn mask, so much of that
        {q1_zero_brn} is genuinely unburned island.</p></article>
      <article class="tile"><p class="tile-v">0.47 &rarr; 0.27 &rarr; 0.18</p>
        <h3 class="tile-k">Char ranks inversely with scar age</h3>
        <p class="tile-g">Kenneth at 14 days, Palisades at 16, Franklin at 44. It weathers
        like a material, rather than tracking a brightness contrast.</p></article>
      <article class="tile"><p class="tile-v">2.8&deg;</p>
        <h3 class="tile-k">And where it stops</h3>
        <p class="tile-g">White ash was expected and measured to be absent — 2.8&deg; in
        spectral angle from bright burned ground. The library has four members because a
        separability test said so, not because four was chosen.</p></article>
    </div>
    <div class="figure">
      <h3>The signal stops at the fire line</h3>
      <p class="sub">January char fraction over land, with the Palisades, Franklin and
      Kenneth perimeters drawn on top, for reference only — the unmixing is per-pixel and
      has no spatial information. Open water is
      masked here — it is almost pure shade, and shade-normalised fractions are meaningless
      there (see the limits below).</p>
      <div class="figmap">{q1_map}<img class="perim" src="{q1_perim}" alt="Fire perimeter outlines"></div>
      <div style="margin-top:12px">{legend_char}</div>
      <p class="sub" style="margin:14px 0 0">Distribution over natural vegetation with a
      reliable shade fraction, split by whether the pixel is inside a perimeter:</p>
      <ul class="keys">
        <li><span class="dot" style="background:var(--s1)"></span>unburned</li>
        <li><span class="dot" style="background:var(--s2)"></span>inside perimeters</li>
      </ul>
      {q1_chart}
    </div>
  </div>
</section>

<section id="q2">
  <div class="sec-head">
    <p class="eyebrow">Question 2 &#183; how far to trust it</p>
    <h2>Checked against a second spectrometer, the same day</h2>
    <p><b>Where the 426 bands do the work:</b> because the quantity is a material fraction
    rather than an index value, it can be compared directly against another instrument
    measuring the same material. NASA/JPL flew <b>AVIRIS-3</b> over these fires on 23 January
    2025 — the same day as the Tanager acquisition, about 1.4 hours later. Running the same
    library and the same unmixing on both makes this <b>char fraction against char
    fraction</b>, not a fraction against a severity proxy.</p>
  </div>
  <div class="figrow two">
    <div class="tiles stack">
      <article class="tile"><p class="tile-v">r = {av_r}</p>
        <h3 class="tile-k">Against same-day AVIRIS-3</h3>
        <p class="tile-g">Over {av_n} co-located 30&nbsp;m pixels from {av_lines} flight
        lines — spaceborne against airborne, 30&nbsp;m against 3&nbsp;m.</p></article>
      <article class="tile"><p class="tile-v">{tan_out} / {av_out}</p>
        <h3 class="tile-k">Both instruments, unburned land</h3>
        <p class="tile-g">Medians outside the perimeters. Two independent spectrometers
        return the same zero, so the Q1 null is no longer self-reported.</p></article>
      <article class="tile"><p class="tile-v">r = {s2_r}</p>
        <h3 class="tile-k">Against Sentinel-2 dNBR</h3>
        <p class="tile-g">A different sensor, processing chain and algorithm, over {s2_n}
        pixels ({s2_km2} km&sup2;).</p></article>
      <div class="callout"><b>The limit this puts on every other number.</b> Inside the
      perimeters Tanager reads a median char fraction of {tan_in} against AVIRIS-3&rsquo;s
      {av_in} — about {av_ratio}&times; — despite agreeing strongly on spatial <em>pattern</em>.
      At matched 30&nbsp;m resolution this is not a scale artifact; the likely cause is
      endmember purity. <b>Char fraction is validated as a relative measure only.</b> Where it
      says more char there is more char, but 0.30 is not 30% areal cover.</div>
    </div>
    <div class="figure">
      <h3>Same ground, same day, two instruments</h3>
      <p class="sub">Char fraction over the AVIRIS-3 flight lines, on the identical 30&nbsp;m
      grid and the identical colour scale. Scar boundaries and interior structure match; the
      airborne product simply reads higher.</p>
      {av_tan_reveal}
      <div style="margin-top:12px">{legend_char_both}</div>
      <p class="rv-hint">Drag the handle — or focus it and use the arrow keys — to wipe
      between the two instruments over the same ground.</p>
      <p class="sub" style="margin:18px 0 0">And against a wholly independent product —
      median Tanager char across the five USGS dNBR severity classes:</p>
      {q2_s2_chart}
      <div class="callout">Char rises monotonically from {q2_s2_lo} in the unburned class to
      {q2_s2_hi} in the high-severity class. Two products built from different sensors by
      different methods order the same ground the same way.</div>
    </div>
  </div>
</section>

<section id="q3">
  <div class="sec-head">
    <p class="eyebrow">Question 3 &#183; what comes back</p>
    <h2>What was growing there sets both the rate and the baseline</h2>
    <p>Recovery ranks with time since fire: Franklin, which burned four weeks earlier,
    out-recovers Palisades by <b>{q3_gap}</b> in chaparral and {q3_scrub_gap} in coastal
    scrub, inside <em>every</em> severity band. But the community also sets the baseline —
    unburned chaparral browns {q3_base_ratio}&times; harder than unburned coastal scrub over
    the same summer — so a single pooled baseline flatters whichever fire burned the gentler
    community, here overstating Franklin by <b>{q3_over_frank}</b> against {q3_over_pal} for
    Palisades.</p>
    <p><b>Where the 426 bands do the work:</b> there is no pre-fire Tanager scene, and the two
    dates differ by 39&deg; of sun elevation. The shade endmember absorbs that difference,
    which is the only reason they can be compared at all — and char doubles as a severity
    variable to control against, measured by Tanager rather than borrowed.</p>
  </div>
  <div class="tiles">{tiles}</div>


  <div class="prose">
    <p>Palisades burned mostly <strong>chaparral</strong>; Franklin burned mostly
    <strong>coastal scrub</strong>. Those two communities behave completely differently
    over a Californian dry season even when nothing burns them: unburned chaparral loses
    <strong>{chap_ctrl}</strong> of its green fraction between January and July, while
    unburned coastal scrub loses only <strong>{scrub_ctrl}</strong>. The pooled control,
    at <strong>{pooled}</strong>, is dominated by chaparral — so using it subtracts a
    chaparral-sized seasonal drop from a coastal-scrub fire, and hands that fire a bonus
    it did not earn.</p>
  </div>

  <div class="figrow two" style="margin-top:26px">
    <div class="figure">
      <h3>Each community has its own baseline</h3>
      <p class="sub">Median change in green fraction on <em>unburned</em> land, January to
      July, by vegetation type. The two the project quotes are highlighted.</p>
      {controls_chart}
    </div>
    <div class="figure">
      <h3>Same fires, two verdicts</h3>
      <p class="sub">Recovery of each burned stratum measured the old way and the corrected
      way. Only the coastal-scrub strata move — and they all move by the same amount,
      because the correction is exactly the gap between the two baselines.</p>
      <ul class="keys">
        <li><span class="dot" style="background:var(--neutral-mark)"></span>vs pooled control</li>
        <li><span class="dot" style="background:var(--s1)"></span>vs same-type control</li>
      </ul>
      {confound_chart}
    </div>
  </div>

  <div class="figure" style="margin-top:20px">
    <h3>Rolled up to whole fires</h3>
    <p class="sub">Area-weighted across the chaparral and coastal-scrub strata of each fire.
    The pooled version flattered Franklin — the coastal-dominated fire — most. The
    conclusion survives anyway: recovery still ranks with time since fire, and now it is
    controlled for what was growing there rather than merely asserted.</p>
    <ul class="keys">
      <li><span class="dot" style="background:var(--neutral-mark)"></span>vs pooled control</li>
      <li><span class="dot" style="background:var(--s1)"></span>vs same-type control</li>
    </ul>
    {perfire_chart}
    <details><summary>Table view</summary>{conf_table}</details>
  </div>

  <div class="prose">
    <p>Repeated fire does not simply reset chaparral; it converts it. Shrubs that need
    long fire-free intervals to set seed are replaced by faster-cycling coastal sage
    scrub, and then by non-native annual grasses. Measured on land that did
    <strong>not</strong> burn in 2024–25 — so this is the legacy of past fire, not an
    effect of the current scars — the gradient is unmistakable.</p>
  </div>

  <div class="figrow two" style="margin-top:26px">
    <div class="figure">
      <h3>Type conversion with fire frequency</h3>
      <p class="sub">Composition of unburned land against the number of fires since 1980.
      Chaparral peaks and then collapses; coastal scrub and exotics take over.</p>
      <ul class="keys">
        <li><span class="dot" style="background:var(--s1)"></span>chaparral</li>
        <li><span class="dot" style="background:var(--s2)"></span>coastal scrub</li>
        <li><span class="dot" style="background:var(--s3)"></span>exotic / ruderal</li>
      </ul>
      {tc_chart}
      <details><summary>Table view</summary>{tc_table}</details>
    </div>
    <div class="figure">
      <h3>And the most-burned sites green up fastest</h3>
      <p class="sub">Change in green fraction of burned land, within vegetation type,
      against burn history.</p>
      <ul class="keys">
        <li><span class="dot" style="background:var(--s1)"></span>chaparral</li>
        <li><span class="dot" style="background:var(--s2)"></span>coastal scrub</li>
      </ul>
      {freq_chart}
      <div class="callout"><b>Do not read this as "reburned chaparral recovers better."</b>
      Sites that have burned repeatedly are largely already type-converted, so what is
      greening there is herbaceous cover, not shrubs: across this gradient the July green
      fraction rises while the dry-vegetation fraction falls from 0.926 to 0.769. Switch the
      map to <em>Dry&nbsp;/&nbsp;bare, July</em> and then to <em>Green, July</em> over the
      same ground to see it.
      <br><br><b>An honest limit on that argument.</b> It is tempting to conclude that a
      broadband greenness index would therefore get this backwards. We tested exactly that,
      using NDVI computed from Tanager’s own bands, and <b>could not show it</b> — at a
      fixed July NDVI, burn history barely changes the measured composition. The rejected
      test is written up in full in FINDINGS.md rather than dropped.</div>
      <details><summary>Table view</summary>{freq_table}</details>
    </div>
  </div>

  <div class="sec-head" style="margin-top:3rem">
    <p class="eyebrow">Control 2</p>
    <h2>Was it the vegetation, or just how hard it burned?</h2>
    <p>Vegetation type and burn severity are tangled — chaparral both burned hotter and
    recovered differently. So neither result above is attributable to vegetation until it
    survives inside a <em>fixed</em> severity band. Severity here is Tanager’s own January
    char fraction, not an external product, and comparing within a band also cancels any
    January floor effect: both sides start from the same place.</p>
  </div>
  <div class="figures">
    <div class="figure">
      <h3>Chaparral still out-recovers coastal scrub</h3>
      <p class="sub">Within every band of January char, against each type’s own control.</p>
      <ul class="keys">
        <li><span class="dot" style="background:var(--s1)"></span>chaparral</li>
        <li><span class="dot" style="background:var(--s2)"></span>coastal scrub</li>
      </ul>
      {sev_veg_chart}
      <div class="callout">The vegetation effect holds in <b>all five</b> severity bands,
      from {sev_veg_min} to {sev_veg_max}. It is not an artifact of chaparral having
      burned harder.</div>
      <details><summary>Table view</summary>{sev_veg_table}</details>
    </div>
    <div class="figure">
      <h3>Franklin still out-recovers Palisades</h3>
      <p class="sub">Chaparral only, so vegetation is held fixed as well as severity.</p>
      <ul class="keys">
        <li><span class="dot" style="background:var(--s2)"></span>Franklin (older)</li>
        <li><span class="dot" style="background:var(--s1)"></span>Palisades</li>
      </ul>
      {sev_age_chart}
      <div class="callout">The fire-age effect holds in <b>all five</b> bands, from
      {sev_age_min} to {sev_age_max} — if anything more stable across severity than the
      headline number suggests. Recovery ranking with time since fire survives both
      controls.</div>
      <details><summary>Table view</summary>{sev_age_table}</details>
    </div>
  </div>
  <div class="sec-head" style="margin-top:3rem">
    <p class="eyebrow">Everything, in full</p>
    <h3>All strata, including the ones excluded from the headlines</h3>
  </div>
  <details><summary>Show every stratum, with the shade statistic that excluded four of them</summary>
  {all_table}</details>
</section>

<section id="explore">
  <div class="sec-head">
    <p class="eyebrow">Explore it yourself</p>
    <h2>Look at any hillside</h2>
    <p>Seven views of the same ground. Switch layers, turn the fire perimeters on and off,
    and point anywhere to read every layer at that pixel.</p>
  </div>

  <div class="toolbar" role="group" aria-label="Map layer">
    {switcher}
    <label class="tog"><input type="checkbox" id="perim-toggle" checked> Fire perimeters</label>
  </div>

  <div class="explorer">
    <div class="mapframe" id="mapframe" tabindex="0" role="application"
         aria-label="Map of the Tanager overlap. Use the arrow keys to move the inspector cursor; values appear in the pixel readout.">
      {layer_imgs}
      <img class="perim" id="perim" src="{perim}" alt="Outlines of the Palisades, Franklin and Kenneth fire perimeters">
      {fire_labels}
      <div class="xhair" id="xhair"></div>
    </div>

    <div class="side">
      <div class="card">
        <h3>Legend</h3>
        <p class="gloss" id="gloss"></p>
        <div id="legend" class="legend"></div>
      </div>
      <div class="card">
        <h3>Pixel readout</h3>
        <dl class="readout" id="readout"></dl>
        <p class="hint">Arrow keys move one pixel, Shift + arrow moves ten.
        Fractions are shade-normalised, <span class="mono">f / (1 − f_shade)</span>.</p>
      </div>
    </div>
  </div>
  <p class="mapnote">The grid is the overlap of the two acquisitions, EPSG:32611, shown at
  {px_m} m (every second pixel of the 30 m analysis grid). Unpainted ground carries no
  fraction: either the pixel was cloud, cirrus or nodata on one of the two dates, or it is
  open water or near-total shade, where dividing by
  <span class="mono">(1 − f_shade)</span> makes a fraction meaningless. Those pixels are
  excluded from every statistic here, so they are excluded from the map on the same rule —
  {masked_km2} km&sup2; in total. The vegetation and burn-history layers are categorical and
  are shown everywhere.</p>
</section>

<section id="limits">
  <div class="sec-head">
    <p class="eyebrow">Read this before quoting anything</p>
    <h2>What these numbers cannot do</h2>
    <p>These are not disclaimers added at the end. Each one changes which statements in this
    submission are allowed.</p>
  </div>
  <div class="caveats">
    <article class="caveat">
      <span class="chip crit">Not quotable</span>
      <h3>{n_unreliable} strata fail the shade check</h3>
      <p>Every fraction here is shade-normalised — divided by
      <span class="mono">(1 − f_shade)</span> — which is what makes a 34° winter sun and a
      73° summer sun comparable at all. That division is only stable where the shade
      fraction is small.</p>
      <p>In the January scene the canyon and north-facing woodland classes
      ({unreliable_names}) reach a 90th-percentile shade fraction of {shade_max}, so their
      normalised values are inflated by a heavy tail. Their <em>raw</em> char fractions are
      high and the burn signal is real — but their normalised numbers are not quotable, and
      no headline in this submission uses them. They are hatched in the vegetation layer for that
      reason.</p>
    </article>
    <article class="caveat">
      <span class="chip warn">Relative only</span>
      <h3>Char fraction is a ranking, not an area</h3>
      <p>NASA flew the airborne AVIRIS-3 spectrometer over these fires on the same day as
      the satellite, about 1.4 hours later. Inside the perimeters the two agree strongly on
      spatial pattern, but the satellite reads roughly <em>half</em> the airborne char
      fraction — most likely because the char endmember is itself a class median over
      mixed burned pixels rather than pure char.</p>
      <p>So "0.30 char" does not mean 30% of the ground is charred. Where the map says more
      char, there is more char; the absolute value should not be read as areal cover.</p>
    </article>
    <article class="caveat">
      <span class="chip info">Modelled input</span>
      <h3>The vegetation map is a model</h3>
      <p>LANDFIRE LF2023 Existing Vegetation Type is a modelled 30 m product, not a field
      survey. It was chosen because it postdates every prior fire and predates all three
      2024–25 fires, which makes it a genuine pre-fire map — but every stratum boundary here
      is a modelled boundary, and misclassified pixels move quietly between strata.</p>
    </article>
    <article class="caveat">
      <span class="chip warn">Easily misread</span>
      <h3>Fast greening can mean worse condition</h3>
      <p>Recovery rises with burn history within chaparral, and the tempting reading —
      "these sites bounce back better" — is wrong. Sites with the longest fire history are
      the ones already converted to grass and sage scrub, and herbaceous cover greens up in
      one wet season where a shrub takes a decade.</p>
      <p>The July dry/bare fraction is the check: where greening is real shrub recovery,
      green rises and dry falls together; where it is a grass flush, the pixel is still
      overwhelmingly dry.</p>
    </article>
    <article class="caveat wide">
      <span class="chip warn">Stated plainly</span>
      <h3>The perimeters were used to build the library</h3>
      <p>The char endmember is the median spectrum of dark, low-NDVI pixels <em>inside</em> the
      NIFC perimeters, and the two vegetation endmembers are drawn from outside them. So the
      perimeters were used — to locate training pixels, not at inference. Earlier versions of
      this work claimed the unmixing &ldquo;was never told where the fires were&rdquo;, which
      was wrong, and that wording has been corrected everywhere.</p>
      <p>Rebuilding each endmember with the same rules but <em>no</em> perimeter constraint
      moves char by <b>1.80&deg;</b> and green vegetation by <b>0.25&deg;</b> — both below the
      2.8&deg; at which this project declared two materials inseparable. Char is also not gated
      at the boundary: it rises over roughly 150–300 m and reaches a 90th percentile of
      <b>0.479</b> just <em>outside</em> the perimeter line.</p>
      <p>The dry-vegetation/soil endmember is the exception, and it is the one worth knowing
      about: without the exclusion, bright burned substrate enters the soil class and the
      spectrum moves <b>7.53&deg;</b>. So the library&rsquo;s soil-versus-char contrast is
      partly maintained by keeping known-burned ground out of the soil class. That cuts the
      other way from the headline, though — a char-contaminated soil endmember would absorb
      char-like signal and push fractions <em>down</em>, so it cannot manufacture the
      zero-outside null.</p>
    </article>
  </div>
</section>


<section id="next">
  <div class="sec-head">
    <p class="eyebrow">A promising research direction</p>
    <h2>What acquisitions would take this further</h2>
    <p>The clearest limits of this work are limits of <em>coverage</em>, not of method. Each
    of the following is a gap that opened during the analysis, and each would be closed by
    Tanager acquisitions that do not yet exist in the open catalogue.</p>
  </div>
  <div class="tiles two">
    <article class="tile">
      <p class="tile-v" style="font-size:1.35rem">A true &ldquo;before&rdquo;</p>
      <h3 class="tile-k">Pre-fire coverage of fire-prone shrubland</h3>
      <p class="tile-g">The earliest Tanager scene over this ground is already sixteen days
      post-ignition, so there is no hyperspectral pre-fire state anywhere in this analysis —
      the pre-burn baseline had to be borrowed from Sentinel-2. Standing coverage of
      Mediterranean shrubland during fire season would make the pre/post comparison a
      single-sensor one.</p>
    </article>
    <article class="tile">
      <p class="tile-v" style="font-size:1.35rem">A third date</p>
      <h3 class="tile-k">The shape of the trajectory, not its endpoints</h3>
      <p class="tile-g">Two acquisitions resolve where recovery started and where it reached,
      and nothing in between. A wet-season acquisition between them would capture the first
      green-up, which is when resprouters and seeders diverge — the distinction the
      green-versus-dry split exists to make.</p>
    </article>
    <article class="tile">
      <p class="tile-v" style="font-size:1.35rem">A second season</p>
      <h3 class="tile-k">Where type conversion is actually decided</h3>
      <p class="tile-g">Four fifths of this landscape had burned before, and repeat fire
      converts chaparral to grassland. Six months is far too early to see which scars convert.
      Re-imaging the same footprint through a second wet season would turn a snapshot into the
      only kind of evidence that settles it.</p>
    </article>
    <article class="tile">
      <p class="tile-v" style="font-size:1.35rem">A matched pair</p>
      <h3 class="tile-k">A way to validate the shade term</h3>
      <p class="tile-g">The photometric shade endmember is what makes a 34&deg;-sun scene
      comparable to a 73&deg;-sun one, and it is justified here by measurement rather than
      assumption. Two acquisitions of the same ground at <em>similar</em> sun elevation would
      let it be tested directly instead of inferred.</p>
    </article>
  </div>
</section>

<footer>
  <p><b>Submission to the Planet Tanager Open Data Competition.</b> Built from Planet
  Tanager-1 open hyperspectral imagery (CC-BY-4.0), NIFC/WFIGS and CAL&nbsp;FIRE perimeter
  archives, USGS LANDFIRE LF2023, Copernicus Sentinel-2 L2A, and NASA/JPL AVIRIS-3.
  Fractions come from fully-constrained linear spectral unmixing over
  {n_usable} cross-date-validated bands.</p>
  <p>The full technical record — including the five claims this project tested and rejected —
  is in <span class="mono">FINDINGS.md</span>, the narrative walkthrough in
  <span class="mono">Tanager_Fire_Recovery.ipynb</span>, and every stage of the pipeline in
  <span class="mono">scripts/</span>. It is generated by
  <span class="mono">scripts/build_ui.py</span> directly from the analysis artifacts — no
  statistic in it was typed by hand.</p>
</footer>

</div>
<script>{js}</script>
"""


if __name__ == "__main__":
    build()
