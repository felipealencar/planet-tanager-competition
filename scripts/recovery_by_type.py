"""Recovery stratified by pre-fire vegetation type, and by how often the site has burned.

analyze.py measures each fire against one pooled unburned control. That comparison is
confounded, and vegtype.py shows exactly how: the two large fires burned close to
opposite vegetation.

    Palisades   13.5 km2 chaparral   3.1 km2 coastal scrub
    Franklin     3.5 km2 chaparral   6.7 km2 coastal scrub

Those two communities also have very different *unburned* seasonal trajectories over the
same six months -- chaparral drops about 0.16 in GV fraction from January to July while
coastal scrub drops about 0.07 -- so a single pooled control subtracts the wrong baseline
from at least one of the fires. This module replaces it with a per-type control: every
burned stratum is compared against unburned pixels of the *same* EVT class inside the
same overlap, which is the comparison the original claim needed.

The result is that the age ordering survives. Franklin still recovers more than Palisades
within each vegetation type separately, so "recovery ranks with time since fire" is no
longer confounded with what was growing there -- it is now controlled for it.

Two further questions this makes askable:

  1. **Does burn history predict what is growing there now?** Measured on land that did
     *not* burn in 2024-25, so the answer is about the legacy of past fire rather than
     about the current scars. This reproduces the documented chaparral -> coastal sage
     scrub -> exotic grassland type-conversion sequence from two independent public
     datasets.
  2. **Does burn history predict recovery rate?** Within chaparral the answer is a clean
     monotonic yes across the four strata that pass the shade check (dGV +0.059 at 1
     prior fire rising to +0.220 at 4; the 0-prior stratum continues the trend downward
     at +0.042 but has p90 shade 0.332 and only 315 px, so it is plotted and flagged
     rather than quoted). It must not
     be read as "reburned chaparral recovers better". Sites that have burned repeatedly
     are largely already type-converted, so what is recovering fast there is herbaceous
     cover, not shrubs -- which is precisely why the GV/NPV split matters and a broadband
     greenness index would draw the wrong conclusion here.

**A shade caveat that bounds which strata are quotable.** Shade normalization divides by
(1 - f_shade), which is stable only where the shade fraction is small. In the January
scene at 34 degrees sun elevation that holds for chaparral and coastal scrub (median
shade 0.000) but not for the canyon and north-facing woodland classes, where the 90th
percentile shade reaches 0.60. Those classes carry a genuinely high *raw* char fraction
(~0.50 before normalization, so the burn signal is real), but their normalized values are
inflated by a heavy shade tail. Every stratum therefore records its own shade statistics
and a `shade_reliable` flag, and the headline comparisons are restricted to chaparral and
coastal scrub.

All fractions are shade-normalized, f / (1 - f_shade), for the same reason as analyze.py:
that is the quantity that can legitimately be compared between a 34-degree-sun January
scene and a 73-degree-sun July scene.

Outputs:
  data/recovery_by_type.json        every statistic below, for the notebook and the UI
  data/quicklook/recovery_by_type.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

# Only force the headless backend when run as a script; importing this module
# from a notebook must not clobber the inline backend.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")

from analyze import IGNITION, per_fire_masks
from vegtype import CHAPARRAL, COASTAL_SCRUB, EXOTIC, NATURAL, load

MIN_PX = 300  # below this a stratum is reported but not interpreted
SHADE_P90_MAX = 0.30  # above this, shade normalization is not trustworthy
# January char bins used as a severity control. Vegetation type and burn severity are
# tangled -- chaparral both burned hotter and recovered differently -- so an effect is
# only attributable to vegetation if it survives inside a fixed severity band.
CHAR_BINS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01)]
OUT_JSON = pathlib.Path("data/recovery_by_type.json")


def fractions():
    d = np.load("data/fractions.npz")
    names = [str(n) for n in d["names"]]
    idx = {n: names.index(n) for n in names}
    norm = {}
    for k in ("jan23", "jul26"):
        frac = d[f"frac_{k}"]
        norm[k] = frac / np.clip(1.0 - frac[idx["SHADE"]], 1e-3, None)
    origin = d["origin"]
    nrow, ncol = d["frac_jan23"].shape[1:]
    shade_jan = d["frac_jan23"][idx["SHADE"]]
    return norm, idx, origin, nrow, ncol, shade_jan


def summarize(mask, norm, idx, dgv, control_dgv=None, shade=None):
    """Median statistics for one stratum, with a shade-reliability flag.

    Shade normalization divides by (1 - f_shade), so a stratum with a heavy shade tail
    has inflated normalized fractions even when its raw fractions are sound. p90 shade
    above SHADE_P90_MAX marks the stratum as not quotable.
    """
    n = int(mask.sum())
    if n == 0:
        return None
    out = {
        "n_px": n,
        "km2": round(n * 900 / 1e6, 2),
        "char_jan": round(float(np.median(norm["jan23"][idx["CHAR"]][mask])), 3),
        "gv_jan": round(float(np.median(norm["jan23"][idx["GV"]][mask])), 3),
        "gv_jul": round(float(np.median(norm["jul26"][idx["GV"]][mask])), 3),
        "npv_jul": round(float(np.median(norm["jul26"][idx["NPV_SOIL"]][mask])), 3),
        "dgv": round(float(np.median(dgv[mask])), 3),
        "interpretable": n >= MIN_PX,
    }
    if shade is not None:
        p90 = float(np.percentile(shade[mask], 90))
        out["shade_med"] = round(float(np.median(shade[mask])), 3)
        out["shade_p90"] = round(p90, 3)
        out["shade_reliable"] = bool(p90 <= SHADE_P90_MAX)
    if control_dgv is not None:
        out["dgv_vs_control"] = round(out["dgv"] - control_dgv, 3)
    return out


def severity_control(evt, masks, any_fire, valid, dgv, norm, idx, controls):
    """Do the vegetation and fire-age effects survive at fixed burn severity?

    Both comparisons are made *within* a January char bin, so the two groups being
    compared burned equally hard. That also cancels any January floor effect: whatever
    the starting GV is, it is the same on both sides of the comparison.
    """
    char = norm["jan23"][idx["CHAR"]]
    out = {"veg_effect": [], "age_effect": []}

    print("\nSeverity control 1 -- does the VEGETATION effect survive at fixed severity?")
    print(f"{'char bin':>14}{'chaparral':>12}{'coastal scrub':>15}{'difference':>12}")
    for lo, hi in CHAR_BINS:
        row = {"char_lo": lo, "char_hi": hi}
        for code, key in ((CHAPARRAL, "chaparral"), (COASTAL_SCRUB, "coastal_scrub")):
            m = valid & any_fire & (evt == code) & (char >= lo) & (char < hi)
            row[f"{key}_n"] = int(m.sum())
            row[key] = (round(float(np.median(dgv[m]) - controls[code]), 3)
                        if m.sum() >= MIN_PX else None)
        if row["chaparral"] is None or row["coastal_scrub"] is None:
            continue
        row["difference"] = round(row["chaparral"] - row["coastal_scrub"], 3)
        out["veg_effect"].append(row)
        print(f"  [{lo:.2f},{hi:.2f}){row['chaparral']:12.3f}"
              f"{row['coastal_scrub']:15.3f}{row['difference']:12.3f}")

    print("\nSeverity control 2 -- does the FIRE-AGE effect survive? (chaparral only)")
    print(f"{'char bin':>14}{'Palisades':>12}{'Franklin':>12}{'difference':>12}")
    for lo, hi in CHAR_BINS:
        row = {"char_lo": lo, "char_hi": hi}
        for fire in ("PALISADES", "Franklin"):
            if fire not in masks:
                continue
            m = valid & masks[fire] & (evt == CHAPARRAL) & (char >= lo) & (char < hi)
            row[f"{fire}_n"] = int(m.sum())
            row[fire] = (round(float(np.median(dgv[m]) - controls[CHAPARRAL]), 3)
                         if m.sum() >= 150 else None)
        if row.get("PALISADES") is None or row.get("Franklin") is None:
            continue
        row["difference"] = round(row["Franklin"] - row["PALISADES"], 3)
        out["age_effect"].append(row)
        print(f"  [{lo:.2f},{hi:.2f}){row['PALISADES']:12.3f}"
              f"{row['Franklin']:12.3f}{row['difference']:12.3f}")

    # Absolute July greenness against January severity. Reported because the obvious
    # objection -- that dGV rises with severity only because severely burned pixels
    # start near zero -- does not hold: GV_jan is 0.000 in every bin, so the trend is
    # in the July state itself, not in the baseline.
    print("\nSecondary: absolute July GV against January severity (chaparral)")
    print(f"{'char bin':>14}{'n':>8}{'GV_jan':>9}{'GV_jul':>9}")
    grad = []
    for lo, hi in CHAR_BINS:
        m = valid & any_fire & (evt == CHAPARRAL) & (char >= lo) & (char < hi)
        if m.sum() < MIN_PX:
            continue
        row = {
            "char_lo": lo, "char_hi": hi, "n_px": int(m.sum()),
            "gv_jan": round(float(np.median(norm["jan23"][idx["GV"]][m])), 3),
            "gv_jul": round(float(np.median(norm["jul26"][idx["GV"]][m])), 3),
            "gv_jul_zero_frac": round(float((norm["jul26"][idx["GV"]][m] == 0).mean()), 3),
        }
        grad.append(row)
        print(f"  [{lo:.2f},{hi:.2f}){row['n_px']:8d}{row['gv_jan']:9.3f}{row['gv_jul']:9.3f}")
    out["july_gv_vs_severity"] = grad
    print("  Caveat: FCLS is zero-inflated, so low-bin medians sit on a floor of exact")
    print("  zeros; gv_jul_zero_frac records how much of each bin that is.")
    return out


def main():
    evt, prior, lut = load()
    norm, idx, origin, nrow, ncol, shade_jan = fractions()
    masks, any_fire = per_fire_masks(nrow, ncol, origin)

    valid = np.isfinite(norm["jan23"][idx["GV"]]) & np.isfinite(norm["jul26"][idx["GV"]])
    dgv = norm["jul26"][idx["GV"]] - norm["jan23"][idx["GV"]]
    unburned = valid & ~any_fire

    report = {
        "grid": {"nrow": nrow, "ncol": ncol, "origin": [float(origin[0]), float(origin[1])]},
        "min_px": MIN_PX,
    }

    # ---------------------------------------------------------------- controls
    print("Per-type unburned control (the seasonal baseline each fire must beat):")
    controls = {}
    for code, label in NATURAL.items():
        m = unburned & (evt == code)
        if m.sum() < MIN_PX:
            continue
        controls[code] = float(np.median(dgv[m]))
        print(f"  {label:28s} n={int(m.sum()):6d}   dGV = {controls[code]:+.3f}")
    pooled = float(np.median(dgv[unburned]))
    print(f"  {'POOLED (what analyze.py used)':28s} n={int(unburned.sum()):6d}   dGV = {pooled:+.3f}")
    report["controls"] = {
        NATURAL[c]: {"dgv": round(v, 3), "n_px": int((unburned & (evt == c)).sum())}
        for c, v in controls.items()
    }
    report["controls_pooled"] = {"dgv": round(pooled, 3), "n_px": int(unburned.sum())}

    # ------------------------------------------------- stratified recovery
    print("\nRecovery by fire x vegetation type, against the SAME-TYPE control:")
    header = f"{'fire':11s}{'vegetation type':28s}{'km2':>7}{'dGV':>9}{'vs ctrl':>10}{'charJan':>9}"
    print(header)
    strata = []
    for fire, fmask in masks.items():
        for code, label in NATURAL.items():
            if code not in controls:
                continue
            m = valid & fmask & (evt == code)
            s = summarize(m, norm, idx, dgv, controls[code], shade_jan)
            if s is None or s["n_px"] < MIN_PX:
                continue
            s.update(fire=fire, evt_code=code, evt_name=label,
                     ignition=IGNITION.get(fire))
            strata.append(s)
            flag = "" if s["shade_reliable"] else "   <- shade p90 %.2f, not quotable" % s["shade_p90"]
            print(f"{fire:11s}{label:28s}{s['km2']:7.1f}{s['dgv']:+9.3f}"
                  f"{s['dgv_vs_control']:+10.3f}{s['char_jan']:9.3f}{flag}")
    report["strata"] = strata

    # the confound, stated as a table
    print("\nWhy the pooled control was invalid -- burned area by type (km2):")
    codes = [CHAPARRAL, COASTAL_SCRUB]
    print(f"{'fire':11s}" + "".join(f"{NATURAL[c]:>26s}" for c in codes))
    composition = {}
    for fire, fmask in masks.items():
        row = {NATURAL[c]: round(float((valid & fmask & (evt == c)).sum() * 900 / 1e6), 2)
               for c in codes}
        composition[fire] = row
        print(f"{fire:11s}" + "".join(f"{row[NATURAL[c]]:26.1f}" for c in codes))
    report["burned_composition"] = composition

    # --------------------------------------------- fire history -> veg type
    print("\nDoes burn history predict what grows there now? (land unburned in 2024-25)")
    conv = []
    for k in range(0, int(prior.max()) + 1):
        m = unburned & (prior == k)
        if m.sum() < MIN_PX:
            continue
        tot = int(m.sum())
        row = {
            "prior_fires": k,
            "n_px": tot,
            "chaparral": round(float((evt[m] == CHAPARRAL).mean()), 4),
            "coastal_scrub": round(float((evt[m] == COASTAL_SCRUB).mean()), 4),
            "exotic": round(float(np.isin(evt[m], list(EXOTIC)).mean()), 4),
        }
        conv.append(row)
        print(f"  {k} prior fires (n={tot:6d}):  chaparral {row['chaparral']:6.1%}"
              f"   coastal scrub {row['coastal_scrub']:6.1%}"
              f"   exotic {row['exotic']:6.1%}")
    report["type_conversion"] = conv

    # ------------------------------------------ fire history -> recovery
    print("\nDoes burn history predict recovery rate? (burned 2024-25, within type)")
    freq = []
    for code in (CHAPARRAL, COASTAL_SCRUB):
        for k in range(0, int(prior.max()) + 1):
            m = valid & any_fire & (evt == code) & (prior == k)
            s = summarize(m, norm, idx, dgv, controls.get(code), shade_jan)
            if s is None or s["n_px"] < MIN_PX:
                continue
            s.update(evt_code=code, evt_name=NATURAL[code], prior_fires=k)
            freq.append(s)
            print(f"  {NATURAL[code]:24s} {k} prior  n={s['n_px']:6d}"
                  f"  dGV={s['dgv']:+.3f}  julGV={s['gv_jul']:.3f}  julNPV={s['npv_jul']:.3f}")
    report["recovery_vs_fire_frequency"] = freq
    print("\n  Monotonic within chaparral, but reburned sites are largely already")
    print("  type-converted, so this is herbaceous flush rather than shrub recovery.")

    # ------------------------------------------------- severity control
    low_shade = valid & (shade_jan < SHADE_P90_MAX)
    sev_controls = {
        c: float(np.median(dgv[low_shade & ~any_fire & (evt == c)]))
        for c in (CHAPARRAL, COASTAL_SCRUB)
    }
    report["severity_control"] = severity_control(
        evt, masks, any_fire, low_shade, dgv, norm, idx, sev_controls
    )
    report["severity_control"]["controls"] = {
        NATURAL[c]: round(v, 3) for c, v in sev_controls.items()
    }

    # ---------------------------------------------------------------- maps
    shade_both = {k: np.load("data/fractions.npz")[f"frac_{k}"][idx["SHADE"]]
                  for k in ("jan23", "jul26")}
    report["maps"] = write_maps(evt, prior, masks, valid, any_fire, dgv, norm, idx,
                                shade_both)

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_JSON}")
    plot(report)


OPEN_WATER = 7292  # LANDFIRE EVT class
SHADE_MAX_DISPLAY = 0.90


def write_maps(evt, prior, masks, valid, any_fire, dgv, norm, idx, shade):
    """Downsampled arrays the UI can render without shipping the full cubes.

    Fraction layers are masked wherever a shade-normalised fraction is not a meaningful
    quantity: open water, and anything whose shade fraction on either date approaches 1.
    Dividing by (1 - f_shade) on an almost-pure-shade surface inflates whatever material
    fraction survives it -- which is why the ocean would otherwise read as the brightest
    char in the scene. Those pixels are already excluded from every published statistic,
    so masking them here keeps the map, the pixel readout and the numbers on one rule
    instead of showing a value the text then has to explain away.

    The categorical layers are left intact: open water is a legitimate vegetation class
    and a legitimate burn-history value, and neither is a shade-normalised quantity.
    """
    step = 2
    sl = (slice(None, None, step), slice(None, None, step))
    meaningless = (evt == OPEN_WATER)
    for k in ("jan23", "jul26"):
        meaningless |= ~np.isfinite(shade[k]) | (shade[k] >= SHADE_MAX_DISPLAY)
    show = valid & ~meaningless

    out = pathlib.Path("data/ui_layers.npz")
    np.savez_compressed(
        out,
        evt=evt[sl].astype(np.uint16),
        prior_fires=prior[sl].astype(np.int8),
        dgv=np.where(show, dgv, np.nan)[sl].astype(np.float32),
        char_jan=np.where(show, norm["jan23"][idx["CHAR"]], np.nan)[sl].astype(np.float32),
        gv_jul=np.where(show, norm["jul26"][idx["GV"]], np.nan)[sl].astype(np.float32),
        npv_jul=np.where(show, norm["jul26"][idx["NPV_SOIL"]], np.nan)[sl].astype(np.float32),
        burned=any_fire[sl],
    )
    masked_km2 = float((valid & meaningless).sum() * 900 / 1e6)
    print(f"wrote {out} (stride {step}); fraction layers masked over "
          f"{masked_km2:.0f} km2 of water and near-total shade")
    return {"file": str(out), "stride": step, "masked_km2": round(masked_km2, 1)}


def plot(report):
    strata = [s for s in report["strata"] if s["evt_code"] in (CHAPARRAL, COASTAL_SCRUB)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    fires = sorted({s["fire"] for s in strata})
    width = 0.36
    for j, code in enumerate((CHAPARRAL, COASTAL_SCRUB)):
        vals = [next((s["dgv_vs_control"] for s in strata
                      if s["fire"] == f and s["evt_code"] == code), np.nan) for f in fires]
        axes[0].bar(np.arange(len(fires)) + (j - 0.5) * width, vals, width,
                    label=NATURAL[code], color=["#16a34a", "#f97316"][j])
    axes[0].set_xticks(range(len(fires)))
    axes[0].set_xticklabels(fires, fontsize=9)
    axes[0].set_ylabel("$\\Delta$GV vs same-type control")
    axes[0].set_title("Recovery, stratified")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis="y")

    conv = report["type_conversion"]
    ks = [c["prior_fires"] for c in conv]
    for key, color, label in [("chaparral", "#16a34a", "chaparral"),
                              ("coastal_scrub", "#f97316", "coastal scrub"),
                              ("exotic", "#dc2626", "exotic/ruderal")]:
        axes[1].plot(ks, [100 * c[key] for c in conv], "o-", color=color, label=label)
    axes[1].set_xlabel("fires 1980-2023")
    axes[1].set_ylabel("% of unburned land")
    axes[1].set_title("Type conversion with fire frequency")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    for code, color in ((CHAPARRAL, "#16a34a"), (COASTAL_SCRUB, "#f97316")):
        rows = [r for r in report["recovery_vs_fire_frequency"] if r["evt_code"] == code]
        if rows:
            axes[2].plot([r["prior_fires"] for r in rows], [r["dgv"] for r in rows],
                         "o-", color=color, label=NATURAL[code])
    axes[2].set_xlabel("prior fires")
    axes[2].set_ylabel("$\\Delta$GV")
    axes[2].set_title("Recovery vs burn history (weak)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    pathlib.Path("data/quicklook").mkdir(parents=True, exist_ok=True)
    plt.savefig("data/quicklook/recovery_by_type.png", dpi=115)
    print("wrote data/quicklook/recovery_by_type.png")


if __name__ == "__main__":
    main()
