"""Independent validation: Tanager char fraction vs Sentinel-2 dNBR.

This is the only fully external check in the project. Every other validation is either
internal to the Tanager cubes or against fire perimeters (which are not burn masks). Here
the comparison is against a different satellite, different optics, different processing
chain and a completely different algorithm:

    Tanager    426-band linear spectral unmixing -> char endmember fraction, 23 Jan 2025
    Sentinel-2 2-band normalized index -> dNBR, 13 Nov 2024 (pre) -> 12 Jan 2025 (post)

If the char fraction is measuring charred material, it must track dNBR. Nothing in the
unmixing was informed by Sentinel-2 or by the fire perimeters, so agreement is meaningful.

Severity classes follow the USGS/MTBS dNBR breakpoints.
"""

from __future__ import annotations

import sys

import matplotlib

# Only force the headless backend when run as a script; importing this module
# from a notebook must not clobber the inline backend.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from analyze import per_fire_masks  # noqa: E402

# USGS / MTBS standard dNBR severity breakpoints.
CLASSES = [
    ("unburned", -np.inf, 0.10, "#94a3b8"),
    ("low", 0.10, 0.27, "#fde047"),
    ("moderate-low", 0.27, 0.44, "#fb923c"),
    ("moderate-high", 0.44, 0.66, "#ef4444"),
    ("high", 0.66, np.inf, "#7f1d1d"),
]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype("f8")
    ry = np.argsort(np.argsort(y)).astype("f8")
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def main():
    fr = np.load("data/fractions.npz")
    s2 = np.load("data/sentinel2.npz", allow_pickle=True)
    names = [str(n) for n in fr["names"]]
    iCH, iGV, iSH = names.index("CHAR"), names.index("GV"), names.index("SHADE")

    frac = fr["frac_jan23"]
    illum = np.clip(1.0 - frac[iSH], 1e-3, None)
    char = frac[iCH] / illum
    dnbr = s2["dnbr_post"]

    ok = np.isfinite(char) & np.isfinite(dnbr) & np.isfinite(fr["rmse_jan23"])
    ok &= frac[iSH] < 0.9
    print(f"pixels comparable in both products: {ok.sum():,} "
          f"({ok.sum() * 900 / 1e6:.1f} km2)\n")

    c, d = char[ok], dnbr[ok]
    pear = float(np.corrcoef(c, d)[0, 1])
    spear = spearman(c, d)
    print(f"Tanager char fraction vs Sentinel-2 dNBR")
    print(f"  Pearson  r = {pear:+.3f}")
    print(f"  Spearman r = {spear:+.3f}\n")

    print(f"{'dNBR severity class':22s} {'px':>8s} {'km2':>7s} {'char p50':>9s} "
          f"{'char p90':>9s} {'GV p50':>8s}")
    gv = frac[iGV] / illum
    rows = []
    for label, lo, hi, color in CLASSES:
        m = ok & (dnbr >= lo) & (dnbr < hi)
        if m.sum() < 100:
            continue
        cc = char[m]
        rows.append((label, int(m.sum()), np.median(cc), np.percentile(cc, 90),
                     np.median(gv[m]), color))
        print(f"{label:22s} {m.sum():8,d} {m.sum() * 900 / 1e6:7.1f} "
              f"{np.median(cc):9.3f} {np.percentile(cc, 90):9.3f} {np.median(gv[m]):8.3f}")

    # Monotonic increase of char with severity class is the substantive claim.
    med = [r[2] for r in rows]
    mono = all(med[i] <= med[i + 1] for i in range(len(med) - 1))
    print(f"\nchar median increases monotonically with dNBR severity class: {mono}")

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    sub = np.random.default_rng(0).choice(np.flatnonzero(ok), size=min(40000, ok.sum()),
                                          replace=False)
    axes[0].hexbin(dnbr.ravel()[sub], char.ravel()[sub], gridsize=60, bins="log",
                   cmap="viridis", extent=(-0.3, 1.2, 0, 1))
    axes[0].set_xlabel("Sentinel-2 dNBR  (13 Nov 2024 $\\rightarrow$ 12 Jan 2025)")
    axes[0].set_ylabel("Tanager char fraction  (23 Jan 2025)")
    axes[0].set_title(f"independent agreement\nPearson r = {pear:+.3f}, "
                      f"Spearman $\\rho$ = {spear:+.3f}")

    for label, n, p50, p90, gvm, color in rows:
        axes[1].bar(label, p50, color=color, edgecolor="#333")
    axes[1].set_ylabel("median Tanager char fraction")
    axes[1].set_title("char fraction by independent dNBR severity class")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.25, axis="y")

    im = axes[2].imshow(np.where(ok, dnbr, np.nan), cmap="inferno", vmin=-0.1, vmax=1.0)
    masks, _ = per_fire_masks(*char.shape, fr["origin"])
    for m in masks.values():
        axes[2].contour(m.astype(float), levels=[0.5], colors="cyan", linewidths=0.9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title("Sentinel-2 dNBR")
    plt.colorbar(im, ax=axes[2], fraction=0.046, label="dNBR")

    plt.tight_layout()
    plt.savefig("data/quicklook/validation.png", dpi=115)
    print("wrote data/quicklook/validation.png")


if __name__ == "__main__":
    main()
