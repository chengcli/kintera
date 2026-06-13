#!/usr/bin/env python3
"""Transport-vs-chemistry diagnostic for the Titan C2 superrotation case.

Isolates the genuine 3-D signature: in a zonal jet, a short-lived radical
(CH3) stays locked to the subsolar point while a long-lived product (C2H6)
is advected downwind (eastward) and smeared in longitude -- structure a
column model cannot produce. Optionally overlays a matched no-wind control.

    VIRTUAL_ENV=~/pyenv ~/pyenv/bin/python transport_diagnostic.py \
        --run /tmp/sr_out [--control /tmp/sr_ctl_out] --out out_superrot
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
MW_AIR = 28.0134e-3


def load(run_dir):
    o2 = sorted(glob.glob(f"{run_dir}/*.out2.*.nc"))
    o1 = sorted(glob.glob(f"{run_dir}/*.out1.*.nc"))
    return xr.open_dataset(o2[-1]), xr.open_dataset(o1[-1]), o2, o1


def lonlat(ds1):
    lon = np.asarray(ds1["lon"].values, dtype=np.float64)
    lat = np.asarray(ds1["lat"].values, dtype=np.float64)
    while lon.ndim > 2:
        lon, lat = lon[0], lat[0]
    if np.abs(lon).max() > 7:
        lon, lat = np.radians(lon), np.radians(lat)
    return lon, lat


def equatorial_profile(field2d, lon, lat, nbin=72):
    """Bin a (x3,x2) field onto a longitude axis within +/-20deg of equator."""
    m = np.abs(lat) < np.radians(20.0)
    lo = (np.degrees(lon[m]) % 360.0)
    val = field2d[m]
    edges = np.linspace(0, 360, nbin + 1)
    idx = np.clip(np.digitize(lo, edges) - 1, 0, nbin - 1)
    out = np.array([val[idx == b].mean() if (idx == b).any() else np.nan
                    for b in range(nbin)])
    return 0.5 * (edges[:-1] + edges[1:]), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--control", default=None)
    ap.add_argument("--out", default=str(HERE / "out_superrot"))
    ap.add_argument("--u", type=float, default=100.0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    d = np.load(HERE / "titan_c2_data.npz")
    mw = dict(zip([str(s) for s in d["storage_species"]], d["mw"]))
    sN, d1, o2, _ = load(args.run)
    lon, lat = lonlat(d1)
    kmid = sN.sizes["x1"] // 2

    def vmr(ds, sp, k=kmid):
        return ds[f"r_{sp}"].values[0, k] * (MW_AIR / mw[sp])

    # control (no wind) once
    ctl = None
    if args.control:
        sC, c1, _, _ = load(args.control)
        lonc, latc = lonlat(c1)
        ctl = (sC, lonc, latc)

    # ---- 1. lifetime-ordered equatorial longitude profiles ----
    # fast radical (locked) -> intermediate (advected) -> long-lived (smeared)
    species = [("CH3", "fast radical"), ("H", "intermediate"),
               ("C2H6", "long-lived product")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for j, (sp, kind) in enumerate(species):
        lo, prof = equatorial_profile(vmr(sN, sp), lon, lat)
        prof = prof / np.nanmax(prof)
        ax[j].plot(lo, prof, color="tab:blue", lw=2,
                   label=f"superrot {args.u:.0f} m/s")
        if ctl is not None:
            loc, pc = equatorial_profile(ctl[0][f"r_{sp}"].values[0, kmid]
                                         * (MW_AIR / mw[sp]), ctl[1], ctl[2])
            ax[j].plot(loc, pc / np.nanmax(pc), "--", color="k", lw=1.6,
                       label="no wind")
        ax[j].axvline(0, color="orange", ls=":", lw=1.2)
        ax[j].axvline(90, color="navy", ls=":", lw=0.7)
        ax[j].axvline(270, color="navy", ls=":", lw=0.7)
        ax[j].set_title(f"{sp} ({kind})")
        ax[j].set_xlabel("longitude (deg; 0=noon, wind→+lon)")
        ax[j].set_ylabel("normalized vmr"); ax[j].set_xlim(0, 360)
        ax[j].legend(fontsize=8); ax[j].grid(alpha=0.3)
    fig.suptitle("Equatorial photochemistry vs longitude: a jet advects "
                 "products downwind while radicals stay locked to noon",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig_zonal_profiles.png", dpi=140)
    plt.close(fig)

    # ---- 2. downwind-tail map: H (intermediate lifetime) trails east of noon ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for a, (ds, lo_, la_, tag) in enumerate(
            [(sN, lon, lat, f"superrot {args.u:.0f} m/s"),
             (ctl[0], ctl[1], ctl[2], "no wind") if ctl else (sN, lon, lat, "")]):
        f = ds["r_H"].values[0, kmid] * (MW_AIR / mw["H"])
        sc = ax[a].scatter((np.degrees(lo_) % 360), np.degrees(la_),
                           c=f.ravel(), s=6, cmap="inferno",
                           norm=matplotlib.colors.LogNorm(
                               vmin=max(f.max() * 1e-3, 1e-30), vmax=f.max()))
        plt.colorbar(sc, ax=ax[a], label="H vmr")
        ax[a].axvline(0, color="cyan", ls=":"); ax[a].axvline(360, color="cyan", ls=":")
        ax[a].set_xlabel("longitude (deg; 0 = noon)"); ax[a].set_title(f"H — {tag}")
    ax[0].set_ylabel("latitude (deg)")
    fig.suptitle("Atomic H (intermediate lifetime): advected into a downwind "
                 "tail by the jet (left) vs locked to the dayside without it (right)")
    fig.tight_layout(); fig.savefig(out / "fig_comet_tail.png", dpi=140)
    plt.close(fig)

    # ---- 3. quantitative summary: night/day homogenization, wind vs no-wind ----
    def night_day(ds, sp, lo_, lat_):
        x, p = equatorial_profile(ds[f"r_{sp}"].values[0, kmid] * (MW_AIR/mw[sp]),
                                  lo_, lat_)
        day = (x < 90) | (x > 270)
        return np.nanmean(p[~day]) / max(np.nanmean(p[day]), 1e-300)
    print(f"{'species':8} {'night/day (wind)':>17} {'night/day (no wind)':>20}  interpretation")
    for sp, kind in species:
        w = night_day(sN, sp, lon, lat)
        c = night_day(ctl[0], sp, ctl[1], ctl[2]) if ctl else float('nan')
        print(f"{sp:8} {w:>17.3f} {c:>20.3f}  {kind}")
    print(f"\nwrote {out}/fig_zonal_profiles.png, fig_comet_tail.png")


if __name__ == "__main__":
    main()
