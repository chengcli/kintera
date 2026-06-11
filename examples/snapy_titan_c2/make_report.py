#!/usr/bin/env python3
"""Figures + HTML report for the 3D Titan C2 snapy case.

Run under snapy's pyenv AFTER a demo run:
    VIRTUAL_ENV=~/pyenv ~/pyenv/bin/python make_report.py \
        --run-dir /tmp/titan_c2_demo_out \
        --config ~/dev/UM-TITAN/titan/titan_c2_dry.yaml \
        --log /tmp/demo_run.log
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

HERE = Path(__file__).resolve().parent
MW_AIR = 28.0134e-3


def lonlat_mosaic(config_path: str, out_shape: tuple[int, int]):
    """(lon, lat, mu0) on the snapshot's 2x3 face mosaic, via snapy."""
    import yaml
    from snapy import Mesh, MeshOptions
    import snapy.coord as scoord
    from paddle import start_dist, close_dist
    import sys
    sys.path.insert(0, str(HERE))

    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = start_dist("nccl")
    options = MeshOptions.from_yaml(config_path)
    options.block().output_dir("/tmp/titan_c2_report_probe")
    mesh = Mesh(options)
    mesh.to(device)
    n3, n2 = out_shape
    bn3, bn2 = n3 // 2, n2 // 3            # mosaic is 2 rows x 3 cols
    lon = np.zeros(out_shape)
    lat = np.zeros(out_shape)
    p = config["problem"]
    for block in mesh.blocks:
        glob_rank = int(block.options.layout().rank())
        face_id = int(block.get_layout().loc_of(glob_rank)[-1])
        face = scoord.get_cs_face_name(face_id)
        coord = block.module("coord")
        x2v = coord.buffer("x2v")[3:-3]
        x3v = coord.buffer("x3v")[3:-3]
        beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
        lo, la = scoord.cs_ab_to_lonlat(face, alpha, beta)
        r, c = divmod(glob_rank, 3)
        lon[r * bn3:(r + 1) * bn3, c * bn2:(c + 1) * bn2] = lo.cpu().numpy()
        lat[r * bn3:(r + 1) * bn3, c * bn2:(c + 1) * bn2] = la.cpu().numpy()
    mu0 = (np.sin(lat) * np.sin(np.radians(p.get("subsolar_lat_deg", 0.0)))
           + np.cos(lat) * np.cos(np.radians(p.get("subsolar_lat_deg", 0.0)))
           * np.cos(lon - np.radians(p.get("subsolar_lon_deg", 0.0))))
    close_dist()
    return lon, lat, mu0


def vmr(ds, name, mw):
    return ds[f"r_{name}"].values * (MW_AIR / mw)


def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(HERE / "titan_c2_data.npz")
    species = [str(s) for s in d["storage_species"]]
    mw = dict(zip(species, d["mw"]))

    files = sorted(glob.glob(f"{args.run_dir}/*.out2.*.nc"))
    ds0, dsN = xr.open_dataset(files[0]), xr.open_dataset(files[-1])
    n3, n2 = ds0.sizes["x3"], ds0.sizes["x2"]
    lon, lat, mu0 = lonlat_mosaic(args.config, (n3, n2))
    lon_d, lat_d = np.degrees(lon), np.degrees(lat)
    ktop = ds0.sizes["x1"] - 1            # top interior layer
    kmid = ds0.sizes["x1"] // 2

    # ---- 1. initial 1D profiles (snapshot 0, horizontal mean) ----
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    x1 = ds0["x1"].values / 1e3 - 2575.0  # km altitude above surface
    for sp, color in (("CH4", "k"), ("C2H2", "tab:blue"), ("C2H4", "tab:cyan"),
                      ("C2H6", "tab:green"), ("CH3", "tab:red"), ("H", "tab:orange")):
        prof = vmr(ds0, sp, mw[sp])[0].mean(axis=(1, 2))
        ax.semilogx(np.maximum(prof, 1e-30), x1, label=sp, color=color)
    ax.set_xlim(1e-14, 1e-1)
    ax.set_xlabel("vmr")
    ax.set_ylabel("altitude (km)")
    ax.set_title("Initial condition (1D photochemical relaxation)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _savefig(fig, out / "fig_init1d.png")

    # ---- 2. day/night terminator maps (final snapshot) ----
    for sp, level, tag in (("CH3", ktop, "top"), ("H", ktop, "top"),
                           ("C2H2", kmid, "mid"), ("C2H6", kmid, "mid")):
        field = vmr(dsN, sp, mw[sp])[0, level]
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        sc = ax.scatter(lon_d.ravel(), lat_d.ravel(), c=field.ravel(),
                        s=4, cmap="inferno",
                        norm=matplotlib.colors.LogNorm(
                            vmin=max(field.max() * 1e-3, 1e-30),
                            vmax=max(field.max(), 1e-29)))
        cs = ax.tricontour(lon_d.ravel(), lat_d.ravel(), mu0.ravel(),
                           levels=[0.05], colors="cyan", linewidths=1.2)
        plt.colorbar(sc, ax=ax, label=f"{sp} vmr")
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)")
        ax.set_title(f"{sp} at {tag} level — final state (cyan = terminator)")
        _savefig(fig, out / f"fig_map_{sp}.png")

    # ---- 3. photochemical growth time series ----
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    times, series = [], {sp: [] for sp in ("C2H2", "C2H4", "C2H6", "CH3", "H2")}
    for f in files:
        ds = xr.open_dataset(f)
        times.append(float(ds["time"].values[0]) / 86400.0)
        for sp in series:
            series[sp].append(float(vmr(ds, sp, mw[sp])[0].mean()))
        ds.close()
    for sp, v in series.items():
        ax.semilogy(times, v, label=sp, marker="o", ms=3)
    ax.set_xlabel("model time (days)")
    ax.set_ylabel("global-mean vmr")
    ax.set_title("Photochemical growth (chem_accel applies to chemistry only)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _savefig(fig, out / "fig_growth.png")

    # ---- 4. CPU vs GPU step benchmark ----
    bench = json.load(open(HERE / "bench_step.json"))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    labels = ["RT (py2sess)", "chemistry (kintera)", "total"]
    cpu = [bench["cpu"]["rt"], bench["cpu"]["chem"],
           bench["cpu"]["rt"] + bench["cpu"]["chem"]]
    gpu = [bench["cuda"]["rt"], bench["cuda"]["chem"],
           bench["cuda"]["rt"] + bench["cuda"]["chem"]]
    x = np.arange(3)
    ax.bar(x - 0.18, cpu, 0.36, label="CPU", color="tab:gray")
    ax.bar(x + 0.18, gpu, 0.36, label="GPU (H100)", color="tab:green")
    for i in range(3):
        ax.text(x[i] + 0.18, gpu[i] * 1.15, f"{cpu[i]/gpu[i]:.0f}×",
                ha="center", color="tab:green", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("ms per block step (2916 cols × 40 lyr × 176 λ)")
    ax.set_title("RT + chemistry step: CPU vs GPU")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _savefig(fig, out / "fig_bench.png")

    # ---- 5. conservation from the run log ----
    cons_txt = ""
    if args.log and Path(args.log).exists():
        mass = [float(m) for m in re.findall(r"mass0=([\d.eE+-]+)",
                                             Path(args.log).read_text())]
        if len(mass) > 2:
            drift = abs(mass[-1] - mass[0]) / mass[0]
            cons_txt = f"total-mass drift over the run: {drift:.2e} (relative)"

    # ---- HTML ----
    speed = (bench["cpu"]["rt"] + bench["cpu"]["chem"]) / (
        bench["cuda"]["rt"] + bench["cuda"]["chem"])
    figs = [
        ("fig_init1d.png", "Initial condition: 1D photochemical column relaxation of the moses05 profile, tiled over the sphere."),
        ("fig_map_CH3.png", "CH₃ radical at the top level — tracks instantaneous photolysis; sharp day/night contrast across the terminator (cyan)."),
        ("fig_map_H.png", "Atomic H at the top level — the other fast photolysis product."),
        ("fig_map_C2H2.png", "C₂H₂ at mid level — accumulating photochemical product."),
        ("fig_map_C2H6.png", "C₂H₆ at mid level — the main CH₃+CH₃ recombination product."),
        ("fig_growth.png", "Global-mean photochemical growth over the run."),
        ("fig_bench.png", "Per-step RT+chemistry wall time, CPU vs GPU."),
    ]
    cards = "".join(
        f'<div class="card"><img src="{f}"><p>{c}</p></div>' for f, c in figs)
    rows = [
        ("grid", f"6 faces × 48×48 × 40 layers ({6*48*48*40:,} cells, 16 scalars)"),
        ("chemistry", "moses05.C2: 42 arrhenius + 4 falloff + 11 KB-custom + 16 photolysis branches"),
        ("RT", "py2sess forward_flux (native, torch), 176 λ-bins, geometry-folded pure absorption"),
        ("GPU speedup (step)", f"RT {bench['cpu']['rt']/bench['cuda']['rt']:.0f}× · chem {bench['cpu']['chem']/bench['cuda']['chem']:.0f}× · total {speed:.0f}×"),
    ]
    if cons_txt:
        rows.append(("conservation", cons_txt))
    table = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>3D Titan C2 photochemistry — snapy × kintera × py2sess</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1419;color:#e6e6e6}}
header{{padding:24px 32px;background:#161b22;border-bottom:1px solid #30363d}}
h1{{margin:0;font-size:21px}} .sub{{color:#9aa4af;font-size:13px;margin-top:6px}}
main{{padding:24px 32px;max-width:1200px;margin:auto}}
.box{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;margin:0 0 22px}}
table{{border-collapse:collapse;font-size:14px}} td{{padding:4px 18px 4px 0}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px}}
.card img{{width:100%;border-radius:6px;background:#fff}} .card p{{font-size:13px;color:#b8c0c8;margin:8px 2px 2px}}
b{{color:#7ee787}}
</style></head><body>
<header><h1>3D cubed-sphere Titan with C2 photochemistry — snapy × kintera × py2sess</h1>
<div class="sub">moses05.C2 network (Gate-A validated vs KINETICS-base) as snapy passive scalars · photolysis from py2sess two-stream actinic flux with per-column solar zenith angle · single H100 (6 blocks) or 8-GPU torchrun</div></header>
<main>
<div class="box"><h3 style="margin:2px 0 10px">Run summary</h3><table>{table}</table></div>
<div class="grid">{cards}</div>
</main></body></html>"""
    (out / "index.html").write_text(html)
    print(f"wrote {out/'index.html'} + {len(figs)} figures")


if __name__ == "__main__":
    main()
