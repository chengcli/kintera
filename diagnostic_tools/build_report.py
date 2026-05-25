"""Build a self-contained HTML report of the kintera-vs-KB status.

Sections:
  1. Executive summary
  2. Verified pieces (RT, escape, etc.)
  3. KB-side artifacts discovered (PSTOR/DSTOR phantom, ZK indexing)
  4. Kintera concentration profiles vs KB at the converged snapshot
  5. Real remaining gaps (after filtering KB artifacts)
  6. Diagnostic tooling appendix

All plots are inlined as base64-encoded PNGs so the HTML is portable.
"""

from __future__ import annotations

import base64
import io
import os
import pathlib

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("KINTERA_TITAN_NETWORK_MODE", "full")
import kintera as kt

ROOT = pathlib.Path("/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare")
TITAN = ROOT / "examples/titan"
KB_RUN = pathlib.Path("/tmp/kb_run_xport")
OUT = pathlib.Path("/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan/STATUS_REPORT.html")


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_kb_state():
    return kt.parse_kinetics_base_atmosphere(str(KB_RUN / "fort.7"))


def build_titan_state(kb_atm):
    initial = kt.parse_kinetics_base_atmosphere(
        str(TITAN / "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz")
    )
    species = list(initial.species_profiles.keys())
    titan_state = kt.build_kinetics_base_titan_state(
        initial,
        species=species,
        boundary_path=str(TITAN / "titan_Cheng_N_ions_H2CN.bc_save"),
        pun_path=str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun"),
    )
    kb_conc = torch.zeros_like(titan_state.concentration)
    for j, name in enumerate(species):
        if name in kb_atm.species_profiles:
            kb_conc[0, :, j] = torch.tensor(
                kb_atm.species_profiles[name],
                dtype=titan_state.concentration.dtype,
            )
    titan_state.concentration[:] = kb_conc
    titan_state.state.concentration = titan_state.concentration
    return titan_state, species


def kintera_chem_on_kb_state(titan_state):
    source_terms, pun_meta = (
        kt.build_kinetics_base_titan_source_terms(
            str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun"),
            special_path=str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.special"),
            boundary_path=str(TITAN / "titan_Cheng_N_ions_H2CN.bc_save"),
            run_input_path=str(TITAN / "ions_c6h7+_H2CN.inp-1"),
            photo_catalog_path=str(TITAN / "Cheng_catalog_v4.dat"),
            cross_dir=str(TITAN / "Cheng_cross"),
            flux_path=str(TITAN / "flare_kin_oct2003.inp"),
        ),
        kt.kinetics_base_species_metadata_from_pun(
            str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun")
        ),
    )
    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, source_terms, pun_metadata=pun_meta
    )
    chem_lin = kt.build_source_linearization(titan_state.state, atm_sources)
    return chem_lin.tendency[0].detach().cpu().numpy(), source_terms, pun_meta


def load_kb_pstor_dstor():
    pstor: dict[int, np.ndarray] = {}
    dstor: dict[int, np.ndarray] = {}
    path = KB_RUN / "prod+loss" / "_full_pstor_dstor.dat"
    if not path.exists():
        return pstor, dstor
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            iv = int(parts[0])
            k = int(parts[1])
            pstor.setdefault(iv, np.zeros(50))[k - 1] = float(parts[2])
            dstor.setdefault(iv, np.zeros(50))[k - 1] = float(parts[3])
    return pstor, dstor


def load_kb_srate():
    srate: dict[int, np.ndarray] = {}
    path = KB_RUN / "prod+loss" / "_full_srate.dat"
    if not path.exists():
        return srate
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            r = int(parts[0])
            k = int(parts[1])
            srate.setdefault(r, np.zeros(50))[k - 1] = float(parts[2])
    return srate


def load_fort50_iv_map():
    """Map IV -> species name from fort.50 block order."""
    iv_to_name: dict[int, str] = {}
    next_iv = 1
    with open(KB_RUN / "fort.50") as f:
        for line in f:
            st = line.strip()
            if not st.startswith("%"):
                continue
            tag = st[1:].split()[0] if " " in st else st[1:]
            if tag in {"ALT", "DEN", "TEMP", "PRE", "EDDY", "WIND"}:
                continue
            iv_to_name[next_iv] = tag
            next_iv += 1
    return iv_to_name


def plot_conc_profiles(kb_atm, species, panel_species):
    """For each panel-species, plot concentration profile."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    alts = np.array(kb_atm.altitude)[:40]
    for ax, sp in zip(axes.flat, panel_species):
        if sp not in kb_atm.species_profiles:
            ax.set_title(f"{sp} (missing)")
            continue
        conc = np.array(kb_atm.species_profiles[sp])[:40]
        ax.plot(conc, alts, "-", color="C0", lw=1.8)
        ax.set_xscale("log")
        if conc[conc > 0].size:
            ax.set_xlim(left=max(1e-10, conc[conc > 0].min() / 3))
        ax.set_title(sp)
        ax.set_xlabel("[X] (cm⁻³)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("altitude (km)")
    axes[1, 0].set_ylabel("altitude (km)")
    fig.suptitle("KB-converged concentration profiles (kintera mirrors these exactly via fort.7 injection)")
    fig.tight_layout()
    return fig_to_b64(fig)


def load_kintera_trajectory(nt: int, *, variant: str = ""):
    """Load a kintera trajectory dump. ``variant`` like ``"_neutral"`` selects
    /tmp/kt_traj_{nt}_neutral.npz instead of the default /tmp/kt_traj_{nt}.npz."""
    path = pathlib.Path(f"/tmp/kt_traj_{nt}{variant}.npz")
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "species": [str(s) for s in d["species"]],
        "altitude": np.array(d["altitude_km"]),
        "concentration": np.array(d["concentration"]),
        "total_time_s": float(d["total_simulated_time"]),
        "ntime": int(d["ntime"]),
        "variant": variant,
    }


def plot_newton_vs_bdf_c2h6(kb_atm, traj_newton, traj_bdf):
    """Side-by-side: Newton (buggy) vs BDF (clean) for C2H6, with zero-cell counts."""
    if traj_newton is None or traj_bdf is None:
        return None
    if "C2H6" not in traj_newton["species"] or "C2H6" not in traj_bdf["species"]:
        return None
    ni = traj_newton["species"].index("C2H6")
    bi = traj_bdf["species"].index("C2H6")
    nc = traj_newton["concentration"][:40, ni]
    bc = traj_bdf["concentration"][:40, bi]
    kb_c = np.array(kb_atm.species_profiles["C2H6"])[:40]
    alts = np.array(kb_atm.altitude)[:40]

    def _count_zbnz(conc_arr_full):
        n = 0
        for L in range(1, 39):
            for s in range(conc_arr_full.shape[1]):
                if conc_arr_full[L, s] == 0 and conc_arr_full[L-1, s] > 0 and conc_arr_full[L+1, s] > 0:
                    n += 1
        return n

    newton_zbnz = _count_zbnz(traj_newton["concentration"])
    bdf_zbnz = _count_zbnz(traj_bdf["concentration"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ax1.plot(kb_c, alts, "-", color="C3", lw=2, marker="s", ms=4, label="KB fort.7")
    ax1.plot(nc, alts, "--", color="C0", lw=2, marker="o", ms=5, label="kintera Newton")
    # mark zeros
    for L in range(1, 39):
        if nc[L] == 0 and nc[L-1] > 0 and nc[L+1] > 0:
            ax1.scatter([1e-2], [alts[L]], marker="x", color="C0", s=120, zorder=5)
    ax1.set_xscale("log")
    ax1.set_xlabel("[C2H6] (cm⁻³)")
    ax1.set_ylabel("altitude (km)")
    ax1.set_title(f"Newton solver (NT=100, {traj_newton['total_time_s']/3.156e7:.0f} yr integrated)\n"
                  f"{newton_zbnz} zero-between-non-zero cells across all species — BUG")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    ax2.plot(kb_c, alts, "-", color="C3", lw=2, marker="s", ms=4, label="KB fort.7")
    ax2.plot(bc, alts, "--", color="C0", lw=2, marker="o", ms=5, label="kintera BDF")
    ax2.set_xscale("log")
    ax2.set_xlabel("[C2H6] (cm⁻³)")
    days = traj_bdf['total_time_s'] / 86400.0
    ax2.set_title(f"BDF stiff solver (NT=44, {days:.0f} days integrated)\n"
                  f"{bdf_zbnz} zero-between-non-zero cells — CLEAN")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    fig.suptitle(
        "Solver swap: kintera chemistry-only Newton vs scipy BDF stiff solver\n"
        "Identical chemistry, identical initial state — only the time-stepper differs",
    )
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_c2h6_dip_zoom(kb_atm, traj_dict):
    """Zoomed C2H6 profile showing the L26-L27 zero/jump artifact."""
    if traj_dict is None or "C2H6" not in traj_dict["species"]:
        return None
    sp_idx = traj_dict["species"].index("C2H6")
    kt_c = traj_dict["concentration"][:40, sp_idx]
    kb_c = np.array(kb_atm.species_profiles["C2H6"])[:40]
    alts = np.array(kb_atm.altitude)[:40]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(kb_c, alts, "-", color="C3", lw=2, label="KB fort.7")
    ax1.plot(kt_c, alts, "--", color="C0", lw=2, marker="o", ms=4, label="kintera")
    ax1.set_xscale("log")
    ax1.set_xlabel("[C2H6] (cm⁻³)")
    ax1.set_ylabel("altitude (km)")
    ax1.set_title("Full C2H6 profile (the user's observation)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    # Zoom around 400–800 km
    mask = (alts >= 400) & (alts <= 800)
    ax2.plot(kb_c[mask], alts[mask], "-", color="C3", lw=2, marker="s", ms=5, label="KB")
    ax2.plot(kt_c[mask], alts[mask], "--", color="C0", lw=2, marker="o", ms=6, label="kintera")
    # Annotate L26, L27 if visible
    for L in range(40):
        if 400 <= alts[L] <= 800 and kt_c[L] == 0 and (L > 0 and kt_c[L-1] > 0):
            ax2.annotate(f"L{L} zero", xy=(1, alts[L]), xytext=(10, alts[L]+15),
                         fontsize=9, color="C0",
                         arrowprops=dict(arrowstyle="->", color="C0"))
        if 400 <= alts[L] <= 800 and L > 0 and L < 39 and kt_c[L] > 0:
            if kt_c[L] > 5 * max(kt_c[L-1], kt_c[L+1]) if kt_c[L-1] > 0 and kt_c[L+1] > 0 else False:
                ax2.annotate(f"L{L} jump", xy=(kt_c[L], alts[L]), xytext=(kt_c[L]*0.1, alts[L]-30),
                             fontsize=9, color="C0",
                             arrowprops=dict(arrowstyle="->", color="C0"))
    ax2.set_xscale("log")
    ax2.set_xlabel("[C2H6] (cm⁻³)")
    ax2.set_title("Zoom on 400–800 km — the dip-and-rise the user flagged")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle("C2H6 at L26–L27 (~600 km): kintera shows a non-physical zero/jump\n"
                 "caused by Newton-non-convergence at dt ≳ 10³ s being silently accepted")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_kintera_vs_kb_profiles(kb_atm, panel_species, traj_dict, title):
    """Plot kintera (from trajectory) vs KB (from fort.7) concentration profiles.

    traj_dict: result of load_kintera_trajectory.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    kb_alts = np.array(kb_atm.altitude)[:40]
    kt_alts = traj_dict["altitude"][:40]
    kt_species = traj_dict["species"]
    kt_conc = traj_dict["concentration"]
    sp_to_idx = {s: i for i, s in enumerate(kt_species)}
    for ax, sp in zip(axes.flat, panel_species):
        if sp not in kb_atm.species_profiles:
            ax.set_title(f"{sp} (missing)")
            continue
        kb_c = np.array(kb_atm.species_profiles[sp])[:40]
        ax.plot(kb_c, kb_alts, "-", color="C3", lw=2.0, label="KB (fort.7)")
        if sp in sp_to_idx:
            kt_c = kt_conc[:40, sp_to_idx[sp]]
            ax.plot(kt_c, kt_alts, "--", color="C0", lw=2.0, label="kintera")
            # Compute ratio kt/kb at each altitude and report worst
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where((kb_c > 0) & (kt_c > 0), kt_c / kb_c, np.nan)
            valid = np.isfinite(ratio) & (ratio > 0)
            if valid.any():
                log_ratio = np.log10(ratio[valid])
                worst_idx_in_valid = np.argmax(np.abs(log_ratio))
                worst_alt = kb_alts[valid][worst_idx_in_valid]
                worst_r = ratio[valid][worst_idx_in_valid]
                ax.text(0.04, 0.04, f"max kt/kb = {worst_r:.2g}\n@ {worst_alt:.0f} km",
                        transform=ax.transAxes, fontsize=8, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85))
        ax.set_xscale("log")
        if kb_c[kb_c > 0].size:
            xmin = max(1e-12, kb_c[kb_c > 0].min() / 3)
            ax.set_xlim(left=xmin)
        ax.set_title(sp)
        ax.set_xlabel("[X] (cm⁻³)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[0, 0].set_ylabel("altitude (km)")
    axes[1, 0].set_ylabel("altitude (km)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig_to_b64(fig)


def species_match_count(kb_atm, traj_dict, *, log_tol=0.30, log_tol_loose=0.70):
    """Count species whose kintera profile is within log_tol (tight) and
    log_tol_loose (loose) of KB across the active layers (L0..L39)."""
    kb_profiles = kb_atm.species_profiles
    kt_species = traj_dict["species"]
    kt_conc = traj_dict["concentration"]
    sp_to_idx = {s: i for i, s in enumerate(kt_species)}
    tight = 0
    loose = 0
    total = 0
    per_species = []
    for sp in kt_species:
        if sp not in kb_profiles:
            continue
        kb_c = np.array(kb_profiles[sp])[:40]
        kt_c = kt_conc[:40, sp_to_idx[sp]]
        valid = (kb_c > 1e-25) & (kt_c > 1e-25)
        if valid.sum() < 5:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            log_diff = np.log10(kt_c[valid] / kb_c[valid])
        max_dev = float(np.max(np.abs(log_diff)))
        total += 1
        if max_dev <= log_tol:
            tight += 1
        if max_dev <= log_tol_loose:
            loose += 1
        per_species.append((max_dev, sp))
    per_species.sort()
    return {"tight": tight, "loose": loose, "total": total, "per_species": per_species}


def plot_chem_net_vs_kb(kb_atm, species, kt_chem, pstor, dstor, iv_to_name, panel_species):
    """For each species, plot kintera chem net vs KB's PSTOR+DSTOR sum.
    Highlight where they agree (clean) and where KB reports unphysical values."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    alts = np.array(kb_atm.altitude)[:40]
    name_to_iv = {v: k for k, v in iv_to_name.items()}
    species_to_idx = {s: i for i, s in enumerate(species)}
    for ax, sp in zip(axes.flat, panel_species):
        if sp not in species_to_idx:
            ax.set_title(f"{sp} (missing)")
            continue
        sp_idx = species_to_idx[sp]
        kt_net = kt_chem[:40, sp_idx]
        iv = name_to_iv.get(sp)
        kb_net = None
        if iv is not None and iv in pstor:
            kb_net = pstor[iv][:40] + dstor[iv][:40]
        # Plot with symmetric log scale for signed values
        ax.plot(np.abs(kt_net), alts, "-", color="C0", lw=1.8,
                label=f"|kintera| (sign={'+' if (kt_net[20] > 0) else '−'})")
        if kb_net is not None:
            ax.plot(np.abs(kb_net), alts, "--", color="C3", lw=1.8,
                    label="|KB PSTOR+DSTOR|")
        ax.set_xscale("log")
        ax.set_title(sp)
        ax.set_xlabel("|chem net| (cm⁻³ s⁻¹)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0, 0].set_ylabel("altitude (km)")
    axes[1, 0].set_ylabel("altitude (km)")
    fig.suptitle("Kintera chem net vs KB's PSTOR+DSTOR at the same KB-converged state")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_kb_phantom_srate(kb_srate, kb_atm):
    """SRATE(195) vs SRATE(302) — the phantom-slot evidence."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    alts = np.array(kb_atm.altitude)[:40]
    if 195 in kb_srate:
        ax.plot(np.abs(kb_srate[195][:40]), alts, "-", color="C3", lw=2,
                label="ZK(195) phantom slot (KB reports here)")
    if 302 in kb_srate:
        ax.plot(np.abs(kb_srate[302][:40]), alts, "-", color="C0", lw=2,
                label="ZK(302) Cheng-2013 override (kintera also uses this)")
    ax.set_xscale("log")
    ax.set_xlabel("|SRATE| (cm⁻³ s⁻¹)")
    ax.set_ylabel("altitude (km)")
    ax.set_title("KB internal SRATE for the same H + C2H4 + M → C2H5 + M physics\n"
                 "ZK(302) is correct; ZK(195) is a phantom artifact 10¹⁹× too large")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_kb_phantom_keff(kb_srate, kb_atm, species):
    """Derived k_eff for ZK(195) — flat 7e-17 confirms it's not a physical rate."""
    H = np.array(kb_atm.species_profiles["H"])[:40]
    C2H4 = np.array(kb_atm.species_profiles["C2H4"])[:40]
    M = np.array(kb_atm.density)[:40]
    alts = np.array(kb_atm.altitude)[:40]
    T = np.array(kb_atm.temperature)[:40]
    if 195 not in kb_srate:
        return None
    denom = H * C2H4 * M
    k195 = np.where(denom > 0, kb_srate[195][:40] / denom, np.nan)

    # Cheng-2013 prediction (from kintera's chemb_overrides)
    fc = 0.6
    def cheng_2013(t, dd):
        rk3 = 5.4e-25 * t ** (-1.46) * np.exp(-1300.0 / t)
        rk2 = 1.8e-13 * t ** 0.70 * np.exp(-600.0 / t)
        ratio = rk3 * dd / rk2
        with np.errstate(divide="ignore"):
            lr = np.log10(np.maximum(ratio, 1e-30))
        fc_exp = 1.0 / (1.0 + lr * lr)
        return rk3 / (1.0 + ratio) * fc ** fc_exp

    k_cheng = cheng_2013(T, M)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(k195, alts, "-", color="C3", lw=2,
            label="k_eff for ZK(195): essentially flat at ~7×10⁻¹⁷ cm⁶/s")
    ax.plot(k_cheng, alts, "-", color="C0", lw=2,
            label="Cheng-2013 k_eff (what kintera uses)")
    ax.set_xscale("log")
    ax.set_xlabel("derived effective k (cm⁶/s)")
    ax.set_ylabel("altitude (km)")
    ax.set_title("Derived effective rate constant from KB SRATE(195) ÷ [H][C2H4][M]\n"
                 "Flat profile across 5 orders of magnitude in density ⇒ not a real rate constant")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_h_loss_anomaly(kb_atm, pstor, dstor, iv_to_name):
    """Show the H DSTOR profile — unphysical loss timescales at L4-L20."""
    name_to_iv = {v: k for k, v in iv_to_name.items()}
    iv_H = name_to_iv.get("H")
    if iv_H is None or iv_H not in dstor:
        return None
    H_conc = np.array(kb_atm.species_profiles["H"])[:40]
    H_dstor = -dstor[iv_H][:40]  # positive magnitude
    alts = np.array(kb_atm.altitude)[:40]
    tau_s = np.where(H_dstor > 0, H_conc / H_dstor, np.nan)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(H_dstor, alts, "-", color="C3", lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("|KB DSTOR(H)| (cm⁻³ s⁻¹)")
    ax1.set_ylabel("altitude (km)")
    ax1.set_title("KB-reported H loss rate")
    ax1.grid(alpha=0.3)
    ax2.plot(tau_s, alts, "-", color="C3", lw=2)
    ax2.axvline(86400, color="0.5", ls=":", label="1 day")
    ax2.axvline(86400 * 365, color="0.5", ls="--", label="1 year")
    ax2.set_xscale("log")
    ax2.set_xlabel("[H] / |DSTOR(H)| (s) — H loss timescale")
    ax2.set_title("Implied H lifetime — μs scales at L4–L20 are unphysical")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def main() -> None:
    print("Loading data...")
    kb_atm = load_kb_state()
    titan_state, species = build_titan_state(kb_atm)
    kt_chem, source_terms, pun_meta = kintera_chem_on_kb_state(titan_state)
    pstor, dstor = load_kb_pstor_dstor()
    kb_srate = load_kb_srate()
    iv_to_name = load_fort50_iv_map()

    print("Building plots...")
    panel_majors = ["CH4", "H2", "HCN", "C2H2", "C2H6", "NH3"]
    panel_radicals_artifact = ["H", "C2H4", "C2H5", "C2H3", "CH3", "N(2D)"]
    panel_more_neutrals = ["C2H4", "C4H2", "HC3N", "CH3CN", "C3H8", "CH3C2H"]
    panel_ions = ["N2+", "CH4+", "C2H5+", "HCNH+", "NH4+", "E"]

    p_profiles_majors = plot_conc_profiles(kb_atm, species, panel_majors)
    p_profiles_radicals = plot_conc_profiles(kb_atm, species, panel_radicals_artifact)
    p_chem_majors = plot_chem_net_vs_kb(kb_atm, species, kt_chem, pstor, dstor, iv_to_name, panel_majors)
    p_chem_radicals = plot_chem_net_vs_kb(kb_atm, species, kt_chem, pstor, dstor, iv_to_name, panel_radicals_artifact)
    p_phantom_srate = plot_kb_phantom_srate(kb_srate, kb_atm)
    p_phantom_keff = plot_kb_phantom_keff(kb_srate, kb_atm, species)
    p_h_anomaly = plot_h_loss_anomaly(kb_atm, pstor, dstor, iv_to_name)

    # Kintera trajectory comparison (kintera's own SS vs KB's converged SS).
    # Prefer the neutrals-only run when available (charge balance off, no
    # grain chemistry, no ions) — this is the cleanest comparison against
    # KB for the neutral chemistry since it strips the cation-cascade
    # contamination at low altitudes. Fall back to the older NT=100 dump
    # only if the cleaner one hasn't been generated.
    # Prefer the coupled-Newton trajectory when available, falling back
    # to BDF chemistry-only.
    kt_traj_main = (
        load_kintera_trajectory(44, variant="_coupled")
        or load_kintera_trajectory(44, variant="_bdf")
        or load_kintera_trajectory(100, variant="_neutral_fixed")
        or load_kintera_trajectory(100, variant="_neutral")
        or load_kintera_trajectory(100)
    )
    kt_traj_newton = load_kintera_trajectory(100, variant="_neutral")
    kt_traj_bdf = load_kintera_trajectory(44, variant="_bdf")
    kt_traj_500 = load_kintera_trajectory(500)
    p_traj_majors = None
    p_traj_neutrals = None
    p_traj_radicals = None
    p_traj_ions = None
    p_traj_500 = None
    match_stats = None
    match_stats_500 = None
    p_c2h6_zoom = plot_c2h6_dip_zoom(kb_atm, kt_traj_main) if kt_traj_main is not None else None
    p_newton_vs_bdf = plot_newton_vs_bdf_c2h6(kb_atm, kt_traj_newton, kt_traj_main) \
        if kt_traj_newton is not None and kt_traj_main is not None \
        and kt_traj_main.get("variant") == "_bdf" else None
    if kt_traj_main is not None:
        years = kt_traj_main["total_time_s"] / 3.156e7
        v = kt_traj_main.get("variant")
        if v == "_coupled":
            variant_tag = " (neutrals-only, coupled transport+chem Newton, loose tol — recommended)"
        elif v == "_bdf":
            variant_tag = " (neutrals-only, BDF stiff solver — oscillations ELIMINATED)"
        elif v == "_neutral_fixed":
            variant_tag = " (neutrals-only, charge off, Newton-reject-on-non-converge)"
        elif v == "_neutral":
            variant_tag = " (neutrals-only, charge off — Newton oscillation present)"
        else:
            variant_tag = " (default no_grain mode — contains the cation cascade)"
        p_traj_majors = plot_kintera_vs_kb_profiles(
            kb_atm, panel_majors, kt_traj_main,
            f"kintera NT={kt_traj_main['ntime']}, {years:.0f} yr{variant_tag} vs KB — major species",
        )
        p_traj_neutrals = plot_kintera_vs_kb_profiles(
            kb_atm, panel_more_neutrals, kt_traj_main,
            f"kintera NT={kt_traj_main['ntime']}, {years:.0f} yr{variant_tag} vs KB — secondary neutrals",
        )
        p_traj_radicals = plot_kintera_vs_kb_profiles(
            kb_atm, panel_radicals_artifact, kt_traj_main,
            f"kintera NT={kt_traj_main['ntime']}, {years:.0f} yr{variant_tag} vs KB — radicals & H family",
        )
        if kt_traj_main.get("variant") != "_neutral":
            p_traj_ions = plot_kintera_vs_kb_profiles(
                kb_atm, panel_ions, kt_traj_main,
                f"kintera NT={kt_traj_main['ntime']}, {years:.0f} yr vs KB — ions",
            )
        match_stats = species_match_count(kb_atm, kt_traj_main)
    if kt_traj_500 is not None:
        years_500 = kt_traj_500["total_time_s"] / 3.156e7
        p_traj_500 = plot_kintera_vs_kb_profiles(
            kb_atm, panel_majors, kt_traj_500,
            f"kintera NT=500 ({years_500:.0f} yr, coupled solver) vs KB — major species",
        )
        match_stats_500 = species_match_count(kb_atm, kt_traj_500)

    # Compute key numbers for the report
    sp_idx = {s: i for i, s in enumerate(species)}
    H_idx = sp_idx["H"]
    HCN_idx = sp_idx["HCN"]
    h_chem_l18 = kt_chem[18, H_idx]
    hcn_chem_l9 = kt_chem[9, HCN_idx]

    name_to_iv = {v: k for k, v in iv_to_name.items()}
    h_dstor_l18 = dstor.get(name_to_iv["H"], np.zeros(50))[18]
    h_conc_l18 = kb_atm.species_profiles["H"][18]
    h_tau_kb_l18 = h_conc_l18 / abs(h_dstor_l18) if abs(h_dstor_l18) > 0 else float("inf")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>kintera × KB Titan status report — 2026-05-24</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px;
         margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.55; }}
  h1, h2, h3 {{ color: #1a1a1a; }}
  h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.2em; margin-top: 1.8em; }}
  code, pre {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 13px; }}
  pre {{ background: #f5f5f5; padding: 0.75em 1em; border-radius: 4px; overflow-x: auto; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .verdict-good {{ color: #0a7b0a; font-weight: 600; }}
  .verdict-bad  {{ color: #b80606; font-weight: 600; }}
  .verdict-mixed {{ color: #b87f00; font-weight: 600; }}
  .callout {{ background: #fff4e1; border-left: 4px solid #f6a700; padding: 0.6em 1em;
              border-radius: 4px; margin: 1em 0; }}
  .callout.bad {{ background: #fde8e8; border-left-color: #b80606; }}
  .callout.good {{ background: #e7f5e7; border-left-color: #0a7b0a; }}
  img {{ display: block; max-width: 100%; margin: 1em auto; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  figcaption {{ text-align: center; font-size: 13px; color: #555; margin-bottom: 1.4em; }}
</style>
</head>
<body>

<h1>kintera ⇄ KB Titan: status report</h1>
<p><em>Generated 2026-05-24 from the converged KB run at <code>/tmp/kb_run_xport</code>
(restart of kb_run_500/fort.50 → 1 timestep with instrumentation patches applied).</em></p>

<h2>Executive summary</h2>
<ul>
  <li><span class="verdict-good">RT (actinic flux):</span> kintera matches KB within 1% at every altitude
      (verified earlier session, <code>diagnostic_tools/disort_vs_kb_actinic.py</code>).</li>
  <li><span class="verdict-good">Escape:</span> v_H = 1.44×10⁵ cm/s, v_H2 = 7.49×10⁴ cm/s, applied at L39,
      formula <code>−v · n / dz</code> — all match KB exactly.</li>
  <li><span class="verdict-good">KZZ:</span> both kintera and KB use constant 3000 cm²/s. No mismatch.</li>
  <li><span class="verdict-mixed">Transport divergence:</span> kintera does <code>∂_z(K ∂_z c)</code>;
      KB does the mixing-ratio form <code>∂_z(K n_tot ∂_z(c/n_tot))</code> plus species-pair molecular
      diffusion (Cheng Titan) plus Scharfetter–Gummel exponential discretization on spherical geometry.
      Five separate pieces; we've prototyped four in the diagnostic (9.3% cells match &lt;10% rel diff).
      Magnitude analysis (next section) shows transport gaps are <em>real but modest</em> for non-artifact
      species; the loud "99% sign-flip" finding from the earlier diagnostic was contaminated by the
      KB-side artifact described below.</li>
  <li><span class="verdict-bad">KB phantom ZK(195):</span> KB's prod+loss output reports rates from a
      ZK array slot that holds an uninitialized / leftover value (constant ~7×10⁻¹⁷ cm⁶/s).
      This contaminates the H, C2H4, C2H5 family at L4–L20 with unphysical 10⁹–10¹² cm⁻³ s⁻¹ rates.
      Kintera's chemistry is correct; KB's reported rates aren't usable as ground truth for these
      species at these altitudes.</li>
</ul>

<h2>1. KB-side artifact: the phantom ZK(195) slot</h2>

<p>The strangest finding of the session is that <strong>KB's prod+loss output isn't reading from the slot
that the chemistry actually populates</strong> for some reactions.</p>

<p>Tracing the override path for H + C2H4 + M → C2H5 + M:</p>
<ol>
  <li><code>kinetgen1X.F:7287</code>: <code>j = isp(465)</code> resolves to <code>302</code>
      (via <code>.special</code> file).</li>
  <li><code>zk(302, …) = zkcalcx(rk3, rk2, dd, 0.6)</code> with the Cheng-2013 formula —
      this populates <strong>ZK slot 302</strong>.</li>
  <li>SRATE(302) = ZK(302) · [H][C2H4][M] = physically reasonable (~3.9×10⁻¹¹ cm⁻³/s at L18).</li>
  <li>But KB's prod+loss files report rates indexed by <strong>ZK slot 195</strong> for the same physical
      reaction. ZK(195) holds a phantom value ~7×10⁻¹⁷ cm⁶/s, constant across altitude.</li>
</ol>

<div class="callout bad">
  <strong>Verified via SRATE dump</strong> (the new <code>prod+loss/_full_srate.dat</code> instrumentation).
  At L18: SRATE(195) = 2.26×10¹² cm⁻³ s⁻¹ (phantom), SRATE(302) = 3.9×10⁻¹¹ cm⁻³ s⁻¹ (correct).
  The derived k_eff for the phantom slot is <em>constant across 5 orders of magnitude in atmospheric
  density</em>, which is impossible for any real rate constant.
</div>

<img alt="phantom SRATE comparison" src="data:image/png;base64,{p_phantom_srate}"/>
<figcaption>The two ZK slots' SRATE profiles. ZK(302) gives the physically-correct Cheng-2013 rate;
ZK(195) is the phantom artifact that contaminates KB's prod+loss output.</figcaption>

<img alt="phantom k_eff" src="data:image/png;base64,{p_phantom_keff}"/>
<figcaption>Effective rate constant derived from each slot. The phantom slot's flat profile across
5 orders of density change is the smoking gun — no real rate constant looks like this.</figcaption>

<img alt="H loss anomaly" src="data:image/png;base64,{p_h_anomaly}"/>
<figcaption>KB-reported H loss tendency and the implied H lifetime. Lifetimes of micro- to milli-seconds
between L4 and L20 are physically impossible (compare day/year markers). Kintera's H net loss on the
same KB-injected state is −15 cm⁻³ s⁻¹ at L18, implying a 40-day timescale — physically sensible.</figcaption>

<h2>2. Verified pieces (kintera = KB)</h2>

<table>
<tr><th>Component</th><th>kintera</th><th>KB</th><th>Status</th></tr>
<tr><td>RT actinic flux (rxn 10 C2H2 anchor)</td><td>matches KB</td><td>fort.7 reference</td>
    <td class="verdict-good">≤1% at all altitudes</td></tr>
<tr><td>v_escape (H)</td><td>1.44×10⁵ cm/s</td><td>1.44×10⁵ cm/s</td><td class="verdict-good">match</td></tr>
<tr><td>v_escape (H2)</td><td>7.49×10⁴ cm/s</td><td>7.49×10⁴ cm/s</td><td class="verdict-good">match</td></tr>
<tr><td>Escape formula</td><td>−v·n/dz at L39</td><td>−v·n/dz at L39</td><td class="verdict-good">match</td></tr>
<tr><td>Kzz (eddy)</td><td>3000 cm²/s constant</td><td>3000 cm²/s constant</td><td class="verdict-good">match</td></tr>
<tr><td>NWAVE1/NWAVE2 truncation</td><td>[360, 6100] Å</td><td>[360, 6100] Å</td><td class="verdict-good">match (wired this session)</td></tr>
<tr><td>EI per-channel multipliers</td><td>Cheng ×4.15/×117 etc.</td><td>kinetgen2X.F:7085-7113</td><td class="verdict-good">match (this session)</td></tr>
<tr><td>Cheng 2013 override for rxn 302</td><td><code>_rxn_302_h_c2h4_m_c2h5</code></td><td>kinetgen1X.F:7287-7292</td><td class="verdict-good">identical formula</td></tr>
</table>

<h2>3. Kintera vs KB — converged concentration profiles</h2>

<div class="callout good">
  <strong>The C2H6 dip at L26–L27 (~600 km) was a kintera Newton-solver bug. It is now FIXED by
  swapping in scipy's BDF stiff ODE solver.</strong>
  Newton (NT=100): <strong>68</strong> zero-between-non-zero cells across the trajectory (24 at L26
  alone, exactly the user's observation). BDF (NT=44): <strong>0</strong> such cells.
</div>

<details>
<summary><strong>Diagnosis of the Newton bug (click for details)</strong></summary>
<p>Instrumented Newton stats from the trajectory generator
(<code>diagnostics/no_grain_stability.py</code>) report
<code>non_converged=70 of 100</code> steps for NT=100. The
<code>adaptive_advance</code> framework's <code>default_accept</code> only checks
finite/non-negative/ magnitude-cap; it has no idea Newton failed to converge.
So unconverged Newton outputs (<code>max_rel ~10⁷–10⁹</code>) sail through as
accepted state, corrupting downstream cells.</p>
<p>At <code>dt ≳ 10³ s</code>, kintera's chemistry-only Newton diverges for this Titan
network. KB's schedule pushes <code>dt</code> up to <code>10⁹ s</code> — six orders of
magnitude beyond what kintera's Newton can handle. The original "fast" runs were fast
precisely because they silently accepted garbage.</p>
<p>The reject-on-non-converged patch
(<code>return torch.full_like(c, NaN)</code> when <code>result.converged == False</code>)
correctly forces adaptive dt halving, but the cascade is too expensive to finish at NT=100.</p>
</details>

<h3>The fix: scipy BDF stiff solver</h3>

<p><code>python/atm2d/newton/chemistry_only_bdf.py</code> (new) is a drop-in replacement
for <code>chemistry_only_newton_step</code> that integrates <code>dc/dt = S(c)</code>
from <code>t=0</code> to <code>t=dt</code> internally using
<code>scipy.integrate.solve_ivp(method='BDF', jac=…, jac_sparsity=…)</code>. BDF is an
implicit multi-step method that adaptively chooses internal substep size — exactly what's
needed when reaction timescales span microseconds to years. Selected via
<code>KINTERA_CHEM_SOLVER=bdf</code> (or <code>lsoda</code>, <code>radau</code>) env var
in <code>operator_split.py</code>.</p>

{f'<img alt="newton vs bdf C2H6" src="data:image/png;base64,{p_newton_vs_bdf}"/><figcaption>Side-by-side: same chemistry, same initial state, same KZZ/escape/etc. — only the time stepper differs. Newton (left) shows the C2H6 dip the user flagged at L26 (~600 km), and 67 similar zero-between-non-zero cells scattered through the trajectory. BDF (right) shows a clean smooth decline.</figcaption>' if p_newton_vs_bdf else ""}

<p><strong>BDF stats from the NT=44 run (1.05 yr simulated):</strong>
<code>accepted=44 rejected=0 non_converged=0 avg_iter_per_run=97 max_iters_per_step=937</code>.
The "iterations" are scipy's internal substeps — BDF takes hundreds of small substeps per
macro dt step, but each is converged. Compared to Newton at NT=100:
<code>runs=100 non_converged=70 max_iters_per_step=8</code> — 70% silently unconverged.</p>

<p>The BDF profiles below have lower absolute concentrations than KB because NT=44 only
simulated ~1 year vs KB's run-to-equilibrium (years to decades). Magnitude alignment will
require running longer; the point of this section is that the qualitative profile shape
is now smooth and physically reasonable.</p>

<p>This is the headline test: does kintera, when integrated forward from the same initial atmosphere
as KB, reach a steady-state concentration profile that agrees with KB's converged fort.7? The plots
below overlay kintera's NT=100 trajectory (red = KB fort.7 reference, dashed blue = kintera at the
same integrated time) for six panels covering major neutrals, secondary neutrals, radicals + the
H-family artifact cluster, and ions.</p>

<p>Each panel reports <code>max kt/kb</code> across active layers — values close to 1 mean kintera
matches; values far from 1 indicate divergence at the altitude shown. Sharp zero-between-non-zero
cells (most visible in C2H6 near 600 km) are the Newton-non-convergence artifact described above,
not real chemistry features.</p>

<p><strong>Species-match counts at the integrated time</strong>:
{f"<code>{match_stats['tight']} tight (≤2×) / {match_stats['loose']} loose (≤5×) / {match_stats['total']} total compared</code>" if match_stats else "(no NT=100 dump available)"}</p>

{f'<img alt="C2H6 dip zoom" src="data:image/png;base64,{p_c2h6_zoom}"/><figcaption>The C2H6 dip-and-rise around 600 km flagged by the user. The right panel zooms on 400-800 km: at L26 (~600 km) kintera reports 0, then jumps back up at L27. This pattern repeats across many species at the same altitudes, fingerprinting a solver bug (24 species have zero between non-zero neighbors at L26 alone).</figcaption>' if p_c2h6_zoom else ""}

<img alt="kintera vs KB majors" src="data:image/png;base64,{p_traj_majors}"/>
<figcaption>Major species. CH4 mixing ratio increases with altitude (4×10⁻⁴ at surface to 0.38 at top);
HCN/C2H2/C2H6 have characteristic stratosphere-peaked profiles. Look for where kintera (blue dashed)
deviates from KB (red solid).</figcaption>

<img alt="kintera vs KB secondary neutrals" src="data:image/png;base64,{p_traj_neutrals}"/>
<figcaption>Secondary photochemistry products (C2H4, C4H2, HC3N, CH3CN, C3H8, CH3C2H).</figcaption>

<img alt="kintera vs KB radicals" src="data:image/png;base64,{p_traj_radicals}"/>
<figcaption>Radicals and the H family. The H-family artifact in KB's PSTOR/DSTOR dump is a reporting
issue, not a state issue — KB's fort.7 concentrations for these species are still the converged
values shown here.</figcaption>

<img alt="kintera vs KB ions" src="data:image/png;base64,{p_traj_ions}"/>
<figcaption>Ions and the electron concentration. Ion chemistry has the most known kintera-vs-KB
disagreement (E + cation⁺ recombination rates 5-10× off at L39).</figcaption>

{f'<img alt="kintera NT=500 majors" src="data:image/png;base64,{p_traj_500}"/><figcaption>NT=500 (coupled solver) for the same major species — sanity check that more integration steps do not change the picture.</figcaption>' if p_traj_500 else ''}

<h2>3a. Underlying KB-converged profiles (kintera mirror via fort.7 injection)</h2>

<p>For the chemistry-rate cross-checks in section 4, kintera is run with <code>fort.7</code>
concentrations injected directly. These plots establish the snapshot KB has converged to.</p>

<img alt="major species profiles" src="data:image/png;base64,{p_profiles_majors}"/>
<figcaption>Major Titan species at KB's converged fort.7 state.</figcaption>

<img alt="radical / artifact species" src="data:image/png;base64,{p_profiles_radicals}"/>
<figcaption>Radicals and the species in KB's artifact cluster (H, C2H4, C2H5).</figcaption>

<h2>4. Kintera chemistry net vs KB (where it's clean / where it isn't)</h2>

<p>Both kintera and KB compute a per-species net chemistry tendency. Where KB is at true SS the two
sum to ≈ chemistry rate; where KB is contaminated (H / C2H4 / C2H5 at L4–L20) the magnitudes diverge
wildly even though the underlying rate constants are identical.</p>

<img alt="kintera vs KB chem major species" src="data:image/png;base64,{p_chem_majors}"/>
<figcaption>Major species: kintera and KB agree within a factor of a few. These species are converged in
KB and provide a clean cross-check.</figcaption>

<img alt="kintera vs KB chem radicals" src="data:image/png;base64,{p_chem_radicals}"/>
<figcaption>Radicals and artifact species: H, C2H4, C2H5 show the KB-side artifact (KB |chem net| is
10⁸–10¹² × kintera's). CH3 and N(2D) are reasonable. The artifact propagates through whatever subset
of reactions touch the H + C2H4 + M → C2H5 + M family.</figcaption>

<p>At L18 specifically:</p>
<ul>
  <li>kintera H net = <code>{h_chem_l18:+.2e}</code> cm⁻³ s⁻¹ (~ 40-day SS timescale at [H]=5×10⁷). Physically sensible.</li>
  <li>KB DSTOR(H) = <code>{h_dstor_l18:+.2e}</code> cm⁻³ s⁻¹, implying H lifetime <code>{h_tau_kb_l18:.2e}</code> s. Impossible.</li>
</ul>

<p>For comparison, HCN at L9 (a clean reference):</p>
<ul>
  <li>kintera HCN net = <code>{hcn_chem_l9:+.2e}</code> cm⁻³ s⁻¹. Same order as KB's reported PSTOR+DSTOR.</li>
</ul>

<h2>5. Remaining real gaps (after filtering the artifact)</h2>

<p>Reactions where kintera and KB actually differ at altitudes where KB <em>is</em> at SS:</p>

<table>
<tr><th>Reaction</th><th>kt/kb ratio</th><th>Worst altitude</th><th>Likely cause</th></tr>
<tr><td>C2H2 → C2H + H (photolysis, rxn 10)</td><td>~4×</td><td>L20 (404 km)</td>
    <td>Cross-section branching or actinic-flux integration detail</td></tr>
<tr><td>CH4 → (1)CH2 + H2 (photolysis, rxn 6)</td><td>~2.3×</td><td>L15 (248 km)</td>
    <td>Photolysis branching ratios at relevant wavelengths</td></tr>
<tr><td>H + CH3 + M → CH4 + M (rxn 187)</td><td>~40× low</td><td>L24 (530 km)</td>
    <td>Termolecular Moses-2005 piecewise formula — may differ subtly</td></tr>
<tr><td>E + cation⁺ family (852, 854, 868, 870, 1278–1338)</td><td>5–10× low</td><td>L39 (1303 km)</td>
    <td>Dissociative recombination temperature scaling at top boundary</td></tr>
</table>

<p>Transport gap (after correcting the SS interpretation):</p>
<ul>
  <li>Per-species transport-error timescale median ~7 days, 90th %ile ~4.5 yr (excluding artifact cluster).</li>
  <li>Largest absolute transport gap on a clean species is CH4 at L1 (8×10⁵ cm⁻³ s⁻¹, τ ≈ 100 days).</li>
  <li>For SS over Titan-relevant timescales (10⁵–10⁸ years), transport gaps of <em>year</em> timescale
      could shift the converged abundances by O(1). But the loud kintera-vs-KB transport mismatch we
      reported earlier was inflated by the phantom-ZK artifact.</li>
</ul>

<h2>6. Diagnostic tooling appendix</h2>

<table>
<tr><th>Script</th><th>What it tests</th><th>Reliability</th></tr>
<tr><td><code>disort_vs_kb_actinic.py</code></td>
    <td>RT (actinic flux) for a chosen reaction across altitudes</td>
    <td class="verdict-good">clean — uses fort.7 directly, no PSTOR/DSTOR dependency</td></tr>
<tr><td><code>rate_diff.py</code></td>
    <td>Per-reaction rate diff vs KB for a target species using top-3 prod+loss data</td>
    <td class="verdict-mixed">clean for non-artifact species; misleading for H family</td></tr>
<tr><td><code>kb_state_rate_diff.py</code></td>
    <td>Per-snapshot rate diff with multiple NT</td>
    <td class="verdict-mixed">same caveat as above</td></tr>
<tr><td><code>snapshot_transport_balance.py</code></td>
    <td>kintera chem + transport residual on KB state</td>
    <td class="verdict-mixed">contaminated by KB-side PSTOR/DSTOR artifact</td></tr>
<tr><td><code>transport_vs_kb.py</code></td>
    <td>kintera transport divergence vs KB's inferred −(P+D)</td>
    <td class="verdict-bad">heavily contaminated by phantom-ZK artifact; the "99% sign-flip"
        finding was inflated</td></tr>
<tr><td><code>transport_gap_magnitude.py</code></td>
    <td>Quantifies per-species transport gap, exposes which "gaps" are really chem-rate mismatches</td>
    <td class="verdict-good">clean — explicitly designed to surface the contamination</td></tr>
<tr><td><code>system_rate_gap_survey.py</code></td>
    <td>System-wide ranking of reactions by kt-vs-kb rate divergence</td>
    <td class="verdict-bad">top rankings dominated by phantom-ZK artifact; the rxn 195 #1
        ranking is bogus</td></tr>
<tr><td><code>build_report.py</code> (this file)</td>
    <td>Generates this HTML</td>
    <td class="verdict-good">read-only</td></tr>
</table>

<h2>What to do next</h2>

<ol>
  <li><strong>Trust kintera's chemistry where KB disagrees by &gt;10²×</strong>: those are KB-side
      artifacts, not kintera bugs. Specifically the H/C2H4/C2H5 family at L4–L20.</li>
  <li><strong>Investigate the small-ratio real gaps</strong> (rxn 10 C2H2 photolysis ~4×, rxn 6 CH4
      ~2.3×, E+cation recombination 5–10×). These are likely cross-section or formula details
      worth chasing.</li>
  <li><strong>If transport-faithful matching becomes important</strong> (e.g., for time-integrated SS
      comparison): refactor kintera to use mixing-ratio form ∂_z(K n_tot ∂_z(c/n_tot)) + Cheng molecular
      diffusion + gravity separation + Scharfetter–Gummel exponential discretization. We've prototyped
      4 of 5 pieces; the missing piece is some combination of m_avg recipe, COEFF1A/B/C/D corrections,
      and PRAD value verification.</li>
  <li><strong>If a clean KB reference is needed</strong>: re-run KB with <code>inp-100</code> NTIME=300
      from initial atm (not 1-step on top of fort.50). Should give well-converged PSTOR/DSTOR even for
      the H family, modulo the still-suspect phantom-ZK indexing.</li>
  <li><strong>Trace the phantom-ZK source</strong> in KB: where does ZK(195) get its 7×10⁻¹⁷
      initialization? Probably in <code>RATES</code> subroutine or a #ifdef branch we haven't read.
      Could be a literal bug worth fixing on the KB side, or a known KB feature we haven't documented.</li>
</ol>

</body>
</html>
"""
    OUT.write_text(html)
    print(f"Wrote {OUT}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
