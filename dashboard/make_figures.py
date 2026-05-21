"""Generate all dashboard figures for the kintera × KB Titan state.

Reads:
  - /tmp/baseline_g29.npz                — current best (G29 config)
  - /tmp/post_phase6b_*.npz              — ion_scale sweep dumps
  - /tmp/kb_run_500/fort.7               — KB oracle (grain ON)

Writes PNGs into /home/sam2/dev/kintera/dashboard/figs/
"""
from __future__ import annotations

import pathlib
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

import kintera as kt


FIGDIR = pathlib.Path("/home/sam2/dev/kintera/dashboard/figs")
FIGDIR.mkdir(parents=True, exist_ok=True)

# ----- load data -----
print("[load] baseline + KB oracle ...")
d = np.load("/tmp/baseline_g29.npz", allow_pickle=True)
SPECIES = [str(x) for x in d["species"]]
C = d["concentration"]  # (nlyr=50, nspecies=128)
KB = kt.parse_kinetics_base_atmosphere("/tmp/kb_run_500/fort.7").species_profiles
INITIAL = kt.parse_kinetics_base_atmosphere(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan/"
    "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz"
)
ALT = np.asarray(INITIAL.altitude)  # km
TEMP = np.asarray(INITIAL.temperature)  # K

NLYR = C.shape[0]
LEVELS_ALL = list(range(NLYR))
LEVELS_SAMPLE = [0, 5, 10, 15, 20, 25, 30, 35]


def ratio_grid(c, species, kb, levels):
    """Return (n_lev, n_species) ratio array with NaN where KB<1 (untrustworthy)."""
    n_lev, n_sp = len(levels), len(species)
    grid = np.full((n_lev, n_sp), np.nan)
    for j, name in enumerate(species):
        kbv = kb.get(name)
        if kbv is None:
            continue
        kbv = np.asarray(kbv)
        for i, lev in enumerate(levels):
            if kbv[lev] < 1.0:
                continue
            grid[i, j] = c[lev, j] / kbv[lev]
    return grid


def matched_count(c, species, kb, lo=0.3, hi=3.0):
    gd = total = 0
    for j, n in enumerate(species):
        kbv = kb.get(n)
        if kbv is None:
            continue
        for lev in LEVELS_SAMPLE:
            if kbv[lev] < 1:
                continue
            total += 1
            r = c[lev, j] / kbv[lev]
            if lo < r < hi:
                gd += 1
    return gd, total


# ----- Figure 1: species ratio heatmap (top-50 most-abundant in KB) -----
print("[fig1] species ratio heatmap ...")
kb_max_per_species = {}
for n in SPECIES:
    if n in KB:
        kb_max_per_species[n] = max(KB[n])

top = sorted(SPECIES, key=lambda s: kb_max_per_species.get(s, 0), reverse=True)[:50]
top_idx = [SPECIES.index(n) for n in top]
grid = ratio_grid(C[:, top_idx], top, KB, LEVELS_ALL)
logr = np.log10(np.where(grid > 0, grid, np.nan))

fig, ax = plt.subplots(figsize=(14, 10))
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
im = ax.imshow(
    logr,
    aspect="auto",
    origin="lower",
    cmap="RdBu_r",
    norm=norm,
    interpolation="nearest",
)
ax.set_xticks(range(len(top)))
ax.set_xticklabels(top, rotation=90, fontsize=8)
ax.set_yticks(range(0, NLYR, 5))
ax.set_yticklabels([f"L{i} ({ALT[i]:.0f}km)" for i in range(0, NLYR, 5)], fontsize=8)
ax.set_xlabel("species (top 50 by KB max)")
ax.set_ylabel("level")
ax.set_title(
    "log₁₀(kt / KB) per species & level — G29 best (full grain, dt=1e+7, NT=100)\n"
    f"matched (0.3-3×): {matched_count(C, SPECIES, KB)[0]}/{matched_count(C, SPECIES, KB)[1]}"
)
cbar = plt.colorbar(im, ax=ax, label="log₁₀(ratio)", extend="both")
cbar.set_ticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
cbar.set_ticklabels(["1e-4", "1e-3", "1e-2", "0.1", "1×", "10×", "100×", "1e3", "1e4"])
plt.tight_layout()
plt.savefig(FIGDIR / "1_ratio_heatmap.png", dpi=110)
plt.close()


# ----- Figure 2: vertical profiles of key species -----
print("[fig2] vertical profiles ...")
key = [
    ("CH4", "neutral"),
    ("HCN", "neutral"),
    ("C2H2", "neutral"),
    ("C2H6", "neutral"),
    ("NH3", "neutral"),
    ("N(2D)", "neutral radical"),
    ("CH3", "neutral radical"),
    ("HCNH+", "cation"),
    ("NH4+", "cation"),
    ("C+", "cation"),
    ("GCH4", "grain"),
    ("GC2H6", "grain"),
]
fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
for ax, (name, tag) in zip(axes.flat, key):
    if name not in SPECIES:
        ax.set_visible(False)
        continue
    j = SPECIES.index(name)
    kt_prof = C[:, j]
    kb_prof = np.asarray(KB.get(name, np.zeros(NLYR)))
    ax.semilogx(np.maximum(kt_prof, 1e-30), ALT[:NLYR], "C0-", label="kintera", lw=2)
    ax.semilogx(np.maximum(kb_prof, 1e-30), ALT[:NLYR], "C1--", label="KB", lw=2)
    ax.set_title(f"{name} ({tag})")
    ax.set_xlim(1e-15, 1e16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
axes[2, 0].set_xlabel("number density [cm⁻³]")
axes[2, 1].set_xlabel("number density [cm⁻³]")
axes[2, 2].set_xlabel("number density [cm⁻³]")
axes[2, 3].set_xlabel("number density [cm⁻³]")
axes[0, 0].set_ylabel("altitude [km]")
axes[1, 0].set_ylabel("altitude [km]")
axes[2, 0].set_ylabel("altitude [km]")
plt.suptitle("Vertical profiles — G29 best config vs KB-500 oracle", y=1.005)
plt.tight_layout()
plt.savefig(FIGDIR / "2_profiles.png", dpi=110)
plt.close()


# ----- Figure 3: cations at L30, bar chart -----
print("[fig3] cations at L30 bar chart ...")
cations = [(j, n) for j, n in enumerate(SPECIES) if n.endswith("+")]
items = []
for j, n in cations:
    kbv = KB.get(n, [0] * NLYR)[30]
    items.append((C[30, j], kbv, n))
items.sort(key=lambda x: max(x[0], x[1]), reverse=True)
items = items[:18]

labels = [it[2] for it in items]
kt_vals = [max(it[0], 1e-20) for it in items]
kb_vals = [max(it[1], 1e-20) for it in items]
xs = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(xs - 0.2, kt_vals, width=0.4, label="kintera", color="C0")
ax.bar(xs + 0.2, kb_vals, width=0.4, label="KB", color="C1")
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("number density [cm⁻³] @ L30")
cat_sum_kt = sum(it[0] for it in items)
cat_sum_kb = sum(it[1] for it in items)
ax.set_title(
    f"Top 18 cations at L30  ·  Σ cation ratio: kt/KB = {cat_sum_kt/cat_sum_kb:.3f}×"
)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGDIR / "3_cations_L30.png", dpi=110)
plt.close()


# ----- Figure 4: ion_scale sweep Pareto -----
print("[fig4] ion_scale Pareto ...")
sweep = {
    "0.0":    {"matched": 145, "cat": 0.519,   "nh3": 0.00},
    "1e-5":   {"matched": 149, "cat": 0.464,   "nh3": 13.8},
    "1e-4":   {"matched": 123, "cat": 298.6,   "nh3": 91.8},
    "1e-3":   {"matched": 146, "cat": 2.01,    "nh3":  4.0},
    "1e-2":   {"matched": 132, "cat": 0.269,   "nh3": 239.6},
    "1e-1":   {"matched": 135, "cat": 1.754,   "nh3": 144.5},
    "1.0":    {"matched": 159, "cat": 0.828,   "nh3": 28.1},
    "2.0":    {"matched": 123, "cat": 0.151,   "nh3": 33.3},
    "5.0":    {"matched": 141, "cat": 2.618,   "nh3": 30.2},
    "10.0":   {"matched": 110, "cat": 2.050,   "nh3":  7.5},
}
xs = [v["matched"] for v in sweep.values()]
ys = [v["cat"] for v in sweep.values()]
labels = list(sweep.keys())
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(xs, ys, s=120, c=range(len(xs)), cmap="viridis", edgecolors="black")
for x, y, lab in zip(xs, ys, labels):
    ax.annotate(f"  {lab}", (x, y), fontsize=9)
ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="cation ratio = 1× KB")
ax.axvline(xs[labels.index("1.0")], color="red", linestyle="--", alpha=0.5, label="baseline matched")
ax.set_yscale("log")
ax.set_xlabel("species-level pairs matched (0.3–3×)")
ax.set_ylabel("cation@L30 ratio (kt / KB)")
ax.set_title("KINTERA_ION_DIFFUSION_SCALE sweep — baseline (1.0) is Pareto-best\n"
             "(NT=100, dt=1e+7, full grain, split solver)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "4_ion_scale_pareto.png", dpi=110)
plt.close()


# ----- Figure 5: ratio histogram (gap decomposition) -----
print("[fig5] ratio histogram ...")
ratios = []
for j, n in enumerate(SPECIES):
    kbv = KB.get(n)
    if kbv is None:
        continue
    for lev in LEVELS_SAMPLE:
        if kbv[lev] < 1:
            continue
        if C[lev, j] <= 0:
            r = 1e-10
        else:
            r = C[lev, j] / kbv[lev]
        ratios.append(r)
ratios = np.array(ratios)
log_ratios = np.log10(ratios.clip(min=1e-10))

bins = np.linspace(-10, 10, 41)
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(log_ratios, bins=bins, color="steelblue", edgecolor="black")
ax.axvline(np.log10(0.3), color="green", linestyle="--", alpha=0.6, label="0.3× / 3× (matched)")
ax.axvline(np.log10(3.0), color="green", linestyle="--", alpha=0.6)
ax.axvline(np.log10(0.1), color="orange", linestyle=":", alpha=0.6, label="0.1× / 10×")
ax.axvline(np.log10(10.0), color="orange", linestyle=":", alpha=0.6)
ax.axvline(0.0, color="black", linestyle="-", alpha=0.4, label="perfect (1×)")
ax.set_xlabel("log₁₀(kt / KB)")
ax.set_ylabel("species-level pair count")
gd, total = matched_count(C, SPECIES, KB, 0.3, 3.0)
gd_loose, _ = matched_count(C, SPECIES, KB, 0.1, 10.0)
gd_extra, _ = matched_count(C, SPECIES, KB, 0.01, 100.0)
ax.set_title(
    f"Distribution of kt/KB ratios over {len(ratios)} species-level pairs  ·  "
    f"matched (0.3-3×): {gd}/{total} ({100*gd/total:.0f}%), "
    f"(0.1-10×): {gd_loose}/{total}, (0.01-100×): {gd_extra}/{total}"
)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGDIR / "5_ratio_histogram.png", dpi=110)
plt.close()


# ----- Figure 6: gap by level (where is the model best/worst?) -----
print("[fig6] gap by level ...")
levels = list(range(NLYR))
matched_pct = []
counts = []
for lev in levels:
    gd = total = 0
    for j, n in enumerate(SPECIES):
        kbv = KB.get(n)
        if kbv is None:
            continue
        if kbv[lev] < 1:
            continue
        total += 1
        r = C[lev, j] / kbv[lev] if C[lev, j] > 0 else 0
        if 0.3 < r < 3:
            gd += 1
    counts.append((gd, total))
    matched_pct.append(100 * gd / total if total else 0)

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.fill_between(ALT[:NLYR], 0, matched_pct, alpha=0.4, color="C0")
ax1.plot(ALT[:NLYR], matched_pct, color="C0", lw=2, marker="o", markersize=4)
ax1.set_xlabel("altitude [km]")
ax1.set_ylabel("% species matched (0.3-3×)", color="C0")
ax1.set_ylim(0, 100)
ax1.tick_params(axis="y", labelcolor="C0")
ax2 = ax1.twinx()
ax2.plot(ALT[:NLYR], TEMP[:NLYR], color="C3", linestyle="--", lw=1.5, label="T [K]")
ax2.set_ylabel("temperature [K]", color="C3")
ax2.tick_params(axis="y", labelcolor="C3")
plt.title("Match rate vs altitude — best at L25–35 (ion-prod zone), worst at L0–10 (NH3 feedback)")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "6_match_by_level.png", dpi=110)
plt.close()


# ----- Figure 7: cation@L30 distribution across ion_scale sweep -----
print("[fig7] sweep NH3 effect ...")
nh3_vals = [v["nh3"] for v in sweep.values()]
fig, ax = plt.subplots(figsize=(10, 5))
xs_label = list(sweep.keys())
xs_num = list(range(len(xs_label)))
bars = ax.bar(xs_num, nh3_vals, color=["#aec7e8" if l != "1.0" else "C3" for l in xs_label],
              edgecolor="black")
ax.set_yscale("log")
ax.set_xticks(xs_num)
ax.set_xticklabels(xs_label, rotation=0)
ax.set_xlabel("ion_diffusion_scale")
ax.set_ylabel("NH3 @ L5 ratio (kt / KB)")
ax.axhline(1.0, color="green", linestyle="--", alpha=0.6, label="KB-match")
ax.set_title("Effect of ion_scale on NH3@L5 — non-monotonic, no simple sweet spot\n"
             "(baseline ion_scale=1.0 highlighted in red)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGDIR / "7_nh3_vs_scale.png", dpi=110)
plt.close()


# ----- Figure 8: grain over-condensation pattern at L5 -----
print("[fig8] grain over-condensation ...")
grain_species = [n for n in SPECIES if n.startswith("G") and n != "G"]
items = []
for n in grain_species:
    j = SPECIES.index(n)
    v = C[5, j]
    kbv = KB.get(n, [0] * NLYR)[5]
    if kbv > 0 or v > 0:
        items.append((v, kbv, n))
items.sort(key=lambda x: max(x[0], x[1]), reverse=True)
items = items[:14]
labels = [it[2] for it in items]
kt_vals = [max(it[0], 1e-3) for it in items]
kb_vals = [max(it[1], 1e-3) for it in items]
xs = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(xs - 0.2, kt_vals, width=0.4, label="kintera", color="C0")
ax.bar(xs + 0.2, kb_vals, width=0.4, label="KB", color="C1")
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("number density [cm⁻³] @ L5")
ax.set_title("Grain ice species at L5 — most are over-condensed in our model\n"
             "(C2H6: 6960× KB; root cause traced upstream to low-alt gas excess)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGDIR / "8_grain_overcondensation.png", dpi=110)
plt.close()


# ----- Figure 9: cation @ different altitudes — the bleed pattern -----
print("[fig9] cation bleed pattern ...")
ions_to_track = ["NH4+", "HCNH+", "C2H5+", "CH5+", "C6H7+"]
fig, ax = plt.subplots(figsize=(11, 6))
for ion in ions_to_track:
    if ion not in SPECIES:
        continue
    j = SPECIES.index(ion)
    kt_prof = C[:, j]
    kb_prof = np.asarray(KB.get(ion, np.zeros(NLYR)))
    ax.semilogx(np.maximum(kt_prof, 1e-15), ALT[:NLYR], "-", label=f"{ion} kt", lw=1.8)
    ax.semilogx(np.maximum(kb_prof, 1e-15), ALT[:NLYR], "--", label=f"{ion} KB", lw=1.2, alpha=0.7)
ax.set_xlim(1e-10, 1e10)
ax.set_xlabel("number density [cm⁻³]")
ax.set_ylabel("altitude [km]")
ax.set_title("Cation profiles — model has ions at L0–15 where KB has zero\n"
             "(this is the G30 bleed driving NH3 feedback loop)")
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "9_cation_bleed.png", dpi=110)
plt.close()


# ----- Figure 10: refactor structure tree -----
print("[fig10] refactor diagram ...")
fig, ax = plt.subplots(figsize=(14, 9))
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Layer 1 (top): kintera/atm2d
def box(x, y, w, h, text, color="#e8f3ec", fontsize=8):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

# Header
ax.text(50, 95, "REFACTOR_SCHEMA Phases 1-6 — landed structure",
        ha="center", fontsize=14, fontweight="bold")
ax.text(50, 91, "All bit-identical against G29 baseline · 14 commits · pushed to origin/ions-support-plan",
        ha="center", fontsize=9, style="italic")

# L1 (kintera core)
ax.text(25, 85, "L1: kintera atm2d (pure physics)", ha="center", fontsize=12,
        fontweight="bold", color="#2a6")
boxes_l1 = [
    (5, 75, 18, 5, "conservation/\natomic.py", "#e8f3ec"),
    (24, 75, 18, 5, "sources/\nprotocol indexed combine\ncharge_balance", "#e8f3ec"),
    (5, 67, 18, 5, "schedule.py", "#e8f3ec"),
    (24, 67, 18, 5, "pins.py\nBoundaryPinSpec", "#e8f3ec"),
    (5, 59, 18, 7, "newton/\nresult coupled\nchemistry_only\noperator_split", "#e8f3ec"),
    (24, 59, 18, 7, "transport.py\nassembly.py\nmatrix.py\nsolver.py", "#e8f3ec"),
    (5, 50, 37, 4, "atm_state2d.py · radiation.py · timestep.py · chemistry.py", "#f0f0e8"),
]
for x, y, w, h, text, color in boxes_l1:
    box(x, y, w, h, text, color)

# L2 (KB adapter)
ax.text(75, 85, "L2: kinetics_base.titan (KB adapter)", ha="center", fontsize=12,
        fontweight="bold", color="#c70")
boxes_l2 = [
    (55, 75, 18, 5, "atmosphere.py\npin specs · ion_scale", "#fff3e0"),
    (74, 75, 18, 5, "atm2d_sources.py\ngrain · adapter", "#fff3e0"),
    (55, 67, 18, 5, "parsing.py\npun · special · bc", "#fff3e0"),
    (74, 67, 18, 5, "source_terms.py\nsource_integration", "#fff3e0"),
    (55, 59, 18, 5, "physics.py\nthermal · sticking", "#fff3e0"),
    (74, 59, 18, 5, "radiation.py\nelectron_impact\nphotochemistry", "#fff3e0"),
    (55, 50, 37, 4, "models.py · schedule.py (KB defaults) · ion_chemistry.py · _core.py", "#f0e8d4"),
]
for x, y, w, h, text, color in boxes_l2:
    box(x, y, w, h, text, color)

# Arrows
ax.annotate("", xy=(50, 70), xytext=(50, 80),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
ax.text(53, 75, "L2 calls L1 primitives", fontsize=8, color="gray")

# L3
ax.text(50, 43, "L3: drivers (diagnostics/)", ha="center", fontsize=12,
        fontweight="bold", color="#66c")
box(15, 35, 70, 5, "no_grain_stability.py — ~600 LOC, wires L2 specs + L1 primitives\n"
                    "operator-split sub-cycling done via atm2d.newton.operator_split_advance",
    "#f0f0f7", fontsize=9)

# Bottom: shim
ax.text(50, 28, "Backward-compat shims:", ha="center", fontsize=10, fontweight="bold")
box(15, 22, 70, 5, "kinetics_base_titan/__init__.py → re-exports kinetics_base.titan",
    "#f7f0f0", fontsize=9)

# Phase numbers
phases = [
    (5, 81, "P1a"), (24, 81, "P1b/P4a"),
    (5, 73, "P1c"), (24, 73, "P3"),
    (5, 65, "P4b/P5"), (24, 65, "(kept)"),
]
for x, y, p in phases:
    ax.text(x + 0.5, y - 0.5, p, fontsize=7, color="#2a6", fontweight="bold")

# Numbers in L2
ax.text(55.5, 80.5, "P3", fontsize=7, color="#c70", fontweight="bold")
ax.text(55.5, 72.5, "(refactored)", fontsize=7, color="#c70", fontweight="bold")

# Bottom annotation
ax.text(50, 12, "Phase 6 physics (open):  "
                "(a) grain audit · (b) full ambipolar · investigated, gaps traced upstream to one root: low-alt cation bleed",
        ha="center", fontsize=9, color="#b00", style="italic")
ax.text(50, 8, "Best metric: 159/531 species-level pairs in 0.3-3× of KB · cation@L30 = 0.83× KB",
        ha="center", fontsize=10, fontweight="bold", color="#2a6")

plt.savefig(FIGDIR / "10_refactor_structure.png", dpi=110, bbox_inches="tight")
plt.close()


# ----- Figure 14: residual oscillation diagnosis (mostly noise) -----
print("[fig14] residual oscillation analysis ...")
import os

def real_osc_flips(profile, kb_profile):
    """Flips where BOTH neighbors KB>=1 and >1 OoM swing."""
    log_p = np.log10(np.maximum(profile, 1e-15))
    flips = []
    for i in range(2, 49):
        if kb_profile[i-1] < 1 or kb_profile[i] < 1 or kb_profile[i+1] < 1:
            continue
        d1 = log_p[i] - log_p[i-1]
        d2 = log_p[i+1] - log_p[i]
        if d1 * d2 < 0 and (abs(d1) > 1.0 or abs(d2) > 1.0):
            flips.append(i)
    return flips

# Tally flips by altitude band, for coupled (current) and split (was)
SPLIT_DUMP = "/tmp/kt_g29_full_dt1e7_100.npz"
if os.path.exists(SPLIT_DUMP):
    d_split_ = np.load(SPLIT_DUMP, allow_pickle=True)
    sp_split_ = [str(x) for x in d_split_["species"]]
    c_split_ = d_split_["concentration"]

bands = [(0, 10), (10, 20), (20, 30), (30, 50)]
band_labels = ["L0-10\n(0-90km)", "L10-20\n(90-400km)", "L20-30\n(400-700km)", "L30-50\n(700+km)"]
SKIP_TYPES = {"N2", "M", "JDUST", "PROD", "SGA", "U", "RAYEAR"}

split_band_flips = [0] * len(bands)
coupled_band_flips = [0] * len(bands)
for name in SPECIES:
    if name.endswith("+") or name.endswith("-") or name.startswith("G") or name in SKIP_TYPES:
        continue
    j = SPECIES.index(name)
    kbprof = np.asarray(KB.get(name, np.zeros(NLYR)))
    # coupled
    flips = real_osc_flips(C[:, j], kbprof)
    for fl in flips:
        for bi, (lo, hi) in enumerate(bands):
            if lo <= fl < hi:
                coupled_band_flips[bi] += 1
                break
    # split
    if os.path.exists(SPLIT_DUMP) and name in sp_split_:
        j_s = sp_split_.index(name)
        flips_s = real_osc_flips(c_split_[:, j_s], kbprof)
        for fl in flips_s:
            for bi, (lo, hi) in enumerate(bands):
                if lo <= fl < hi:
                    split_band_flips[bi] += 1
                    break

# Match rate by altitude band
def matched_in_band(c, sp_, kb, lo, hi):
    gd = total = 0
    for j, n in enumerate(sp_):
        for L in range(lo, hi):
            kbv = kb.get(n, [0]*50)[L]
            if kbv < 1: continue
            total += 1
            r = c[L, j]/kbv if kbv else 0
            if 0.3 < r < 3: gd += 1
    return gd, total

split_match = [matched_in_band(c_split_, sp_split_, KB, lo, hi) for lo, hi in bands]
coupled_match = [matched_in_band(C, SPECIES, KB, lo, hi) for lo, hi in bands]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))
xs = np.arange(len(bands))
axL.bar(xs - 0.2, split_band_flips, width=0.4, color="C3", label="split (was)")
axL.bar(xs + 0.2, coupled_band_flips, width=0.4, color="C2", label="coupled (now)")
axL.set_xticks(xs)
axL.set_xticklabels(band_labels)
axL.set_ylabel("real oscillation flips (KB≥1 neighbors, >1 OoM)")
axL.set_title("Oscillation flips per altitude band — split vs coupled\n"
              "Coupled wins everywhere; residual at L30-50 in upper-mesosphere is real but small")
axL.legend()
axL.grid(True, alpha=0.3, axis="y")

# Right: matched percentage per band
split_pct = [100 * g / t if t else 0 for g, t in split_match]
coupled_pct = [100 * g / t if t else 0 for g, t in coupled_match]
axR.bar(xs - 0.2, split_pct, width=0.4, color="C3", label="split (was)")
axR.bar(xs + 0.2, coupled_pct, width=0.4, color="C2", label="coupled (now)")
for i, ((sg, st), (cg, ct)) in enumerate(zip(split_match, coupled_match)):
    axR.text(i - 0.2, split_pct[i] + 1, f"{sg}/{st}", ha="center", fontsize=8)
    axR.text(i + 0.2, coupled_pct[i] + 1, f"{cg}/{ct}", ha="center", fontsize=8)
axR.set_xticks(xs)
axR.set_xticklabels(band_labels)
axR.set_ylabel("% species-lev pairs matched (0.3-3×)")
axR.set_ylim(0, 60)
axR.set_title("Match rate per altitude band — coupled improves L0-30,\nslight regression at L30-50 (escape zone)")
axR.legend()
axR.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGDIR / "14_osc_by_band.png", dpi=110)
plt.close()


# ----- Figure 13: zigzag verification (coupled vs split for the worst species) -----
print("[fig13] zigzag verification (coupled vs split) ...")

# Reload split-mode baseline for comparison
import os
SPLIT_DUMP = "/tmp/kt_g29_full_dt1e7_100.npz"  # leftover from earlier G29 work
if os.path.exists(SPLIT_DUMP):
    d_split = np.load(SPLIT_DUMP, allow_pickle=True)
    sp_split = [str(x) for x in d_split["species"]]
    c_split = d_split["concentration"]
else:
    sp_split = None
    c_split = None

# Pick the species that were originally worst oscillators
zigzag_targets = ["H2", "HCN", "C6H6", "CN", "C8H3", "c-C3H2",
                  "NH3", "NH2", "C5H3", "C3H7", "C4N2", "N(2D)"]
fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharey=True)
for ax, name in zip(axes.flat, zigzag_targets):
    if name not in SPECIES:
        ax.set_visible(False)
        continue
    j = SPECIES.index(name)
    kb_prof = np.asarray(KB.get(name, np.zeros(NLYR)))
    ax.semilogx(np.maximum(C[:, j], 1e-15), ALT[:NLYR],
                "C2-", lw=2, label="coupled (now)", marker="o", markersize=3)
    if c_split is not None and name in sp_split:
        j_split = sp_split.index(name)
        ax.semilogx(np.maximum(c_split[:, j_split], 1e-15), ALT[:NLYR],
                    "C3-", lw=1.2, alpha=0.6, label="split (was)", marker="x", markersize=3)
    ax.semilogx(np.maximum(kb_prof, 1e-15), ALT[:NLYR],
                "k--", lw=1.5, alpha=0.6, label="KB")
    ax.set_title(name)
    ax.set_xlim(1e-3, 1e15)
    ax.set_ylim(0, 700)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

axes[2, 0].set_xlabel("number density [cm⁻³]")
axes[2, 1].set_xlabel("number density [cm⁻³]")
axes[2, 2].set_xlabel("number density [cm⁻³]")
axes[2, 3].set_xlabel("number density [cm⁻³]")
axes[0, 0].set_ylabel("altitude [km]")
axes[1, 0].set_ylabel("altitude [km]")
axes[2, 0].set_ylabel("altitude [km]")

plt.suptitle(
    "Zigzag verification — 12 worst-oscillating species\n"
    "GREEN = current (coupled), RED = previous (split, sawtooth), BLACK DASH = KB",
    y=1.005,
)
plt.tight_layout()
plt.savefig(FIGDIR / "13_zigzag_verify.png", dpi=110)
plt.close()


# ----- Figure 12: oscillation in lower atmosphere -----
print("[fig12] oscillation in lower atm ...")
def osc_score(profile, hi=30):
    log_p = np.log10(np.maximum(profile[:hi + 1], 1e-15))
    diffs = np.diff(log_p)
    signs = np.sign(diffs)
    return sum(
        1 for i in range(1, len(signs))
        if signs[i] * signs[i - 1] < 0 and abs(diffs[i]) > 1.0
    )

# Score all neutral non-fixed species
osc_scores = []
SKIP = {"N2", "M", "JDUST", "PROD", "SGA", "U", "RAYEAR"}
for name in SPECIES:
    if name.endswith("+") or name.endswith("-") or name.startswith("G"):
        continue
    if name in SKIP:
        continue
    j = SPECIES.index(name)
    sc = osc_score(C[:, j])
    if sc >= 3:
        kbsc = osc_score(np.asarray(KB.get(name, np.zeros(NLYR))))
        osc_scores.append((sc, kbsc, name))
osc_scores.sort(reverse=True)

# Plot 1: oscillation count histogram
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5))
top_osc = osc_scores[:20]
labels = [s[2] for s in top_osc]
kt_flips = [s[0] for s in top_osc]
kb_flips = [s[1] for s in top_osc]
xs = np.arange(len(labels))
axL.bar(xs - 0.2, kt_flips, width=0.4, label="kintera", color="C3")
axL.bar(xs + 0.2, kb_flips, width=0.4, label="KB", color="C2")
axL.set_xticks(xs)
axL.set_xticklabels(labels, rotation=45, ha="right")
axL.set_ylabel(">1 OoM zigzag flips (L0-30)")
axL.set_title("Sawtooth oscillation count — top 20 neutrals\nKB profiles are smooth (=0 flips)")
axL.legend()
axL.grid(True, alpha=0.3, axis="y")

# Plot 2: H2 + HCN + C6H6 profiles showing the zigzag
example_names = ["H2", "HCN", "C6H6", "CN"]
for name in example_names:
    if name not in SPECIES:
        continue
    j = SPECIES.index(name)
    prof = C[:, j]
    kb_prof = np.asarray(KB.get(name, np.zeros(NLYR)))
    axR.semilogx(np.maximum(prof, 1e-3), ALT[:NLYR], "-", label=f"{name} kt", lw=2)
    axR.semilogx(np.maximum(kb_prof, 1e-3), ALT[:NLYR], ":", alpha=0.6, lw=1.5,
                 label=f"{name} KB")
axR.set_xlabel("number density [cm⁻³]")
axR.set_ylabel("altitude [km]")
axR.set_ylim(0, 700)
axR.set_xlim(1e-3, 1e13)
axR.set_title("Examples of profile zigzag (solid lines)\nzeros at random levels between high values")
axR.legend(fontsize=8, ncol=2)
axR.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "12_oscillation.png", dpi=110)
plt.close()


# ----- Figure 11: next-steps decision tree -----
print("[fig11] next steps ...")
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

ax.text(50, 96, "Next-step options — recommended order",
        ha="center", fontsize=14, fontweight="bold")

def opt_box(x, y, w, h, num, title, body, color="#e8f3ec"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", lw=1.5))
    ax.text(x + 1, y + h - 2, f"#{num}", fontsize=11, fontweight="bold", color="#2a6")
    ax.text(x + 6, y + h - 2, title, fontsize=11, fontweight="bold")
    ax.text(x + 1, y + h - 5.5, body, fontsize=8.5, va="top", wrap=True)

# Option 1: Coupled Newton (G18 redux)
opt_box(2, 75, 96, 18, 1,
        "Coupled transport+chemistry Newton — the only fix for sawtooth (G18 redux)",
        "Diagnosis (5 hypotheses ruled out): zigzag NOT caused by transport (transport-only smooths 7→0 flips), NOT by atomic\n"
        "projection (column/soft variants worse), NOT by apply_pins (removing it changes nothing). Root cause = chemistry-only Newton\n"
        "per-cell independence — adjacent cells lock into different attractor basins, transport between outer steps doesn't keep up.\n"
        "Plan: revisit G18. Use new Phase 3 BoundaryPinSpec / Phase 4b newton.coupled module to retry with Phase 6b ion_scale knob\n"
        "and look at WHICH coupled config (max_iter, damping, dt cap) survives — G18 had instability at small dt, not at moderate.\n"
        "Effort: 1-2 sessions    Risk: medium-high (G18 history)    Reward: smooth profiles + closes whole NH3-feedback root.",
        "#fff3e0")

# Option 2: Diagnose NT>100 regression
opt_box(2, 54, 96, 18, 2,
        "Diagnose NT>100 regression",
        "Symptom: NT=100→200 (dt=1e+7) matched drops 159→141 even though physical time grows from 32y to 63y.\n"
        "Hypothesis A: Newton finds a non-physical fixed-point at long time. Hypothesis B: atomic projection over-trims slow species.\n"
        "Plan: dump intermediate NT=120/140/160/180 → diff which species cross the boundary first.\n"
        "Effort: ~1 hour    Risk: medium    Reward: if root-cause found, may unlock NT&gt;500 = direct KB-time comparison.",
        "#e8f3ec")

# Option 3: Real ambipolar
opt_box(2, 33, 96, 18, 3,
        "Real ambipolar diffusion (co-solved ions+electrons)",
        "Physics: cations and electrons drift together with effective D = D_i × (1 + T_e/T_i), constrained by partner mobility.\n"
        "Needs: a per-cell ambipolar coefficient computed from local [E], [cation], T; modify atm2d.transport for charged species.\n"
        "Sweep showed uniform-scale is wrong shape — only proper coupling can fix G30 NH3 feedback.\n"
        "Effort: 1–2 sessions    Risk: high (Newton convergence at small dt)    Reward: closes biggest known physics gap.",
        "#f0f0f7")

# Option 4: Per-reaction audit
opt_box(2, 12, 96, 18, 4,
        "Per-reaction audit for mid-altitude neutrals (CH3, HCN @ L15-20)",
        "Tool: diagnostic_tools/rate_diff.py (already built). Trace specific production/loss reactions vs KB at offending levels.\n"
        "Mirrors the G25 audit that found C+ cascade root cause. Already done for NH3 (G30).\n"
        "Next targets: CH3 lev 20 (0.73× KB), HCN lev 15 (zero in our model but ~3e+9 in KB).\n"
        "Effort: ~1 hour each species    Risk: low    Reward: variable.",
        "#f7f0f0")

plt.tight_layout()
plt.savefig(FIGDIR / "11_next_steps.png", dpi=110, bbox_inches="tight")
plt.close()


print("[done] generated", len(list(FIGDIR.glob("*.png"))), "figures in", FIGDIR)
