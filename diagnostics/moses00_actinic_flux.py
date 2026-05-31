"""Dump kintera's actinic flux per wavelength at key altitudes, and
compare against an analytical Beer-Lambert calculation (top-flux × exp(-τ))
using kintera's own opacity sources. If the DISORT result differs from the
analytic exp(-τ) by a lot, the radiative-transfer setup has a bug.

For a direct-beam (single-stream, no scattering) approximation, the actinic
flux at altitude z should be F_top × exp(-τ(z, λ) / μ_0), where τ is the
vertical optical depth from top down to z, and μ_0 is the cosine of the
solar zenith angle.
"""
from __future__ import annotations
import sys, math
sys.path.insert(0, '/home/sam2/dev/kintera/diagnostics')
exec(open('/home/sam2/dev/kintera/diagnostics/moses00_perturb.py').read().split('def main()')[0])

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kintera.kinetics_base.titan.radiation import _kinetics_base_pyharp_actinic_flux
from kintera.kinetics_base.titan.models import KBTitanState

ts, filtered, species = build_state_and_sources()
ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))
ts.state.concentration = inject_state(ts, ref)
ts.concentration = ts.state.concentration
species_index = {s: i for i, s in enumerate(species)}

# Pick a representative photo term (CH4 → CH3 + H)
ch4_term = next(t for t in filtered if t.kind == "pun_photo_rate_reaction"
                and t.reactants == ["CH4"] and "CH3" in t.products)

wavelengths = ch4_term.parameters["wavelengths"]  # Å
nwave = len(wavelengths)
flux_top = ch4_term.parameters["flux"]  # photons / cm² / s / wave-bin at top

src_state = KBTitanState(
    species=ts.species, fixed_species=ts.fixed_species,
    varying_species=ts.varying_species, conversion=ts.conversion,
    concentration=ts.concentration, density=ts.density,
    kzz=ts.kzz, state=ts.state,
)

# 1. Kintera's DISORT actinic flux
flux_tensor = torch.tensor(flux_top, dtype=torch.float64)
actinic = _kinetics_base_pyharp_actinic_flux(
    ch4_term, src_state, ts.state.concentration, species_index,
    flux_tensor, nwave, torch.float64, torch.device("cpu"),
)  # shape (ncol, nlyr, nwave)

# 2. Analytical Beer-Lambert: compute τ(z, λ) from top down for each wavelength
xs_by_sp = ch4_term.parameters["total_cross_section_by_species"]
nlyr = 91
dz_cm = ts.state.dx1f
extinction_per_layer = torch.zeros(nlyr, nwave, dtype=torch.float64)
for sp_name, sigma_list in xs_by_sp.items():
    if sp_name not in species_index:
        continue
    i = species_index[sp_name]
    sigma = torch.tensor(sigma_list, dtype=torch.float64)
    c_col = ts.state.concentration[0, :, i]
    extinction_per_layer += c_col.unsqueeze(-1) * sigma.unsqueeze(0)

# tau from top to each layer using reverse cumulative sum
# tau at L = sum of ext × dz for layers L..L_top (column at and above L)
ext_dz = extinction_per_layer * dz_cm.unsqueeze(-1)  # (nlyr, nwave)
tau_from_top = torch.flip(torch.cumsum(torch.flip(ext_dz, dims=[0]), dim=0), dims=[0])

# Solar mu0
mu0 = float(ch4_term.parameters.get("solar_mu0", 0.5))
print(f"Solar mu0 = {mu0}")

# Beer-Lambert analytic actinic flux: F_top × exp(-τ/μ0)
analytic_flux = flux_tensor.view(1, -1) * torch.exp(-tau_from_top / mu0)

# Comparison at key altitudes for a sample of wavelengths
altitudes_km = ts.state.x1v.numpy() / 1.0e5
LAYERS_OF_INTEREST = [10, 20, 30, 40, 50, 60, 70, 80, 85]
# Pick wavelengths where CH4 cross-section is non-negligible
ch4_sigma = torch.tensor(xs_by_sp["CH4"])
imp_waves = ch4_sigma.argsort(descending=True)[:8].tolist()

print(f"\n{'='*100}")
print(f"Actinic flux comparison (kintera DISORT vs Beer-Lambert exp(-τ/μ0))")
print(f"{'='*100}")
for k in imp_waves:
    wl = wavelengths[k]
    print(f"\nλ = {wl:>7.1f} Å:")
    print(f"  {'L':<4} {'z (km)':<8} {'τ/μ0':<12} {'kt DISORT':<15} {'analytic':<15} {'ratio kt/an':<12}")
    for L in LAYERS_OF_INTEREST:
        kt_f = actinic[0, L, k].item()
        an_f = analytic_flux[L, k].item()
        ratio = kt_f / an_f if abs(an_f) > 1e-100 else float('nan')
        print(f"  L{L:<3d} {altitudes_km[L]:<8.0f} {tau_from_top[L,k].item()/mu0:<12.2e} "
              f"{kt_f:<15.3e} {an_f:<15.3e} {ratio:<12.3f}")

# Plot: kintera vs analytic, several wavelengths, vs altitude
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, k in zip(axes.flat, imp_waves):
    wl = wavelengths[k]
    kt_f = actinic[0, :, k].numpy()
    an_f = analytic_flux[:, k].numpy()
    ax.semilogx(np.clip(kt_f, 1e-60, None), altitudes_km, "o-",
                color="#cc3333", label="kintera DISORT", ms=2)
    ax.semilogx(np.clip(an_f, 1e-60, None), altitudes_km, "s-",
                color="#000080", label="analytic exp(-τ/μ0)", ms=2)
    ax.axhline(altitudes_km[20], color="orange", lw=0.5, alpha=0.5)
    ax.text(np.clip(kt_f, 1e-60, None).max(), altitudes_km[20], "L20", fontsize=7)
    ax.set_xlabel("actinic flux (photons cm⁻² s⁻¹)")
    ax.set_ylabel("altitude (km)")
    ax.set_title(f"λ={wl:.0f} Å (σ_CH4={ch4_sigma[k].item():.2e})")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/moses00_actinic_flux.png", dpi=72, bbox_inches="tight")
print(f"\nSaved comparison plot to /tmp/moses00_actinic_flux.png")
