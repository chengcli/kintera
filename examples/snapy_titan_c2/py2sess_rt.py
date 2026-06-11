"""py2sess radiative-transfer driver for the Titan C2 snapy case.

Computes the per-wavelength actinic flux that drives the photolysis rates in
:mod:`titan_c2_chem`, using py2sess's native level-flux kernels
(``TwoStreamEss.forward_flux``; torch backend, CPU or CUDA).

Conventions (verified against Beer-Lambert in test_titan_c2.py, Gate C1):
- ``forward_flux`` returns the level *horizontal-plane* downward flux
  ``flux_down = mu0 * fbeam * exp(-tau/mu0)`` (TOA->BOA levels) for the
  pure-absorption limit. The actinic flux (beam intensity) is therefore
  ``flux_down / mu0``. Titan's UV photolysis region is absorption-dominated
  (CH4/C2 cross-sections; ssa ~ 0), so this is exact; adding haze scattering
  would require a mean-intensity output instead.
- The geometry axis of forward_flux is an outer product with the profile
  batch, so columns are grouped into SZA bins (``sza_bin_deg`` wide) and one
  batched call is made per bin: total work stays ncols x nwave.
- ``fbeam`` is normalized to 1; the spectral TOA flux (photons/cm^2/s at
  Titan's distance, from the validated KB pipeline) plus the transmission of
  the overhead atmosphere ABOVE the model lid multiply the result per bin.

All tensors stay on the construction device; CUDA end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from py2sess.api import TwoStreamEss, TwoStreamEssOptions

SI_TO_CM3 = 6.02214076e23 / 1.0e6   # mol/m^3 -> molecule/cm^3


class TitanC2Radiation:
    """Actinic-flux driver: absorber columns -> (ncol, nlyr, nwave)."""

    def __init__(self, data_npz: str | Path, nlyr: int,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64,
                 mu_min: float = 0.05, sza_bin_deg: float = 1.0,
                 lid_altitude_km: float | None = None):
        self.device = torch.device(device)
        self.dtype = dtype
        self.mu_min = float(mu_min)
        self.sza_bin_deg = float(sza_bin_deg)
        d = np.load(data_npz, allow_pickle=False)
        self.wavelengths = torch.tensor(d["wavelengths"], dtype=dtype,
                                        device=self.device)
        self.nwave = self.wavelengths.numel()
        self.abs_species = [str(s) for s in d["abs_species"]]
        self.abs_sigma = torch.tensor(d["abs_sigma"], dtype=dtype,
                                      device=self.device)   # (nabs, nwave) cm^2
        self.toa_flux = torch.tensor(d["toa_flux"], dtype=dtype,
                                     device=self.device)    # ph/cm^2/s at Titan

        # transmission of the overhead column above the model lid, from the
        # moses05 atm profile (a fixed background; the lid column is dominated
        # by CH4 whose vmr is nearly constant up to the homopause).
        if lid_altitude_km is not None:
            self.toa_transmission = self._overhead_transmission(
                d, lid_altitude_km)
        else:
            self.toa_transmission = torch.ones_like(self.toa_flux)

        self.nlyr = int(nlyr)
        opt = TwoStreamEssOptions(
            nlyr=self.nlyr, backend="native", mode="solar",
            plane_parallel=True,
            torch_device=str(self.device), torch_dtype="float64")
        self._ess = TwoStreamEss(opt)
        # dummy height grid (km, TOA->BOA); plane-parallel solar fluxes only
        # depend on tau, not z spacing.
        self._z = torch.linspace(float(self.nlyr), 0.0, self.nlyr + 1,
                                 dtype=torch.float64)

    def _overhead_transmission(self, d, lid_km: float) -> torch.Tensor:
        z = np.asarray(d["atm_altitude_km"], dtype=np.float64)
        dens = np.asarray(d["atm_density"], dtype=np.float64)      # cm^-3
        names = [str(s) for s in d["atm_profile_species"]]
        vmr = np.asarray(d["atm_profile_vmr"], dtype=np.float64)
        order = np.argsort(z)
        z, dens = z[order], dens[order]
        above = z >= lid_km
        if not above.any():
            return torch.ones_like(self.toa_flux)
        col_tau = np.zeros(self.nwave)
        zc = z[above]
        dz_cm = np.gradient(zc) * 1.0e5
        for ia, sp in enumerate(self.abs_species):
            if sp not in names:
                continue
            prof = vmr[names.index(sp)][order][above] * dens[above]
            col = float(np.sum(prof * dz_cm))                       # cm^-2
            col_tau += col * self.abs_sigma[ia].cpu().numpy()
        return torch.tensor(np.exp(-col_tau), dtype=self.dtype,
                            device=self.device)

    @staticmethod
    def cos_sza(lon: torch.Tensor, lat: torch.Tensor,
                subsolar_lon: float, subsolar_lat: float) -> torch.Tensor:
        """mu0 on the sphere; lon/lat and subsolar point in RADIANS."""
        return (torch.sin(lat) * np.sin(subsolar_lat)
                + torch.cos(lat) * np.cos(subsolar_lat)
                * torch.cos(lon - subsolar_lon))

    def optical_depth(self, conc_abs: torch.Tensor,
                      dz_cm: torch.Tensor) -> torch.Tensor:
        """tau (ncol, nlyr, nwave), layers ordered TOA->BOA.

        Args:
            conc_abs: (ncol, nlyr, nabs) absorber number densities
                      [molecule/cm^3], layer index 0 = TOP of atmosphere.
            dz_cm: scalar | (nlyr,) | (ncol, nlyr) layer thickness [cm].
        """
        if dz_cm.dim() == 1:
            dz_cm = dz_cm.view(1, -1, 1)
        elif dz_cm.dim() == 2:
            dz_cm = dz_cm.unsqueeze(-1)
        return torch.einsum("cla,aw->clw", conc_abs * dz_cm, self.abs_sigma)

    def actinic_flux(self, conc_abs: torch.Tensor, dz_cm: torch.Tensor,
                     mu0: torch.Tensor) -> torch.Tensor:
        """Layer-center actinic flux (ncol, nlyr, nwave) [photons/cm^2/s].

        Args:
            conc_abs: (ncol, nlyr, nabs) [molecule/cm^3], layer 0 = TOA.
            dz_cm: layer thickness [cm] (scalar/(nlyr,)/(ncol,nlyr)).
            mu0: (ncol,) cosine of solar zenith angle (<= mu_min -> night).
        Output layer index 0 = TOA (same ordering as conc_abs).
        """
        ncol = conc_abs.shape[0]
        tau = self.optical_depth(conc_abs, dz_cm)            # (ncol,nlyr,nw)
        out = torch.zeros(ncol, self.nlyr, self.nwave,
                          dtype=self.dtype, device=self.device)
        day = mu0 >= self.mu_min
        if not day.any():
            return out
        day_idx = day.nonzero(as_tuple=True)[0]
        mu_day = mu0[day_idx]
        # group day columns into SZA bins; one batched call per bin
        sza_deg = torch.rad2deg(torch.acos(mu_day.clamp(-1.0, 1.0)))
        bins = torch.round(sza_deg / self.sza_bin_deg).to(torch.int64)
        for b in bins.unique():
            sel = day_idx[bins == b]
            sza_c = float(b) * self.sza_bin_deg
            mu_c = float(np.cos(np.radians(sza_c)))
            if mu_c < self.mu_min:
                continue
            t = tau[sel].movedim(-1, 1).reshape(-1, self.nlyr)   # (nsel*nw, nlyr)
            ssa = torch.full_like(t, 1e-9)
            res = self._ess.forward_flux(
                tau=t, ssa=ssa, g=torch.zeros_like(t), z=self._z,
                angles=np.array([sza_c, 0.0, 0.0]), fbeam=1.0, albedo=0.0)
            fd = res.flux_down                                   # (nsel*nw, nlyr+1)
            fact_lvl = fd / mu_c                                  # beam intensity
            fact = 0.5 * (fact_lvl[:, :-1] + fact_lvl[:, 1:])     # layer centers
            out[sel] = fact.reshape(len(sel), self.nwave, self.nlyr).movedim(1, -1)
        return out * (self.toa_flux * self.toa_transmission).view(1, 1, -1)
