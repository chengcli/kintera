"""
kintera radiative transfer module using pyharp DISORT.

Provides actinic flux and J-value computation for photochemistry
via pyharp's discrete ordinates solver (pure absorption, no scattering).
"""
import numpy as np
import torch
from pyharp import pydisort

_disort_cache = {}


def _get_disort(nz, nwave):
    """Return a cached DISORT solver for the given (nz, nwave) dimensions."""
    key = (nz, nwave)
    if key not in _disort_cache:
        nstr = 4
        opts = pydisort.DisortOptions()
        opts.ds().nlyr = nz
        opts.ds().nstr = nstr
        opts.ds().nmom = nstr
        opts.ds().nphase = 1
        opts.nwave(nwave)
        opts.ncol(1)
        opts.flags("lamber,onlyfl,quiet")
        _disort_cache[key] = pydisort.Disort(opts)
    return _disort_cache[key]


def compute_actinic_flux(y, cross_O2, cross_O3, stellar_flux, dzi,
                         cos_zen, wavelengths, absorber_ids=(1, 2)):
    """
    Compute actinic flux using pyharp DISORT.

    Parameters
    ----------
    y : ndarray (nz, ni)
        Number densities at each layer (bottom-up ordering).
    cross_O2 : ndarray (nwave,)
        O2 absorption cross-section [cm^2].
    cross_O3 : ndarray (nwave,)
        O3 absorption cross-section [cm^2].
    stellar_flux : ndarray (nwave,)
        TOA stellar flux [photons/cm^2/s/nm].
    dzi : ndarray (nz-1,)
        Interface spacing [cm].
    cos_zen : float
        Cosine of solar zenith angle.
    wavelengths : ndarray (nwave,)
        Wavelength grid [nm].
    absorber_ids : tuple
        Indices into y's species axis for (O2, O3).

    Returns
    -------
    aflux : ndarray (nz, nwave)
        Actinic flux at layer centers [photons/cm^2/s/nm].
    """
    nz, ni = y.shape
    nwave = len(wavelengths)

    dz = np.zeros(nz)
    dz[0] = dzi[0]
    dz[-1] = dzi[-1]
    for j in range(1, nz - 1):
        dz[j] = (dzi[j - 1] + dzi[j]) / 2.0

    n_O2 = y[:, absorber_ids[0]]
    n_O3 = y[:, absorber_ids[1]]
    alpha = n_O2[:, None] * cross_O2[None, :] + n_O3[:, None] * cross_O3[None, :]
    dtau = alpha * dz[:, None]  # (nz, nwave)

    nstr = 4
    nprop = 2 + nstr + 1

    d = _get_disort(nz, nwave)

    prop = torch.zeros(nwave, 1, nz, nprop, dtype=torch.float64)
    for k in range(nz):
        layer = nz - 1 - k  # bottom-up to DISORT top-down
        prop[:, 0, k, 0] = torch.from_numpy(dtau[layer].copy())

    result = d.forward(
        prop, "", None,
        fbeam=torch.from_numpy(stellar_flux.copy()).reshape(nwave, 1).to(torch.float64),
        umu0=torch.tensor([cos_zen], dtype=torch.float64),
        phi0=torch.zeros(1, dtype=torch.float64),
        albedo=torch.zeros(nwave, 1, dtype=torch.float64),
    )

    # result: (nwave, 1, nz+1, 2) — [upflux, downflux] at each level
    # DISORT level 0 = TOA, level nz = surface
    flux_dn = result[:, 0, :, 1].numpy()  # (nwave, nz+1)

    # Convert horizontal flux to actinic flux; average interfaces for layer centers
    aflux = np.zeros((nz, nwave))
    for j in range(nz):
        k_top = nz - 1 - j
        k_bot = nz - j
        aflux[j] = (flux_dn[:, k_top] + flux_dn[:, k_bot]) / (2.0 * cos_zen)

    return aflux


def compute_J(y, cross_O2, cross_O3, stellar_flux, dzi, cos_zen, wavelengths,
              absorber_ids=(1, 2)):
    """
    Compute photolysis rates J_O2 and J_O3 using pyharp DISORT.

    Returns
    -------
    J_O2 : ndarray (nz,)
    J_O3 : ndarray (nz,)
    """
    aflux = compute_actinic_flux(y, cross_O2, cross_O3, stellar_flux, dzi,
                                 cos_zen, wavelengths, absorber_ids)
    J_O2 = np.trapz(cross_O2[None, :] * aflux, wavelengths, axis=1)
    J_O3 = np.trapz(cross_O3[None, :] * aflux, wavelengths, axis=1)
    return J_O2, J_O3
