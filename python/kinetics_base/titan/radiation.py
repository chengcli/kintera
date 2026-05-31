from __future__ import annotations

import math
from typing import Any

import torch
import pyharp

from .parsing import _align_kinetics_base_absorption_cross_section, _linear_interpolate


def _kinetics_base_pyharp_actinic_flux(
    term: KBTitanSourceTerm,
    titan_state: KBTitanState,
    concentration: torch.Tensor,
    species_index: dict[str, int],
    top_flux: torch.Tensor,
    nwave: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    total_cross_sections = term.parameters.get("total_cross_section_by_species")
    if not isinstance(total_cross_sections, dict):
        return None

    dz = titan_state.state.dx1f.to(dtype=dtype, device=device).view(1, -1)
    gas_extinction = torch.zeros(
        (*concentration.shape[:2], nwave), dtype=dtype, device=device
    )
    for species_name, values in total_cross_sections.items():
        idx = species_index.get(str(species_name))
        if idx is None or not isinstance(values, list) or len(values) != nwave:
            continue
        sigma = torch.tensor(values, dtype=dtype, device=device)
        gas_extinction = gas_extinction + torch.clamp(
            concentration[:, :, idx], min=0.0
        ).unsqueeze(-1) * sigma.view(1, 1, -1)

    aerosol_extinction_profile = torch.zeros_like(gas_extinction)
    aerosol_extinction = _aerosol_extinction_on_state_grid(
        term, titan_state, nwave, dtype, device
    )
    jdust_idx = species_index.get("JDUST")
    if aerosol_extinction is not None and jdust_idx is not None:
        aerosol_extinction_profile = torch.clamp(
            concentration[:, :, jdust_idx], min=0.0
        ).unsqueeze(-1) * aerosol_extinction.unsqueeze(0)
    extinction = gas_extinction + aerosol_extinction_profile
    if bool(term.parameters.get("kinetics_direct_radiation", False)):
        return _kinetics_base_direct_actinic_flux(
            term,
            titan_state,
            concentration,
            species_index,
            top_flux,
            dtype,
            device,
        )

    optical_depth = torch.clamp(
        extinction
        * dz.unsqueeze(-1)
        * float(term.parameters.get("optical_depth_scale", 1.0)),
        min=0.0,
        max=700.0,
    )
    active_nlyr = term.parameters.get("radiation_active_nlyr")
    if active_nlyr is not None:
        try:
            active_nlyr_int = int(active_nlyr)
        except (TypeError, ValueError):
            active_nlyr_int = titan_state.state.nlyr
        if 0 < active_nlyr_int < titan_state.state.nlyr:
            optical_depth = optical_depth.clone()
            optical_depth[:, active_nlyr_int:, :] = 0.0

    pydisort = pyharp.pydisort
    options = pydisort.DisortOptions().flags("onlyfl,lamber,quiet")
    options.ds().nlyr = titan_state.state.nlyr
    options.ds().nstr = int(term.parameters.get("radiation_streams", 4))
    options.ds().nmom = options.ds().nstr
    options.ds().nphase = options.ds().nstr
    options.nwave(nwave).ncol(titan_state.state.ncol)
    disort = pydisort.Disort(options)

    nprop = 2 + options.ds().nmom
    prop = torch.zeros(
        (nwave, titan_state.state.ncol, titan_state.state.nlyr, nprop),
        dtype=dtype,
        device=device,
    )
    prop[..., 0] = torch.flip(optical_depth, dims=[1]).permute(2, 0, 1)
    scattering_optical_depth = torch.zeros_like(optical_depth)
    aerosol_albedo = _aerosol_property_on_state_grid(
        term, "aerosol_albedo", titan_state, nwave, dtype, device
    )
    if aerosol_albedo is not None:
        scattering_optical_depth = aerosol_extinction_profile * dz.unsqueeze(-1)
        scattering_optical_depth = scattering_optical_depth * torch.clamp(
            aerosol_albedo.unsqueeze(0), min=0.0, max=1.0
        )
    scattering_optical_depth = scattering_optical_depth * float(
        term.parameters.get("optical_depth_scale", 1.0)
    )
    if active_nlyr is not None and 0 < active_nlyr_int < titan_state.state.nlyr:
        scattering_optical_depth = scattering_optical_depth.clone()
        scattering_optical_depth[:, active_nlyr_int:, :] = 0.0
    positive_tau = optical_depth > 0.0
    single_scattering_albedo = torch.zeros_like(optical_depth)
    single_scattering_albedo[positive_tau] = (
        scattering_optical_depth[positive_tau] / optical_depth[positive_tau]
    )
    prop[..., 1] = torch.flip(
        torch.clamp(single_scattering_albedo, min=0.0, max=1.0), dims=[1]
    ).permute(2, 0, 1)

    aerosol_asymmetry = _aerosol_property_on_state_grid(
        term, "aerosol_asymmetry", titan_state, nwave, dtype, device
    )
    if aerosol_asymmetry is not None:
        g = torch.flip(torch.clamp(aerosol_asymmetry, min=-0.999, max=0.999), dims=[0])
        for moment in range(options.ds().nmom):
            prop[..., 2 + moment] = (
                g.pow(moment + 1).transpose(0, 1).unsqueeze(1)
            )

    top_flux_2d = top_flux.view(nwave, 1).expand(nwave, titan_state.state.ncol)
    zeros = torch.zeros(titan_state.state.ncol, dtype=dtype, device=device)
    albedo = torch.full(
        (nwave, titan_state.state.ncol),
        max(min(float(term.parameters.get("surface_albedo", 0.0)), 1.0), 0.0),
        dtype=dtype,
        device=device,
    )
    actinic_flux = torch.zeros(
        (*concentration.shape[:2], nwave), dtype=dtype, device=device
    )
    for mu0, weight in _kinetics_base_solar_mu0_weights(term.parameters):
        if weight <= 0.0:
            continue
        disort.forward(
            prop,
            umu0=torch.full_like(zeros, mu0),
            phi0=zeros,
            fbeam=top_flux_2d,
            albedo=albedo,
        )
        gathered = disort.gather_flx()
        average_intensity_levels = gathered[..., pydisort.kIUAVG]
        # KB-2012's JPHOTO accumulates Σ(σ × downward_flux) where downward_flux
        # = π × J̄ for the diffuse component (plus direct-beam projection). The
        # Cheng_cross/ σ tables were tabulated against that convention, so to
        # reproduce KB rates we use π × J̄ here, not the physics-standard
        # actinic flux 4π × J̄. Direct comparison kintera vs KB per-reaction
        # at fort.7 SS gave a constant 4.000× over-rate that this corrects.
        # See project-moses00-kb-layer-offset memory for the trace.
        actinic_flux_levels = torch.flip(
            (torch.pi * average_intensity_levels).permute(1, 2, 0),
            dims=[1],
        )
        actinic_flux = actinic_flux + weight * actinic_flux_levels[:, 1:, :]
    return actinic_flux

def _kinetics_base_direct_actinic_flux(
    term: KBTitanSourceTerm,
    titan_state: KBTitanState,
    concentration: torch.Tensor,
    species_index: dict[str, int],
    top_flux: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ncol, nlyr, nwave = concentration.shape[0], concentration.shape[1], top_flux.numel()
    active_nlyr = min(int(term.parameters.get("radiation_active_nlyr") or nlyr), nlyr)
    opacity = term.parameters.get("total_cross_section_by_species", {})
    if not isinstance(opacity, dict) or active_nlyr <= 0:
        return top_flux.view(1, 1, nwave).expand(ncol, nlyr, nwave)

    alt_km = titan_state.state.x1v.to(dtype=dtype, device=device) / 1.0e5
    tau = torch.zeros((ncol, nlyr, nwave), dtype=dtype, device=device)
    species_ext: list[torch.Tensor] = []
    species_concentration: list[torch.Tensor] = []
    for name, values in opacity.items():
        idx = species_index.get(str(name))
        if idx is None:
            continue
        sigma = torch.tensor(values, dtype=dtype, device=device)
        if sigma.numel() != nwave:
            continue
        conc = torch.clamp(concentration[:, :active_nlyr, idx], min=0.0)
        species_concentration.append(conc)
        species_ext.append(conc.unsqueeze(-1) * sigma.view(1, 1, nwave))
    if not species_ext:
        return top_flux.view(1, 1, nwave).expand(ncol, nlyr, nwave)

    total_ext = torch.stack(species_ext, dim=0).sum(dim=0)
    column = torch.zeros((ncol, active_nlyr, nwave), dtype=dtype, device=device)
    if active_nlyr > 1:
        top_tau = torch.zeros((ncol, nwave), dtype=dtype, device=device)
        for conc, ext in zip(species_concentration, species_ext):
            c0 = conc[:, active_nlyr - 2]
            c1 = conc[:, active_nlyr - 1]
            dz_top = alt_km[active_nlyr - 1] - alt_km[active_nlyr - 2]
            scale_height = torch.full_like(c1, 10.0)
            valid = (c0 != 0.0) & (c1 != 0.0) & (c0 != c1)
            ratio = torch.zeros_like(c1)
            ratio[valid] = torch.abs(c1[valid]) / torch.abs(c0[valid])
            scale_height[valid] = torch.abs(dz_top / torch.log(ratio[valid]))
            top_tau = top_tau + scale_height.unsqueeze(-1) * ext[:, active_nlyr - 1, :]
        column[:, active_nlyr - 1, :] = top_tau * 1.0e5
        for i in range(active_nlyr - 2, -1, -1):
            dz_km = alt_km[i + 1] - alt_km[i]
            layer_tau = 0.5 * (total_ext[:, i, :] + total_ext[:, i + 1, :])
            layer_tau = layer_tau * dz_km * 1.0e5
            column[:, i, :] = column[:, i + 1, :] + layer_tau

    column = torch.clamp(
        column * float(term.parameters.get("optical_depth_scale", 1.0)),
        min=0.0,
        max=700.0,
    )
    lat = term.parameters.get("solar_latitude_deg")
    dec = term.parameters.get("solar_declination_deg")
    if lat is None or dec is None:
        mu0 = max(min(float(term.parameters.get("solar_mu0", 1.0)), 1.0), 1.0e-6)
        attenuation = torch.exp(-column / mu0)
    else:
        latitude = math.radians(float(lat))
        declination = math.radians(float(dec))
        a5 = math.sin(declination) * math.sin(latitude)
        b5 = math.cos(declination) * math.cos(latitude)
        apb = max(a5 + b5, 1.0e-12)
        cof0 = a5 * math.log(2.0) / b5 if abs(b5) > 1.0e-12 else 0.0
        cf = column / apb
        arg = torch.clamp((cf - cof0) / (cf + math.log(2.0)), min=-1.0, max=1.0)
        attenuation = torch.exp(-cf) * torch.acos(arg) / math.pi
    actinic = torch.zeros((ncol, nlyr, nwave), dtype=dtype, device=device)
    # KB-2012's photolysis uses ALTFLX × FLUX directly. Empirically the result
    # is 1/4 of what kintera's direct path produces with the same Cogley-Borucki
    # attenuation. The DISORT path (after the 4π→π fix at line ~158) matches
    # KB exactly, so the direct path needs the same 1/4 to match.
    # The 4× factor is the convention difference between actinic flux (4π × J̄,
    # photochemistry standard) and downward irradiance (π × F_down) that KB-2012
    # implicitly assumes via its tabulated cross sections.
    actinic[:, :active_nlyr, :] = 0.25 * top_flux.view(1, 1, nwave) * attenuation
    if active_nlyr < nlyr:
        actinic[:, active_nlyr:, :] = 0.25 * top_flux.view(1, 1, nwave)
    return actinic

def _kinetics_base_solar_mu0_weights(parameters: dict[str, Any]) -> list[tuple[float, float]]:
    default_mu0 = max(min(float(parameters.get("solar_mu0", 1.0)), 1.0), 1.0e-6)
    if not parameters.get("diurnal_average"):
        return [(default_mu0, 1.0)]

    latitude = parameters.get("solar_latitude_deg")
    declination = parameters.get("solar_declination_deg")
    if latitude is None or declination is None:
        return [(default_mu0, 1.0)]

    lat = math.radians(float(latitude))
    dec = math.radians(float(declination))
    a5 = math.sin(dec) * math.sin(lat)
    b5 = math.cos(dec) * math.cos(lat)
    if abs(b5) < 1.0e-12:
        mu0 = max(min(a5, 1.0), 0.0)
        return [(max(mu0, 1.0e-6), 1.0 if mu0 > 0.0 else 0.0)]

    terminator = -a5 / b5
    if terminator <= -1.0:
        hour_angle = math.pi
    elif terminator >= 1.0:
        return [(1.0e-6, 0.0)]
    else:
        hour_angle = math.acos(terminator)

    npoint = max(int(parameters.get("diurnal_quadrature_points", 8)), 1)
    daylight_weight = hour_angle / math.pi / float(npoint)
    samples: list[tuple[float, float]] = []
    for index in range(npoint):
        frac = (float(index) + 0.5) / float(npoint)
        local_hour_angle = -hour_angle + 2.0 * hour_angle * frac
        mu0 = a5 + b5 * math.cos(local_hour_angle)
        if mu0 > 0.0:
            samples.append((max(min(mu0, 1.0), 1.0e-6), daylight_weight))
    return samples or [(1.0e-6, 0.0)]

def _aerosol_extinction_on_state_grid(
    term: KBTitanSourceTerm,
    titan_state: KBTitanState,
    nwave: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    return _aerosol_property_on_state_grid(
        term, "aerosol_extinction", titan_state, nwave, dtype, device
    )

def _aerosol_property_on_state_grid(
    term: KBTitanSourceTerm,
    parameter_name: str,
    titan_state: KBTitanState,
    nwave: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    aerosol = term.parameters.get(parameter_name)
    if not isinstance(aerosol, dict):
        return None
    altitudes = aerosol.get("altitude_km")
    values = aerosol.get("values")
    if not isinstance(altitudes, list) or not isinstance(values, list):
        return None
    if not values or len(values[0]) != nwave:
        return None

    source_alt = torch.tensor(altitudes, dtype=dtype, device=device)
    source_values = torch.tensor(values, dtype=dtype, device=device)
    target_alt = titan_state.state.x1v.to(dtype=dtype, device=device) / 1.0e5
    output = torch.zeros((target_alt.numel(), nwave), dtype=dtype, device=device)
    for i, altitude in enumerate(target_alt):
        if altitude <= source_alt[0]:
            output[i] = source_values[0]
        elif altitude >= source_alt[-1]:
            output[i] = source_values[-1]
        else:
            upper = int(torch.searchsorted(source_alt, altitude).item())
            lower = upper - 1
            frac = (altitude - source_alt[lower]) / (
                source_alt[upper] - source_alt[lower]
            )
            output[i] = source_values[lower] + frac * (
                source_values[upper] - source_values[lower]
            )
    return output

