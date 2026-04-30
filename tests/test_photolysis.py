"""
Python integration tests for photolysis module.
"""

import pytest
import torch


def test_import_photolysis():
    from kintera import (
        PhotolysisOptions,
        Photolysis,
        ActinicFluxOptions,
        create_actinic_flux,
        create_uniform_flux,
        create_solar_flux,
        interpolate_actinic_flux,
    )

    assert PhotolysisOptions is not None
    assert Photolysis is not None
    assert ActinicFluxOptions is not None
    assert create_actinic_flux is not None
    assert create_uniform_flux is not None
    assert create_solar_flux is not None
    assert interpolate_actinic_flux is not None


def test_photolysis_options_creation():
    from kintera import PhotolysisOptions

    opts = PhotolysisOptions()
    opts.wavelength([100.0, 150.0, 200.0])
    assert opts.wavelength() == [100.0, 150.0, 200.0]

    opts.temperature([200.0, 300.0])
    assert opts.temperature() == [200.0, 300.0]


def test_actinic_flux_options():
    from kintera import ActinicFluxOptions

    opts = ActinicFluxOptions()
    opts.wavelength([100.0, 200.0, 300.0])
    opts.default_flux([1e14, 2e14, 1e14])
    opts.wave_min(50.0)
    opts.wave_max(400.0)

    assert opts.wavelength() == [100.0, 200.0, 300.0]
    assert opts.wave_min() == 50.0
    assert opts.wave_max() == 400.0


def test_create_uniform_flux():
    from kintera import create_uniform_flux

    wavelength = torch.linspace(100.0, 300.0, 21)
    flux = create_uniform_flux(wavelength, 1e14)

    assert flux.shape[0] == 21


def test_create_solar_flux():
    from kintera import create_solar_flux

    wavelength = torch.linspace(100.0, 800.0, 71)
    flux = create_solar_flux(wavelength, 1e14)

    assert flux.shape[0] == 71


def test_create_actinic_flux():
    from kintera import ActinicFluxOptions, create_actinic_flux

    opts = ActinicFluxOptions()
    opts.wavelength([100.0, 200.0, 300.0])
    opts.default_flux([1e14, 2e14, 1e14])

    target = torch.tensor([150.0, 250.0])
    flux = create_actinic_flux(opts, target)

    assert flux.shape[0] == 2


def test_actinic_flux_interpolation():
    from kintera import interpolate_actinic_flux

    wavelength = torch.tensor([100.0, 200.0, 300.0])
    flux_vals = torch.tensor([1e14, 2e14, 1e14])
    new_wave = torch.tensor([150.0, 250.0])
    interp_flux = interpolate_actinic_flux(wavelength, flux_vals, new_wave)

    assert interp_flux.shape[0] == 2
    assert interp_flux[0].item() > 1e14
    assert interp_flux[0].item() < 2e14


def test_photolysis_module_creation():
    from kintera import PhotolysisOptions, Photolysis, Reaction, set_species_names

    set_species_names(["N2", "O2"])

    opts = PhotolysisOptions()
    opts.wavelength([100.0, 150.0, 200.0])
    opts.temperature([200.0, 300.0])
    opts.reactions([Reaction("N2 => N2")])
    opts.cross_section([1e-18, 2e-18, 1e-18])
    opts.branches([[{"N2": 1.0}]])

    module = Photolysis(opts)

    assert module is not None
    assert module.options is not None


def test_photolysis_forward():
    from kintera import (
        PhotolysisOptions,
        Photolysis,
        Reaction,
        set_species_names,
        create_uniform_flux,
    )

    set_species_names(["N2", "O2"])

    opts = PhotolysisOptions()
    opts.wavelength([100.0, 150.0, 200.0])
    opts.temperature([200.0, 300.0])
    opts.reactions([Reaction("N2 => N2")])
    opts.cross_section([1e-18, 2e-18, 1e-18])
    opts.branches([[{"N2": 1.0}]])

    module = Photolysis(opts)

    temp = torch.tensor([250.0])
    actinic_flux = create_uniform_flux(module.wavelength, 1.0)

    module.update_xs_diss_stacked(temp)
    rate = module.forward(temp, actinic_flux)

    assert rate.dim() == 2
    assert rate.size(-1) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
