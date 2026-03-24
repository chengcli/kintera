"""
Checkpoint Test 4: Python integration tests for photolysis module.

Tests the Python bindings for:
- PhotolysisOptions
- Photolysis module
- ActinicFluxOptions
- ActinicFluxData
"""

import pytest
import torch


def test_import_photolysis():
    """Test that photolysis classes can be imported."""
    from kintera import (
        PhotolysisOptions,
        Photolysis,
        ActinicFluxOptions,
        ActinicFluxData,
        create_uniform_flux,
        create_solar_flux,
    )

    # Verify classes exist
    assert PhotolysisOptions is not None
    assert Photolysis is not None
    assert ActinicFluxOptions is not None
    assert ActinicFluxData is not None


def test_photolysis_options_creation():
    """Test PhotolysisOptions creation and configuration."""
    from kintera import PhotolysisOptions

    opts = PhotolysisOptions()

    # Set wavelength grid
    opts.wavelength([100.0, 150.0, 200.0])
    assert opts.wavelength() == [100.0, 150.0, 200.0]

    # Set temperature grid
    opts.temperature([200.0, 300.0])
    assert opts.temperature() == [200.0, 300.0]


def test_actinic_flux_options():
    """Test ActinicFluxOptions creation and configuration."""
    from kintera import ActinicFluxOptions

    opts = ActinicFluxOptions()

    opts.wavelength([100.0, 200.0, 300.0])
    opts.default_flux([1e14, 2e14, 1e14])
    opts.wave_min(50.0)
    opts.wave_max(400.0)

    assert opts.wavelength() == [100.0, 200.0, 300.0]
    assert opts.wave_min() == 50.0
    assert opts.wave_max() == 400.0


def test_actinic_flux_data():
    """Test ActinicFluxData creation and methods."""
    from kintera import ActinicFluxData

    # Test empty flux
    flux = ActinicFluxData()
    assert not flux.is_valid()
    assert flux.nwave() == 0

    # Test with tensors
    wavelength = torch.tensor([100.0, 200.0, 300.0])
    flux_vals = torch.tensor([1e14, 2e14, 1e14])
    flux = ActinicFluxData(wavelength, flux_vals)

    assert flux.is_valid()
    assert flux.nwave() == 3


def test_create_uniform_flux():
    """Test create_uniform_flux helper function."""
    from kintera import create_uniform_flux

    flux = create_uniform_flux(100.0, 300.0, 21, 1e14)

    assert flux.is_valid()
    assert flux.nwave() == 21


def test_create_solar_flux():
    """Test create_solar_flux helper function."""
    from kintera import create_solar_flux

    flux = create_solar_flux(100.0, 800.0, 71, 1e14)

    assert flux.is_valid()
    assert flux.nwave() == 71


def test_actinic_flux_to_map():
    """Test ActinicFluxData.to_map() method."""
    from kintera import ActinicFluxData

    wavelength = torch.tensor([100.0, 200.0, 300.0])
    flux_vals = torch.tensor([1e14, 2e14, 1e14])
    flux = ActinicFluxData(wavelength, flux_vals)

    flux_map = flux.to_map()

    assert "wavelength" in flux_map
    assert "actinic_flux" in flux_map
    assert flux_map["wavelength"].shape[0] == 3


def test_actinic_flux_interpolation():
    """Test ActinicFluxData interpolation."""
    from kintera import ActinicFluxData

    wavelength = torch.tensor([100.0, 200.0, 300.0])
    flux_vals = torch.tensor([1e14, 2e14, 1e14])
    flux = ActinicFluxData(wavelength, flux_vals)

    # Interpolate to midpoints
    new_wave = torch.tensor([150.0, 250.0])
    interp_flux = flux.interpolate_to(new_wave)

    assert interp_flux.shape[0] == 2
    # Interpolated values should be between neighbors
    assert interp_flux[0].item() > 1e14
    assert interp_flux[0].item() < 2e14


def test_photolysis_module_creation():
    """Test Photolysis module creation."""
    from kintera import PhotolysisOptions, Photolysis, Reaction, set_species_names

    # Initialize species
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
    """Test Photolysis.forward() method."""
    from kintera import PhotolysisOptions, Photolysis, Reaction, set_species_names

    # Initialize species
    set_species_names(["N2", "O2"])

    opts = PhotolysisOptions()
    opts.wavelength([100.0, 150.0, 200.0])
    opts.temperature([200.0, 300.0])
    opts.reactions([Reaction("N2 => N2")])
    opts.cross_section([1e-18, 2e-18, 1e-18])
    opts.branches([[{"N2": 1.0}]])

    module = Photolysis(opts)

    # Create inputs
    temp = torch.tensor([250.0])
    pres = torch.tensor([1e5])
    conc = torch.zeros(1, 2)

    wave = torch.tensor([100.0, 150.0, 200.0])
    aflux = torch.ones(3)

    other = {"wavelength": wave, "actinic_flux": aflux}

    rate = module.forward(temp, pres, conc, other)

    assert rate.dim() == 2
    assert rate.size(-1) == 1  # One reaction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

