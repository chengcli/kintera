"""
Python integration tests for the KINETICS-base reader.

Tests the Python bindings for:
- KineticsOptions.from_kinetics_base()
- Kinetics module with KINETICS-base data
"""

import os
import pytest
import torch

torch.set_default_dtype(torch.float64)

DATA_DIR = os.path.join(os.path.dirname(__file__), "kinetics_base", "data")


@pytest.fixture
def master_path():
    return os.path.join(DATA_DIR, "test_master.inp")


@pytest.fixture
def catalog_path():
    return os.path.join(DATA_DIR, "test_catalog.dat")


@pytest.fixture
def cross_dir():
    return os.path.join(DATA_DIR, "cross") + "/"


def _count_reversible(reactions):
    """Count reversible reactions by checking equation string for <=>."""
    return sum(1 for r in reactions if "<=>" in r.equation())


def test_from_kinetics_base_no_xsec(master_path):
    """Load KINETICS-base master input without cross-sections."""
    import kintera as kt

    opts = kt.KineticsOptions.from_kinetics_base(master_path)

    species = opts.species()
    assert len(species) == 10
    assert "O" in species
    assert "O2" in species
    assert "O3" in species
    assert "H2O" in species
    assert "N2" in species
    assert "O(1D)" in species

    reactions = opts.reactions()
    assert len(reactions) == 12

    n_rev = _count_reversible(reactions)
    assert n_rev == 12  # all thermal reactions are reversible


def test_from_kinetics_base_with_xsec(master_path, catalog_path, cross_dir):
    """Load KINETICS-base master input with cross-sections."""
    import kintera as kt

    opts = kt.KineticsOptions.from_kinetics_base(
        master_path, catalog_path, cross_dir
    )

    reactions = opts.reactions()
    assert len(reactions) == 12


def test_kinetics_module_creation(master_path, catalog_path, cross_dir):
    """Create Kinetics module from KINETICS-base data."""
    import kintera as kt

    opts = kt.KineticsOptions.from_kinetics_base(
        master_path, catalog_path, cross_dir
    )
    kinet = kt.Kinetics(opts)
    assert kinet is not None

    species = opts.species()
    nspecies = len(species)
    nrxn = len(opts.reactions())
    n_rev = _count_reversible(opts.reactions())

    stoich = kinet.stoich
    assert stoich.size(0) == nspecies
    assert stoich.size(1) == nrxn + n_rev


def test_photochem_forward_with_xsec(master_path, catalog_path, cross_dir):
    """Forward pass with cross-section data loaded."""
    import kintera as kt

    kinet_opts = kt.KineticsOptions.from_kinetics_base(
        master_path, catalog_path, cross_dir
    )
    photo_opts = kt.PhotoChemOptions.from_kinetics_base(
        master_path, catalog_path, cross_dir
    )
    kinet = kt.Kinetics(kinet_opts)
    photo = kt.PhotoChem(photo_opts)

    species = kinet_opts.species()
    nspecies = len(species)

    temp = 300.0 * torch.ones(1)
    pres = 1.0e5 * torch.ones(1)
    conc = 1e18 * torch.ones(1, nspecies)

    wave = photo.module("photolysis").buffer("wavelength")
    aflux = 1e14 * torch.ones_like(wave)
    rate, rc_ddC, rc_ddT = kinet.forward_nogil(temp, pres, conc, {})
    photo_rate = photo.forward(temp, conc, aflux)

    nrxn = len(kinet_opts.reactions())
    n_rev = _count_reversible(kinet_opts.reactions())
    assert rate.dim() == 2
    assert rate.size(0) == 1
    assert rate.size(1) == nrxn + n_rev
    assert photo_rate.size(1) == len(photo_opts.reactions())

    du = rate @ kinet.stoich.t() + photo_rate @ photo.stoich.t()
    assert du.size(-1) == nspecies


def test_kinetics_species_consistency(master_path):
    """Check that species names and weights are consistent."""
    import kintera as kt

    opts = kt.KineticsOptions.from_kinetics_base(master_path)

    species = opts.species()
    weights = kt.species_weights()
    assert len(species) == len(weights)
    assert all(w > 0 for w in weights)

    idx_o = species.index("O")
    assert abs(weights[idx_o] - 16.0) < 0.1

    idx_o2 = species.index("O2")
    assert abs(weights[idx_o2] - 32.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
