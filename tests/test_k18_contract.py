"""Cross-reader absorption contracts and self-product dataset validation."""
from pathlib import Path

import pytest
import torch
import yaml

import kintera as kt
from kintera.atm2d.radiation import _total_cross_section_by_species

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device(request.param)


@pytest.mark.parametrize("ratio", [0.25, 1.000002, 1.0000005])
def test_shared_absorber_total_consistency(tmp_path, device, ratio):
    deck = yaml.safe_load((TEST_DIR / "chapman_cycle.yaml").read_text())
    reactions = []
    for total in [1.e-30, ratio * 1.e-30]:
        reactions.append({
            "equation": "O3 => O2 + O", "type": "photolysis",
            "cross-section": [{"format": "YAML", "data": [
                [300., total, 0.25e-30], [400., total, 0.25e-30]
            ]}],
        })
    deck["reactions"] = reactions
    path = tmp_path / "shared.yaml"
    path.write_text(yaml.safe_dump(deck))
    photo = kt.PhotoChem(kt.PhotoChemOptions.from_yaml(str(path)))
    photo.to(torch.float64)
    photo.to(device)
    temperature = torch.tensor([[298.]], dtype=torch.float64, device=device)
    wave = torch.tensor([300., 400.], dtype=torch.float64, device=device)
    if abs(ratio - 1.) > 1.e-6:
        with pytest.raises(ValueError, match=
                "column 0 must be the total absorption; reactions of O3 disagree"):
            _total_cross_section_by_species(photo, temperature, wave)
    else:
        photolysis = photo.module("photolysis")
        before = [photolysis.interp_cross_section(i, wave, temperature.flatten()).clone()
                  for i in range(len(reactions))]
        sigma = _total_cross_section_by_species(photo, temperature, wave)
        idx = photo.options.species().index("O3")
        torch.testing.assert_close(sigma[..., idx], torch.full_like(sigma[..., idx], 1.e-30),
                                   rtol=1.e-14, atol=0.)
        for i, xs in enumerate(before):
            torch.testing.assert_close(
                photolysis.interp_cross_section(i, wave, temperature.flatten()), xs,
                rtol=0., atol=0.,
            )


def test_vulcan_cho_rejects_per_channel_totals(device):
    photo = kt.PhotoChem(kt.PhotoChemOptions.from_yaml(str(TEST_DIR / "vulcan" / "cho_photo.yaml")))
    photo.to(torch.float64)
    photo.to(device)
    temperature = torch.tensor([[298.]], dtype=torch.float64, device=device)
    wave = photo.module("photolysis").buffer("wavelength")
    with pytest.raises(ValueError, match=
            "column 0 must be the total absorption; reactions of .* disagree"):
        _total_cross_section_by_species(photo, temperature, wave)


@pytest.mark.parametrize("dataset_type,yield_value", [(0, 1.), (2, 0.25)])
def test_equal_products_require_type_zero_absorption(tmp_path, device, dataset_type, yield_value):
    # The catalog can supply a parent absorption table as well as a yield table
    # for the same equation. A yield table on a reaction whose products equal
    # its reactants is rejected at load; a type-0 table alone loads as absorption.
    master = (TEST_DIR / "kinetics_base" / "data" / "test_master.inp").read_text()
    species_section = master.split("      O2      =2O")[0]
    master_path = tmp_path / "master.inp"
    master_path.write_text(species_section + f"{'      O3      = O3':<66}0.00E+00   0.0       0.\n")
    absorption = 1.e-20
    (tmp_path / "abs.dat").write_text(
        "O3 = O3\n1\n0 298.0\n1 2\n1000.0 1.e-20\n2000.0 1.e-20\n"
    )
    filename = "abs.dat"
    catalog = f"{'O3 = O3':<60}abs.dat\n"
    if dataset_type == 2:
        filename = "yield.dat"
        (tmp_path / filename).write_text(
            f"O3 = O3\n1\n2 298.0\n1 2\n1000.0 {yield_value}\n2000.0 {yield_value}\n"
        )
        catalog += f"{'O3 = O3':<60}{filename}\n"
    catalog_path = tmp_path / "catalog.dat"
    catalog_path.write_text(catalog)
    if dataset_type == 2:
        with pytest.raises(RuntimeError, match="O3 => O3.*supply it as type-0 absorption"):
            kt.PhotoChemOptions.from_kinetics_base(str(master_path), str(catalog_path), str(tmp_path))
        return
    opts = kt.PhotoChemOptions.from_kinetics_base(str(master_path), str(catalog_path), str(tmp_path))
    photo = kt.PhotoChem(opts)
    photo.to(torch.float64)
    photo.to(device)
    assert len(opts.reactions()) == 1
    temp = torch.tensor([298.], dtype=torch.float64, device=device)
    concentration = torch.ones((1, len(opts.species())), dtype=torch.float64, device=device)
    photolysis = photo.module("photolysis")
    wave = photolysis.buffer("wavelength")
    photolysis.update_xs_diss_stacked(temp)
    rate = photo.forward(temp, concentration, torch.full_like(wave, 1.e14))
    expected = absorption * yield_value * 100. * 1.e14
    assert rate.item() == pytest.approx(expected, rel=1.e-12, abs=0.)
    # A nonzero event rate with identical products/reactants has zero net source.
    assert torch.count_nonzero(photo.buffer("stoich")).item() == 0
    xs = photolysis.interp_cross_section(0, wave, temp)
    torch.testing.assert_close(xs[:, 0], torch.full_like(wave, absorption), rtol=1.e-14, atol=0.)


@pytest.mark.parametrize("catalog_state", ["original", "missing_first", "all_missing"])
def test_missing_catalog_totals_on_full_grid(tmp_path, device, catalog_state):
    data = TEST_DIR / "kinetics_base" / "data"
    master_path = data / "test_master.inp"
    catalog_path = data / "test_catalog.dat"
    valid_o2 = 0
    if catalog_state == "missing_first":
        lines = master_path.read_text().splitlines(keepends=True)
        first = next(i for i, line in enumerate(lines) if line.startswith("      O2      =2O"))
        # The O2 -> O + O(1D) reaction has no catalog entry. Put it first.
        lines[first], lines[first + 1] = lines[first + 1], lines[first]
        master_path = tmp_path / "master.inp"
        master_path.write_text("".join(lines))
        valid_o2 = 1
    elif catalog_state == "all_missing":
        lines = master_path.read_text().splitlines(keepends=True)
        first = next(i for i, line in enumerate(lines) if line.startswith("      O2      =2O"))
        master_path = tmp_path / "master.inp"
        master_path.write_text("".join(lines[:first + 2]))
        lines = catalog_path.read_text().splitlines(keepends=True)
        catalog_path = tmp_path / "catalog.dat"
        catalog_path.write_text("".join(line for line in lines if not line.lstrip().startswith("O2")))
    opts = kt.PhotoChemOptions.from_kinetics_base(str(master_path), str(catalog_path), str(data / "cross"))
    photo = kt.PhotoChem(opts)
    photo.to(torch.float64)
    photo.to(device)
    temperature = torch.tensor([[298.]], dtype=torch.float64, device=device)
    photolysis = photo.module("photolysis")
    wave = photolysis.buffer("wavelength")  # Entire shared in-tree grid.
    o2_totals = [photolysis.interp_cross_section(i, wave, temperature.flatten())[:, 0]
                 for i in (0, 1)]
    if catalog_state == "all_missing":
        assert all(torch.count_nonzero(xs).item() == 0 for xs in o2_totals)
    else:
        assert torch.count_nonzero(o2_totals[valid_o2]).item() > 0
        assert torch.count_nonzero(o2_totals[1 - valid_o2]).item() == 0
    sigma = _total_cross_section_by_species(photo, temperature, wave)
    expected = torch.zeros_like(wave) if catalog_state == "all_missing" else o2_totals[valid_o2]
    torch.testing.assert_close(sigma[0, 0, :, opts.species().index("O2")], expected,
                               rtol=0., atol=0.)
    assert torch.isfinite(sigma).all()
    if catalog_state == "all_missing":
        assert torch.count_nonzero(sigma).item() == 0
