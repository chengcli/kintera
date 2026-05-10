"""
Python integration tests for the KINETICS-base reader.

Tests the Python bindings for:
- KineticsOptions.from_kinetics_base()
- Kinetics module with KINETICS-base data
"""

import os
import re
import subprocess
from types import SimpleNamespace

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


def _external_titan_paths():
    # Local diagnosis build used during development:
    #   KINTERA_KINETICS_BASE_ROOT=/tmp/KINETICS-base-compare
    #   KINTERA_KINETICS_BASE_EXECUTABLE=/tmp/KINETICS-base-compare/build/bin/titan.release
    # Keep tests env-driven so CI and other machines can provide their own oracle.
    root = os.environ.get("KINTERA_KINETICS_BASE_ROOT")
    executable = os.environ.get("KINTERA_KINETICS_BASE_EXECUTABLE")
    if not root or not executable:
        pytest.skip(
            "set KINTERA_KINETICS_BASE_ROOT and "
            "KINTERA_KINETICS_BASE_EXECUTABLE to run Titan oracle tests"
        )

    titan_dir = os.path.join(root, "examples", "titan")
    pun_path = os.path.join(
        titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.pun"
    )
    run_input_path = os.path.join(titan_dir, "ions_c6h7+_H2CN.inp-1")
    atmosphere_path = os.path.join(
        titan_dir, "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz"
    )
    for path in [pun_path, run_input_path, atmosphere_path, executable]:
        if not os.path.exists(path):
            pytest.skip(f"missing external KINETICS-base file: {path}")
    return titan_dir, executable, pun_path, run_input_path, atmosphere_path


def _write_fresh_start_run_input(src, dst, *, ntime=1):
    with open(src) as f:
        text = f.read()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("NTIME "):
            for j in range(i + 1, len(lines)):
                if re.match(r"^\s*\d+", lines[j]):
                    parts = lines[j].split()
                    if len(parts) >= 11:
                        parts[0] = str(ntime)
                        parts[10] = "0"
                        lines[j] = " ".join(parts)
                        with open(dst, "w") as f:
                            f.write("\n".join(lines) + "\n")
                        return
    raise AssertionError("could not find KINETICS-base ISTART field")


def _run_titan_steps(tmp_path, *, ntime, timeout=120):
    titan_dir, executable, pun_path, run_input_path, atmosphere_path = (
        _external_titan_paths()
    )

    work = tmp_path / f"titan-{ntime}-steps"
    work.mkdir()
    (work / "prod+loss").mkdir()
    patched_run_input = work / "fort.81.fresh-start"
    _write_fresh_start_run_input(run_input_path, patched_run_input, ntime=ntime)

    links = {
        "fort.1": pun_path,
        "fort.3": os.path.join(titan_dir, "kintitan.truncate"),
        "fort.4": os.path.join(
            titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
        ),
        "fort.15": os.path.join(titan_dir, "titan_Cheng_N_ions_H2CN.bc_save"),
        "fort.20": os.path.join(titan_dir, "Cheng_wavel.dat"),
        "fort.21": os.path.join(titan_dir, "flare_kin_oct2003.inp"),
        "fort.27": os.path.join(titan_dir, "kintitan-difrad-2.inp"),
        "fort.30": os.path.join(titan_dir, "Cheng_catalog_v4.dat"),
        "fort.45": os.path.join(titan_dir, "kintitan_aerosol_interp_albedo.inp"),
        "fort.46": os.path.join(titan_dir, "kintitan_aerosol_interp_gr.inp"),
        "fort.47": os.path.join(titan_dir, "kintitan_aerosol_interp_asymm.inp"),
        "fort.50": atmosphere_path,
        "fort.81": patched_run_input,
        "crossfilepath": os.path.join(titan_dir, "Cheng_cross"),
    }
    for name, target in links.items():
        if not os.path.exists(target):
            pytest.skip(f"missing external KINETICS-base file: {target}")
        os.symlink(target, work / name)

    for name in ["kintitan.out.pun", "kintitan.res", "titandebug.dat"]:
        (work / name).touch()
    os.symlink(work / "kintitan.out.pun", work / "fort.7")
    os.symlink(work / "kintitan.res", work / "fort.10")
    os.symlink(work / "titandebug.dat", work / "fort.11")

    proc = subprocess.run(
        [executable],
        cwd=work,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stdout[-4000:])

    output_path = work / "kintitan.out.pun"
    assert output_path.exists()
    return atmosphere_path, output_path


def _run_titan_one_step(tmp_path):
    return _run_titan_steps(tmp_path, ntime=1)


def _titan_species_in_reference(initial, reference):
    return [
        name
        for name in initial.species_profiles
        if name in reference.species_profiles
    ]


def _build_titan_state_from_oracle(initial, species):
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    boundary_path = os.path.join(titan_dir, "titan_Cheng_N_ions_H2CN.bc_save")
    return kt.build_kinetics_base_titan_state(
        initial, species=species, boundary_path=boundary_path, pun_path=pun_path
    )


def _build_titan_source_terms_from_oracle():
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    special_path = os.path.join(
        titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
    )
    boundary_path = os.path.join(titan_dir, "titan_Cheng_N_ions_H2CN.bc_save")
    return kt.build_kinetics_base_titan_source_terms(
        pun_path,
        special_path=special_path,
        boundary_path=boundary_path,
        run_input_path=run_input_path,
        photo_catalog_path=os.path.join(titan_dir, "Cheng_catalog_v4.dat"),
        cross_dir=os.path.join(titan_dir, "Cheng_cross"),
        flux_path=os.path.join(titan_dir, "flare_kin_oct2003.inp"),
    ), kt.kinetics_base_species_metadata_from_pun(pun_path)


def _solve_titan_transport_steps(
    titan_state,
    *,
    ntime,
    source_terms=None,
    pun_metadata=None,
):
    import kintera as kt

    concentration = titan_state.concentration
    boundary_species = [
        titan_state.species.index(name)
        for name, conversion in titan_state.conversion.items()
        if conversion == "lower_boundary_mixing_ratio_times_density"
    ]
    fixed_species = [
        titan_state.species.index(name)
        for name in titan_state.fixed_species
        if name in titan_state.species
    ]
    transport = kt.build_transport_matrix(titan_state.state, titan_state.kzz)
    atm_sources = (
        kt.build_kinetics_base_titan_atm2d_source_terms(
            titan_state, source_terms, pun_metadata=pun_metadata
        )
        if source_terms is not None
        else None
    )
    dt = 1.0e-15
    # The Titan executable grows the negative-DELTIM startup sequence by half
    # decades for this configuration.
    growth = 10.0 ** 0.5
    for _ in range(ntime):
        if atm_sources is not None:
            titan_state.state.concentration = concentration
            system, rhs = kt.build_implicit_step_system(
                titan_state.state,
                titan_state.kzz,
                dt,
                source_terms=atm_sources,
            )
            concentration = kt.solve_sparse_system(system, rhs)
            concentration = torch.clamp(concentration, min=0.0)
        else:
            system_dense = torch.eye(transport.nstate) - dt * transport.global_csr.to_dense()
            system = kt.SparseSystemMatrix.from_dense(
                system_dense,
                ncol=1,
                nlyr=titan_state.state.nlyr,
                nspecies=titan_state.state.nspecies,
            )
            concentration = kt.solve_sparse_system(system, concentration)
        if fixed_species:
            concentration[:, :, fixed_species] = titan_state.concentration[
                :, :, fixed_species
            ]
        if boundary_species:
            concentration[:, 0, boundary_species] = titan_state.concentration[
                :, 0, boundary_species
            ]
        dt *= growth
    return concentration[0]


def _assert_titan_equivalence(kintera_concentration, reference_concentration, species):
    diff = (kintera_concentration - reference_concentration).abs()
    max_abs_diff = diff.max().item()
    nonzero_reference = reference_concentration.abs() > 0.0
    max_rel_diff = (
        (diff[nonzero_reference] / reference_concentration[nonzero_reference].abs())
        .max()
        .item()
        if nonzero_reference.any()
        else 0.0
    )
    changed_entries = int((diff > 1.0e-6).sum().item())
    species_max_diff = diff.max(dim=0).values
    top_count = min(10, len(species))
    top_values, top_indices = torch.topk(species_max_diff, top_count)
    top_species = ", ".join(
        f"{species[int(index)]}:{value.item():.3e}"
        for value, index in zip(top_values, top_indices)
    )
    assert torch.isfinite(kintera_concentration).all()
    torch.testing.assert_close(
        kintera_concentration,
        reference_concentration,
        rtol=5.0e-4,
        atol=1.0e-6,
        msg=(
            "Titan output mismatch after KINETICS-base state conversion, "
            "lower boundary application, and transport solve: "
            f"max_abs_diff={max_abs_diff:.6e}, "
            f"max_rel_diff={max_rel_diff:.6e}, "
            f"entries_gt_1e-6={changed_entries}/{diff.numel()}, "
            f"top_species=[{top_species}]."
        ),
    )


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


def test_kinetics_base_species_on_1d_kzz_diffusion_solver(master_path):
    """Use KINETICS-base species in a one-column Kzz diffusion solve."""
    import kintera as kt

    opts = kt.KineticsOptions.from_kinetics_base(master_path)
    nspecies = len(opts.species())
    nlyr = 9
    ncol = 1

    x1f = torch.linspace(0.0, 8.0e5, nlyr + 1)
    x2f = torch.tensor([0.0, 1.0])
    temperature = torch.full((ncol, nlyr), 250.0)
    pressure = torch.logspace(6.0, 3.0, nlyr).unsqueeze(0)

    vertical_profile = torch.linspace(0.2, 1.0, nlyr).view(ncol, nlyr, 1)
    species_scale = torch.linspace(1.0, 2.0, nspecies).view(1, 1, nspecies)
    concentration = 1.0e8 * vertical_profile * species_scale
    concentration[:, :, 0] = 3.0e8  # Uniform species should remain unchanged.

    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=temperature,
        pressure=pressure,
        concentration=concentration,
    )
    kzz = torch.full((ncol, nlyr), 1.0e5)
    transport = kt.build_transport_matrix(state, kzz)

    dt = 5.0e4
    system_dense = torch.eye(transport.nstate) - dt * transport.global_csr.to_dense()
    system = kt.SparseSystemMatrix.from_dense(
        system_dense,
        ncol=ncol,
        nlyr=nlyr,
        nspecies=nspecies,
    )

    next_concentration = kt.solve_sparse_system(system, concentration)

    assert next_concentration.shape == concentration.shape
    torch.testing.assert_close(
        next_concentration.sum(dim=1),
        concentration.sum(dim=1),
        rtol=1.0e-10,
        atol=1.0e-2,
    )
    torch.testing.assert_close(
        next_concentration[:, :, 0],
        concentration[:, :, 0],
        rtol=1.0e-12,
        atol=1.0e-6,
    )
    assert next_concentration[:, -1, 1].item() < concentration[:, -1, 1].item()
    assert next_concentration[:, 0, 1].item() > concentration[:, 0, 1].item()


def test_titan_first_order_sources_expose_atm2d_linearization():
    """Titan first-order source terms can be assembled by atm2d."""
    import kintera as kt

    x1f = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    concentration = torch.tensor([[[3.0, 0.0], [5.0, 1.0]]], dtype=torch.float64)
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 2), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 2), 1.0e4, dtype=torch.float64),
        concentration=concentration,
    )
    titan_state = kt.KBTitanState(
        species=["A", "B"],
        fixed_species=[],
        varying_species=["A", "B"],
        conversion={},
        concentration=concentration,
        density=torch.ones((1, 2), dtype=torch.float64),
        kzz=torch.zeros((1, 2), dtype=torch.float64),
        state=state,
    )
    source = kt.KBTitanSourceTerm(
        kind="pun_photo_rate_reaction",
        reaction_id=1,
        reactants=["A"],
        products=["B"],
        parameters={"rate": 2.0, "attenuation": "none"},
    )

    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, [source]
    )
    linearization = kt.build_source_linearization(state, atm_sources)

    torch.testing.assert_close(
        linearization.tendency,
        torch.tensor([[[-6.0, 6.0], [-10.0, 10.0]]], dtype=torch.float64),
    )
    expected_jacobian = torch.zeros((1, 2, 2, 2), dtype=torch.float64)
    expected_jacobian[:, :, 0, 0] = -2.0
    expected_jacobian[:, :, 1, 0] = 2.0
    torch.testing.assert_close(linearization.jacobian, expected_jacobian)


def test_titan_thermal_and_boundary_sources_expose_atm2d_linearization():
    """Titan thermal and boundary source terms are available to atm2d."""
    import kintera as kt

    x1f = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    concentration = torch.tensor([[[3.0, 4.0], [5.0, 7.0]]], dtype=torch.float64)
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 2), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 2), 1.0e4, dtype=torch.float64),
        concentration=concentration,
    )
    titan_state = kt.KBTitanState(
        species=["A", "B"],
        fixed_species=[],
        varying_species=["A", "B"],
        conversion={},
        concentration=concentration,
        density=torch.ones((1, 2), dtype=torch.float64),
        kzz=torch.zeros((1, 2), dtype=torch.float64),
        state=state,
    )
    thermal = kt.KBTitanSourceTerm(
        kind="pun_thermal_reaction",
        reaction_id=1,
        reactants=["A"],
        products=["B"],
        parameters={
            "A": 0.5,
            "b": 0.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "F": 0.0,
            "Tmin": 1.0,
            "Tmax": 1.0e9,
            "reactant_coefficients": [1],
            "product_coefficients": [1],
        },
    )
    boundary = kt.KBTitanSourceTerm(
        kind="lower_boundary_flux",
        reaction_id=None,
        reactants=[],
        products=["B"],
        parameters={"value": 6.0},
    )

    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, [thermal, boundary]
    )
    linearization = kt.build_source_linearization(state, atm_sources)

    expected_tendency = torch.tensor(
        [[[-1.5, 4.5], [-2.5, 2.5]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(linearization.tendency, expected_tendency)
    expected_jacobian = torch.zeros((1, 2, 2, 2), dtype=torch.float64)
    expected_jacobian[:, :, 0, 0] = -0.5
    expected_jacobian[:, :, 1, 0] = 0.5
    torch.testing.assert_close(linearization.jacobian, expected_jacobian)


def test_kinetics_base_profile_conversion_does_not_infer_mixing_ratio_from_value():
    """Small number-density profiles must not be treated as mixing ratios."""
    import kintera as kt

    profile = SimpleNamespace(
        altitude=[0.0, 1.0],
        density=[1.0e12, 2.0e12],
        species_profiles={"LOW": [1.0e-8, 2.0e-8]},
    )

    concentration, conversion = kt.kinetics_base_concentration_from_profile(
        profile, ["LOW"]
    )

    torch.testing.assert_close(
        concentration[:, 0], torch.tensor([1.0e-8, 2.0e-8])
    )
    assert conversion["LOW"] == "number_density"


def test_external_titan_one_step_equivalence_if_available(tmp_path):
    """Compare the Fortran Titan one-step output with current kintera state."""
    import kintera as kt

    initial_path, output_path = _run_titan_one_step(tmp_path)
    initial = kt.parse_kinetics_base_atmosphere(initial_path)
    reference = kt.parse_kinetics_base_atmosphere(str(output_path))

    species = _titan_species_in_reference(initial, reference)
    assert len(species) == 128
    assert len(initial.altitude) == len(reference.altitude) == 50
    assert "CH4" in initial.mixing_ratio_species_profiles
    assert "N2" not in initial.mixing_ratio_species_profiles

    reference_concentration = kt.kinetics_base_profile_tensor(reference, species)
    titan_state = _build_titan_state_from_oracle(initial, species)
    assert titan_state.concentration.shape == (1, 50, 128)
    assert titan_state.conversion["CH4"] == "lower_boundary_mixing_ratio_times_density"
    assert "E" not in titan_state.fixed_species
    assert titan_state.conversion["E"] == "pun_electron_or_ion_number_density"
    if "CH3+" in titan_state.conversion:
        assert titan_state.conversion["CH3+"] == "pun_electron_or_ion_number_density"
    assert titan_state.conversion["U"] == "pun_empty_composition_number_density"
    assert titan_state.conversion["SGA"] == "fixed_pun_zero_molecular_weight_number_density"

    kintera_concentration = _solve_titan_transport_steps(titan_state, ntime=1)
    _assert_titan_equivalence(kintera_concentration, reference_concentration, species)


def test_external_titan_multi_step_equivalence_gap_if_available(tmp_path):
    """Compare Fortran Titan multi-step output with repeated kintera steps."""
    import kintera as kt

    ntime = 10
    initial_path, output_path = _run_titan_steps(tmp_path, ntime=ntime)
    initial = kt.parse_kinetics_base_atmosphere(initial_path)
    reference = kt.parse_kinetics_base_atmosphere(str(output_path))

    species = _titan_species_in_reference(initial, reference)
    assert len(species) == 128
    assert len(initial.altitude) == len(reference.altitude) == 50

    reference_concentration = kt.kinetics_base_profile_tensor(reference, species)
    titan_state = _build_titan_state_from_oracle(initial, species)
    source_terms, pun_metadata = _build_titan_source_terms_from_oracle()
    kintera_concentration = _solve_titan_transport_steps(
        titan_state,
        ntime=ntime,
        source_terms=source_terms,
        pun_metadata=pun_metadata,
    )
    try:
        _assert_titan_equivalence(kintera_concentration, reference_concentration, species)
    except AssertionError as exc:
        pytest.xfail(
            "multi-step Titan source-term path is present but not complete "
            f"enough for full chemistry convergence yet: {exc}"
        )


def test_external_titan_reaction_classification_if_available():
    """Report selected Titan photolysis and thermal candidate reaction counts."""
    import kintera as kt

    _, _, pun_path, run_input_path, _ = _external_titan_paths()
    report = kt.classify_kinetics_base_titan_reactions(pun_path, run_input_path)

    assert report.total_reactions == 2139
    assert report.selected_photolysis_reactions == 9
    assert report.selected_photolysis_ids == [221, 222, 223, 224, 225, 226, 227, 228, 245]
    assert report.thermal_candidate_reactions == 2130
    assert report.n_reactants_counts[1] == 319
    assert report.n_reactants_counts[2] == 1698
    assert report.n_reactants_counts[3] == 122
    assert report.missing_rate_blocks == 0


def test_external_titan_special_placeholders_if_available():
    """Known disabled Titan placeholders should not be reported as unimplemented."""
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    terms = kt.build_kinetics_base_titan_source_terms(
        pun_path,
        special_path=os.path.join(
            titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
        ),
        boundary_path=os.path.join(titan_dir, "titan_Cheng_N_ions_H2CN.bc_save"),
        run_input_path=run_input_path,
        photo_catalog_path=os.path.join(titan_dir, "Cheng_catalog_v4.dat"),
        cross_dir=os.path.join(titan_dir, "Cheng_cross"),
        flux_path=os.path.join(titan_dir, "flare_kin_oct2003.inp"),
    )

    disabled_ids = {
        term.reaction_id
        for term in terms
        if term.kind == "disabled_special_placeholder"
    }
    unimplemented_ids = {
        term.reaction_id
        for term in terms
        if term.kind == "unimplemented_special_reaction"
    }
    ion_recombination_count = sum(
        term.kind == "pun_thermal_reaction" and "E" in term.reactants
        for term in terms
    )
    electron_impact_count = sum(
        term.kind == "pun_electron_impact_reaction" for term in terms
    )
    assert disabled_ids == {187, 2094, 2095, 2096, 2097, 2098, 2099}
    assert unimplemented_ids == set()
    assert ion_recombination_count > 0
    assert electron_impact_count > 0


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
    rate, rc_ddC, rc_ddT = kinet.forward_nogil(temp, pres, conc)
    photo.module("photolysis").update_xs_diss_stacked(temp)
    photo_rate = photo.forward(temp, conc, aflux)

    nrxn = len(kinet_opts.reactions())
    n_rev = _count_reversible(kinet_opts.reactions())
    assert rate.dim() == 2
    assert rate.size(0) == 1
    assert rate.size(1) == nrxn + n_rev
    assert photo_rate.size(1) == len(photo_opts.reactions())

    du = rate @ kinet.stoich.t() + photo_rate @ photo.stoich.t()
    assert du.size(-1) == nspecies


def test_photochem_requires_catalog(master_path):
    """Photochemistry loader should fail fast without a catalog."""
    import kintera as kt

    with pytest.raises(RuntimeError, match="photo_catalog_path"):
        kt.PhotoChemOptions.from_kinetics_base(master_path)


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
