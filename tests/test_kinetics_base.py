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
    #   KINTERA_KINETICS_BASE_ROOT=diagnostics/KINETICS-base-compare
    #   KINTERA_KINETICS_BASE_EXECUTABLE=diagnostics/KINETICS-base-compare/build/bin/titan.release
    # Keep tests env-driven so CI and other machines can provide their own oracle.
    default_root = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "diagnostics",
        "KINETICS-base-compare",
    )
    root = os.environ.get("KINTERA_KINETICS_BASE_ROOT", default_root)
    executable = os.environ.get(
        "KINTERA_KINETICS_BASE_EXECUTABLE",
        os.path.join(root, "build", "bin", "titan.release"),
    )
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


def _build_titan_state_from_oracle(initial, species, boundary_path=None):
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    if boundary_path is None:
        boundary_path = os.path.join(titan_dir, "titan_Cheng_N_ions_H2CN.bc_save")
    return kt.build_kinetics_base_titan_state(
        initial, species=species, boundary_path=boundary_path, pun_path=pun_path
    )


def _write_titan_ch4_effective_boundary_input(src, dst, atmosphere):
    """Copy the KB boundary file; Cheng cold-trap handling is applied separately."""
    del atmosphere
    with open(src) as f:
        lines = f.readlines()
    with open(dst, "w") as f:
        f.writelines(lines)


def _build_titan_source_terms_from_oracle():
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    special_path = os.path.join(
        titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
    )
    boundary_path = os.path.join(titan_dir, "titan_Cheng_N_ions_H2CN.bc_save")
    truncate_path = os.path.join(titan_dir, "kintitan.truncate")
    return kt.build_kinetics_base_titan_source_terms(
        pun_path,
        special_path=special_path,
        boundary_path=boundary_path,
        run_input_path=run_input_path,
        photo_catalog_path=os.path.join(titan_dir, "Cheng_catalog_v4.dat"),
        cross_dir=os.path.join(titan_dir, "Cheng_cross"),
        flux_path=os.path.join(titan_dir, "flare_kin_oct2003.inp"),
        truncate_path=truncate_path,
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
    transport = kt.build_transport_matrix(titan_state.state, titan_state.kzz)
    atm_sources = (
        kt.build_kinetics_base_titan_atm2d_source_terms(
            titan_state, source_terms, pun_metadata=pun_metadata
        )
        if source_terms is not None
        else None
    )
    species_diffusion_scale = kt.kinetics_base_titan_species_diffusion_scale(
        titan_state.species,
        dtype=titan_state.state.dtype,
        device=titan_state.state.device,
    )
    dt = 1.0e-15
    # KINETICS-base's Titan run grows the negative-DELTIM startup sequence by
    # half decades; the stdout sequence is 1e-15, 3.2e-15, 1e-14, ...
    growth = 10.0 ** 0.5
    for _ in range(ntime):
        if atm_sources is not None:
            titan_state.state.concentration = concentration
            system, rhs = kt.build_implicit_step_system(
                titan_state.state,
                titan_state.kzz,
                dt,
                species_diffusion_scale=species_diffusion_scale,
                source_terms=atm_sources,
            )
            system, rhs = kt.apply_kinetics_base_titan_dirichlet_rows(
                system, rhs, titan_state
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
            system, concentration = kt.apply_kinetics_base_titan_dirichlet_rows(
                system, concentration, titan_state
            )
            concentration = kt.solve_sparse_system(system, concentration)
        kt.apply_kinetics_base_titan_boundary_pins(concentration, titan_state)
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


def test_titan_boundary_pins_are_dirichlet_rows():
    """Titan fixed cells are enforced inside the linear solve, not only after it."""
    import kintera as kt

    x1f = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    current = torch.full((1, 3, 2), 9.0, dtype=torch.float64)
    pinned = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float64)
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 3), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 3), 1.0e4, dtype=torch.float64),
        concentration=current,
    )
    titan_state = kt.KBTitanState(
        species=["FIXED", "LOWER"],
        fixed_species=["FIXED"],
        varying_species=["FIXED", "LOWER"],
        conversion={"LOWER": "lower_boundary_mixing_ratio_times_density"},
        concentration=pinned,
        density=torch.ones((1, 3), dtype=torch.float64),
        kzz=torch.zeros((1, 3), dtype=torch.float64),
        state=state,
    )
    system = kt.SparseSystemMatrix.from_dense(
        torch.eye(6, dtype=torch.float64),
        ncol=1,
        nlyr=3,
        nspecies=2,
    )
    rhs = current.clone()
    system, rhs = kt.apply_kinetics_base_titan_dirichlet_rows(
        system, rhs, titan_state
    )
    solved = kt.solve_sparse_system(system, rhs)

    torch.testing.assert_close(solved[:, :, 0], pinned[:, :, 0])
    assert solved[0, 0, 1].item() == pinned[0, 0, 1].item()
    assert solved[0, 1, 1].item() == current[0, 1, 1].item()


def test_titan_first_order_sources_preserve_product_multiplicity():
    """Repeated photolysis products contribute stoichiometric coefficients."""
    import kintera as kt

    x1f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    concentration = torch.tensor([[[3.0, 0.0]]], dtype=torch.float64)
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 1), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 1), 1.0e4, dtype=torch.float64),
        concentration=concentration,
    )
    titan_state = kt.KBTitanState(
        species=["A", "B"],
        fixed_species=[],
        varying_species=["A", "B"],
        conversion={},
        concentration=concentration,
        density=torch.ones((1, 1), dtype=torch.float64),
        kzz=torch.zeros((1, 1), dtype=torch.float64),
        state=state,
    )
    source = kt.KBTitanSourceTerm(
        kind="pun_photo_rate_reaction",
        reaction_id=1,
        reactants=["A"],
        products=["B", "B"],
        parameters={"rate": 2.0, "attenuation": "none"},
    )

    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, [source]
    )
    linearization = kt.build_source_linearization(state, atm_sources)

    torch.testing.assert_close(
        linearization.tendency,
        torch.tensor([[[-6.0, 12.0]]], dtype=torch.float64),
    )
    assert linearization.jacobian[0, 0, 1, 0].item() == 4.0


def test_titan_photolysis_can_freeze_actinic_opacity():
    """KINETICS-base RAD=0 keeps photolysis attenuation tied to the initial state."""
    import kintera as kt

    x1f = torch.tensor([0.0, 1.0e5, 2.0e5], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    initial_concentration = torch.zeros((1, 2, 2), dtype=torch.float64)
    current_concentration = torch.tensor([[[3.0, 0.0], [5.0, 0.0]]], dtype=torch.float64)
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 2), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 2), 1.0e4, dtype=torch.float64),
        concentration=current_concentration,
    )
    titan_state = kt.KBTitanState(
        species=["A", "B"],
        fixed_species=[],
        varying_species=["A", "B"],
        conversion={},
        concentration=initial_concentration,
        density=torch.ones((1, 2), dtype=torch.float64),
        kzz=torch.zeros((1, 2), dtype=torch.float64),
        state=state,
    )
    source = kt.KBTitanSourceTerm(
        kind="pun_photo_rate_reaction",
        reaction_id=1,
        reactants=["A"],
        products=["B"],
        parameters={
            "wavelengths": [100.0],
            "cross_section": [1.0],
            "flux": [2.0],
            "total_cross_section_by_species": {"A": [100.0]},
            "radiation_active_nlyr": 2,
            "solar_mu0": 1.0,
            "freeze_actinic_flux": True,
        },
    )

    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, [source]
    )
    linearization = kt.build_source_linearization(state, atm_sources)

    torch.testing.assert_close(
        linearization.tendency,
        torch.tensor([[[-6.0, 6.0], [-10.0, 10.0]]], dtype=torch.float64),
    )


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


def test_titan_condensation_prefers_explicit_pun_rate():
    """Special SGA condensation reactions can carry literal KINETICS-base rates."""
    import kintera as kt

    x1f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    concentration = torch.tensor([[[3.0, 0.2, 0.0]]], dtype=torch.float64)
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 1), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 1), 1.0e4, dtype=torch.float64),
        concentration=concentration,
    )
    titan_state = kt.KBTitanState(
        species=["H", "SGA", "GH"],
        fixed_species=[],
        varying_species=["H", "SGA", "GH"],
        conversion={},
        concentration=concentration,
        density=torch.ones((1, 1), dtype=torch.float64),
        kzz=torch.zeros((1, 1), dtype=torch.float64),
        state=state,
    )
    condensation = kt.KBTitanSourceTerm(
        kind="titan_condensation",
        reaction_id=351,
        reactants=["H", "SGA"],
        products=["GH"],
        parameters={"source": "pun_special", "A": 10.0},
    )

    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, [condensation]
    )
    linearization = kt.build_source_linearization(state, atm_sources)

    expected_rate = torch.tensor([[[6.0]]], dtype=torch.float64)
    torch.testing.assert_close(linearization.tendency[:, :, 0], -expected_rate[:, :, 0])
    torch.testing.assert_close(linearization.tendency[:, :, 1], torch.zeros((1, 1), dtype=torch.float64))
    torch.testing.assert_close(linearization.tendency[:, :, 2], expected_rate[:, :, 0])


def test_titan_ch4_grain_special_sources_are_product_only_and_shifted():
    """Cheng CH4 grain loading/release has non-local special semantics."""
    import kintera as kt

    x1f = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    concentration = torch.tensor(
        [
            [
                [10.0, 0.5, 2.0, 1.0],
                [20.0, 0.5, 3.0, 1.0],
                [30.0, 0.5, 5.0, 1.0],
            ]
        ],
        dtype=torch.float64,
    )
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 3), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 3), 1.0e4, dtype=torch.float64),
        concentration=concentration,
    )
    titan_state = kt.KBTitanState(
        species=["CH4", "SGA", "GCH4", "U"],
        fixed_species=[],
        varying_species=["CH4", "SGA", "GCH4", "U"],
        conversion={},
        concentration=concentration,
        density=torch.ones((1, 3), dtype=torch.float64),
        kzz=torch.zeros((1, 3), dtype=torch.float64),
        state=state,
    )
    loading = kt.KBTitanSourceTerm(
        kind="titan_condensation",
        reaction_id=475,
        reactants=["CH4", "SGA"],
        products=["GCH4"],
        parameters={"source": "pun_special", "A": 2.0},
    )
    release = kt.KBTitanSourceTerm(
        kind="titan_sublimation",
        reaction_id=1361,
        reactants=["GCH4", "U"],
        products=["CH4"],
        parameters={
            "source": "pun_special",
            "vapor_A": 0.0,
            "vapor_B": 0.0,
            "vapor_C": 300.0,
            "vapor_Tmin_C": 0.0,
            "vapor_Tmax_C": 1000.0,
        },
    )

    loading_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, [loading]
    )
    loading_linearization = kt.build_source_linearization(state, loading_sources)
    expected_loading = 2.0 * concentration[:, :, 1] * concentration[:, :, 0]
    torch.testing.assert_close(
        loading_linearization.tendency[:, :, 0],
        torch.zeros((1, 3), dtype=torch.float64),
    )
    torch.testing.assert_close(
        loading_linearization.tendency[:, :, 2],
        expected_loading,
    )

    release_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, [release]
    )
    release_linearization = kt.build_source_linearization(state, release_sources)
    loss = -release_linearization.tendency[:, :, 2]
    expected_ch4 = torch.zeros((1, 3), dtype=torch.float64)
    expected_ch4[:, 1:] = loss[:, :-1]
    torch.testing.assert_close(
        release_linearization.tendency[:, :, 0],
        expected_ch4,
    )
    torch.testing.assert_close(
        release_linearization.tendency[:, -1, 2],
        torch.zeros((1,), dtype=torch.float64),
    )
    system, rhs = kt.build_implicit_step_system(
        state,
        torch.zeros((1, 3), dtype=torch.float64),
        1.0,
        source_terms=release_sources,
    )
    dense = system.global_csr.to_dense()
    row = (1 * state.nspecies) + 0
    col = (0 * state.nspecies) + 2
    torch.testing.assert_close(dense[row, col], -loss[0, 0] / concentration[0, 0, 2])
    torch.testing.assert_close(rhs[:, 1, 0], concentration[:, 1, 0])


def test_titan_sublimation_uses_total_grain_ice_abundance_limiter():
    """Sublimation switches from site capacity to total ice abundance above a monolayer."""
    import kintera as kt
    from kintera.kinetics_base_titan.physics import _titan_sublimation_rate_profile

    x1f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    concentration = torch.tensor([[[1.0, 1.0, 1.2e16, 0.0]]], dtype=torch.float64)
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 1), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 1), 1.0e4, dtype=torch.float64),
        concentration=concentration,
    )
    params = {
        "vapor_A": 0.0,
        "vapor_B": 0.0,
        "vapor_C": 300.0,
    }
    species_index = {"SGA": 0, "GCH4": 1, "GC2H2": 2, "GH": 3}
    limited = _titan_sublimation_rate_profile(state, species_index, params, "CH4", 16.0)

    site_only_concentration = concentration.clone()
    site_only_concentration[:, :, 2] = 0.0
    site_only_state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=state.temperature,
        pressure=state.pressure,
        concentration=site_only_concentration,
    )
    site_only = _titan_sublimation_rate_profile(
        site_only_state, species_index, params, "CH4", 16.0
    )

    torch.testing.assert_close(limited, site_only / 2.0)


def test_titan_named_grain_species_still_use_eddy_diffusion():
    """Cheng Titan G-prefixed species are names, not KINETICS grain flags."""
    import kintera as kt

    scale = kt.kinetics_base_titan_species_diffusion_scale(
        ["H", "GH", "GCH4", "CH4"],
        dtype=torch.float64,
    )

    torch.testing.assert_close(scale, torch.ones(4, dtype=torch.float64))


def test_titan_charged_thermal_sources_are_typed_and_linearized():
    """Charged .pun thermal reactions reuse the mass-action source kernel."""
    import kintera as kt

    x1f = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    x2f = torch.tensor([0.0, 1.0], dtype=torch.float64)
    concentration = torch.tensor(
        [
            [
                [2.0, 3.0, 0.0, 0.0],
                [5.0, 7.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    state = kt.AtmState2D(
        x1f=x1f,
        x2f=x2f,
        temperature=torch.full((1, 2), 250.0, dtype=torch.float64),
        pressure=torch.full((1, 2), 1.0e4, dtype=torch.float64),
        concentration=concentration,
    )
    titan_state = kt.KBTitanState(
        species=["E", "X+", "A", "B"],
        fixed_species=[],
        varying_species=["E", "X+", "A", "B"],
        conversion={},
        concentration=concentration,
        density=torch.ones((1, 2), dtype=torch.float64),
        kzz=torch.zeros((1, 2), dtype=torch.float64),
        state=state,
    )
    recombination = kt.KBTitanSourceTerm(
        kind="pun_dissociative_recombination",
        reaction_id=1,
        reactants=["E", "X+"],
        products=["A", "B"],
        parameters={
            "A": 0.5,
            "b": 0.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "F": 0.0,
            "Tmin": 1.0,
            "Tmax": 1.0e9,
            "reactant_coefficients": [1, 1],
            "product_coefficients": [1, 1],
            "reactant_charge": 0,
            "product_charge": 0,
            "net_charge_delta": 0,
            "charge_balanced": True,
        },
    )

    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, [recombination]
    )
    linearization = kt.build_source_linearization(state, atm_sources)

    expected_rate = 0.5 * concentration[:, :, 0] * concentration[:, :, 1]
    expected_tendency = torch.zeros_like(concentration)
    expected_tendency[:, :, 0] = -expected_rate
    expected_tendency[:, :, 1] = -expected_rate
    expected_tendency[:, :, 2] = expected_rate
    expected_tendency[:, :, 3] = expected_rate
    torch.testing.assert_close(linearization.tendency, expected_tendency)
    assert torch.count_nonzero(linearization.jacobian).item() > 0


def test_titan_source_builder_classifies_charged_pun_reactions(tmp_path):
    """Ion mass-action and recombination terms get explicit source kinds."""
    import kintera as kt

    pun_path = tmp_path / "charged.pun"
    pun_path.write_text(
        "\n".join(
            [
                " NATOM NMOL NREACT NPART VER",
                "    2    5      2     0 5",
                "N   14.01 C   12.01",
                "   1. E             1    1      0.00   0  0",
                "   2. N2+           1    1     28.02   0  0",
                "   3. N             1    1     14.01   1  0",
                "   4. CH4           1    1     16.05   0  1",
                "   5. CH4+          1    1     16.05   0  1",
                "   1. 2  2  (  )   1+(  )   2=( 2)   3   1.00E-07   0.00       0.0",
                "   2. 2  2  (  )   2+(  )   4=(  )   5+(  )   3   1.00E-09   0.00       0.0",
                "",
            ]
        )
    )

    terms = kt.build_kinetics_base_titan_source_terms(pun_path)
    by_id = {term.reaction_id: term for term in terms}

    assert by_id[1].kind == "pun_dissociative_recombination"
    assert by_id[1].parameters["charge_balanced"] is True
    assert by_id[2].kind == "pun_ion_mass_action_reaction"
    assert by_id[2].parameters["reactant_charge"] == 1
    assert by_id[2].parameters["product_charge"] == 1

    run_input_path = tmp_path / "charged.inp"
    run_input_path.write_text(
        "\n".join(
            [
                "NFIX NVARYS NVARYF",
                "0 0 5",
                "NPHOTO NPHOTS NPHOTR NPHOTD",
                "0 0 0 0",
                "IFIX,IVARYS,IVARYF:",
                "1 2 3 4 5",
                "",
            ]
        )
    )
    report = kt.classify_kinetics_base_titan_reactions(
        str(pun_path), str(run_input_path)
    )
    assert report.charged_species == ["E", "N2+", "CH4+"]
    assert report.charged_reactions == 2
    assert report.charged_thermal_candidate_reactions == 2
    assert report.dissociative_recombination_reactions == 1
    assert report.ion_mass_action_reactions == 1
    assert report.charge_balanced_reactions == 2
    assert report.charge_imbalanced_reaction_ids == []


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
    titan_dir, _, _, _, _ = _external_titan_paths()
    boundary_path = tmp_path / "titan_Cheng_N_ions_H2CN.effective.bc"
    _write_titan_ch4_effective_boundary_input(
        os.path.join(titan_dir, "titan_Cheng_N_ions_H2CN.bc_save"),
        boundary_path,
        initial,
    )
    titan_state = _build_titan_state_from_oracle(initial, species, boundary_path)
    assert titan_state.concentration.shape == (1, 50, 128)
    # KINETICS-base __CHENG pins CH4's cold-trap boundary row, so CH4's
    # conversion type reflects that boundary rather than the surface
    # mixing-ratio input.
    assert titan_state.conversion["CH4"] == "kinetics_base_cheng_cold_trap_mixing_ratio"
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
    titan_dir, _, _, _, _ = _external_titan_paths()
    boundary_path = tmp_path / "titan_Cheng_N_ions_H2CN.effective.bc"
    _write_titan_ch4_effective_boundary_input(
        os.path.join(titan_dir, "titan_Cheng_N_ions_H2CN.bc_save"),
        boundary_path,
        initial,
    )
    titan_state = _build_titan_state_from_oracle(initial, species, boundary_path)
    source_terms, pun_metadata = _build_titan_source_terms_from_oracle()
    kintera_concentration = _solve_titan_transport_steps(
        titan_state,
        ntime=ntime,
        source_terms=source_terms,
        pun_metadata=pun_metadata,
    )
    _assert_titan_equivalence(kintera_concentration, reference_concentration, species)


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
    assert report.charged_species_count == len(report.charged_species)
    assert "E" in report.charged_species
    assert report.charged_reactions > 0
    assert (
        report.charge_balanced_reactions + report.charge_imbalanced_reactions
        == report.charged_reactions
    )
    assert report.charged_thermal_candidate_reactions == (
        report.ion_mass_action_reactions
        + report.dissociative_recombination_reactions
    )
    assert report.dissociative_recombination_reactions > 0
    assert report.ion_mass_action_reactions > 0


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
        term.kind == "pun_dissociative_recombination" for term in terms
    )
    ion_mass_action_count = sum(
        term.kind == "pun_ion_mass_action_reaction" for term in terms
    )
    electron_impact_count = sum(
        term.kind == "pun_electron_impact_reaction" for term in terms
    )
    assert disabled_ids == {187, 2094, 2095, 2096, 2097, 2098, 2099}
    assert unimplemented_ids == set()
    assert ion_recombination_count > 0
    assert ion_mass_action_count > 0
    assert electron_impact_count > 0


def test_external_titan_truncate_active_network_if_available():
    """Titan source terms can be restricted to KINETICS-base's active network."""
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    truncate_path = os.path.join(titan_dir, "kintitan.truncate")
    if not os.path.exists(truncate_path):
        pytest.skip(f"missing external KINETICS-base file: {truncate_path}")

    active_network = kt.parse_kinetics_base_truncate(truncate_path)
    assert active_network.reaction_mapping[64] == 63
    assert active_network.reaction_mapping[143] == 120
    assert active_network.reaction_mapping[260] == 167
    assert active_network.reaction_mapping[1048] == 0
    assert active_network.reaction_mapping[1056] == 0
    assert active_network.reaction_mapping[1057] == 606
    assert len(active_network.active_reaction_ids) == 1364

    terms = kt.build_kinetics_base_titan_source_terms(
        pun_path,
        special_path=os.path.join(
            titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
        ),
        run_input_path=run_input_path,
        photo_catalog_path=os.path.join(titan_dir, "Cheng_catalog_v4.dat"),
        cross_dir=os.path.join(titan_dir, "Cheng_cross"),
        flux_path=os.path.join(titan_dir, "flare_kin_oct2003.inp"),
        truncate_path=truncate_path,
    )
    by_id = {term.reaction_id: term for term in terms}

    for reaction_id in {64, 65, 143, 260, 1057}:
        assert reaction_id in by_id
    for reaction_id in {1048, 1056}:
        assert reaction_id not in by_id


def test_external_titan_special_index_if_available():
    """Cheng runtime ISP mappings are available to source-term construction."""
    import kintera as kt

    titan_dir, _, _, _, _ = _external_titan_paths()
    special_path = os.path.join(
        titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
    )

    special = kt.parse_kinetics_base_special_index(special_path)

    assert special.target_id(381, kind=1) == 9
    assert special.target_id(503, kind=2) == 10
    assert special.target_id(537, kind=2) == 202
    assert special.target_id(538, kind=2) == 203
    assert special.target_id(539, kind=2) == 247
    assert special.target_id(540, kind=2) == 246
    assert special.target_id(229, kind=3) == 222


def test_external_titan_ch4_photolysis_branch_loss_if_available():
    """KINETICS-base Cheng CH4 photo branches share products but not parent loss."""
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    truncate_path = os.path.join(titan_dir, "kintitan.truncate")
    terms = kt.build_kinetics_base_titan_source_terms(
        pun_path,
        special_path=os.path.join(
            titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
        ),
        run_input_path=run_input_path,
        photo_catalog_path=os.path.join(titan_dir, "Cheng_catalog_v4.dat"),
        cross_dir=os.path.join(titan_dir, "Cheng_cross"),
        flux_path=os.path.join(titan_dir, "flare_kin_oct2003.inp"),
        truncate_path=truncate_path,
    )
    ch4_photo = {
        term.reaction_id: term
        for term in terms
        if term.kind == "pun_photo_rate_reaction" and term.reactants == ["CH4"]
    }

    assert set(ch4_photo) == {5, 6, 7, 8, 9, 222}
    assert ch4_photo[6].parameters["source"] == "cheng_branch_rate_profile"
    assert ch4_photo[6].parameters["rate_profile_multiplier"]
    for reaction_id in {5, 6, 7, 8, 9}:
        assert ch4_photo[reaction_id].parameters["suppress_reactant_loss"] is True
    assert ch4_photo[8].products == ["(3)CH2"]
    assert ch4_photo[222].products == ["CH4"]
    assert not ch4_photo[222].parameters.get("suppress_reactant_loss", False)


def test_external_titan_electron_impact_uses_special_runtime_indices_if_available():
    """Only Cheng runtime-referenced ionization branches are enabled."""
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    truncate_path = os.path.join(titan_dir, "kintitan.truncate")
    terms = kt.build_kinetics_base_titan_source_terms(
        pun_path,
        special_path=os.path.join(
            titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
        ),
        run_input_path=run_input_path,
        photo_catalog_path=os.path.join(titan_dir, "Cheng_catalog_v4.dat"),
        cross_dir=os.path.join(titan_dir, "Cheng_cross"),
        flux_path=os.path.join(titan_dir, "flare_kin_oct2003.inp"),
        truncate_path=truncate_path,
    )
    electron_impact_ids = {
        term.reaction_id
        for term in terms
        if term.kind == "pun_electron_impact_reaction"
    }

    assert electron_impact_ids == {202, 203, 246, 247}


def test_external_titan_catalog_photolysis_branches_if_available():
    """Catalog-mapped unary A=0 branches are exposed as Titan photolysis."""
    import kintera as kt

    titan_dir, _, pun_path, run_input_path, _ = _external_titan_paths()
    truncate_path = os.path.join(titan_dir, "kintitan.truncate")
    terms = kt.build_kinetics_base_titan_source_terms(
        pun_path,
        special_path=os.path.join(
            titan_dir, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
        ),
        run_input_path=run_input_path,
        photo_catalog_path=os.path.join(titan_dir, "Cheng_catalog_v4.dat"),
        cross_dir=os.path.join(titan_dir, "Cheng_cross"),
        flux_path=os.path.join(titan_dir, "flare_kin_oct2003.inp"),
        truncate_path=truncate_path,
    )
    photolysis_branches = {
        (tuple(term.reactants), tuple(term.products)): term
        for term in terms
        if term.kind == "pun_photo_rate_reaction"
    }

    expected_branches = [
        (("C4H4",), ("C4H2", "H2")),
        (("NH3",), ("NH2", "H")),
        (("HCN",), ("H", "CN")),
    ]
    for branch in expected_branches:
        term = photolysis_branches[branch]
        assert term.parameters["source"] == "catalog_flux"
        assert term.parameters["rate"] > 0.0
        assert term.parameters["radiation_active_nlyr"] == 40
        assert term.parameters["freeze_actinic_flux"] is True
        assert not term.parameters.get("suppress_reactant_loss", False)


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
