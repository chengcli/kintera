from pathlib import Path

import pytest
import scipy.sparse.linalg
import torch

import kintera as kt
from kintera.atm2d import radiation as atm2d_radiation


torch.set_default_dtype(torch.float64)
TEST_DIR = Path(__file__).resolve().parent
CHAPMAN_CYCLE_YAML = TEST_DIR / "chapman_cycle.yaml"


def _make_state(ncol: int = 3, nlyr: int = 5, ns: int = 2) -> kt.AtmState2D:
    x2f = torch.linspace(0.0, 2.0e5, ncol + 1, dtype=torch.float64)
    x1f = torch.linspace(0.0, 4.0e5, nlyr + 1, dtype=torch.float64)
    temp = torch.full((ncol, nlyr), 250.0, dtype=torch.float64)
    pres = torch.logspace(5.0, 3.0, nlyr, dtype=torch.float64).unsqueeze(0).expand(ncol, nlyr)
    conc = torch.full((ncol, nlyr, ns), 0.5e-6, dtype=torch.float64)
    return kt.AtmState2D(x1f=x1f, x2f=x2f, temperature=temp, pressure=pres, concentration=conc)


def test_vertical_eddy_transport_matches_columnwise_reference():
    state = _make_state(ncol=2, nlyr=5, ns=2)
    conc = torch.tensor(
        [
            [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.35, 0.65], [0.25, 0.75]],
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.65, 0.35], [0.75, 0.25]],
        ],
        dtype=torch.float64,
    )
    state.concentration = conc
    kzz = torch.tensor(
        [[1.0e5, 2.0e5, 3.0e5, 4.0e5, 5.0e5], [1.5e5, 2.5e5, 3.5e5, 4.5e5, 5.5e5]],
        dtype=torch.float64,
    )

    matrix = kt.build_eddy_diffusion_matrix(state, kzz)
    tendency = matrix.matvec(conc)
    kzz_face = 0.5 * (kzz[:, :-1] + kzz[:, 1:])
    for icol in range(state.ncol):
        reference = kt.diffusion_tendency(conc[icol], kzz_face[icol], state.x1v[1:] - state.x1v[:-1])
        torch.testing.assert_close(tendency[icol], reference, atol=1e-12, rtol=1e-12)


def test_horizontal_and_cross_diffusion_create_2d_coupling():
    state = _make_state(ncol=3, nlyr=4, ns=2)
    conc = torch.zeros((state.ncol, state.nlyr, state.nspecies), dtype=state.dtype)
    conc[0, :, 0] = 1.0
    conc[2, :, 0] = -1.0
    state.concentration = conc

    kzz = torch.zeros((state.ncol, state.nlyr), dtype=state.dtype)
    kyy = torch.full((state.ncol, state.nlyr), 2.0e5, dtype=state.dtype)
    kzy = torch.full((state.ncol, state.nlyr), 5.0e4, dtype=state.dtype)

    matrix = kt.build_eddy_diffusion_matrix(state, kzz, kyy=kyy, kzy=kzy)
    tendency = matrix.matvec(conc)

    assert torch.count_nonzero(tendency[:, :, 0]).item() > 0
    assert matrix.global_csr._nnz() > state.ncol * state.nlyr * state.nspecies


def test_binary_diffusion_creates_species_coupling():
    state = _make_state(ncol=2, nlyr=4, ns=3)
    binary = torch.zeros((state.ncol, state.nlyr, state.nspecies, state.nspecies), dtype=state.dtype)
    binary[..., 0, 0] = 1.0e4
    binary[..., 1, 1] = 1.5e4
    binary[..., 2, 2] = 0.8e4
    binary[..., 0, 1] = 0.3e4
    binary[..., 1, 0] = 0.2e4
    weights = torch.tensor([28.0, 32.0, 16.0], dtype=state.dtype)

    matrix = kt.build_binary_diffusion_matrix(state, binary, weights, include_gravity=False)
    dense = matrix.global_csr.to_dense()
    row0 = dense[0 : state.nspecies, state.nspecies : 2 * state.nspecies]
    assert row0[0, 1].abs().item() > 0.0


def test_sparse_solver_matches_dense_solution():
    ncol, nlyr, ns = 2, 3, 2
    nstate = ncol * nlyr * ns
    dense = torch.eye(nstate, dtype=torch.float64) * 5.0
    dense[0, 1] = -0.2
    dense[1, 0] = -0.1
    dense[3, 7] = 0.15
    dense[7, 3] = 0.05
    rhs = torch.arange(1, nstate + 1, dtype=torch.float64).reshape(ncol, nlyr, ns)

    matrix = kt.SparseSystemMatrix.from_dense(dense, ncol=ncol, nlyr=nlyr, nspecies=ns)
    sol = kt.solve_sparse_system(matrix, rhs)
    ref = torch.linalg.solve(dense, rhs.reshape(-1)).reshape(ncol, nlyr, ns)
    torch.testing.assert_close(sol, ref, atol=1e-12, rtol=1e-12)


def test_sparse_solver_reuses_cpu_factorization(monkeypatch):
    ncol, nlyr, ns = 1, 3, 1
    dense = torch.tensor(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=torch.float64,
    )
    matrix = kt.SparseSystemMatrix.from_dense(dense, ncol=ncol, nlyr=nlyr, nspecies=ns)
    calls = {"count": 0}
    factorized_impl = scipy.sparse.linalg.factorized

    def counting_factorized(*args, **kwargs):
        calls["count"] += 1
        return factorized_impl(*args, **kwargs)

    monkeypatch.setattr(scipy.sparse.linalg, "factorized", counting_factorized)
    rhs1 = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64)
    rhs2 = torch.tensor([[[3.0], [2.0], [1.0]]], dtype=torch.float64)

    sol1 = kt.solve_sparse_system(matrix, rhs1)
    sol2 = kt.solve_sparse_system(matrix, rhs2)

    ref1 = torch.linalg.solve(dense, rhs1.reshape(-1)).reshape(ncol, nlyr, ns)
    ref2 = torch.linalg.solve(dense, rhs2.reshape(-1)).reshape(ncol, nlyr, ns)
    torch.testing.assert_close(sol1, ref1, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(sol2, ref2, atol=1.0e-12, rtol=1.0e-12)
    assert calls["count"] == 1


def test_steady_1d_advection_diffusion_dirichlet_matches_analytic_solution():
    ncol, nlyr, ns = 1, 161, 1
    x = torch.linspace(0.0, 1.0, nlyr, dtype=torch.float64)
    dx = float(x[1] - x[0])
    diffusivity = 2.0e-2
    velocity = 3.0e-1
    c_left = 1.0
    c_right = 0.2
    dt = 0.5

    operator = torch.zeros((nlyr, nlyr), dtype=torch.float64)
    lower = diffusivity / (dx * dx) + velocity / (2.0 * dx)
    diag = -2.0 * diffusivity / (dx * dx)
    upper = diffusivity / (dx * dx) - velocity / (2.0 * dx)
    for i in range(1, nlyr - 1):
        operator[i, i - 1] = lower
        operator[i, i] = diag
        operator[i, i + 1] = upper

    system = torch.eye(nlyr, dtype=torch.float64) - dt * operator
    system[0] = 0.0
    system[-1] = 0.0
    system[0, 0] = 1.0
    system[-1, -1] = 1.0

    rhs_override_mask = torch.zeros((ncol, nlyr, ns), dtype=torch.bool)
    rhs_override_values = torch.zeros((ncol, nlyr, ns), dtype=torch.float64)
    rhs_override_mask[0, 0, 0] = True
    rhs_override_mask[0, -1, 0] = True
    rhs_override_values[0, 0, 0] = c_left
    rhs_override_values[0, -1, 0] = c_right

    matrix = kt.SparseSystemMatrix.from_dense(
        system,
        ncol=ncol,
        nlyr=nlyr,
        nspecies=ns,
        rhs_override_mask=rhs_override_mask,
        rhs_override_values=rhs_override_values,
    )

    state = torch.zeros((ncol, nlyr, ns), dtype=torch.float64)
    state[0, 0, 0] = c_left
    state[0, -1, 0] = c_right
    for _ in range(400):
        next_state = kt.solve_sparse_system(matrix, state)
        if torch.max(torch.abs(next_state - state)).item() < 1.0e-11:
            state = next_state
            break
        state = next_state

    profile = state[0, :, 0]
    analytic = c_left + (c_right - c_left) * (
        torch.exp((velocity / diffusivity) * x) - 1.0
    ) / (torch.exp(torch.tensor(velocity / diffusivity, dtype=torch.float64)) - 1.0)
    torch.testing.assert_close(profile, analytic, atol=2.5e-3, rtol=2.5e-3)


def test_steady_2d_diffusion_four_side_dirichlet_matches_linear_solution():
    ncol, nlyr, ns = 31, 25, 1
    x2f = torch.linspace(0.0, 2.0, ncol + 1, dtype=torch.float64)
    x1f = torch.linspace(0.0, 1.5, nlyr + 1, dtype=torch.float64)
    x2v = 0.5 * (x2f[:-1] + x2f[1:])
    x1v = 0.5 * (x1f[:-1] + x1f[1:])
    temp = torch.full((ncol, nlyr), 250.0, dtype=torch.float64)
    pres = torch.full((ncol, nlyr), 1.0e4, dtype=torch.float64)
    conc = torch.zeros((ncol, nlyr, ns), dtype=torch.float64)
    state = kt.AtmState2D(x1f=x1f, x2f=x2f, temperature=temp, pressure=pres, concentration=conc)

    x1_grid = x1v.unsqueeze(0).expand(ncol, nlyr)
    x2_grid = x2v.unsqueeze(1).expand(ncol, nlyr)
    analytic = 0.3 + 0.7 * (x1_grid / x1f[-1]) - 0.4 * (x2_grid / x2f[-1])

    left_bc = analytic[0, :].unsqueeze(-1)
    right_bc = analytic[-1, :].unsqueeze(-1)
    bottom_bc = analytic[:, 0].unsqueeze(-1)
    top_bc = analytic[:, -1].unsqueeze(-1)
    bc = kt.SpeciesBoundaryConditions2D(
        left=kt.SpeciesBoundaryCondition(kind="dirichlet", value=left_bc),
        right=kt.SpeciesBoundaryCondition(kind="dirichlet", value=right_bc),
        bottom=kt.SpeciesBoundaryCondition(kind="dirichlet", value=bottom_bc),
        top=kt.SpeciesBoundaryCondition(kind="dirichlet", value=top_bc),
    )

    kzz = torch.full((ncol, nlyr), 4.0e-2, dtype=torch.float64)
    kyy = torch.full((ncol, nlyr), 9.0e-2, dtype=torch.float64)
    transport = kt.build_transport_matrix(state, kzz, kyy=kyy)
    dt = 0.2
    system = torch.eye(transport.nstate, dtype=torch.float64) - dt * transport.global_csr.to_dense()
    matrix = kt.SparseSystemMatrix.from_dense(system, ncol=ncol, nlyr=nlyr, nspecies=ns)

    row_values: dict[int, float] = {}
    rhs_override_mask = torch.zeros((ncol, nlyr, ns), dtype=torch.bool)
    rhs_override_values = torch.zeros((ncol, nlyr, ns), dtype=torch.float64)

    def add_dirichlet_row(icol: int, ilev: int, value: float) -> None:
        row = (icol * nlyr + ilev) * ns
        row_values[row] = 1.0
        rhs_override_mask[icol, ilev, 0] = True
        rhs_override_values[icol, ilev, 0] = value

    for ilev in range(nlyr):
        add_dirichlet_row(0, ilev, float(left_bc[ilev, 0]))
        add_dirichlet_row(ncol - 1, ilev, float(right_bc[ilev, 0]))
    for icol in range(ncol):
        add_dirichlet_row(icol, 0, float(bottom_bc[icol, 0]))
        add_dirichlet_row(icol, nlyr - 1, float(top_bc[icol, 0]))

    rows = torch.tensor(sorted(row_values), dtype=torch.int64)
    matrix = matrix.replace_rows(
        rows,
        rows.clone(),
        torch.ones(rows.numel(), dtype=torch.float64),
        rhs_override_mask=rhs_override_mask,
        rhs_override_values=rhs_override_values,
    )

    solution = torch.zeros((ncol, nlyr, ns), dtype=torch.float64)
    solution[:, :, 0] = analytic
    solution[1:-1, 1:-1, 0] = 0.0
    for _ in range(600):
        next_solution = kt.solve_sparse_system(matrix, solution)
        if torch.max(torch.abs(next_solution - solution)).item() < 1.0e-11:
            solution = next_solution
            break
        solution = next_solution

    torch.testing.assert_close(solution[:, :, 0], analytic, atol=2.5e-3, rtol=2.5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_cusolver_binding_matches_dense_solution():
    dense = torch.tensor(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.0, 4.5, -0.5, 0.0],
            [0.0, -0.25, 3.5, -0.75],
            [0.0, 0.0, -1.0, 2.5],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
    csr = dense.cuda().to_sparse_csr()
    sol = kt.cuda_csr_solve_cusolver(
        csr.crow_indices().to(dtype=torch.int32),
        csr.col_indices().to(dtype=torch.int32),
        csr.values(),
        rhs.cuda(),
        0.0,
        0,
    ).cpu()
    ref = torch.linalg.solve(dense, rhs)
    torch.testing.assert_close(sol, ref, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_steady_1d_advection_diffusion_dirichlet_matches_analytic_solution():
    ncol, nlyr, ns = 1, 161, 1
    x = torch.linspace(0.0, 1.0, nlyr, dtype=torch.float64, device="cuda")
    dx = float((x[1] - x[0]).item())
    diffusivity = 2.0e-2
    velocity = 3.0e-1
    c_left = 1.0
    c_right = 0.2
    dt = 0.5

    operator = torch.zeros((nlyr, nlyr), dtype=torch.float64, device="cuda")
    lower = diffusivity / (dx * dx) + velocity / (2.0 * dx)
    diag = -2.0 * diffusivity / (dx * dx)
    upper = diffusivity / (dx * dx) - velocity / (2.0 * dx)
    for i in range(1, nlyr - 1):
        operator[i, i - 1] = lower
        operator[i, i] = diag
        operator[i, i + 1] = upper

    system = torch.eye(nlyr, dtype=torch.float64, device="cuda") - dt * operator
    system[0] = 0.0
    system[-1] = 0.0
    system[0, 0] = 1.0
    system[-1, -1] = 1.0

    rhs_override_mask = torch.zeros((ncol, nlyr, ns), dtype=torch.bool, device="cuda")
    rhs_override_values = torch.zeros((ncol, nlyr, ns), dtype=torch.float64, device="cuda")
    rhs_override_mask[0, 0, 0] = True
    rhs_override_mask[0, -1, 0] = True
    rhs_override_values[0, 0, 0] = c_left
    rhs_override_values[0, -1, 0] = c_right

    matrix = kt.SparseSystemMatrix.from_dense(
        system,
        ncol=ncol,
        nlyr=nlyr,
        nspecies=ns,
        rhs_override_mask=rhs_override_mask,
        rhs_override_values=rhs_override_values,
    )

    state = torch.zeros((ncol, nlyr, ns), dtype=torch.float64, device="cuda")
    state[0, 0, 0] = c_left
    state[0, -1, 0] = c_right
    for _ in range(400):
        next_state = kt.solve_sparse_system(matrix, state)
        if torch.max(torch.abs(next_state - state)).item() < 1.0e-11:
            state = next_state
            break
        state = next_state

    profile = state[0, :, 0]
    analytic = c_left + (c_right - c_left) * (
        torch.exp((velocity / diffusivity) * x) - 1.0
    ) / (torch.exp(torch.tensor(velocity / diffusivity, dtype=torch.float64, device="cuda")) - 1.0)
    torch.testing.assert_close(profile.cpu(), analytic.cpu(), atol=2.5e-3, rtol=2.5e-3)


def test_boundary_conditions_apply_on_left_and_top_edges():
    state = _make_state(ncol=3, nlyr=4, ns=3)
    kzz = torch.full((state.ncol, state.nlyr), 1.0e5, dtype=torch.float64)
    kyy = torch.full((state.ncol, state.nlyr), 2.0e5, dtype=torch.float64)
    bc = kt.SpeciesBoundaryConditions2D(
        left=kt.SpeciesBoundaryCondition(
            kind=["dirichlet", "neumann", "none"],
            value=torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0], [7.0, 8.0, 0.0]], dtype=torch.float64),
        ),
        top=kt.SpeciesBoundaryCondition(
            kind=["neumann", "dirichlet", "none"],
            value=torch.tensor([[9.0, 10.0, 0.0], [11.0, 12.0, 0.0], [13.0, 14.0, 0.0]], dtype=torch.float64),
        ),
    )
    matrix = kt.build_eddy_diffusion_matrix(state, kzz, kyy=kyy, boundary_conditions=bc)
    rhs = torch.zeros((state.ncol, state.nlyr, state.nspecies), dtype=torch.float64)
    applied = matrix.apply_rhs_overrides(rhs)
    dense = matrix.global_csr.to_dense()

    assert torch.allclose(applied[0, :3, 0], torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64))
    assert torch.allclose(applied[0, :3, 1], torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64))
    assert applied[0, -1, 0].item() == 9.0
    assert applied[0, -1, 1].item() == 10.0
    top_mid_dirichlet = ((1 * state.nlyr + (state.nlyr - 1)) * state.nspecies) + 1
    left_mid_dirichlet = ((0 * state.nlyr + 1) * state.nspecies) + 0
    assert dense[left_mid_dirichlet, left_mid_dirichlet].item() == 1.0
    assert dense[top_mid_dirichlet, top_mid_dirichlet].item() == 1.0


def test_boundary_corner_precedence_overrides_left_with_top():
    state = _make_state(ncol=3, nlyr=4, ns=1)
    kzz = torch.full((state.ncol, state.nlyr), 1.0e5, dtype=torch.float64)
    kyy = torch.full((state.ncol, state.nlyr), 2.0e5, dtype=torch.float64)
    bc = kt.SpeciesBoundaryConditions2D(
        left=kt.SpeciesBoundaryCondition(kind="dirichlet", value=7.0),
        top=kt.SpeciesBoundaryCondition(kind="neumann", value=torch.tensor([[5.0], [6.0], [7.0]], dtype=torch.float64)),
    )
    matrix = kt.build_eddy_diffusion_matrix(state, kzz, kyy=kyy, boundary_conditions=bc)
    dense = matrix.global_csr.to_dense()
    rhs = torch.zeros((state.ncol, state.nlyr, state.nspecies), dtype=torch.float64)
    applied = matrix.apply_rhs_overrides(rhs)

    corner_row = (0 * state.nlyr + (state.nlyr - 1)) * state.nspecies
    corner_neighbor = (0 * state.nlyr + (state.nlyr - 2)) * state.nspecies
    row = dense[corner_row]
    nnz = torch.nonzero(row, as_tuple=False).squeeze(-1)

    assert applied[0, state.nlyr - 1, 0].item() == 5.0
    assert nnz.numel() == 2
    assert set(nnz.tolist()) == {corner_neighbor, corner_row}
    expected = 1.0 / state.dx1v[-1]
    torch.testing.assert_close(row[corner_neighbor], -expected)
    torch.testing.assert_close(row[corner_row], expected)


def test_actinic_flux_from_disort_supports_2d_state():
    photo_opts = kt.PhotoChemOptions.from_yaml(str(CHAPMAN_CYCLE_YAML))
    photo = kt.PhotoChem(photo_opts)
    species = photo_opts.species()
    idx = {name: i for i, name in enumerate(species)}

    ncol, nlyr = 2, 4
    x2f = torch.linspace(0.0, 1.0e5, ncol + 1, dtype=torch.float64)
    x1f = torch.linspace(0.0, 3.0e5, nlyr + 1, dtype=torch.float64)
    temp = torch.full((ncol, nlyr), 250.0, dtype=torch.float64)
    pres = torch.logspace(5.0, 3.0, nlyr, dtype=torch.float64).unsqueeze(0).expand(ncol, nlyr)
    conc = torch.zeros((ncol, nlyr, len(species)), dtype=torch.float64)
    conc[..., idx["N2"]] = 0.79
    conc[..., idx["O2"]] = torch.tensor(
        [[0.21, 0.24, 0.27, 0.30], [0.30, 0.27, 0.24, 0.21]], dtype=torch.float64
    )
    conc[..., idx["O"]] = 1.0e-12
    conc[..., idx["O3"]] = torch.tensor(
        [[1.0e-10, 3.0e-10, 1.0e-9, 3.0e-9], [3.0e-9, 1.0e-9, 3.0e-10, 1.0e-10]],
        dtype=torch.float64,
    )
    state = kt.AtmState2D(x1f=x1f, x2f=x2f, temperature=temp, pressure=pres, concentration=conc)

    wave = photo.module("photolysis").buffer("wavelength")
    top_flux = torch.full((wave.numel(), ncol), 1.0e12, dtype=torch.float64)
    rt = kt.compute_actinic_flux_disort(photo, state, top_flux, concentration_unit="molecules_cm3")

    assert rt.optical_depth.shape == (ncol, nlyr, wave.numel())
    assert rt.actinic_flux.shape == (wave.numel(), ncol, nlyr)
    absorb_idx = int(torch.argmin(torch.abs(wave - 200.0)).item())
    assert rt.actinic_flux[absorb_idx, 0, 0].item() > rt.actinic_flux[absorb_idx, 0, -1].item()


def test_implicit_operator_adds_chemistry_and_photochemistry():
    kinetics = kt.Kinetics(kt.KineticsOptions.from_yaml(str(CHAPMAN_CYCLE_YAML)))
    photo_opts = kt.PhotoChemOptions.from_yaml(str(CHAPMAN_CYCLE_YAML))
    photo = kt.PhotoChem(photo_opts)
    species = photo_opts.species()
    kt.set_species_names(species)

    ncol, nlyr = 2, 3
    x2f = torch.linspace(0.0, 1.0e5, ncol + 1, dtype=torch.float64)
    x1f = torch.linspace(0.0, 2.0e5, nlyr + 1, dtype=torch.float64)
    temp = torch.full((ncol, nlyr), 250.0, dtype=torch.float64)
    pres = torch.full((ncol, nlyr), 1.0e4, dtype=torch.float64)
    conc = torch.full((ncol, nlyr, len(species)), 1.0e-6, dtype=torch.float64)
    state = kt.AtmState2D(x1f=x1f, x2f=x2f, temperature=temp, pressure=pres, concentration=conc)
    kzz = torch.full((ncol, nlyr), 1.0e5, dtype=torch.float64)
    kyy = torch.full((ncol, nlyr), 2.0e5, dtype=torch.float64)

    wave = photo.module("photolysis").buffer("wavelength")
    actinic_flux = torch.ones((wave.numel(), ncol, nlyr), dtype=torch.float64)

    transport = kt.build_transport_matrix(state, kzz, kyy=kyy)
    implicit = kt.build_implicit_operator(
        state,
        kzz,
        kyy=kyy,
        kinetics=kinetics,
        photo_chem=photo,
        actinic_flux=actinic_flux,
    )

    transport_dense = transport.global_csr.to_dense()
    implicit_dense = implicit.global_csr.to_dense()
    assert implicit_dense.shape == transport_dense.shape
    assert torch.isfinite(implicit_dense).all()
    assert not torch.allclose(implicit_dense, transport_dense)


def test_total_cross_section_uses_absorption_branch_only():
    opts = kt.PhotoChemOptions.from_yaml(str(CHAPMAN_CYCLE_YAML))
    module = kt.PhotoChem(opts)
    temperature = torch.full((1, 1), 250.0, dtype=torch.float64)
    wavelength = module.module("photolysis").buffer("wavelength").to(dtype=torch.float64)

    sigma = atm2d_radiation._total_cross_section_by_species(module, temperature, wavelength)
    species = opts.species()
    absorber_idx = species.index("O2")
    xs = module.module("photolysis").interp_cross_section(0, wavelength, temperature.reshape(-1))
    expected_absorption = xs[:, 0].reshape(1, 1, wavelength.numel())
    wrong_summed = xs.sum(-1).reshape(1, 1, wavelength.numel())

    torch.testing.assert_close(sigma[..., absorber_idx], expected_absorption, atol=0.0, rtol=0.0)
    assert torch.max(torch.abs(sigma[..., absorber_idx] - wrong_summed)).item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_sparse_solver_matches_cpu():
    ncol, nlyr, ns = 2, 3, 2
    nstate = ncol * nlyr * ns
    dense = torch.eye(nstate, dtype=torch.float64) * 4.0
    dense[0, 1] = -0.1
    dense[5, 3] = 0.2
    rhs = torch.randn((ncol, nlyr, ns), dtype=torch.float64)

    cpu_matrix = kt.SparseSystemMatrix.from_dense(dense, ncol=ncol, nlyr=nlyr, nspecies=ns)
    cpu_sol = kt.solve_sparse_system(cpu_matrix, rhs)

    gpu_matrix = kt.SparseSystemMatrix.from_dense(
        dense.cuda(), ncol=ncol, nlyr=nlyr, nspecies=ns
    )
    gpu_sol = kt.solve_sparse_system(gpu_matrix, rhs.cuda()).cpu()
    torch.testing.assert_close(cpu_sol, gpu_sol, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_sparse_solver_reuses_cached_int32_csr_indices():
    dense = torch.tensor(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=torch.float64,
        device="cuda",
    )
    matrix = kt.SparseSystemMatrix.from_dense(dense, ncol=1, nlyr=3, nspecies=1)

    crow1, col1 = matrix.cuda_csr_indices_int32()
    crow2, col2 = matrix.cuda_csr_indices_int32()
    assert crow1.dtype == torch.int32
    assert col1.dtype == torch.int32
    assert crow1.data_ptr() == crow2.data_ptr()
    assert col1.data_ptr() == col2.data_ptr()

    rhs1 = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64, device="cuda")
    rhs2 = torch.tensor([[[3.0], [2.0], [1.0]]], dtype=torch.float64, device="cuda")
    kt.solve_sparse_system(matrix, rhs1)
    crow3, col3 = matrix.cuda_csr_indices_int32()
    kt.solve_sparse_system(matrix, rhs2)
    crow4, col4 = matrix.cuda_csr_indices_int32()

    assert crow1.data_ptr() == crow3.data_ptr() == crow4.data_ptr()
    assert col1.data_ptr() == col3.data_ptr() == col4.data_ptr()
