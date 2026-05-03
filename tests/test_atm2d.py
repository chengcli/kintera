import pytest
import torch

import kintera as kt


torch.set_default_dtype(torch.float64)


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
    kzz = torch.full((state.ncol, state.nlyr - 1), 1.0e5, dtype=torch.float64)

    matrix = kt.build_eddy_diffusion_matrix(state, kzz)
    tendency = matrix.matvec(conc)
    for icol in range(state.ncol):
        reference = kt.diffusion_tendency(conc[icol], kzz[icol], state.x1v[1:] - state.x1v[:-1])
        torch.testing.assert_close(tendency[icol], reference, atol=1e-12, rtol=1e-12)


def test_horizontal_and_cross_diffusion_create_2d_coupling():
    state = _make_state(ncol=3, nlyr=4, ns=2)
    conc = torch.zeros((state.ncol, state.nlyr, state.nspecies), dtype=state.dtype)
    conc[0, :, 0] = 1.0
    conc[2, :, 0] = -1.0
    state.concentration = conc

    kzz = torch.zeros((state.ncol, state.nlyr - 1), dtype=state.dtype)
    kyy = torch.full((state.ncol - 1, state.nlyr), 2.0e5, dtype=state.dtype)
    kzy = torch.full((state.ncol, state.nlyr), 5.0e4, dtype=state.dtype)

    matrix = kt.build_eddy_diffusion_matrix(state, kzz, kyy=kyy, kzy=kzy)
    tendency = matrix.matvec(conc)

    assert torch.count_nonzero(tendency[:, :, 0]).item() > 0
    assert matrix.global_csr._nnz() > state.ncol * state.nlyr * state.nspecies


def test_binary_diffusion_creates_species_coupling():
    state = _make_state(ncol=2, nlyr=4, ns=3)
    binary = torch.zeros((state.ncol, state.nlyr - 1, state.nspecies, state.nspecies), dtype=state.dtype)
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


def test_boundary_conditions_apply_on_left_and_top_edges():
    state = _make_state(ncol=3, nlyr=4, ns=3)
    kzz = torch.full((state.ncol, state.nlyr - 1), 1.0e5, dtype=torch.float64)
    kyy = torch.full((state.ncol - 1, state.nlyr), 2.0e5, dtype=torch.float64)
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


def test_actinic_flux_from_disort_supports_2d_state():
    photo_opts = kt.PhotoChemOptions.from_yaml("tests/chapman_cycle.yaml")
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
    kinetics = kt.Kinetics(kt.KineticsOptions.from_yaml("tests/chapman_cycle.yaml"))
    photo_opts = kt.PhotoChemOptions.from_yaml("tests/chapman_cycle.yaml")
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
    kzz = torch.full((ncol, nlyr - 1), 1.0e5, dtype=torch.float64)
    kyy = torch.full((ncol - 1, nlyr), 2.0e5, dtype=torch.float64)

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
