import pytest
import torch

import kintera as kt


torch.set_default_dtype(torch.float64)


def _make_state(nz: int = 5, ns: int = 2) -> kt.ColumnState1D:
    z = torch.linspace(0.0, 4.0e5, nz, dtype=torch.float64)
    temp = torch.full((nz,), 250.0, dtype=torch.float64)
    pres = torch.logspace(5.0, 3.0, nz, dtype=torch.float64)
    conc = torch.full((nz, ns), 0.5e-6, dtype=torch.float64)
    return kt.ColumnState1D(z=z, temperature=temp, pressure=pres, concentration=conc)


def test_eddy_transport_matches_constant_density_diffusion():
    state = _make_state()
    conc = torch.tensor(
        [
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.35, 0.65],
            [0.25, 0.75],
        ],
        dtype=torch.float64,
    )
    state.concentration = conc
    kzz = torch.full((state.nz - 1,), 1.0e5, dtype=torch.float64)

    matrix = kt.build_eddy_diffusion_blocks(state, kzz)
    tend_matrix = matrix.matvec(conc)
    tend_ref = kt.diffusion_tendency(conc, kzz, state.z[1:] - state.z[:-1])
    torch.testing.assert_close(tend_matrix, tend_ref, atol=1e-12, rtol=1e-12)


def test_binary_diffusion_creates_species_coupling():
    state = _make_state(nz=4, ns=3)
    binary = torch.zeros((state.nz - 1, state.nspecies, state.nspecies), dtype=state.dtype)
    binary[:, 0, 0] = 1.0e4
    binary[:, 1, 1] = 1.5e4
    binary[:, 2, 2] = 0.8e4
    binary[:, 0, 1] = 0.3e4
    binary[:, 1, 0] = 0.2e4
    weights = torch.tensor([28.0, 32.0, 16.0], dtype=state.dtype)

    matrix = kt.build_binary_diffusion_blocks(state, binary, weights, include_gravity=False)
    dense_upper = matrix.upper_packed[0]
    assert dense_upper[0, 1].abs().item() > 0.0
    assert matrix.upper_blocks[0]._nnz() > state.nspecies


def test_block_solver_matches_dense_solution():
    nz, ns = 4, 3
    lower = torch.zeros((nz, ns, ns), dtype=torch.float64)
    diag = torch.zeros((nz, ns, ns), dtype=torch.float64)
    upper = torch.zeros((nz, ns, ns), dtype=torch.float64)
    eye = torch.eye(ns, dtype=torch.float64)

    for i in range(nz):
        diag[i] = (4.0 + i) * eye
        if i > 0:
            lower[i] = -0.2 * eye
        if i < nz - 1:
            upper[i] = -0.1 * eye

    matrix = kt.BlockTridiagonalMatrix.from_dense(lower, diag, upper)
    rhs = torch.arange(1, nz * ns + 1, dtype=torch.float64).reshape(nz, ns)

    dense = torch.zeros((nz * ns, nz * ns), dtype=torch.float64)
    for i in range(nz):
        dense[i * ns : (i + 1) * ns, i * ns : (i + 1) * ns] = diag[i]
        if i > 0:
            dense[i * ns : (i + 1) * ns, (i - 1) * ns : i * ns] = lower[i]
        if i < nz - 1:
            dense[i * ns : (i + 1) * ns, (i + 1) * ns : (i + 2) * ns] = upper[i]

    sol = kt.solve_block_tridiagonal(matrix, rhs)
    ref = torch.linalg.solve(dense, rhs.reshape(-1)).reshape(nz, ns)
    torch.testing.assert_close(sol, ref, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_solver_matches_cpu():
    nz, ns = 3, 2
    eye = torch.eye(ns, dtype=torch.float64)
    lower = torch.zeros((nz, ns, ns), dtype=torch.float64)
    diag = torch.stack([3.0 * eye, 3.5 * eye, 4.0 * eye], dim=0)
    upper = torch.zeros((nz, ns, ns), dtype=torch.float64)
    lower[1:] = -0.1 * eye
    upper[:-1] = -0.15 * eye
    rhs = torch.randn((nz, ns), dtype=torch.float64)

    cpu_matrix = kt.BlockTridiagonalMatrix.from_dense(lower, diag, upper)
    cpu_sol = kt.solve_block_tridiagonal_cpu(cpu_matrix, rhs)

    gpu_matrix = kt.BlockTridiagonalMatrix.from_dense(
        lower.cuda(), diag.cuda(), upper.cuda()
    )
    gpu_sol = kt.solve_block_tridiagonal_cuda(gpu_matrix, rhs.cuda()).cpu()
    torch.testing.assert_close(cpu_sol, gpu_sol, atol=1e-12, rtol=1e-12)


def test_actinic_flux_from_disort_attenuates_downward():
    photo_opts = kt.PhotoChemOptions.from_yaml("tests/chapman_cycle.yaml")
    photo = kt.PhotoChem(photo_opts)
    species = photo_opts.species()
    idx = {name: i for i, name in enumerate(species)}

    nz = 4
    z = torch.linspace(0.0, 3.0e5, nz, dtype=torch.float64)
    temp = torch.full((nz,), 250.0, dtype=torch.float64)
    pres = torch.logspace(5.0, 3.0, nz, dtype=torch.float64)
    conc = torch.zeros((nz, len(species)), dtype=torch.float64)
    conc[:, idx["N2"]] = 0.79
    conc[:, idx["O2"]] = torch.tensor([0.21, 0.24, 0.27, 0.30], dtype=torch.float64)
    conc[:, idx["O"]] = 1.0e-12
    conc[:, idx["O3"]] = torch.tensor([1.0e-10, 3.0e-10, 1.0e-9, 3.0e-9], dtype=torch.float64)
    state = kt.ColumnState1D(z=z, temperature=temp, pressure=pres, concentration=conc)

    wave = photo.module("photolysis").buffer("wavelength")
    top_flux = torch.full((wave.numel(),), 1.0e12, dtype=torch.float64)
    rt = kt.compute_actinic_flux_disort(photo, state, top_flux, concentration_unit="molecules_cm3")

    assert rt.optical_depth.shape == (nz, wave.numel())
    assert rt.actinic_flux.shape == (wave.numel(), nz)
    assert torch.all(rt.optical_depth >= 0.0)

    absorb_idx = int(torch.argmin(torch.abs(wave - 200.0)).item())
    assert rt.actinic_flux[absorb_idx, 0].item() > rt.actinic_flux[absorb_idx, -1].item()


def test_implicit_operator_adds_chemistry_and_photochemistry():
    kinetics = kt.Kinetics(kt.KineticsOptions.from_yaml("tests/chapman_cycle.yaml"))
    photo_opts = kt.PhotoChemOptions.from_yaml("tests/chapman_cycle.yaml")
    photo = kt.PhotoChem(photo_opts)
    species = photo_opts.species()
    kt.set_species_names(species)

    nz = 3
    z = torch.linspace(0.0, 2.0e5, nz, dtype=torch.float64)
    temp = torch.full((nz,), 250.0, dtype=torch.float64)
    pres = torch.full((nz,), 1.0e4, dtype=torch.float64)
    conc = torch.full((nz, len(species)), 1.0e-6, dtype=torch.float64)
    state = kt.ColumnState1D(z=z, temperature=temp, pressure=pres, concentration=conc)
    kzz = torch.full((nz - 1,), 1.0e5, dtype=torch.float64)

    wave = photo.module("photolysis").buffer("wavelength")
    actinic_flux = torch.ones((wave.numel(), nz), dtype=torch.float64)

    transport = kt.build_transport_matrix(state, kzz)
    implicit = kt.build_implicit_operator(
        state,
        kzz,
        kinetics=kinetics,
        photo_chem=photo,
        actinic_flux=actinic_flux,
    )

    assert implicit.diag_packed.shape == transport.diag_packed.shape
    assert torch.isfinite(implicit.diag_packed).all()
    assert not torch.allclose(implicit.diag_packed, transport.diag_packed)
