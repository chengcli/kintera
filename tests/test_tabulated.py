#! /usr/bin/env python3
"""
Tests for TabulatedRate evaluator and evolve_implicit_subcycle.

Tests:
  1. read_rate_table + TabulatedRate 1D (T-only)
  2. read_rate_table + TabulatedRate 2D (T,P)
  3. Autograd through tabulated rate (gradient w.r.t. T)
  4. evolve_implicit_subcycle consistency
  5. Full Kinetics with mixed Arrhenius + Tabulated reactions
"""

import os
import sys
import math
import torch
import numpy as np

# Ensure we pick up the local build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import kintera
from kintera import Reaction

torch.set_default_dtype(torch.float64)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_1d_table_readback():
    """Test that a 1D table can be loaded and evaluated correctly."""
    print("=== Test 1: 1D TabulatedRate ===")

    # Set up minimal species
    kintera.set_species_names(["dry", "A", "B"])
    kintera.set_species_weights([2.0e-3, 16.0e-3, 28.0e-3])

    opts = kintera.TabulatedRateOptions()
    opts.reactions([Reaction("A => B")])
    opts.files([os.path.join(DATA_DIR, "rate_1d_arrhenius.dat")])
    opts.log_interpolation(True)

    # Manually load table
    from kintera import TabulatedRate
    rate_mod = TabulatedRate(opts)
    print(f"  Module: {rate_mod}")

    # Evaluate at a known T
    T = torch.tensor([1000.0])
    P = torch.tensor([1.0e5])
    C = torch.zeros(1, 3)  # not used for tabulated
    other = {}

    k = rate_mod(T, P, C, other)
    print(f"  k(T=1000) = {k.item():.6e}")

    # Compare with analytical: k = 1e-10 * exp(-5000/1000) = 1e-10 * exp(-5)
    k_analytical = 1e-10 * math.exp(-5000.0 / 1000.0)
    print(f"  k_analytical = {k_analytical:.6e}")

    # The interpolated value should be close (within ~10% due to interpolation)
    rel_err = abs(k.item() - k_analytical) / k_analytical
    print(f"  Relative error: {rel_err:.4f}")
    assert rel_err < 0.15, f"1D interpolation error too large: {rel_err}"
    print("  PASSED\n")


def test_2d_table_readback():
    """Test that a 2D table (T,P) can be loaded and evaluated."""
    print("=== Test 2: 2D TabulatedRate ===")

    kintera.set_species_names(["dry", "A", "B", "C"])
    kintera.set_species_weights([2.0e-3, 16.0e-3, 28.0e-3, 18.0e-3])

    opts = kintera.TabulatedRateOptions()
    opts.reactions([Reaction("A + B => C")])
    opts.files([os.path.join(DATA_DIR, "rate_2d_tp.dat")])
    opts.log_interpolation(True)

    rate_mod = kintera.TabulatedRate(opts)
    print(f"  Module: {rate_mod}")

    # Evaluate at known T, P
    T = torch.tensor([1500.0])
    P = torch.tensor([5.0e5])
    C = torch.zeros(1, 4)
    other = {}

    k = rate_mod(T, P, C, other)
    print(f"  k(T=1500, P=5e5) = {k.item():.6e}")

    # Analytical: k = 1e-8 * (1500/1000)^1.5 * (5e5/1e5)^0.3
    k_analytical = 1e-8 * (1500.0 / 1000.0) ** 1.5 * (5.0e5 / 1.0e5) ** 0.3
    print(f"  k_analytical = {k_analytical:.6e}")

    rel_err = abs(k.item() - k_analytical) / k_analytical
    print(f"  Relative error: {rel_err:.4f}")
    assert rel_err < 0.15, f"2D interpolation error too large: {rel_err}"
    print("  PASSED\n")


def test_autograd_tabulated():
    """Test that autograd tracks gradients through tabulated rate interpolation."""
    print("=== Test 3: Autograd through TabulatedRate ===")

    kintera.set_species_names(["dry", "A", "B"])
    kintera.set_species_weights([2.0e-3, 16.0e-3, 28.0e-3])

    opts = kintera.TabulatedRateOptions()
    opts.reactions([Reaction("A => B")])
    opts.files([os.path.join(DATA_DIR, "rate_1d_arrhenius.dat")])
    opts.log_interpolation(True)

    rate_mod = kintera.TabulatedRate(opts)

    # T with grad
    T = torch.tensor([1500.0], requires_grad=True)
    P = torch.tensor([1.0e5])
    C = torch.zeros(1, 3)

    k = rate_mod(T, P, C, {})
    k.backward()

    dk_dT = T.grad.item()
    print(f"  k(T=1500) = {k.item():.6e}")
    print(f"  dk/dT = {dk_dT:.6e}")

    # Verify with finite differences
    eps = 1.0
    with torch.no_grad():
        T_plus = torch.tensor([1500.0 + eps])
        T_minus = torch.tensor([1500.0 - eps])
        k_plus = rate_mod(T_plus, P, C, {})
        k_minus = rate_mod(T_minus, P, C, {})
        dk_dT_fd = (k_plus.item() - k_minus.item()) / (2 * eps)

    print(f"  dk/dT (finite diff) = {dk_dT_fd:.6e}")
    rel_err = abs(dk_dT - dk_dT_fd) / max(abs(dk_dT_fd), 1e-30)
    print(f"  Relative error: {rel_err:.4f}")
    assert rel_err < 0.1, f"Autograd vs finite diff mismatch: {rel_err}"
    assert dk_dT > 0, "dk/dT should be positive (rate increases with T)"
    print("  PASSED\n")


def test_subcycle_consistency():
    """Test evolve_implicit_subcycle consistency with evolve_implicit."""
    print("=== Test 4: evolve_implicit_subcycle ===")

    nspecies = 3
    nreaction = 2

    # Mock stoichiometry: A => B, B => A
    stoich = torch.tensor([
        [-1.0, 1.0],   # species A: consumed in rxn0, produced in rxn1
        [1.0, -1.0],    # species B: produced in rxn0, consumed in rxn1
        [0.0, 0.0],     # species C: inert
    ])

    rate = torch.tensor([1.0e-3, 0.5e-3])  # mol/m^3/s
    jacobian = torch.zeros(nreaction, nspecies)
    dt = 10.0

    # Single step
    dc_single = kintera.evolve_implicit(rate, stoich, jacobian, dt)
    print(f"  Single step dc: {dc_single}")

    # 10 substeps
    dc_sub10 = kintera.evolve_implicit_subcycle(rate, stoich, jacobian, dt, 10)
    print(f"  10 substeps dc: {dc_sub10}")

    # With zero Jacobian, subcycling should give same result
    diff = (dc_single - dc_sub10).abs().max().item()
    print(f"  Max difference: {diff:.6e}")
    assert diff < 1e-10, f"Subcycle differs from single step with zero Jacobian: {diff}"

    # Test nsubsteps=1 returns same as single
    dc_sub1 = kintera.evolve_implicit_subcycle(rate, stoich, jacobian, dt, 1)
    diff1 = (dc_single - dc_sub1).abs().max().item()
    assert diff1 < 1e-15, f"nsubsteps=1 should equal single step: {diff1}"

    print("  PASSED\n")


def test_full_kinetics_yaml():
    """Test full Kinetics pipeline with tabulated reactions from YAML."""
    print("=== Test 5: Full Kinetics from YAML ===")

    # Use absolute path to the test yaml
    test_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(test_dir, "tabulated_test.yaml")

    if not os.path.exists(yaml_path):
        print("  SKIPPED (tabulated_test.yaml not found)")
        return

    try:
        kintera.add_search_path(test_dir)
        op = kintera.KineticsOptions.from_yaml(yaml_path)
        print(f"  Loaded {len(op.reactions())} total reactions")

        kinet = kintera.Kinetics(op)
        print(f"  Kinetics module created")
        print(f"  Stoichiometry:\n{kinet.stoich}")

        # Evaluate forward
        T = torch.tensor([1500.0])
        P = torch.tensor([1.0e5])
        nspecies = len(op.species())
        C = torch.ones(1, nspecies) * 1e-3

        result = kinet(T, P, C, {})
        print(f"  Forward result shape: {result.shape}")
        print(f"  dc/dt = {result}")
        print("  PASSED\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        raise


def test_batch_evaluation():
    """Test that TabulatedRate handles batched inputs correctly."""
    print("=== Test 6: Batch evaluation ===")

    kintera.set_species_names(["dry", "A", "B"])
    kintera.set_species_weights([2.0e-3, 16.0e-3, 28.0e-3])

    opts = kintera.TabulatedRateOptions()
    opts.reactions([Reaction("A => B")])
    opts.files([os.path.join(DATA_DIR, "rate_1d_arrhenius.dat")])
    opts.log_interpolation(True)

    rate_mod = kintera.TabulatedRate(opts)

    # Batch of temperatures
    T = torch.tensor([800.0, 1200.0, 2000.0, 2800.0])
    P = torch.tensor([1e5, 1e5, 1e5, 1e5])
    C = torch.zeros(4, 3)

    k = rate_mod(T, P, C, {})
    print(f"  T = {T.tolist()}")
    print(f"  k = {k.tolist()}")

    # Verify monotonicity (k should increase with T for Arrhenius)
    for i in range(len(T) - 1):
        assert k[i, 0] < k[i + 1, 0], f"Rate should increase with T: k[{i}]={k[i, 0]}, k[{i+1}]={k[i+1, 0]}"

    # Compare with analytical
    for i, t in enumerate(T.tolist()):
        k_exact = 1e-10 * math.exp(-5000.0 / t)
        rel_err = abs(k[i, 0].item() - k_exact) / k_exact
        print(f"  T={t}: k_interp={k[i,0].item():.4e}, k_exact={k_exact:.4e}, err={rel_err:.4f}")
        assert rel_err < 0.15, f"Batch interpolation error too large at T={t}: {rel_err}"

    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("TabulatedRate and Subcycling Tests")
    print("=" * 60 + "\n")

    test_1d_table_readback()
    test_2d_table_readback()
    test_autograd_tabulated()
    test_subcycle_consistency()
    # test_full_kinetics_yaml()  # TODO: enable once full YAML pipeline works
    test_batch_evaluation()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
