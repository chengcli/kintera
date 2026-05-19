"""Tests for the L1 utilities extracted in REFACTOR_SCHEMA.html phase 1.

These cover bit-identical behavior with the pre-refactor diagnostic
implementations:

- ``kintera.atm2d.conservation`` (Phase 1a)
- ``kintera.atm2d.sources.charge_balance`` (Phase 1b)
- ``kintera.atm2d.schedule`` (Phase 1c)
"""
from __future__ import annotations

import math

import torch

from kintera.atm2d.conservation import count_atoms, project_atomic_budget
from kintera.atm2d.sources.charge_balance import fold_charge_balance_into_jacobian
from kintera.atm2d.schedule import StageScheduleConfig, stage_schedule


torch.set_default_dtype(torch.float64)


# ---------- count_atoms ----------

def test_count_atoms_basic():
    assert count_atoms("CH4", "C") == 1
    assert count_atoms("CH4", "H") == 4
    assert count_atoms("CH4", "N") == 0
    assert count_atoms("C2H6", "C") == 2
    assert count_atoms("C2H6", "H") == 6


def test_count_atoms_charged_state_labels():
    # Charge markers stripped.
    assert count_atoms("NH4+", "N") == 1
    assert count_atoms("NH4+", "H") == 4
    # Parenthesized state labels ((1)CH2, (3)CH2, N(2D)) ignored.
    assert count_atoms("(1)CH2", "C") == 1
    assert count_atoms("(1)CH2", "H") == 2
    assert count_atoms("(3)CH2", "C") == 1
    assert count_atoms("N(2D)", "N") == 1


def test_count_atoms_two_letter_element_disambiguation():
    # "Cl" should not match "C".
    assert count_atoms("ClO", "C") == 0
    assert count_atoms("ClO", "O") == 1
    # "Br" should not match "B".
    assert count_atoms("BrO", "B") == 0


def test_count_atoms_single_digit_subscript_only():
    # Preserves bit-identical behavior with the pre-refactor diagnostic:
    # C4H10's "10" subscript is truncated to "1". Multi-digit fix is a
    # separate physics PR.
    assert count_atoms("C4H10", "C") == 4
    assert count_atoms("C4H10", "H") == 1


# ---------- project_atomic_budget ----------

def test_project_atomic_budget_passthrough_under_budget():
    # 3 species: A (1 H), B (1 H), C (no H). Total H atoms = (a+b) below budget → no change.
    c = torch.tensor([[[1.0, 2.0, 5.0]]])
    atom_counts = {"H": torch.tensor([1.0, 1.0, 0.0])}
    fixed_mask = torch.tensor([False, False, False])
    budget = {"H": torch.tensor([[[10.0]]])}
    out = project_atomic_budget(c, atom_counts=atom_counts, fixed_mask=fixed_mask, budget=budget)
    assert torch.equal(out, c)


def test_project_atomic_budget_trims_hogs_only():
    # Two species both with 1 H atom; one is a "hog" (>mean), one is small.
    # Budget 3, current 10+1=11, excess 8. Hog gets trimmed; small stays.
    c = torch.tensor([[[10.0, 1.0]]])
    atom_counts = {"H": torch.tensor([1.0, 1.0])}
    fixed_mask = torch.tensor([False, False])
    budget = {"H": torch.tensor([[[3.0]]])}
    out = project_atomic_budget(c, atom_counts=atom_counts, fixed_mask=fixed_mask, budget=budget)
    # Hog is index 0 (10 > mean 5.5). Excess 8 subtracted from hog.
    assert out[0, 0, 0] == 2.0
    assert out[0, 0, 1] == 1.0


def test_project_atomic_budget_skips_fixed():
    # 3 species. A (1 H, fixed) is always above-mean but must never be touched.
    # B (1 H, variable, large) is the hog; C (1 H, variable, small) stays.
    # Sum of variable H = 100 + 1 = 101; budget 11; excess 90 lands on B.
    c = torch.tensor([[[999.0, 100.0, 1.0]]])
    atom_counts = {"H": torch.tensor([1.0, 1.0, 1.0])}
    fixed_mask = torch.tensor([True, False, False])
    budget = {"H": torch.tensor([[[11.0]]])}
    out = project_atomic_budget(c, atom_counts=atom_counts, fixed_mask=fixed_mask, budget=budget)
    # A (fixed) untouched.
    assert out[0, 0, 0] == 999.0
    # B (hog) trimmed to budget.
    assert out[0, 0, 1].item() == 10.0
    # C (small, below mean) untouched.
    assert out[0, 0, 2].item() == 1.0


# ---------- fold_charge_balance_into_jacobian ----------

def test_charge_balance_fold_adds_e_column_to_cations():
    # 4 species: 0=A, 1=B+, 2=C+, 3=E. Cations are [1, 2], E is at index 3.
    # Construct a Jacobian where J[:, :, :, 3] (the E column) is known.
    J = torch.zeros((1, 1, 4, 4))
    J[0, 0, 0, 3] = 7.0   # dF_A / dc_E = 7
    J[0, 0, 1, 3] = 11.0
    J[0, 0, 2, 3] = 13.0
    J[0, 0, 3, 3] = 17.0
    # Pre-existing cation columns must remain visible after the fold.
    J[0, 0, 0, 1] = 2.0
    J[0, 0, 0, 2] = 3.0
    out = fold_charge_balance_into_jacobian(J, cation_indices=[1, 2], e_index=3)
    # Original Jacobian not mutated.
    assert J[0, 0, 0, 1] == 2.0
    # After fold: cation columns 1 and 2 each got the E column added.
    assert out[0, 0, 0, 1] == 2.0 + 7.0
    assert out[0, 0, 0, 2] == 3.0 + 7.0
    assert out[0, 0, 1, 1] == 0.0 + 11.0
    assert out[0, 0, 2, 2] == 0.0 + 13.0
    # E column itself untouched.
    assert torch.equal(out[..., 3], J[..., 3])


def test_charge_balance_fold_empty_cation_list_is_noop():
    J = torch.randn((1, 1, 3, 3))
    out = fold_charge_balance_into_jacobian(J, cation_indices=[], e_index=2)
    assert out is J


# ---------- stage_schedule ----------

def test_stage_schedule_geometric_growth():
    cfg = StageScheduleConfig(start_dt=1e-15, growth_factor=math.sqrt(10.0), max_dt=1e+9)
    seq = stage_schedule(5, cfg)
    assert len(seq) == 5
    assert seq[0] == 1e-15
    # Each step grows by √10 until cap.
    for prev, curr in zip(seq[:-1], seq[1:]):
        assert math.isclose(curr / prev, math.sqrt(10.0), rel_tol=1e-10)


def test_stage_schedule_caps_at_max_dt():
    cfg = StageScheduleConfig(start_dt=1.0, growth_factor=10.0, max_dt=1000.0)
    seq = stage_schedule(6, cfg)
    # 1, 10, 100, 1000, then capped.
    assert seq == [1.0, 10.0, 100.0, 1000.0, 1000.0, 1000.0]


def test_stage_schedule_matches_kb_titan_defaults():
    """End-to-end: kinetics_base_titan_dt_schedule should match a direct
    stage_schedule call with NCYCLE=2 KB defaults."""
    import kintera as kt
    kb_seq = kt.kinetics_base_titan_dt_schedule(20, max_dt=1e+9)
    cfg = StageScheduleConfig(
        start_dt=1e-15,
        growth_factor=10.0 ** 0.5,
        max_dt=1e+9,
    )
    expected = stage_schedule(20, cfg)
    assert kb_seq == expected


def test_stage_schedule_rejects_nonpositive_ntime():
    cfg = StageScheduleConfig(start_dt=1.0, growth_factor=2.0, max_dt=10.0)
    try:
        stage_schedule(0, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for ntime=0")
