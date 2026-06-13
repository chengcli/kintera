"""
Unit tests for the core multi-range Arrhenius rate option.

Covers OpenSpec change `unify-titan-chem-onto-core`, capability
`core-multirange-arrhenius`:

- Task 1.3: single-range parity (a one-range reaction reproduces the legacy
  single-range Arrhenius result exactly).
- Task 1.4: KB ZK1 (B>0) / ZK2 (B<0) parity across temperature ranges to 1e-3
  relative.

The core rate within a range is ``A * (T / Tref)**b * exp(-Ea_R / T)``. KB's
ZK1 (B>0) form ``A*(T/T0)**B*exp(C/T)`` and ZK2 (B<0) form
``A*(T0/T)**|B|*exp(C/T)`` are the same expression once the sign of ``B`` is
carried in ``b`` and ``C`` maps to ``Ea_R = -C``.
"""

import numpy as np
import pytest
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)


def _arrhenius(op):
    return kt.Arrhenius(op)


def _eval(module, temp):
    """Evaluate an Arrhenius module on a (ncol, nlyr) temperature grid.

    Returns the rate tensor of shape (ncol, nlyr, nreaction).
    """
    pres = torch.ones_like(temp)
    # concentration is unused by the rate-constant evaluator; shape only needs a
    # trailing species axis.
    conc = torch.zeros(temp.shape + (1,), dtype=temp.dtype)
    return module.forward(temp, pres, conc, {})


def _single_range_options(A, b, Ea_R, Tref=300.0):
    op = kt.ArrheniusOptions()
    op.Tref(Tref)
    op.A(list(A))
    op.b(list(b))
    op.Ea_R(list(Ea_R))
    return op


def _multi_range_options(A_ranges, b_ranges, Ea_R_ranges, T_ranges, Tref=300.0):
    op = kt.ArrheniusOptions()
    op.Tref(Tref)
    op.A_ranges([list(x) for x in A_ranges])
    op.b_ranges([list(x) for x in b_ranges])
    op.Ea_R_ranges([list(x) for x in Ea_R_ranges])
    op.T_ranges([list(x) for x in T_ranges])
    return op


TEMPS = torch.tensor([[80.0, 150.0, 296.0], [300.0, 600.0, 1200.0]])


# ---------------------------------------------------------------------------
# Task 1.3 — single-range parity
# ---------------------------------------------------------------------------


def test_single_range_matches_analytic():
    """Legacy single-range option reproduces A*(T/Tref)^b*exp(-Ea_R/T)."""
    A = [2.5e-11, 1.0, 7.0e-12]
    b = [0.0, 0.5, -1.3]
    Ea_R = [0.0, 1200.0, -250.0]
    Tref = 298.0

    op = _single_range_options(A, b, Ea_R, Tref=Tref)
    rate = _eval(_arrhenius(op), TEMPS)  # (2, 3, 3)

    A_t = torch.tensor(A)
    b_t = torch.tensor(b)
    Ea_t = torch.tensor(Ea_R)
    temp = TEMPS.unsqueeze(-1)  # (2, 3, 1)
    expected = A_t * (temp / Tref).pow(b_t) * torch.exp(-Ea_t / temp)

    assert rate.shape == expected.shape
    assert torch.allclose(rate, expected, rtol=1e-12, atol=0.0)


def test_one_range_equals_legacy_single_range():
    """A multi-range option with a single range per reaction is identical to
    the legacy single-range option (Scenario: single-range reaction unchanged).
    """
    A = [2.5e-11, 1.0, 7.0e-12]
    b = [0.0, 0.5, -1.3]
    Ea_R = [0.0, 1200.0, -250.0]
    Tref = 298.0

    legacy = _single_range_options(A, b, Ea_R, Tref=Tref)
    # one range each; the single upper bound is ignored (last range -> +inf)
    multi = _multi_range_options(
        A_ranges=[[a] for a in A],
        b_ranges=[[x] for x in b],
        Ea_R_ranges=[[e] for e in Ea_R],
        T_ranges=[[1.0e30] for _ in A],
        Tref=Tref,
    )

    legacy_rate = _eval(_arrhenius(legacy), TEMPS)
    multi_rate = _eval(_arrhenius(multi), TEMPS)

    assert legacy_rate.shape == multi_rate.shape
    # exact: the single-range fast path performs the same ops as the legacy path
    assert torch.equal(legacy_rate, multi_rate)


# ---------------------------------------------------------------------------
# Task 1.4 — KB ZK1/ZK2 parity across ranges
# ---------------------------------------------------------------------------


def _kb_zk(T, A, B, C, T0):
    """KB ZKT value. ZK1 (B>0) and ZK2 (B<0) collapse to the same expression."""
    return A * (T / T0) ** B * np.exp(C / T)


def _kb_reference(temp_np, A, B, C, bounds, T0):
    """Reference KB rate selecting the active range by temperature.

    `bounds` are the per-range upper bounds; range r is active on
    [bounds[r-1], bounds[r]) with the first extending to 0 and the last to inf.
    """
    out = np.empty_like(temp_np)
    lowers = [0.0] + list(bounds[:-1])
    uppers = list(bounds[:-1]) + [np.inf]
    for idx, T in np.ndenumerate(temp_np):
        r = next(i for i in range(len(A)) if lowers[i] <= T < uppers[i])
        out[idx] = _kb_zk(T, A[r], B[r], C[r], T0)
    return out


def test_kb_zk_three_ranges_mixed_signs():
    """Three temperature ranges exercising ZK1 (B>0) and ZK2 (B<0) match KB."""
    T0 = 300.0
    # range params: r0 ZK1 (B>0), r1 ZK2 (B<0), r2 ZK1 (B>0)
    A = [1.2e-10, 3.4e-11, 9.0e-12]
    B = [0.75, -1.40, 0.30]
    C = [-180.0, 250.0, -50.0]  # KB exp(C/T)
    # upper bounds: r0 -> [0,200), r1 -> [200,500), r2 -> [500, inf)
    bounds = [200.0, 500.0, 1.0e30]

    # core uses exp(-Ea_R/T), so Ea_R = -C
    op = _multi_range_options(
        A_ranges=[A],
        b_ranges=[B],
        Ea_R_ranges=[[-c for c in C]],
        T_ranges=[bounds],
        Tref=T0,
    )

    temp = torch.tensor([[120.0, 199.9, 200.0], [350.0, 499.9, 800.0]])
    rate = _eval(_arrhenius(op), temp)[..., 0].numpy()

    expected = _kb_reference(temp.numpy(), A, B, C, bounds, T0)

    rel = np.abs(rate - expected) / np.abs(expected)
    assert rel.max() < 1e-3, f"max rel diff {rel.max():.2e}\n{rate}\n{expected}"


def test_kb_zk1_and_zk2_single_form():
    """A pure ZK1 (B>0) and a pure ZK2 (B<0) reaction each match KB across T."""
    T0 = 300.0
    # reaction 0: ZK1, reaction 1: ZK2
    A = [[2.0e-11], [5.0e-12]]
    B = [[1.10], [-0.85]]
    C = [[-300.0], [120.0]]

    op = _multi_range_options(
        A_ranges=A,
        b_ranges=B,
        Ea_R_ranges=[[-c for c in row] for row in C],
        T_ranges=[[1.0e30], [1.0e30]],
        Tref=T0,
    )

    temp = torch.tensor([[90.0, 175.0, 300.0], [450.0, 700.0, 1100.0]])
    rate = _eval(_arrhenius(op), temp).numpy()  # (2, 3, 2)

    t = temp.numpy()
    exp0 = _kb_zk(t, A[0][0], B[0][0], C[0][0], T0)  # ZK1
    exp1 = _kb_zk(t, A[1][0], B[1][0], C[1][0], T0)  # ZK2

    rel0 = np.abs(rate[..., 0] - exp0) / np.abs(exp0)
    rel1 = np.abs(rate[..., 1] - exp1) / np.abs(exp1)
    assert rel0.max() < 1e-3
    assert rel1.max() < 1e-3


def test_multi_range_rates_are_finite():
    """Scenario: building multi-range reactions yields finite rates."""
    op = _multi_range_options(
        A_ranges=[[1.0e-10, 2.0e-11], [3.0e-12, 4.0e-12, 5.0e-12]],
        b_ranges=[[0.5, -0.5], [0.0, 1.0, -1.0]],
        Ea_R_ranges=[[100.0, -100.0], [0.0, 200.0, -200.0]],
        T_ranges=[[400.0, 1.0e30], [150.0, 600.0, 1.0e30]],
    )
    rate = _eval(_arrhenius(op), TEMPS)
    assert rate.shape == TEMPS.shape + (2,)
    assert torch.isfinite(rate).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
