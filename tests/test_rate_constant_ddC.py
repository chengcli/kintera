"""Per-reaction correctness of ``rc_ddC``, the rate-constant derivative
with respect to concentration.

``KineticsImpl::forward`` expands the concentration tensor to
``(..., nspecies, nreaction)`` and differentiates the *summed* rate in a
single backward pass (``rate.backward(ones_like(rate))``). That only
recovers ``d(k_r)/dC_s`` separately for each reaction if the evaluator
carries the trailing reaction axis all the way through. Every
concentration-dependent evaluator (three-body and the four falloff
forms) used to start with ``C.select(last_dim, 0)``, dropping the axis --
so the whole ``sum_r d(k_r)/dC_s`` collapsed onto the block's *first*
reaction column and every other reaction in the block got exactly zero.

The rates themselves were unaffected, which is why this hid: it is
purely a Jacobian defect. Its effect is worst on reverse reactions,
where ``kinetics.cpp`` forms ``rc_ddC_rev = rc_ddC_fwd / Kc``; a
misplaced ``rc_ddC`` divided by a tiny ``Kc`` produced Jacobian entries
around 1e97 and made the backward-Euler matrix singular.

Each test compares autograd's ``rc_ddC`` against a central finite
difference of the evaluator on the *unexpanded* concentration path, and
uses two reactions with very different ``k0`` so a collapse onto column
0 cannot pass by coincidence.
"""

import pytest
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

SPECIES = ["A", "B", "C"]
TEMP = torch.tensor([[300.0]])
PRES = torch.tensor([[1.0e5]])
CONC = torch.tensor([[[1.0, 2.0, 3.0]]])
NRXN = 2


@pytest.fixture(autouse=True)
def _register_species():
    kt.set_species_names(SPECIES)
    kt.set_species_weights([1.0e-3, 2.0e-3, 3.0e-3])


def _rc_ddC(module):
    """Reproduce the expanded-concentration autograd path of KineticsImpl."""
    conc1 = (
        CONC.unsqueeze(-1)
        .expand(list(CONC.shape) + [NRXN])
        .clone()
        .requires_grad_(True)
    )
    rate = module.forward(TEMP, PRES, conc1, {})
    rate.backward(torch.ones_like(rate))
    return conc1.grad[0, 0], rate.detach()


def _rc_ddC_findiff(module, h=1.0e-6):
    fd = torch.zeros(len(SPECIES), NRXN)
    for s in range(len(SPECIES)):
        cp, cm = CONC.clone(), CONC.clone()
        cp[0, 0, s] += h
        cm[0, 0, s] -= h
        fd[s] = (
            module.forward(TEMP, PRES, cp, {})[0, 0]
            - module.forward(TEMP, PRES, cm, {})[0, 0]
        ) / (2.0 * h)
    return fd


def _check(module):
    grad, rate = _rc_ddC(module)
    fd = _rc_ddC_findiff(module)
    torch.testing.assert_close(grad, fd, rtol=1e-6, atol=0.0)
    # The collapse signature: everything in column 0, zeros elsewhere.
    assert grad[:, 1].abs().max() > 0.0
    return rate


def _falloff_common(op, equations):
    op.Tref(300.0)
    op.reactions([kt.Reaction(e) for e in equations])
    # deliberately dissimilar so a summed-and-misplaced gradient is obvious
    op.k0_A([1.0e-3, 5.0e2])
    op.k0_b([0.0, 0.0])
    op.k0_Ea_R([0.0, 0.0])
    op.kinf_A([1.0e2, 1.0e2])
    op.kinf_b([0.0, 0.0])
    op.kinf_Ea_R([0.0, 0.0])
    return op


FALLOFF_EQNS = ["A + A (+M) => B (+M)", "B + A (+M) => C (+M)"]


def test_three_body_rc_ddC_is_per_reaction():
    op = kt.ThreeBodyOptions()
    op.Tref(300.0)
    op.reactions([kt.Reaction("A + A + M => B + M"), kt.Reaction("B + A + M => C + M")])
    op.k0_A([1.0e-3, 5.0e2])
    op.k0_b([0.0, 0.0])
    op.k0_Ea_R([0.0, 0.0])
    op.efficiencies([{}, {"A": 2.0}])

    module = kt.ThreeBody(op)
    rate = _check(module)

    # For k = k0 * [M]_eff the derivative is exactly k0 * efficiency[r, s],
    # so this one can be asserted in closed form as well.
    expected = torch.tensor([[1.0e-3, 1.0e3], [1.0e-3, 5.0e2], [1.0e-3, 5.0e2]])
    torch.testing.assert_close(_rc_ddC(module)[0], expected)
    # Rates are unaffected by the fix: M_eff = 6 and 2*1 + 2 + 3 = 7.
    torch.testing.assert_close(
        rate.flatten(), torch.tensor([1.0e-3 * 6.0, 5.0e2 * 7.0])
    )


def test_lindemann_falloff_rc_ddC_is_per_reaction():
    op = _falloff_common(kt.LindemannFalloffOptions(), FALLOFF_EQNS)
    op.efficiencies([{}, {"A": 2.0}])
    _check(kt.LindemannFalloff(op))


def test_troe_falloff_rc_ddC_is_per_reaction():
    op = _falloff_common(kt.TroeFalloffOptions(), FALLOFF_EQNS)
    op.efficiencies([{}, {"A": 2.0}])
    op.troe_A([0.5, 0.5])
    op.troe_T1([100.0, 100.0])
    op.troe_T2([1000.0, 1000.0])
    op.troe_T3([10.0, 10.0])
    _check(kt.TroeFalloff(op))


def test_sri_falloff_rc_ddC_is_per_reaction():
    op = _falloff_common(kt.SRIFalloffOptions(), FALLOFF_EQNS)
    op.efficiencies([{}, {"A": 2.0}])
    op.sri_A([1.0, 1.0])
    op.sri_B([10.0, 10.0])
    op.sri_C([100.0, 100.0])
    op.sri_D([1.0, 1.0])
    op.sri_E([0.0, 0.0])
    _check(kt.SRIFalloff(op))


def test_kb_falloff_rc_ddC_is_per_reaction():
    op = _falloff_common(kt.KBFalloffOptions(), FALLOFF_EQNS)
    op.fc(0.6)
    _check(kt.KBFalloff(op))
