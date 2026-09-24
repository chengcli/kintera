"""Regression: VU->T and PV->T for `use-h2-dissociation` converge on a strongly dissociating grid.

From the cold constant-cv guess, plain Newton on the S-shaped U(T) jumped from the flat hot tail
across the latent peak to T < 0 (NaN) in 503 of these 2400 cells and ended far from the root in
117 more; the damped PV->T step left up to 0.36% error after the default 10 iterations. Both
solves are now a bracketed (rtsafe-style) Newton, PV->T with the exact f' from the Mayer relation.
"""
import warnings

import numpy as np
import pytest
import torch
import kintera
from kintera import ThermoOptions, ThermoY

torch.set_default_dtype(torch.float64)
NH, NHE = 1.6667, 0.16667


def thermo(tmp_path, extra):
    card = tmp_path / "h2diss.yaml"
    card.write_text("reference-state: {Tref: 300.0, Pref: 1.0e5, use-h2-dissociation: true%s}\n"
                    "species:\n- {name: H2, composition: {H: %r, He: %r}, cv_R: 2.5}\n"
                    % (extra, NH, NHE))
    return ThermoOptions.from_yaml(str(card))


@pytest.mark.parametrize("extra", ["", ", fused-h2diss: true"])
def test_inversions_converge_on_dissociating_grid(tmp_path, extra):
    op = thermo(tmp_path, extra)  # default max-iter
    TT, CC = np.meshgrid(np.linspace(1100., 5500., 60), np.logspace(-2, 4, 40), indexing="ij")
    T = torch.tensor(TT.ravel())
    rho = torch.tensor(CC.ravel()) * kintera.species_weights()[0]
    th = ThermoY(op)
    V = th.compute("DY->V", (rho, torch.zeros(0, T.numel())))
    U, P = th.compute("VT->U", (V, T)), th.compute("VT->P", (V, T))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        T_vu = ThermoY(op).compute("VU->T", (V, U))  # fresh objects: no warm-start seeds
        T_pv = ThermoY(op).compute("PV->T", (P, V))
    assert not [x for x in w if "max" in str(x.message)], [str(x.message) for x in w]
    for name, Ts in [("VU->T", T_vu), ("PV->T", T_pv)]:
        err = ((Ts - T).abs() / T).numpy()
        assert np.isfinite(err).all(), "%s: %d NaN" % (name, (~np.isfinite(err)).sum())
        assert err.max() < 1e-9, "%s: %d cells off, max rel err %g" % (name, (err > 1e-9).sum(), err.max())


def test_full_dissociation_does_not_overflow(tmp_path):
    # Kc^2 overflowed above ~44 kK and the root returned [H] = 0, i.e. cz fell back to 1
    op = thermo(tmp_path, "")
    th = ThermoY(op)
    T = torch.tensor([3.0e4, 4.5e4, 6.0e4])
    V = th.compute("DY->V", (torch.full((3,), kintera.species_weights()[0]), torch.zeros(0, 3)))
    cz = th.compute("VT->P", (V, T)) / (8.31446 * T)
    np.testing.assert_allclose(cz.numpy(), NH + NHE, rtol=1e-6)
