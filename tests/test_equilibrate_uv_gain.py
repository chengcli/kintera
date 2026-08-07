"""equilibrate_uv must export a gain matrix that is supported only on the ACTIVE reactions.

mmdot() writes gain as an (nactive x nactive) row-major block, but the scatter that restores
reaction order used to read it with leading dimension `nreaction` and to loop the column index
to `nreaction`. Both are wrong whenever nactive < nreaction: real entries land in slots that
belong to reactions which are in equilibrium and contribute nothing. The same defect was fixed
in equilibrate_tp by #105; this is the equilibrate_uv half.

`nactive` is also an in/out parameter that ThermoY warm-starts across calls, so a solve that
converges on its first pass (the common case: nothing is supersaturated) used to scatter with
the PREVIOUS call's nactive over a freshly-allocated, uninitialized gain buffer.
"""
import torch
from kintera import ThermoY, ThermoOptions

torch.set_default_dtype(torch.float64)

# Three condensible vapours. The inline Antoine form is
#   logsvp = log(1e5) + (A - B/(T+C)) * log(10)   =>   svp = 1e5 * 10^A  for B = 0,
# so `A` alone sets the saturation concentration and the state below makes reactions 0 and 1
# supersaturated (svp = 100 Pa) while reaction 2 cannot condense (svp = 1e15 Pa): nactive = 2,
# nreaction = 3.
YAML = """
reference-state: {Tref: 0., Pref: 1.e5}
species:
- {name: dry, composition: {N: 1.56, O: 0.42}, cv_R: 2.5}
- {name: A, composition: {H: 2, O: 1}, cv_R: 1.5, u0_R: 0.}
- {name: B, composition: {H: 3, N: 1}, cv_R: 1.5, u0_R: 0.}
- {name: C, composition: {H: 4, C: 1}, cv_R: 1.5, u0_R: 0.}
- {name: A(l), composition: {H: 2, O: 1}, cv_R: 7.5, u0_R: -6786.66}
- {name: B(l), composition: {H: 3, N: 1}, cv_R: 7.5, u0_R: -5000.00}
- {name: C(l), composition: {H: 4, C: 1}, cv_R: 7.5, u0_R: -4000.00}
reactions:
- {equation: 'A <=> A(l)', type: nucleation, rate-constant: {formula: antoine, A: -3.0, B: 0., C: 1.}}
- {equation: 'B <=> B(l)', type: nucleation, rate-constant: {formula: antoine, A: -3.0, B: 0., C: 1.}}
- {equation: 'C <=> C(l)', type: nucleation, rate-constant: {formula: antoine, A: 10.0, B: 0., C: 1.}}
"""

INACTIVE = 2  # reaction index that can never condense


def _state(th, tmp_path):
    rho = torch.tensor([1.0])
    yfrac = torch.zeros(6, 1)
    yfrac[0:3] = 0.02  # vapour A, B, C — supersaturated w.r.t. svp = 100 Pa
    V = th.compute("DY->V", (rho, yfrac))
    U = th.compute("VT->U", (V, torch.tensor([300.0])))
    return rho, U, yfrac


def _th(tmp_path):
    p = tmp_path / "three_vapour.yaml"
    p.write_text(YAML)
    return ThermoY(ThermoOptions.from_yaml(str(p)))


def test_gain_is_zero_outside_the_active_set(tmp_path):
    th = _th(tmp_path)
    rho, U, yfrac = _state(th, tmp_path)
    gain = th.forward(rho, U, yfrac, False)[0]
    assert gain.shape == (3, 3)
    assert float(gain[INACTIVE, :].abs().max()) == 0.0, (
        "reaction %d is in equilibrium but owns a gain ROW: %s" % (INACTIVE, gain))
    assert float(gain[:, INACTIVE].abs().max()) == 0.0, (
        "reaction %d is in equilibrium but owns a gain COLUMN: %s" % (INACTIVE, gain))


def test_gain_is_zero_when_nothing_is_active(tmp_path):
    th = _th(tmp_path)
    rho, U, yfrac = _state(th, tmp_path)
    th.forward(rho, U, yfrac, False)          # equilibrates in place; leaves nactive = 2
    gain = th.forward(rho, U, yfrac, True)[0]  # warm start: every reaction now in equilibrium
    assert float(gain.abs().max()) == 0.0, (
        "converged solve exported a non-empty gain: %s" % gain)
