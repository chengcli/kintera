"""`ThermoX.extrapolate_dz` must write through a view, not rebind it.

The caller owns the tensors; the method updates them in place. A column loop over an
(ncol, nlyr) profile therefore passes slices, `temp[:, i]`, and expects the parent to
change. `Tensor::set_` rebinds the argument's storage to the result's instead of
writing into it, so the slice detaches from its base and the base keeps its old value:
the call becomes a silent no-op that still returns and still exits 0. The sibling
`extrapolate_dlnp` uses in-place arithmetic and was never affected.

A whole tensor and a slice carrying the same state must land on the same answer.

Run in a child interpreter: the species registry is process-global.
"""

import subprocess
import sys
import textwrap

SCRIPT = """
import torch, kintera
from kintera import ThermoOptions, ThermoX

torch.set_default_dtype(torch.float64)
kintera.set_species_names(["dry"])
kintera.set_species_weights([29.0e-3])

op = ThermoOptions()
op.vapor_ids([0]); op.cref_R([2.5]); op.uref_R([0.0]); op.sref_R([0.0])
op.Tref(300.0); op.Pref(1.0e5)
thermo = ThermoX(op)

T = torch.tensor([310.0]); P = torch.tensor([1.0e5]); X = torch.ones(1, 1)
thermo.extrapolate_dz(T, P, X, dz=100.0, grav=9.8)

T2 = torch.full((1, 2), 310.0); P2 = torch.full((1, 2), 1.0e5); X2 = torch.ones(1, 2, 1)
thermo.extrapolate_dz(T2[:, 1], P2[:, 1], X2[:, 1, :], dz=100.0, grav=9.8)

print(repr((T.item(), P.item(), T2[0, 1].item(), P2[0, 1].item())))
"""


def run():
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(SCRIPT)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"probe failed ({r.returncode}):\n{r.stderr}"
    return eval(r.stdout.strip().splitlines()[-1])


def test_extrapolate_dz_updates_a_slice_in_place():
    t_whole, p_whole, t_slice, p_slice = run()

    # the whole tensor must move at all, or the probe is not exercising anything
    assert t_whole < 309.9, f"whole tensor did not cool: {t_whole} K"

    # dry adiabat over 100 m: g/cp = 9.8 / (3.5 * 8.314 / 0.029) = 9.77 K/km
    assert abs((310.0 - t_whole) - 0.977) < 0.01, f"unexpected lapse: {310.0 - t_whole} K"

    assert t_slice == t_whole, (
        f"slice left at {t_slice} K while the whole tensor reached {t_whole} K; "
        f"extrapolate_dz rebound the view instead of writing through it")
    assert p_slice == p_whole, (
        f"slice left at {p_slice} Pa while the whole tensor reached {p_whole} Pa")
