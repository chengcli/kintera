"""Na2S evaporation in an isothermal closed box, using the existing integrator.

Transport and caloric parameters are illustrative, not material calibration.
Timestep rejection protects inventories and avoids crossing equilibrium for
this isolated reaction; evolve_implicit itself provides no positivity guarantee.
"""

from pathlib import Path

import torch
import kintera as kt


def run(dt=1000., duration=100000., device="cpu"):
    options = kt.KineticsOptions.from_yaml(str(Path(__file__).with_suffix(".yaml")))
    kinetics = kt.Kinetics(options)
    kinetics.to(torch.device(device))
    names = options.species()
    initial = {"H2": 1., "Na": 0., "H2S": .005, "Na2S(s)": .02}
    conc = torch.tensor([[initial[name] for name in names]],
                        dtype=torch.float64, device=device)
    start = conc.clone()
    temp = torch.full((1,), 1200., dtype=conc.dtype, device=device)
    ngas = len(options.vapor_ids())

    def residual(c):
        # Zero Na correctly gives -inf before any evaporation.
        return kt.relative_humidity(
            temp, c, -kinetics.buffer("stoich"), options.evaporation(), ngas
        )[..., 0].log()

    # Element totals in the runtime species order, not assumed array slots.
    atoms = {"H2": [2, 0, 0], "Na": [0, 1, 0],
             "H2S": [2, 0, 1], "Na2S(s)": [0, 2, 1]}
    elements = conc.new_tensor([atoms[name] for name in names])
    time, accepted, rejected = 0., 0, 0
    while time < duration:
        step = min(dt, duration - time)
        for _ in range(60):
            pressure = kt.constants.Rgas * temp * conc[:, :ngas].sum(-1)
            rate, dc, _ = kinetics.forward(temp, pressure, conc)
            jac = kinetics.jacobian(temp, conc, torch.ones_like(temp), rate, dc)
            delta = kt.evolve_implicit(rate, kinetics.buffer("stoich"), jac, step)
            trial = conc + delta
            if (torch.isfinite(trial).all() and (trial >= 0).all() and
                    (residual(trial) <= 1.e-10).all()):
                break
            step *= .5
            rejected += 1
        else:
            raise RuntimeError("Unable to find a feasible evaporation timestep")
        if time + step == time:
            raise RuntimeError("Evaporation timestep underflow")
        conc = trial
        time += step
        accepted += 1
    torch.testing.assert_close(conc @ elements, start @ elements,
                               rtol=1.e-12, atol=1.e-14)
    assert torch.isfinite(conc).all() and (conc >= 0).all()
    print(f"time={time:g} s; accepted={accepted}, rejected={rejected}")
    print("concentrations [mol/m^3]:", dict(zip(names, conc[0].tolist())))
    print("log(Q/Qsat):", residual(conc).item())
    return conc


if __name__ == "__main__":
    run()
