#! /usr/bin/env python3

import torch
import kintera
import numpy as np
from kintera import (
        Reaction,
        NucleationOptions,
        SpeciesThermo,
        ThermoOptions,
        ThermoX,
        relative_humidity
        )
torch.set_default_dtype(torch.float64)

def setup_earth_thermo():
    kintera.set_species_names(["dry", "H2O", "H2O(l)"])
    kintera.set_species_weights([29.e-3, 18.e-3, 18.e-3])

    nucleation = NucleationOptions()
    nucleation.reactions([Reaction("H2O <=> H2O(l)")])
    nucleation.minT([200.0])
    nucleation.maxT([400.0])
    nucleation.set_logsvp(["h2o_ideal"])

    op = ThermoOptions().max_iter(15).ftol(1.e-8)
    op.vapor_ids([0, 1])
    op.cloud_ids([2])
    op.cref_R([2.5, 2.5, 9.0])
    op.uref_R([0.0, 0.0, -3430.])
    op.sref_R([0.0, 0.0, 0.0])
    op.Tref(300.0)
    op.Pref(1.e5)
    op.nucleation(nucleation)

    return ThermoX(op)

if __name__ == "__main__":
    thermo = setup_earth_thermo()
    print(thermo.options)
    print(thermo.options.species())

    ncol = 1
    nlyr = 40
    pmax = 1.e5
    pmin = 1.e4
    Tbot = 310.0  # Surface temperature in Kelvin
    nspecies = len(thermo.options.species())
    dlnp = np.log(pmax / pmin) / (nlyr - 1)
    grav = 9.8
    dz = 100.

    temp = Tbot * torch.ones((ncol, nlyr))
    pres = pmax * torch.ones((ncol, nlyr))
    xfrac = torch.zeros((ncol, nlyr, nspecies))

    # set bottom concentration
    xfrac[:, 0, :] = torch.tensor([0.98, 0.02, 0.])

    # calculate equilibrium condensation at the bottom layer
    thermo.forward(temp[:, 0], pres[:, 0], xfrac[:, 0, :])
    print("xfrac bottom = ", xfrac[:, 0, :])

    # loop through layers
    for i in range(1, nlyr):
        # set state to previous layer
        temp[:, i] = temp[:, i - 1]
        pres[:, i] = pres[:, i - 1]
        xfrac[:, i, :] = xfrac[:, i - 1, :]

        # adiabatic extrapolation
        #thermo.extrapolate_ad(temp[:, i], pres[:, i], xfrac[:, i, :], -dlnp);
        thermo.extrapolate_ad(temp[:, i], pres[:, i], xfrac[:, i, :], grav, dz);

    # compute molar concentration
    conc = thermo.compute("TPX->V", [temp, pres, xfrac])

    # comptue volumetric entropy
    entropy_vol = thermo.compute("TPV->S", [temp, pres, conc])

    # compute molar entropy
    entropy_mol = entropy_vol / conc.sum(dim=-1)

    # compute relative humdity
    stoich = thermo.get_buffer("stoich")
    rh = relative_humidity(temp, conc, stoich, thermo.options.nucleation())

    print("temp = ", temp)
    print("pres = ", pres)
    print("xfrac = ", xfrac)
    print("entropy_mol = ", entropy_mol)
    print("rh = ", rh)
