#! /usr/bin/env python3

import torch
import kintera
from kintera import (
        Reaction,
        NucleationOptions,
        SpeciesThermo,
        ThermoOptions,
        ThermoX,
        )

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
    op.Rd(287.)
    op.mu_ratio([1.0, 0.621, 0.621])
    op.nucleation(nucleation)

    return ThermoX(op)

if __name__ == "__main__":
    thermo = setup_earth_thermo()
    print(thermo.options)

    temp = torch.tensor([300.], dtype=torch.float64)
    pres = torch.tensor([1.e5], dtype=torch.float64)
    xfrac = torch.tensor([[0.9, 0.1, 0.0]], dtype=torch.float64)

    print("xfrac before = ", xfrac)
    print("temp = ", temp)
    thermo.extrapolate_ad(temp, pres, xfrac, -0.1);
    print("xfrac after = ", xfrac)
