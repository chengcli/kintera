#! /usr/bin/env python3

import torch
from kintera import SpeciesThermo, ThermoOptions, ThermoX

if __name__ == "__main__":
    # Create an instance of ThermoOptions
    op = ThermoOptions.from_yaml("jupiter.yaml").max_iter(15).ftol(1.e-8)
    thermo = ThermoX(op)

    temp = torch.tensor([200.], dtype=torch.float64)
    pres = torch.tensor([1.e5], dtype=torch.float64)

    species = op.species()
    print(species)

    nspecies = len(species)
    xfrac = torch.rand((1, 1, nspecies), dtype=torch.float64)
    xfrac /= xfrac.sum(dim=-1, keepdim=True)

    print("xfrac before = ", xfrac)
    thermo.forward(temp, pres, xfrac);
    print("xfrac after = ", xfrac)
