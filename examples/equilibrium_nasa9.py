"""Gas equilibrium sample using the standard NASA-9 database."""

from pathlib import Path

import torch
from kintera import EquilibriumOptions, EquilibriumTP
from kintera.equilibrium import Nasa9LogK


def main() -> None:
    config = Path(__file__).with_suffix(".yaml")
    options = EquilibriumOptions.from_yaml(str(config))
    species = options.components()
    solver = EquilibriumTP(options)
    log_k_model = Nasa9LogK(options)

    temp = torch.tensor(4000.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    initial_moles = torch.tensor([0.60, 0.20, 0.20], dtype=torch.float64)
    log_k = log_k_model(temp, pressure)
    moles, _, diagnostics = solver(temp, pressure, initial_moles, log_k)

    print(f"T = {temp.item():.0f} K, P = {pressure.item():.0f} Pa")
    print("log(K):", dict(zip(options.reactions(), log_k.tolist())))
    print("initial moles:", dict(zip(species, initial_moles.tolist())))
    print("equilibrium moles:", dict(zip(species, moles.tolist())))
    print("diagnostics [status, iterations, error]:", diagnostics.tolist())


if __name__ == "__main__":
    main()
