"""NASA-9 equilibrium-constant models."""

from collections.abc import Sequence

import torch

from ..kintera import EquilibriumOptions, EquilibriumTP, nasa9_gibbs_rt


class Nasa9LogK:
    """Compute reaction log(K) from bundled ideal-gas NASA-9 data."""

    def __init__(
        self,
        options: EquilibriumOptions,
        nasa9_species: Sequence[str] | None = None,
    ) -> None:
        self.species = list(nasa9_species or options.components())
        if len(self.species) != len(options.components()):
            raise ValueError("nasa9_species must contain one name per component")
        self.stoich = EquilibriumTP(options).buffer("stoich")

    def __call__(self, temp: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        """Return natural-log equilibrium constants in option reaction order."""
        del pressure
        gibbs_rt = nasa9_gibbs_rt(temp, self.species)
        stoich = self.stoich.to(device=temp.device, dtype=temp.dtype)
        return -(gibbs_rt @ stoich)
