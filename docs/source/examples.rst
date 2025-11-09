Examples
========

This page provides complete, working examples demonstrating various features of KINTERA.

Example 1: Simple Jupiter Atmosphere
-------------------------------------

This example demonstrates basic usage with a pre-configured YAML file.

.. code-block:: python

   import torch
   from kintera import ThermoOptions, ThermoX

   # Set default precision
   torch.set_default_dtype(torch.float64)

   # Load configuration from YAML
   op = ThermoOptions.from_yaml("jupiter.yaml")
   op.max_iter(15).ftol(1.e-8)

   # Create thermodynamics object
   thermo = ThermoX(op)

   # Set up state variables
   temp = torch.tensor([200.], dtype=torch.float64)
   pres = torch.tensor([1.e5], dtype=torch.float64)

   # Get species and create random initial composition
   species = op.species()
   print("Species:", species)

   nspecies = len(species)
   xfrac = torch.rand((1, 1, nspecies), dtype=torch.float64)
   xfrac /= xfrac.sum(dim=-1, keepdim=True)

   print("Initial composition:", xfrac)

   # Compute equilibrium
   thermo.forward(temp, pres, xfrac)

   print("Equilibrium composition:", xfrac)

Expected output shows the equilibrium composition of Jupiter's atmosphere at the specified temperature and pressure.

Example 2: Earth Atmosphere with Water Condensation
----------------------------------------------------

A complete example of modeling Earth's atmosphere with water phase transitions.

.. code-block:: python

   import torch
   import kintera
   import numpy as np
   from kintera import (
       Reaction,
       NucleationOptions,
       ThermoOptions,
       ThermoX,
       relative_humidity
   )

   # Set default precision
   torch.set_default_dtype(torch.float64)

   def setup_earth_thermo():
       """Set up thermodynamics for Earth atmosphere."""
       # Define species
       kintera.set_species_names(["dry", "H2O", "H2O(l)"])
       kintera.set_species_weights([29.e-3, 18.e-3, 18.e-3])  # kg/mol

       # Configure phase transition
       nucleation = NucleationOptions()
       nucleation.reactions([Reaction("H2O <=> H2O(l)")])
       nucleation.minT([200.0])
       nucleation.maxT([400.0])
       nucleation.set_logsvp(["h2o_ideal"])

       # Configure thermodynamics
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
       print("Configuration:", thermo.options)
       print("Species:", thermo.options.species())

       # Set up grid
       ncol = 1
       nlyr = 40
       pmax = 1.e5  # Surface pressure
       pmin = 1.e4  # Top pressure
       Tbot = 310.0  # Surface temperature (K)
       nspecies = len(thermo.options.species())

       # Initialize state
       temp = Tbot * torch.ones((ncol, nlyr), dtype=torch.float64)
       pres = pmax * torch.ones((ncol, nlyr), dtype=torch.float64)
       xfrac = torch.zeros((ncol, nlyr, nspecies), dtype=torch.float64)

       # Set bottom concentration
       xfrac[:, 0, :] = torch.tensor([0.98, 0.02, 0.])

       # Calculate equilibrium at bottom
       thermo.forward(temp[:, 0], pres[:, 0], xfrac[:, 0, :])
       print("Bottom layer after condensation:", xfrac[:, 0, :])

       # Compute molar concentration
       conc = thermo.compute("TPX->V", [temp, pres, xfrac])

       # Compute relative humidity
       stoich = thermo.get_buffer("stoich")
       rh = relative_humidity(temp, conc, stoich, thermo.options.nucleation())

       print("Relative humidity at bottom:", rh[:, 0])

Example 3: Vertical Profile with Adiabatic Extrapolation
---------------------------------------------------------

Computing a vertical atmospheric profile with adiabatic lapse rate.

.. code-block:: python

   import torch
   import kintera
   import numpy as np
   from kintera import (
       Reaction,
       NucleationOptions,
       ThermoOptions,
       ThermoX,
       relative_humidity
   )
   import matplotlib.pyplot as plt

   torch.set_default_dtype(torch.float64)

   def setup_earth_thermo():
       """Set up thermodynamics for Earth atmosphere."""
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

   def compute_vertical_profile():
       """Compute vertical atmospheric profile."""
       thermo = setup_earth_thermo()

       # Grid parameters
       ncol = 1
       nlyr = 40
       pmax = 1.e5
       pmin = 1.e4
       Tbot = 310.0
       nspecies = len(thermo.options.species())
       grav = 9.8  # m/s²
       dz = 100.0  # m

       # Initialize
       temp = Tbot * torch.ones((ncol, nlyr), dtype=torch.float64)
       pres = pmax * torch.ones((ncol, nlyr), dtype=torch.float64)
       xfrac = torch.zeros((ncol, nlyr, nspecies), dtype=torch.float64)

       # Bottom layer
       xfrac[:, 0, :] = torch.tensor([0.98, 0.02, 0.])
       thermo.forward(temp[:, 0], pres[:, 0], xfrac[:, 0, :])

       # Loop through layers
       for i in range(1, nlyr):
           temp[:, i] = temp[:, i - 1]
           pres[:, i] = pres[:, i - 1]
           xfrac[:, i, :] = xfrac[:, i - 1, :]

           # Adiabatic extrapolation
           thermo.extrapolate_ad(temp[:, i], pres[:, i], xfrac[:, i, :], grav, dz)

       # Compute derived quantities
       conc = thermo.compute("TPX->V", [temp, pres, xfrac])
       entropy_vol = thermo.compute("TPV->S", [temp, pres, conc])
       entropy_mol = entropy_vol / conc.sum(dim=-1)

       # Relative humidity
       stoich = thermo.get_buffer("stoich")
       rh = relative_humidity(temp, conc, stoich, thermo.options.nucleation())

       return temp, pres, xfrac, entropy_mol, rh

   if __name__ == "__main__":
       temp, pres, xfrac, entropy, rh = compute_vertical_profile()

       print("Temperature profile:", temp[0, :])
       print("Pressure profile:", pres[0, :])
       print("Water vapor profile:", xfrac[0, :, 1])
       print("Liquid water profile:", xfrac[0, :, 2])
       print("Relative humidity profile:", rh[0, :])

       # Optional: Plot results
       # fig, axes = plt.subplots(1, 4, figsize=(16, 6))
       # axes[0].plot(temp[0, :].numpy(), pres[0, :].numpy())
       # axes[0].set_xlabel('Temperature (K)')
       # axes[0].set_ylabel('Pressure (Pa)')
       # ... (add more plots)
       # plt.show()

Example 4: Custom Reaction Mechanism
-------------------------------------

Defining and working with custom chemical reactions.

.. code-block:: python

   from kintera import Reaction

   # Create reactions
   rxn1 = Reaction("H2 + O2 => H2O2")
   rxn2 = Reaction("2H2 + O2 => 2H2O")
   rxn3 = Reaction()  # Empty reaction

   # Examine reaction 1
   print("Equation:", rxn1.equation())
   print("Reactants:", rxn1.reactants())
   print("Products:", rxn1.products())

   # Manually set reaction components
   rxn3.reactants({"N2": 1.0, "O2": 1.0})
   rxn3.products({"NO": 2.0})
   print("Custom reaction:", rxn3.equation())

   # Use in phase transitions
   from kintera import NucleationOptions

   nucleation = NucleationOptions()
   nucleation.reactions([
       Reaction("H2O <=> H2O(l)"),
       Reaction("NH3 <=> NH3(l)"),
       Reaction("CH4 <=> CH4(l)")
   ])
   nucleation.minT([200.0, 150.0, 80.0])
   nucleation.maxT([400.0, 300.0, 200.0])
   nucleation.set_logsvp(["h2o_ideal", "nh3_ideal", "ch4_ideal"])

   print("Configured", len(nucleation.reactions()), "phase transitions")

Example 5: GPU Acceleration
----------------------------

Leveraging GPU acceleration for large-scale computations.

.. code-block:: python

   import torch
   from kintera import ThermoOptions, ThermoX

   torch.set_default_dtype(torch.float64)

   # Check GPU availability
   if not torch.cuda.is_available():
       print("GPU not available, using CPU")
       device = torch.device("cpu")
   else:
       print("Using GPU")
       device = torch.device("cuda")

   # Load configuration
   op = ThermoOptions.from_yaml("jupiter.yaml")
   thermo = ThermoX(op)

   # Create large batch on GPU
   ncol = 100  # Many columns
   nlyr = 50   # Many layers
   nspecies = len(op.species())

   # Initialize on GPU
   temp = torch.full((ncol, nlyr), 200.0, dtype=torch.float64, device=device)
   pres = torch.full((ncol, nlyr), 1.e5, dtype=torch.float64, device=device)
   xfrac = torch.rand((ncol, nlyr, nspecies), dtype=torch.float64, device=device)
   xfrac /= xfrac.sum(dim=-1, keepdim=True)

   # Compute equilibrium (automatically on GPU)
   import time
   start = time.time()

   for i in range(nlyr):
       thermo.forward(temp[:, i], pres[:, i], xfrac[:, i, :])

   elapsed = time.time() - start
   print(f"Computed {ncol * nlyr} equilibrium states in {elapsed:.3f} seconds")
   print(f"Rate: {ncol * nlyr / elapsed:.1f} states/second")

   # Move results back to CPU if needed
   xfrac_cpu = xfrac.cpu()

Example 6: Type Checking with MyPy
-----------------------------------

Using type hints for code validation.

.. code-block:: python

   # save as: example_typed.py
   import torch
   from kintera import ThermoOptions, ThermoX
   from typing import Tuple

   def setup_atmosphere(config_file: str) -> ThermoX:
       """Set up atmosphere from config file."""
       op: ThermoOptions = ThermoOptions.from_yaml(config_file)
       op.max_iter(15).ftol(1.e-8)
       return ThermoX(op)

   def compute_equilibrium(
       thermo: ThermoX,
       temp: torch.Tensor,
       pres: torch.Tensor,
       xfrac: torch.Tensor
   ) -> torch.Tensor:
       """Compute equilibrium composition."""
       thermo.forward(temp, pres, xfrac)
       return xfrac

   def main() -> None:
       """Main function."""
       torch.set_default_dtype(torch.float64)

       thermo = setup_atmosphere("jupiter.yaml")

       temp = torch.tensor([200.0])
       pres = torch.tensor([1.e5])
       nspecies = len(thermo.options.species())
       xfrac = torch.ones((1, 1, nspecies)) / nspecies

       result = compute_equilibrium(thermo, temp, pres, xfrac)
       print("Result:", result)

   if __name__ == "__main__":
       main()

Run type checking:

.. code-block:: bash

   mypy example_typed.py

Example 7: Integration with External Data
------------------------------------------

Loading and using external atmospheric data.

.. code-block:: python

   import torch
   import numpy as np
   from kintera import ThermoOptions, ThermoX

   torch.set_default_dtype(torch.float64)

   # Load pressure-temperature profile from file
   def load_profile(filename: str):
       """Load T-P profile from CSV."""
       data = np.loadtxt(filename, delimiter=',', skiprows=1)
       pressure = data[:, 0]  # Pa
       temperature = data[:, 1]  # K
       return pressure, temperature

   # Set up thermodynamics
   op = ThermoOptions.from_yaml("jupiter.yaml")
   thermo = ThermoX(op)

   # Load external profile
   pres_data, temp_data = load_profile("atmosphere_profile.csv")

   # Convert to tensors
   nlyr = len(pres_data)
   ncol = 1
   nspecies = len(op.species())

   temp = torch.from_numpy(temp_data).reshape(ncol, nlyr)
   pres = torch.from_numpy(pres_data).reshape(ncol, nlyr)
   xfrac = torch.zeros((ncol, nlyr, nspecies), dtype=torch.float64)

   # Set initial composition
   xfrac[:, :, 0] = 0.89  # H2
   xfrac[:, :, 1] = 0.11  # He
   xfrac /= xfrac.sum(dim=-1, keepdim=True)

   # Compute equilibrium at each level
   for i in range(nlyr):
       thermo.forward(temp[:, i], pres[:, i], xfrac[:, i, :])

   # Save results
   results = np.column_stack([
       pres[0, :].numpy(),
       temp[0, :].numpy(),
       xfrac[0, :, 0].numpy(),  # H2
       xfrac[0, :, 1].numpy(),  # He
   ])

   np.savetxt("equilibrium_results.csv", results,
              delimiter=',',
              header="Pressure(Pa),Temperature(K),H2,He",
              comments='')

Running the Examples
--------------------

All examples are available in the `examples/ <https://github.com/chengcli/kintera/tree/main/examples>`_ directory of the repository.

To run an example:

.. code-block:: bash

   cd examples
   python example_earth.py

Some examples require data files (like ``jupiter.yaml``), which are included in the examples directory.

Contributing Examples
---------------------

If you have an example that demonstrates KINTERA usage, please consider contributing it to the repository via a pull request!
