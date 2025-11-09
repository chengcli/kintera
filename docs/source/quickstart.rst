Quickstart Guide
================

This guide will help you get started with KINTERA quickly by walking through basic usage patterns.

Basic Setup
-----------

Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import kintera
   from kintera import (
       SpeciesThermo,
       ThermoOptions,
       ThermoX,
       Reaction,
       NucleationOptions
   )
   
   # Set default dtype for PyTorch tensors
   torch.set_default_dtype(torch.float64)

Configuring Species
~~~~~~~~~~~~~~~~~~~

KINTERA requires you to define the species in your system:

.. code-block:: python

   # Define species names and molecular weights
   kintera.set_species_names(["dry", "H2O", "H2O(l)"])
   kintera.set_species_weights([29.e-3, 18.e-3, 18.e-3])  # kg/mol

Simple Thermodynamic Calculations
----------------------------------

Example 1: Jupiter Atmosphere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using a pre-configured YAML file:

.. code-block:: python

   import torch
   from kintera import ThermoOptions, ThermoX

   # Load configuration from YAML
   op = ThermoOptions.from_yaml("jupiter.yaml").max_iter(15).ftol(1.e-8)
   thermo = ThermoX(op)

   # Set up state variables
   temp = torch.tensor([200.], dtype=torch.float64)
   pres = torch.tensor([1.e5], dtype=torch.float64)

   # Get species and create composition array
   species = op.species()
   nspecies = len(species)
   xfrac = torch.rand((1, 1, nspecies), dtype=torch.float64)
   xfrac /= xfrac.sum(dim=-1, keepdim=True)

   print("Initial composition:", xfrac)
   
   # Compute equilibrium
   thermo.forward(temp, pres, xfrac)
   
   print("Equilibrium composition:", xfrac)

Example 2: Earth Atmosphere with Water Condensation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting up a system with phase transitions:

.. code-block:: python

   import torch
   import kintera
   from kintera import (
       Reaction,
       NucleationOptions,
       ThermoOptions,
       ThermoX
   )

   # Configure species
   kintera.set_species_names(["dry", "H2O", "H2O(l)"])
   kintera.set_species_weights([29.e-3, 18.e-3, 18.e-3])

   # Set up phase transition
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

   # Create thermodynamics object
   thermo = ThermoX(op)

   # Set initial state
   temp = torch.tensor([[310.0]], dtype=torch.float64)
   pres = torch.tensor([[1.e5]], dtype=torch.float64)
   xfrac = torch.tensor([[[0.98, 0.02, 0.0]]], dtype=torch.float64)

   # Compute equilibrium condensation
   thermo.forward(temp, pres, xfrac)
   
   print("After condensation:", xfrac)

Working with Atmospheric Profiles
----------------------------------

Computing Vertical Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import numpy as np
   from kintera import ThermoX, ThermoOptions, relative_humidity

   # Set up thermodynamics (using setup from previous example)
   thermo = setup_earth_thermo()  # Assume this is defined

   # Define vertical grid
   ncol = 1
   nlyr = 40
   pmax = 1.e5  # Surface pressure (Pa)
   pmin = 1.e4  # Top pressure (Pa)
   Tbot = 310.0  # Surface temperature (K)
   nspecies = len(thermo.options.species())
   
   grav = 9.8  # m/s²
   dz = 100.0  # vertical spacing (m)

   # Initialize arrays
   temp = Tbot * torch.ones((ncol, nlyr), dtype=torch.float64)
   pres = pmax * torch.ones((ncol, nlyr), dtype=torch.float64)
   xfrac = torch.zeros((ncol, nlyr, nspecies), dtype=torch.float64)

   # Set bottom layer composition
   xfrac[:, 0, :] = torch.tensor([0.98, 0.02, 0.])

   # Compute equilibrium at bottom layer
   thermo.forward(temp[:, 0], pres[:, 0], xfrac[:, 0, :])

   # Propagate upward using adiabatic extrapolation
   for i in range(1, nlyr):
       temp[:, i] = temp[:, i - 1]
       pres[:, i] = pres[:, i - 1]
       xfrac[:, i, :] = xfrac[:, i - 1, :]
       
       # Adiabatic extrapolation
       thermo.extrapolate_ad(temp[:, i], pres[:, i], xfrac[:, i, :], grav, dz)

   print("Temperature profile:", temp)
   print("Pressure profile:", pres)
   print("Composition profile:", xfrac)

Computing Derived Quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute molar concentration from TPX
   conc = thermo.compute("TPX->V", [temp, pres, xfrac])

   # Compute volumetric entropy
   entropy_vol = thermo.compute("TPV->S", [temp, pres, conc])

   # Compute molar entropy
   entropy_mol = entropy_vol / conc.sum(dim=-1)

   # Compute relative humidity
   stoich = thermo.get_buffer("stoich")
   rh = relative_humidity(temp, conc, stoich, thermo.options.nucleation())

   print("Relative humidity:", rh)

Advanced Usage
--------------

Custom Reaction Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~

Define custom reactions:

.. code-block:: python

   from kintera import Reaction

   # Create individual reactions
   rxn1 = Reaction("H2 + O2 => H2O2")
   rxn2 = Reaction("2H2 + O2 => 2H2O")

   # Access reaction properties
   print("Equation:", rxn1.equation())
   print("Reactants:", rxn1.reactants())
   print("Products:", rxn1.products())

Using Kinetics Module
~~~~~~~~~~~~~~~~~~~~~

For time-dependent chemistry:

.. code-block:: python

   from kintera import Kinetics, KineticsOptions, Arrhenius

   # Set up kinetics with reactions
   kop = KineticsOptions()
   # Configure kinetics options...
   
   kinetics = Kinetics(kop)
   
   # Compute reaction rates
   # (See API reference for detailed usage)

Best Practices
--------------

1. **Always use torch.float64**: For numerical stability in atmospheric calculations

   .. code-block:: python

      torch.set_default_dtype(torch.float64)

2. **Normalize compositions**: Ensure mole fractions sum to 1.0

   .. code-block:: python

      xfrac /= xfrac.sum(dim=-1, keepdim=True)

3. **Use YAML configuration**: For complex systems, use YAML files to define species and reactions

4. **Check convergence**: Monitor the convergence of iterative solvers

   .. code-block:: python

      op.max_iter(15).ftol(1.e-8)

5. **Handle GPU arrays**: KINTERA works with both CPU and GPU PyTorch tensors

   .. code-block:: python

      # Move to GPU if available
      if torch.cuda.is_available():
          temp = temp.cuda()
          pres = pres.cuda()
          xfrac = xfrac.cuda()

Next Steps
----------

* Explore the :doc:`user_guide` for detailed explanations
* Check the :doc:`api_reference` for complete API documentation
* See :doc:`examples` for more complete examples
* Review the `GitHub examples <https://github.com/chengcli/kintera/tree/main/examples>`_ directory
