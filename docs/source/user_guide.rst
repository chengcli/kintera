User Guide
==========

This guide provides in-depth information about KINTERA's concepts, architecture, and usage patterns.

Core Concepts
-------------

Species and Composition
~~~~~~~~~~~~~~~~~~~~~~~

KINTERA works with chemical species defined by:

* **Names**: Unique identifiers for each species
* **Molecular weights**: Used in thermodynamic calculations
* **Thermodynamic reference states**: Reference values for entropy, enthalpy, etc.

Species are categorized into:

* **Vapor species**: Gaseous phase
* **Cloud species**: Condensed phase (liquid or solid)

Thermodynamic State
~~~~~~~~~~~~~~~~~~~

The thermodynamic state is defined by:

* **Temperature (T)**: In Kelvin
* **Pressure (P)**: In Pascal
* **Composition (X)**: Mole fractions for each species

KINTERA supports tensor operations, so you can work with:

* Single points: ``(T, P, X)``
* 1D profiles: ``(ncol, nlyr)``
* Multi-dimensional grids: ``(ncol, nlyr, nlvl)``

Phase Equilibrium
~~~~~~~~~~~~~~~~~

Phase transitions are handled through:

* **Nucleation**: Vapor-to-liquid/solid transitions
* **Evaporation**: Liquid/solid-to-vapor transitions
* **Saturation vapor pressure**: Clausius-Clapeyron or custom formulations

Working with ThermoOptions
---------------------------

The ``ThermoOptions`` class configures the thermodynamic solver.

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kintera import ThermoOptions

   op = ThermoOptions()

   # Convergence criteria
   op.max_iter(15)  # Maximum iterations
   op.ftol(1.e-8)   # Convergence tolerance

   # Reference state
   op.Tref(300.0)   # Reference temperature (K)
   op.Pref(1.e5)    # Reference pressure (Pa)

Species Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define which species are vapors vs clouds
   op.vapor_ids([0, 1])  # Indices of vapor species
   op.cloud_ids([2])     # Indices of cloud species

   # Thermodynamic reference values (normalized by R)
   op.cref_R([2.5, 2.5, 9.0])        # Specific heat capacity
   op.uref_R([0.0, 0.0, -3430.])     # Internal energy
   op.sref_R([0.0, 0.0, 0.0])        # Entropy

Loading from YAML
~~~~~~~~~~~~~~~~~

For complex configurations, use YAML files:

.. code-block:: python

   op = ThermoOptions.from_yaml("jupiter.yaml")

Example YAML structure:

.. code-block:: yaml

   species:
     - name: H2
       weight: 2.016e-3
       vapor: true
     - name: He
       weight: 4.003e-3
       vapor: true

   thermodynamics:
     Tref: 200.0
     Pref: 1.e5
     max_iter: 15
     ftol: 1.e-8

Phase Transitions with NucleationOptions
-----------------------------------------

Configuring Phase Transitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kintera import NucleationOptions, Reaction

   nucleation = NucleationOptions()

   # Define phase transition reactions
   nucleation.reactions([
       Reaction("H2O <=> H2O(l)"),
       Reaction("NH3 <=> NH3(l)")
   ])

   # Temperature range for each transition
   nucleation.minT([200.0, 150.0])
   nucleation.maxT([400.0, 300.0])

   # Saturation vapor pressure model
   nucleation.set_logsvp(["h2o_ideal", "nh3_ideal"])

Available SVP Models
~~~~~~~~~~~~~~~~~~~~

* ``h2o_ideal``: Ideal water vapor pressure
* ``nh3_ideal``: Ideal ammonia vapor pressure
* Custom models can be defined via configuration files

Attaching to ThermoOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   op = ThermoOptions()
   # ... other configuration ...
   op.nucleation(nucleation)

Computing Thermodynamic Properties
-----------------------------------

The ThermoX Class
~~~~~~~~~~~~~~~~~

``ThermoX`` is the main class for thermodynamic calculations:

.. code-block:: python

   from kintera import ThermoX

   thermo = ThermoX(op)  # op is a ThermoOptions object

Forward Computation
~~~~~~~~~~~~~~~~~~~

Compute equilibrium composition at given T and P:

.. code-block:: python

   import torch

   temp = torch.tensor([300.0], dtype=torch.float64)
   pres = torch.tensor([1.e5], dtype=torch.float64)
   xfrac = torch.tensor([[0.98, 0.02, 0.0]], dtype=torch.float64)

   # Modifies xfrac in-place to equilibrium composition
   thermo.forward(temp, pres, xfrac)

State Conversions
~~~~~~~~~~~~~~~~~

Convert between different state representations:

.. code-block:: python

   # Temperature, Pressure, Mole fraction -> Volume (concentration)
   conc = thermo.compute("TPX->V", [temp, pres, xfrac])

   # Temperature, Pressure, Volume -> Entropy
   entropy = thermo.compute("TPV->S", [temp, pres, conc])

   # Temperature, Pressure, Volume -> Internal energy
   energy = thermo.compute("TPV->U", [temp, pres, conc])

Available conversions:

* ``TPX->V``: Mole fractions to concentrations
* ``TPV->S``: State to entropy
* ``TPV->U``: State to internal energy
* ``TPV->H``: State to enthalpy

Adiabatic Extrapolation
~~~~~~~~~~~~~~~~~~~~~~~~

Propagate state adiabatically:

.. code-block:: python

   # Using pressure coordinate
   dlnp = 0.1  # Change in log(pressure)
   thermo.extrapolate_ad(temp, pres, xfrac, -dlnp)

   # Using height coordinate
   grav = 9.8  # m/s²
   dz = 100.0  # m
   thermo.extrapolate_ad(temp, pres, xfrac, grav, dz)

Working with Chemical Kinetics
-------------------------------

The Kinetics Module
~~~~~~~~~~~~~~~~~~~

For time-dependent chemistry:

.. code-block:: python

   from kintera import Kinetics, KineticsOptions, ArrheniusOptions

Reaction Types
~~~~~~~~~~~~~~

KINTERA supports several types of chemical reactions, each with different rate constant formulations.

Arrhenius Reactions
~~~~~~~~~~~~~~~~~~~

Standard temperature-dependent rate constants:

.. code-block:: python

   from kintera import Arrhenius

   # k = A * T^b * exp(-Ea/RT)
   rate = Arrhenius()
   rate.A(1.0e13)      # Pre-exponential factor
   rate.b(0.0)         # Temperature exponent
   rate.Ea(50.0e3)     # Activation energy (J/mol)

YAML configuration:

.. code-block:: yaml

   reactions:
   - equation: O + H2 <=> H + OH
     type: arrhenius
     rate-constant: {A: 38.7, b: 2.7, Ea_R: 6260.0}

Three-Body Reactions
~~~~~~~~~~~~~~~~~~~~

Simple three-body reactions have the form ``A + B + M → products + M``, where M is a third body (collision partner). The rate is:

.. math::

   k = k_0 \times [M]_{eff}

where :math:`[M]_{eff} = \sum_i \alpha_i [species_i]` is the effective third-body concentration, and :math:`\alpha_i` are collision efficiencies.

YAML configuration:

.. code-block:: yaml

   reactions:
   - equation: H2O2 + M <=> O + H2O + M
     type: three-body
     rate-constant: {A: 1.2e+11, b: -1.0, Ea_R: 0.0}
     efficiencies: {AR: 0.83, H2: 2.4, H2O: 15.4}

Falloff Reactions
~~~~~~~~~~~~~~~~~

Falloff reactions transition between low-pressure and high-pressure limits:

.. math::

   k = \frac{k_0 [M]_{eff}}{1 + k_0 [M]_{eff} / k_{\infty}} \times F(T)

where :math:`F(T)` is a broadening factor (1.0 for Lindemann, or Troe/SRI for enhanced accuracy).

YAML configuration examples:

.. code-block:: yaml

   reactions:
   # Lindemann falloff (no broadening)
   - equation: 2 OH (+ M) <=> H2O2 (+ M)
     type: falloff
     low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: -1700.0}
     high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
     efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}

   # Troe falloff (3-parameter)
   - equation: 2 OH (+ M) <=> H2O2 (+ M)
     type: falloff
     low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: -1700.0}
     high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
     Troe: {A: 0.51, T3: 1.0e-30, T1: 1.0e+30}
     efficiencies: {AR: 0.3, H2: 1.5, H2O: 2.7}

   # Troe falloff (4-parameter)
   - equation: 2 OH (+ M) <=> H2O2 (+ M)
     type: falloff
     low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea_R: -1700.0}
     high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea_R: 0.0}
     Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}
     efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}

   # SRI falloff (3-parameter)
   - equation: O + H2 (+ M) <=> H + OH (+ M)
     type: falloff
     low-P-rate-constant: {A: 4.0e+19, b: -3.0, Ea: 0.0 cal/mol}
     high-P-rate-constant: {A: 1.0e+15, b: -2.0, Ea: 1000.0 cal/mol}
     SRI: {A: 0.54, B: 201.0, C: 1024.0}

   # SRI falloff (5-parameter)
   - equation: H + HO2 (+ M) <=> H2 + O2 (+ M)
     type: falloff
     low-P-rate-constant: {A: 7.0e+20, b: -1.0, Ea: 0.0 cal/mol}
     high-P-rate-constant: {A: 4.0e+15, b: -0.5, Ea: 100.0 cal/mol}
     SRI: {A: 1.1, B: 700.0, C: 1234.0, D: 56.0, E: 0.7}
     efficiencies: {AR: 0.7, H2: 2.0, H2O: 6.0}

**Notes:**

* Use ``type: falloff`` with both ``low-P-rate-constant`` and ``high-P-rate-constant``
* Optional ``efficiencies`` map specifies third-body collision efficiencies (defaults to 1.0)
* Optional ``Troe`` or ``SRI`` nodes add temperature-dependent broadening
* Activation energies can be ``Ea_R`` (K) or ``Ea`` with units (``cal/mol``, ``kJ/mol``)
* Input units: ``molecule, cm, s``; internal: ``mol, m, s``

Configuring Kinetics
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   kop = KineticsOptions()

   # Set species information (inherits from SpeciesThermo)
   kop.vapor_ids([0, 1, 2])

   # Add reactions with rate constants
   # (See API reference for detailed usage)

Utility Functions
-----------------

Global Species Management
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import kintera

   # Set species globally
   kintera.set_species_names(["H2", "O2", "H2O"])
   kintera.set_species_weights([2.016e-3, 32.0e-3, 18.015e-3])

   # Get current species
   names = kintera.species_names()
   weights = kintera.species_weights()

Resource Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add resource directories for data files
   kintera.add_resource_directory("/path/to/data", prepend=True)

   # Find resource files
   file_path = kintera.find_resource("jupiter.yaml")

   # Get/set search paths
   paths = kintera.get_search_paths()
   kintera.set_search_paths("/path1:/path2")

Computing Relative Humidity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kintera import relative_humidity

   # Get stoichiometry buffer from thermo object
   stoich = thermo.get_buffer("stoich")

   # Compute relative humidity
   rh = relative_humidity(temp, conc, stoich, thermo.options.nucleation())

GPU Acceleration
----------------

KINTERA supports GPU computation through PyTorch:

.. code-block:: python

   import torch

   # Check GPU availability
   if torch.cuda.is_available():
       device = torch.device("cuda")

       # Move tensors to GPU
       temp = temp.to(device)
       pres = pres.to(device)
       xfrac = xfrac.to(device)

       # Computations automatically use GPU
       thermo.forward(temp, pres, xfrac)

Type Hints and IDE Support
---------------------------

KINTERA provides complete type annotations:

.. code-block:: python

   from kintera import ThermoOptions
   import torch

   # Type checking with mypy
   def setup_thermo(temp: torch.Tensor) -> ThermoOptions:
       op = ThermoOptions()
       op.Tref(temp.item())
       return op

Run type checking:

.. code-block:: bash

   mypy your_script.py

Error Handling
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Convergence Failures**

If thermodynamic solver doesn't converge:

.. code-block:: python

   # Increase iterations or relax tolerance
   op.max_iter(30).ftol(1.e-6)

   # Check initial conditions
   print("Initial state:", temp, pres, xfrac)

**Numerical Instabilities**

Use double precision:

.. code-block:: python

   torch.set_default_dtype(torch.float64)

**Invalid Compositions**

Ensure mole fractions are valid:

.. code-block:: python

   # Check for negative values
   assert (xfrac >= 0).all()

   # Normalize
   xfrac /= xfrac.sum(dim=-1, keepdim=True)

Performance Tips
----------------

1. **Batch computations**: Process multiple atmospheric columns together
2. **GPU acceleration**: Use CUDA for large-scale problems
3. **Minimize data transfers**: Keep data on GPU between operations
4. **Vectorize operations**: Use PyTorch's vectorized functions

Advanced Topics
---------------

Custom Thermodynamic Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend KINTERA with custom EOS or SVP models by subclassing base classes and implementing required methods.

Coupling with Other Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

KINTERA can be integrated with:

* Radiative transfer models
* Dynamical cores
* Cloud microphysics schemes

See the examples directory for integration patterns.
