API Reference
=============

This page provides detailed API documentation for all KINTERA classes and functions.

Core Classes
------------

ThermoX
~~~~~~~

.. class:: kintera.ThermoX

   Main thermodynamics computation class for equilibrium calculations.

   :param options: Configuration options for thermodynamics
   :type options: ThermoOptions

   .. method:: forward(temp: torch.Tensor, pres: torch.Tensor, xfrac: torch.Tensor) -> None

      Compute equilibrium composition at given temperature and pressure.
      Modifies xfrac in-place.

      :param temp: Temperature (K)
      :param pres: Pressure (Pa)
      :param xfrac: Mole fractions (modified in-place)

   .. method:: compute(mode: str, state: list[torch.Tensor]) -> torch.Tensor

      Convert between different state representations.

      :param mode: Conversion mode (e.g., "TPX->V", "TPV->S")
      :param state: List of state tensors [temp, pres, composition]
      :return: Computed property tensor

      Available modes:

      * ``"TPX->V"``: Mole fractions to concentrations
      * ``"TPV->S"``: State to entropy
      * ``"TPV->U"``: State to internal energy
      * ``"TPV->H"``: State to enthalpy

   .. method:: extrapolate_ad(temp: torch.Tensor, pres: torch.Tensor, xfrac: torch.Tensor, *args) -> None

      Perform adiabatic extrapolation.

      **Using pressure coordinate:**

      :param temp: Temperature (modified in-place)
      :param pres: Pressure (modified in-place)
      :param xfrac: Mole fractions (modified in-place)
      :param dlnp: Change in log(pressure)

      **Using height coordinate:**

      :param temp: Temperature (modified in-place)
      :param pres: Pressure (modified in-place)
      :param xfrac: Mole fractions (modified in-place)
      :param grav: Gravitational acceleration (m/s²)
      :param dz: Height increment (m)

   .. method:: get_buffer(name: str) -> torch.Tensor

      Get internal buffer by name.

      :param name: Buffer name (e.g., "stoich")
      :return: Buffer tensor

   .. attribute:: options

      The ThermoOptions configuration object.

ThermoY
~~~~~~~

.. class:: kintera.ThermoY

   Alternative thermodynamics class (implementation may vary from ThermoX).
   See ThermoX for method documentation.

Configuration Classes
---------------------

ThermoOptions
~~~~~~~~~~~~~

.. class:: kintera.ThermoOptions

   Configuration options for thermodynamic calculations.
   Inherits from SpeciesThermo.

   .. method:: __init__() -> ThermoOptions

      Create a new ThermoOptions object.

   .. classmethod:: from_yaml(filename: str) -> ThermoOptions

      Load configuration from YAML file.

      :param filename: Path to YAML configuration file
      :return: Configured ThermoOptions object

   .. method:: max_iter(value: int) -> ThermoOptions

      Set maximum number of iterations for solver.

      :param value: Maximum iterations
      :return: Self for method chaining

   .. method:: ftol(value: float) -> ThermoOptions

      Set convergence tolerance.

      :param value: Tolerance value
      :return: Self for method chaining

   .. method:: Tref(value: float) -> ThermoOptions

      Set reference temperature.

      :param value: Reference temperature (K)
      :return: Self for method chaining

   .. method:: Pref(value: float) -> ThermoOptions

      Set reference pressure.

      :param value: Reference pressure (Pa)
      :return: Self for method chaining

   .. method:: nucleation(value: NucleationOptions) -> ThermoOptions

      Set nucleation/phase transition options.

      :param value: NucleationOptions object
      :return: Self for method chaining

   .. method:: species() -> list[str]

      Get list of species names.

      :return: List of species names

SpeciesThermo
~~~~~~~~~~~~~

.. class:: kintera.SpeciesThermo

   Base class for species thermodynamic properties.

   .. method:: __init__() -> SpeciesThermo

      Create a new SpeciesThermo object.

   .. method:: species() -> list[str]

      Get the list of species names.

      :return: List of species names

   .. method:: vapor_ids(value: list[int] = None) -> list[int] | SpeciesThermo

      Get or set vapor species IDs.

      :param value: List of vapor species IDs (optional)
      :return: List of IDs if no argument, self if setting

   .. method:: cloud_ids(value: list[int] = None) -> list[int] | SpeciesThermo

      Get or set cloud species IDs.

      :param value: List of cloud species IDs (optional)
      :return: List of IDs if no argument, self if setting

   .. method:: cref_R(value: list[float] = None) -> list[float] | SpeciesThermo

      Get or set specific heat capacity reference values (normalized by R).

      :param value: List of specific heat values (optional)
      :return: List of values if no argument, self if setting

   .. method:: uref_R(value: list[float] = None) -> list[float] | SpeciesThermo

      Get or set internal energy reference values (normalized by R).

      :param value: List of internal energy values (optional)
      :return: List of values if no argument, self if setting

   .. method:: sref_R(value: list[float] = None) -> list[float] | SpeciesThermo

      Get or set entropy reference values (normalized by R).

      :param value: List of entropy values (optional)
      :return: List of values if no argument, self if setting

NucleationOptions
~~~~~~~~~~~~~~~~~

.. class:: kintera.NucleationOptions

   Configuration for phase transition (nucleation/condensation).

   .. method:: __init__() -> NucleationOptions

      Create a new NucleationOptions object.

   .. method:: reactions(value: list[Reaction]) -> NucleationOptions

      Set phase transition reactions.

      :param value: List of Reaction objects
      :return: Self for method chaining

   .. method:: minT(value: list[float]) -> NucleationOptions

      Set minimum temperatures for each reaction.

      :param value: List of minimum temperatures (K)
      :return: Self for method chaining

   .. method:: maxT(value: list[float]) -> NucleationOptions

      Set maximum temperatures for each reaction.

      :param value: List of maximum temperatures (K)
      :return: Self for method chaining

   .. method:: set_logsvp(models: list[str]) -> NucleationOptions

      Set saturation vapor pressure models.

      :param models: List of model names (e.g., ["h2o_ideal"])
      :return: Self for method chaining

Reaction Classes
----------------

Reaction
~~~~~~~~

.. class:: kintera.Reaction

   Represents a chemical reaction.

   .. method:: __init__(equation: str = "") -> Reaction

      Create a reaction from equation string.

      :param equation: Chemical equation (e.g., "H2 + O2 => H2O2")

   .. method:: equation() -> str

      Get the chemical equation.

      :return: Equation string

   .. method:: reactants(value: dict[str, float] = None) -> dict[str, float] | Reaction

      Get or set reactants with stoichiometric coefficients.

      :param value: Dictionary of reactants (optional)
      :return: Reactants dict if no argument, self if setting

   .. method:: products(value: dict[str, float] = None) -> dict[str, float] | Reaction

      Get or set products with stoichiometric coefficients.

      :param value: Dictionary of products (optional)
      :return: Products dict if no argument, self if setting

Kinetics Classes
----------------

Kinetics
~~~~~~~~

.. class:: kintera.Kinetics

   Chemical kinetics solver for time-dependent chemistry.

   .. method:: __init__(options: KineticsOptions) -> Kinetics

      Create kinetics solver with configuration.

      :param options: KineticsOptions object

KineticsOptions
~~~~~~~~~~~~~~~

.. class:: kintera.KineticsOptions

   Configuration for chemical kinetics calculations.
   Inherits from SpeciesThermo.

ArrheniusOptions
~~~~~~~~~~~~~~~~

.. class:: kintera.ArrheniusOptions

   Configuration for Arrhenius rate expressions.
   Inherits from SpeciesThermo.

Arrhenius
~~~~~~~~~

.. class:: kintera.Arrhenius

   Arrhenius rate constant: k = A * T^b * exp(-Ea/RT)

   .. method:: A(value: float = None) -> float | Arrhenius

      Get or set pre-exponential factor.

      :param value: Pre-exponential factor (optional)
      :return: Value if no argument, self if setting

   .. method:: b(value: float = None) -> float | Arrhenius

      Get or set temperature exponent.

      :param value: Temperature exponent (optional)
      :return: Value if no argument, self if setting

   .. method:: Ea(value: float = None) -> float | Arrhenius

      Get or set activation energy.

      :param value: Activation energy in J/mol (optional)
      :return: Value if no argument, self if setting

CoagulationOptions
~~~~~~~~~~~~~~~~~~

.. class:: kintera.CoagulationOptions

   Configuration for coagulation processes.
   Inherits from ArrheniusOptions.

Evaporation Classes
-------------------

Evaporation
~~~~~~~~~~~

.. class:: kintera.Evaporation

   Evaporation rate calculations.

EvaporationOptions
~~~~~~~~~~~~~~~~~~

.. class:: kintera.EvaporationOptions

   Configuration for evaporation calculations.
   Inherits from NucleationOptions.

Module-Level Functions
----------------------

Species Management
~~~~~~~~~~~~~~~~~~

.. function:: kintera.species_names() -> list[str]

   Get current global species names.

   :return: List of species names

.. function:: kintera.set_species_names(names: list[str]) -> list[str]

   Set global species names.

   :param names: List of species names
   :return: Updated list of species names

.. function:: kintera.species_weights() -> list[float]

   Get current global species molecular weights.

   :return: List of molecular weights (kg/mol)

.. function:: kintera.set_species_weights(weights: list[float]) -> list[float]

   Set global species molecular weights.

   :param weights: List of molecular weights (kg/mol)
   :return: Updated list of molecular weights

.. function:: kintera.species_cref_R() -> list[float]

   Get global species specific heat capacity reference values.

   :return: List of cref/R values

.. function:: kintera.set_species_cref_R(cref_R: list[float]) -> list[float]

   Set global species specific heat capacity reference values.

   :param cref_R: List of cref/R values
   :return: Updated list

.. function:: kintera.species_uref_R() -> list[float]

   Get global species internal energy reference values.

   :return: List of uref/R values

.. function:: kintera.set_species_uref_R(uref_R: list[float]) -> list[float]

   Set global species internal energy reference values.

   :param uref_R: List of uref/R values
   :return: Updated list

.. function:: kintera.species_sref_R() -> list[float]

   Get global species entropy reference values.

   :return: List of sref/R values

.. function:: kintera.set_species_sref_R(sref_R: list[float]) -> list[float]

   Set global species entropy reference values.

   :param sref_R: List of sref/R values
   :return: Updated list

Resource Management
~~~~~~~~~~~~~~~~~~~

.. function:: kintera.get_search_paths() -> str

   Get current resource search paths.

   :return: Colon-separated path string

.. function:: kintera.set_search_paths(path: str) -> str

   Set resource search paths.

   :param path: Colon-separated path string
   :return: Updated path string

.. function:: kintera.add_resource_directory(path: str, prepend: bool = True) -> str

   Add a directory to resource search paths.

   :param path: Directory path to add
   :param prepend: If True, add to front of search path
   :return: Updated path string

.. function:: kintera.find_resource(filename: str) -> str

   Find a resource file in search paths.

   :param filename: Filename to find
   :return: Full path to resource file

Utility Functions
~~~~~~~~~~~~~~~~~

.. function:: kintera.relative_humidity(temp: torch.Tensor, conc: torch.Tensor, stoich: torch.Tensor, nucleation: NucleationOptions) -> torch.Tensor

   Calculate relative humidity.

   :param temp: Temperature tensor (K)
   :param conc: Concentration tensor
   :param stoich: Stoichiometry tensor
   :param nucleation: NucleationOptions with phase transition info
   :return: Relative humidity tensor (0-1)

.. function:: kintera.evolve_implicit(...)

   Implicit time evolution for chemical kinetics.

   (Parameters depend on specific usage - see source for details)

Constants
---------

.. class:: kintera.constants

   Physical and chemical constants.

   Common constants include:

   * Gas constant
   * Boltzmann constant
   * Avogadro's number
   * Universal constants used in calculations

   Access constants as attributes of this class.

Type Aliases
------------

.. type:: Composition = dict[str, float]

   Dictionary mapping species names to their stoichiometric coefficients or mole fractions.
