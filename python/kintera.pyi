"""
Atmospheric Thermodynamics and Chemistry Library

This module provides Python bindings for the kintera C++ library,
offering efficient implementations of chemical kinetics calculations,
thermodynamic equation of state, and phase equilibrium computations.
"""

from typing import Dict, List, Optional, Union, overload
import torch

# Type aliases
Composition = Dict[str, float]

class SpeciesThermo:
    """
    Species thermodynamics configuration.

    This class manages thermodynamic properties and species information
    for atmospheric chemistry calculations.
    """

    def __init__(self) -> None:
        """
        Initialize a new SpeciesThermo object.

        Returns:
            SpeciesThermo: class object

        Examples:
            >>> from kintera import SpeciesThermo
            >>> op = SpeciesThermo()
        """
        ...

    def __repr__(self) -> str: ...

    def species(self) -> List[str]:
        """
        Get the list of species names.

        Returns:
            list[str]: list of species names

        Examples:
            >>> from kintera import SpeciesThermo
            >>> op = SpeciesThermo()
            >>> op.species()
            ['H2', 'O2', 'N2', 'Ar']
        """
        ...

    def narrow_copy(self) -> SpeciesThermo:
        """Create a narrow copy of the SpeciesThermo object."""
        ...

    def accumulate(self) -> SpeciesThermo:
        """Accumulate thermodynamic properties."""
        ...

    @overload
    def vapor_ids(self) -> List[int]:
        """
        Get the vapor species IDs.

        Returns:
            list[int]: List of vapor species IDs
        """
        ...

    @overload
    def vapor_ids(self, value: List[int]) -> SpeciesThermo:
        """
        Set the vapor species IDs.

        Args:
            value (list[int]): List of vapor species IDs

        Returns:
            SpeciesThermo: class object for method chaining

        Examples:
            >>> from kintera import SpeciesThermo
            >>> op = SpeciesThermo().vapor_ids([1, 2, 3])
            >>> print(op.vapor_ids())
            [1, 2, 3]
        """
        ...

    @overload
    def cloud_ids(self) -> List[int]:
        """
        Get the cloud species IDs.

        Returns:
            list[int]: List of cloud species IDs
        """
        ...

    @overload
    def cloud_ids(self, value: List[int]) -> SpeciesThermo:
        """
        Set the cloud species IDs.

        Args:
            value (list[int]): List of cloud species IDs

        Returns:
            SpeciesThermo: class object for method chaining

        Examples:
            >>> from kintera import SpeciesThermo
            >>> op = SpeciesThermo().cloud_ids([1, 2, 3])
            >>> print(op.cloud_ids())
            [1, 2, 3]
        """
        ...

    @overload
    def cref_R(self) -> List[float]:
        """
        Get the specific heat ratio for the reference state.

        Returns:
            list[float]: List of specific heat ratios
        """
        ...

    @overload
    def cref_R(self, value: List[float]) -> SpeciesThermo:
        """
        Set the specific heat ratio for the reference state.

        Args:
            value (list[float]): List of specific heat ratios

        Returns:
            SpeciesThermo: class object for method chaining

        Examples:
            >>> from kintera import SpeciesThermo
            >>> op = SpeciesThermo().cref_R([2.5, 2.7, 2.9])
            >>> print(op.cref_R())
            [2.5, 2.7, 2.9]
        """
        ...

    @overload
    def uref_R(self) -> List[float]:
        """
        Get the internal energy for the reference state.

        Returns:
            list[float]: List of internal energies
        """
        ...

    @overload
    def uref_R(self, value: List[float]) -> SpeciesThermo:
        """
        Set the internal energy for the reference state.

        Args:
            value (list[float]): List of internal energies

        Returns:
            SpeciesThermo: class object for method chaining

        Examples:
            >>> from kintera import SpeciesThermo
            >>> op = SpeciesThermo().uref_R([0.0, 1.0, 2.0])
            >>> print(op.uref_R())
            [0.0, 1.0, 2.0]
        """
        ...

    @overload
    def sref_R(self) -> List[float]:
        """
        Get the entropy for the reference state.

        Returns:
            list[float]: List of entropies
        """
        ...

    @overload
    def sref_R(self, value: List[float]) -> SpeciesThermo:
        """
        Set the entropy for the reference state.

        Args:
            value (list[float]): List of entropies

        Returns:
            SpeciesThermo: class object for method chaining

        Examples:
            >>> from kintera import SpeciesThermo
            >>> op = SpeciesThermo().sref_R([0.0, 1.0, 2.0])
            >>> print(op.sref_R())
            [0.0, 1.0, 2.0]
        """
        ...

class Reaction:
    """
    Chemical reaction representation.

    This class represents a chemical reaction with reactants and products.
    """

    @overload
    def __init__(self) -> None:
        """
        Initialize an empty Reaction object.

        Returns:
            Reaction: class object

        Examples:
            >>> from kintera import Reaction
            >>> op = Reaction()
        """
        ...

    @overload
    def __init__(self, equation: str) -> None:
        """
        Initialize a Reaction from a chemical equation string.

        Args:
            equation (str): The chemical equation of the reaction

        Returns:
            Reaction: class object

        Examples:
            >>> from kintera import Reaction
            >>> op = Reaction("H2 + O2 => H2O2")
        """
        ...

    def __repr__(self) -> str: ...

    def equation(self) -> str:
        """
        Get the chemical equation of the reaction.

        Returns:
            str: The chemical equation

        Examples:
            >>> from kintera import Reaction
            >>> op = Reaction("H2 + O2 => H2O2")
            >>> print(op.equation())
            H2 + O2 => H2O2
        """
        ...

    @overload
    def reactants(self) -> Composition:
        """
        Get the reactants of the reaction.

        Returns:
            dict[str, float]: The reactants with their stoichiometric coefficients
        """
        ...

    @overload
    def reactants(self, value: Composition) -> Reaction:
        """
        Set the reactants of the reaction.

        Args:
            value (dict[str, float]): The reactants with their stoichiometric coefficients

        Returns:
            Reaction: class object for method chaining

        Examples:
            >>> from kintera import Reaction
            >>> op = Reaction("H2 + O2 => H2O2")
            >>> print(op.reactants())
            {'H2': 1.0, 'O2': 1.0}
        """
        ...

    @overload
    def products(self) -> Composition:
        """
        Get the products of the reaction.

        Returns:
            dict[str, float]: The products with their stoichiometric coefficients
        """
        ...

    @overload
    def products(self, value: Composition) -> Reaction:
        """
        Set the products of the reaction.

        Args:
            value (dict[str, float]): The products with their stoichiometric coefficients

        Returns:
            Reaction: class object for method chaining

        Examples:
            >>> from kintera import Reaction
            >>> op = Reaction("H2 + O2 => H2O2")
            >>> print(op.products())
            {'H2O2': 1.0}
        """
        ...

class NucleationOptions:
    """
    Configuration options for nucleation reactions.

    This class manages parameters for heterogeneous and homogeneous
    nucleation processes in atmospheric chemistry.
    """

    def __init__(self) -> None:
        """
        Initialize a new NucleationOptions object.

        Returns:
            NucleationOptions: class object

        Examples:
            >>> from kintera import NucleationOptions
            >>> nucleation = NucleationOptions().minT([200.0]).maxT([300.0])
        """
        ...

    def __repr__(self) -> str: ...

    @overload
    def minT(self) -> List[float]:
        """
        Get the minimum temperature for nucleation reactions.

        Returns:
            list[float]: List of minimum temperatures in Kelvin
        """
        ...

    @overload
    def minT(self, value: List[float]) -> NucleationOptions:
        """
        Set the minimum temperature for nucleation reactions.

        Args:
            value (list[float]): List of minimum temperatures in Kelvin

        Returns:
            NucleationOptions: class object for method chaining

        Examples:
            >>> from kintera import NucleationOptions
            >>> nucleation = NucleationOptions().minT([200.0])
            >>> print(nucleation.minT())
            [200.0]
        """
        ...

    @overload
    def maxT(self) -> List[float]:
        """
        Get the maximum temperature for nucleation reactions.

        Returns:
            list[float]: List of maximum temperatures in Kelvin
        """
        ...

    @overload
    def maxT(self, value: List[float]) -> NucleationOptions:
        """
        Set the maximum temperature for nucleation reactions.

        Args:
            value (list[float]): List of maximum temperatures in Kelvin

        Returns:
            NucleationOptions: class object for method chaining

        Examples:
            >>> from kintera import NucleationOptions
            >>> nucleation = NucleationOptions().maxT([300.0])
            >>> print(nucleation.maxT())
            [300.0]
        """
        ...

    @overload
    def logsvp(self) -> List[str]:
        """
        Get the log saturation vapor pressure functions.

        Returns:
            list[str]: List of log SVP function names
        """
        ...

    @overload
    def logsvp(self, value: List[str]) -> NucleationOptions:
        """
        Set the log saturation vapor pressure functions.

        Args:
            value (list[str]): List of log SVP function names

        Returns:
            NucleationOptions: class object for method chaining

        Examples:
            >>> from kintera import NucleationOptions
            >>> nucleation = NucleationOptions().logsvp(["h2o_ideal"])
            >>> print(nucleation.logsvp())
            ["h2o_ideal"]
        """
        ...

    @overload
    def reactions(self) -> List[Reaction]:
        """
        Get the nucleation reactions.

        Returns:
            list[Reaction]: List of Reaction objects
        """
        ...

    @overload
    def reactions(self, value: List[Reaction]) -> NucleationOptions:
        """
        Set the nucleation reactions.

        Args:
            value (list[Reaction]): List of Reaction objects

        Returns:
            NucleationOptions: class object for method chaining

        Examples:
            >>> from kintera import NucleationOptions, Reaction
            >>> reaction = Reaction("H2O + CO2 -> H2CO3")
            >>> nucleation = NucleationOptions().reactions([reaction])
            >>> print(nucleation.reactions())
            [Reaction(H2O + CO2 -> H2CO3)]
        """
        ...

class ThermoOptions(SpeciesThermo):
    """
    Configuration options for thermodynamic calculations.

    This class extends SpeciesThermo with additional parameters for
    saturation adjustment and phase equilibrium calculations.
    """

    def __init__(self) -> None:
        """
        Initialize a new ThermoOptions object.

        Returns:
            ThermoOptions: class object

        Examples:
            >>> from kintera import ThermoOptions
            >>> op = ThermoOptions().Tref(300.0).Pref(1.e5)
        """
        ...

    def __repr__(self) -> str: ...

    def reactions(self) -> List[Reaction]:
        """
        Get the list of thermodynamic reactions.

        Returns:
            list[Reaction]: list of reactions

        Examples:
            >>> from kintera import ThermoOptions, Reaction
            >>> op = ThermoOptions().reactions([Reaction("H2 + O2 -> H2O")])
            >>> print(op.reactions())
            [Reaction(H2 + O2 -> H2O)]
        """
        ...

    @staticmethod
    def from_yaml(filename: str) -> ThermoOptions:
        """
        Create a ThermoOptions object from a YAML file.

        Args:
            filename (str): Path to the YAML file

        Returns:
            ThermoOptions: class object

        Examples:
            >>> from kintera import ThermoOptions
            >>> op = ThermoOptions.from_yaml("thermo_options.yaml")
        """
        ...

    @overload
    def Tref(self) -> float:
        """
        Get the reference temperature.

        Returns:
            float: Reference temperature in Kelvin (default: 300.0)
        """
        ...

    @overload
    def Tref(self, value: float) -> ThermoOptions:
        """
        Set the reference temperature.

        Args:
            value (float): Reference temperature in Kelvin

        Returns:
            ThermoOptions: class object for method chaining

        Examples:
            >>> from kintera import ThermoOptions
            >>> op = ThermoOptions().Tref(300.0)
            >>> print(op.Tref())
            300.0
        """
        ...

    @overload
    def Pref(self) -> float:
        """
        Get the reference pressure.

        Returns:
            float: Reference pressure in Pa (default: 1.e5)
        """
        ...

    @overload
    def Pref(self, value: float) -> ThermoOptions:
        """
        Set the reference pressure.

        Args:
            value (float): Reference pressure in Pa

        Returns:
            ThermoOptions: class object for method chaining

        Examples:
            >>> from kintera import ThermoOptions
            >>> op = ThermoOptions().Pref(1.e5)
            >>> print(op.Pref())
            100000.0
        """
        ...

    @overload
    def nucleation(self) -> NucleationOptions:
        """
        Get the nucleation options.

        Returns:
            NucleationOptions: nucleation reaction options
        """
        ...

    @overload
    def nucleation(self, value: NucleationOptions) -> ThermoOptions:
        """
        Set the nucleation options.

        Args:
            value (NucleationOptions): nucleation reaction options

        Returns:
            ThermoOptions: class object for method chaining
        """
        ...

    @overload
    def max_iter(self) -> int:
        """
        Get the maximum number of iterations for convergence.

        Returns:
            int: Maximum number of iterations (default: 10)
        """
        ...

    @overload
    def max_iter(self, value: int) -> ThermoOptions:
        """
        Set the maximum number of iterations for convergence.

        Args:
            value (int): Maximum number of iterations

        Returns:
            ThermoOptions: class object for method chaining

        Examples:
            >>> from kintera import ThermoOptions
            >>> op = ThermoOptions().max_iter(10)
            >>> print(op.max_iter())
            10
        """
        ...

    @overload
    def ftol(self) -> float:
        """
        Get the convergence tolerance for free energy.

        Returns:
            float: Convergence tolerance (default: 1.e-6)
        """
        ...

    @overload
    def ftol(self, value: float) -> ThermoOptions:
        """
        Set the convergence tolerance for free energy.

        Args:
            value (float): Convergence tolerance

        Returns:
            ThermoOptions: class object for method chaining

        Examples:
            >>> from kintera import ThermoOptions
            >>> op = ThermoOptions().ftol(1.e-6)
            >>> print(op.ftol())
            1e-06
        """
        ...

class ThermoY:
    """
    Saturation adjustment using mass fractions.

    This class performs thermodynamic equilibrium calculations using
    mass fraction as the primary variable.
    """

    def __init__(self, options: ThermoOptions) -> None:
        """
        Initialize ThermoY with thermodynamic options.

        Args:
            options (ThermoOptions): Configuration options
        """
        ...

    def forward(
        self,
        rho: torch.Tensor,
        intEng: torch.Tensor,
        yfrac: torch.Tensor,
        warm_start: bool = False,
        diag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform saturation adjustment.

        Args:
            rho (torch.Tensor): Density tensor [kg/m^3]
            intEng (torch.Tensor): Internal energy tensor [J/m^3]
            yfrac (torch.Tensor): Mass fraction tensor
            warm_start (bool): Whether to use warm start (default: False)
            diag (torch.Tensor, optional): Diagnostic output tensor

        Returns:
            torch.Tensor: Changes in mass fraction

        Examples:
            >>> from kintera import ThermoY, ThermoOptions
            >>> options = ThermoOptions().Tref(300.0).Pref(1.e5)
            >>> thermo_y = ThermoY(options)
            >>> rho = torch.tensor([1.0, 2.0, 3.0])
            >>> intEng = torch.tensor([1000.0, 2000.0, 3000.0])
            >>> yfrac = torch.tensor([0.1, 0.2, 0.3])
            >>> dyfrac = thermo_y.forward(rho, intEng, yfrac)
        """
        ...

    def compute(self, ab: str, args: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute thermodynamic transformations.

        Args:
            ab (str): Transformation string. Options:
                     ["C->Y", "Y->X", "DY->C", "DPY->U", "DUY->P", "DPY->T", "DTY->P"]
            args (list): List of arguments for the transformation

        Returns:
            torch.Tensor: Resulting tensor after transformation

        Examples:
            >>> from kintera import ThermoY, ThermoOptions
            >>> options = ThermoOptions().Tref(300.0).Pref(1.e5)
            >>> thermo_y = ThermoY(options)
            >>> result = thermo_y.compute("C->Y", [torch.tensor([1.0, 2.0, 3.0])])
        """
        ...

class ThermoX:
    """
    Equilibrium condensation using mole fractions.

    This class performs thermodynamic equilibrium calculations using
    mole fraction as the primary variable.
    """

    def __init__(self, options: ThermoOptions) -> None:
        """
        Initialize ThermoX with thermodynamic options.

        Args:
            options (ThermoOptions): Configuration options
        """
        ...

    def forward(
        self,
        temp: torch.Tensor,
        pres: torch.Tensor,
        xfrac: torch.Tensor,
        warm_start: bool = False,
        diag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform equilibrium condensation.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            pres (torch.Tensor): Pressure tensor [Pa]
            xfrac (torch.Tensor): Mole fraction tensor
            warm_start (bool): Whether to use warm start (default: False)
            diag (torch.Tensor, optional): Diagnostic output tensor

        Returns:
            torch.Tensor: Changes in mole fraction

        Examples:
            >>> from kintera import ThermoX, ThermoOptions
            >>> options = ThermoOptions().Tref(300.0).Pref(1.e5)
            >>> thermo_x = ThermoX(options)
            >>> temp = torch.tensor([300.0, 310.0, 320.0])
            >>> pres = torch.tensor([1.e5, 1.e6, 1.e7])
            >>> xfrac = torch.tensor([0.1, 0.2, 0.3])
            >>> dxfrac = thermo_x.forward(temp, pres, xfrac)
        """
        ...

    def compute(self, ab: str, args: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute thermodynamic transformations.

        Args:
            ab (str): Transformation string. Options: ["X->Y", "TPX->D"]
            args (list): List of arguments for the transformation

        Returns:
            torch.Tensor: Resulting tensor after transformation

        Examples:
            >>> from kintera import ThermoX, ThermoOptions
            >>> options = ThermoOptions().Tref(300.0).Pref(1.e5)
            >>> thermo_x = ThermoX(options)
            >>> result = thermo_x.compute("X->Y", [torch.tensor([0.1, 0.2, 0.3])])
        """
        ...

    def effective_cp(
        self,
        temp: torch.Tensor,
        pres: torch.Tensor,
        xfrac: torch.Tensor,
        gain: torch.Tensor,
        conc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute effective specific heat capacity at constant pressure.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            pres (torch.Tensor): Pressure tensor [Pa]
            xfrac (torch.Tensor): Mole fraction tensor
            gain (torch.Tensor): Gain tensor
            conc (torch.Tensor, optional): Concentration tensor

        Returns:
            torch.Tensor: Effective molar heat capacity [J/(mol*K)]
        """
        ...

    @overload
    def extrapolate_ad(
        self,
        temp: torch.Tensor,
        pres: torch.Tensor,
        xfrac: torch.Tensor,
        dlnp: float,
        verbose: bool = False
    ) -> None:
        """
        Extrapolate temperature and pressure along an adiabatic path.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            pres (torch.Tensor): Pressure tensor [Pa]
            xfrac (torch.Tensor): Mole fraction tensor
            dlnp (float): Delta ln pressure
            verbose (bool): Print diagnostic information (default: False)

        Returns:
            None

        Examples:
            >>> from kintera import ThermoX, ThermoOptions
            >>> options = ThermoOptions().Tref(300.0).Pref(1.e5)
            >>> thermo = ThermoX(options)
            >>> temp = torch.tensor([300.0, 310.0, 320.0])
            >>> pres = torch.tensor([1.e5, 1.e6, 1.e7])
            >>> xfrac = torch.tensor([0.1, 0.2, 0.3])
            >>> thermo.extrapolate_ad(temp, pres, xfrac, -0.01)
        """
        ...

    @overload
    def extrapolate_ad(
        self,
        temp: torch.Tensor,
        pres: torch.Tensor,
        xfrac: torch.Tensor,
        grav: float,
        dz: float,
        verbose: bool = False
    ) -> None:
        """
        Extrapolate temperature and pressure along an adiabatic path.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            pres (torch.Tensor): Pressure tensor [Pa]
            xfrac (torch.Tensor): Mole fraction tensor
            grav (float): Gravitational acceleration [m/s^2]
            dz (float): Delta height [m]
            verbose (bool): Print diagnostic information (default: False)

        Returns:
            None

        Examples:
            >>> from kintera import ThermoX, ThermoOptions
            >>> options = ThermoOptions().Tref(300.0).Pref(1.e5)
            >>> thermo = ThermoX(options)
            >>> temp = torch.tensor([300.0, 310.0, 320.0])
            >>> pres = torch.tensor([1.e5, 1.e6, 1.e7])
            >>> xfrac = torch.tensor([0.1, 0.2, 0.3])
            >>> thermo.extrapolate_ad(temp, pres, xfrac, 9.8, 100.0)
        """
        ...

class ArrheniusOptions:
    """
    Configuration options for Arrhenius rate kinetics.

    This class manages parameters for temperature-dependent reaction rates
    following the Arrhenius equation.
    """

    def __init__(self) -> None:
        """Initialize ArrheniusOptions."""
        ...

    @overload
    def Tref(self) -> float:
        """Get reference temperature for rate constant."""
        ...

    @overload
    def Tref(self, value: float) -> ArrheniusOptions:
        """
        Set reference temperature for rate constant.

        Args:
            value (float): Reference temperature [K]

        Returns:
            ArrheniusOptions: class object for method chaining
        """
        ...

    @overload
    def reactions(self) -> List[Reaction]:
        """Get reactions for which rate constants are defined."""
        ...

    @overload
    def reactions(self, value: List[Reaction]) -> ArrheniusOptions:
        """
        Set reactions for which rate constants are defined.

        Args:
            value (list[Reaction]): List of reactions

        Returns:
            ArrheniusOptions: class object for method chaining
        """
        ...

    @overload
    def A(self) -> List[float]:
        """Get pre-exponential factors for rate constants."""
        ...

    @overload
    def A(self, value: List[float]) -> ArrheniusOptions:
        """
        Set pre-exponential factors for rate constants.

        Args:
            value (list[float]): Pre-exponential factors

        Returns:
            ArrheniusOptions: class object for method chaining
        """
        ...

    @overload
    def b(self) -> List[float]:
        """Get dimensionless temperature exponents."""
        ...

    @overload
    def b(self, value: List[float]) -> ArrheniusOptions:
        """
        Set dimensionless temperature exponents.

        Args:
            value (list[float]): Temperature exponents

        Returns:
            ArrheniusOptions: class object for method chaining
        """
        ...

    @overload
    def Ea_R(self) -> List[float]:
        """Get activation energies in K."""
        ...

    @overload
    def Ea_R(self, value: List[float]) -> ArrheniusOptions:
        """
        Set activation energies in K.

        Args:
            value (list[float]): Activation energies [K]

        Returns:
            ArrheniusOptions: class object for method chaining
        """
        ...

    @overload
    def E4_R(self) -> List[float]:
        """Get additional 4th parameters in rate expression."""
        ...

    @overload
    def E4_R(self, value: List[float]) -> ArrheniusOptions:
        """
        Set additional 4th parameters in rate expression.

        Args:
            value (list[float]): 4th parameters

        Returns:
            ArrheniusOptions: class object for method chaining
        """
        ...

class Arrhenius:
    """
    Arrhenius rate kinetics model.

    This class implements temperature-dependent reaction rates using
    the Arrhenius equation.
    """

    def __init__(self, options: ArrheniusOptions) -> None:
        """
        Initialize Arrhenius with kinetics options.

        Args:
            options (ArrheniusOptions): Configuration options
        """
        ...

    def forward(
        self,
        temp: torch.Tensor,
        pres: torch.Tensor,
        conc: torch.Tensor,
        other: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Arrhenius reaction rates.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            pres (torch.Tensor): Pressure tensor [Pa]
            conc (torch.Tensor): Concentration tensor [mol/m^3]
            other (torch.Tensor): Additional state tensor

        Returns:
            torch.Tensor: Reaction rates
        """
        ...

class CoagulationOptions(ArrheniusOptions):
    """
    Configuration options for coagulation reactions.

    Extends ArrheniusOptions for particle coagulation processes.
    """

    def __init__(self) -> None:
        """Initialize CoagulationOptions."""
        ...

class EvaporationOptions(NucleationOptions):
    """
    Configuration options for evaporation reactions.

    This class manages parameters for cloud particle evaporation.
    """

    def __init__(self) -> None:
        """Initialize EvaporationOptions."""
        ...

    @overload
    def Tref(self) -> float:
        """Get reference temperature [K] for evaporation rate."""
        ...

    @overload
    def Tref(self, value: float) -> EvaporationOptions:
        """
        Set reference temperature [K] for evaporation rate.

        Args:
            value (float): Reference temperature [K]

        Returns:
            EvaporationOptions: class object for method chaining
        """
        ...

    @overload
    def Pref(self) -> float:
        """Get reference pressure [Pa] for evaporation rate."""
        ...

    @overload
    def Pref(self, value: float) -> EvaporationOptions:
        """
        Set reference pressure [Pa] for evaporation rate.

        Args:
            value (float): Reference pressure [Pa]

        Returns:
            EvaporationOptions: class object for method chaining
        """
        ...

    @overload
    def diff_c(self) -> List[float]:
        """Get diffusivities [m^2/s] at reference conditions."""
        ...

    @overload
    def diff_c(self, value: List[float]) -> EvaporationOptions:
        """
        Set diffusivities [m^2/s] at reference conditions.

        Args:
            value (list[float]): Diffusivities [m^2/s]

        Returns:
            EvaporationOptions: class object for method chaining
        """
        ...

    @overload
    def diff_T(self) -> List[float]:
        """Get diffusivity temperature exponents."""
        ...

    @overload
    def diff_T(self, value: List[float]) -> EvaporationOptions:
        """
        Set diffusivity temperature exponents.

        Args:
            value (list[float]): Temperature exponents

        Returns:
            EvaporationOptions: class object for method chaining
        """
        ...

    @overload
    def diff_P(self) -> List[float]:
        """Get diffusivity pressure exponents."""
        ...

    @overload
    def diff_P(self, value: List[float]) -> EvaporationOptions:
        """
        Set diffusivity pressure exponents.

        Args:
            value (list[float]): Pressure exponents

        Returns:
            EvaporationOptions: class object for method chaining
        """
        ...

    @overload
    def vm(self) -> List[float]:
        """Get molar volumes [m^3/mol]."""
        ...

    @overload
    def vm(self, value: List[float]) -> EvaporationOptions:
        """
        Set molar volumes [m^3/mol].

        Args:
            value (list[float]): Molar volumes [m^3/mol]

        Returns:
            EvaporationOptions: class object for method chaining
        """
        ...

    @overload
    def diameter(self) -> List[float]:
        """Get particle diameters [m]."""
        ...

    @overload
    def diameter(self, value: List[float]) -> EvaporationOptions:
        """
        Set particle diameters [m].

        Args:
            value (list[float]): Particle diameters [m]

        Returns:
            EvaporationOptions: class object for method chaining
        """
        ...

class Evaporation:
    """
    Evaporation rate kinetics model.

    This class implements cloud particle evaporation processes.
    """

    def __init__(self, options: EvaporationOptions) -> None:
        """
        Initialize Evaporation with kinetics options.

        Args:
            options (EvaporationOptions): Configuration options
        """
        ...

    def forward(
        self,
        temp: torch.Tensor,
        pres: torch.Tensor,
        conc: torch.Tensor,
        other: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute evaporation rates.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            pres (torch.Tensor): Pressure tensor [Pa]
            conc (torch.Tensor): Concentration tensor [mol/m^3]
            other (torch.Tensor): Additional state tensor

        Returns:
            torch.Tensor: Evaporation rates
        """
        ...

class KineticsOptions(SpeciesThermo):
    """
    Configuration options for chemical kinetics calculations.

    This class extends SpeciesThermo with parameters for kinetic
    reaction mechanisms.
    """

    def __init__(self) -> None:
        """Initialize KineticsOptions."""
        ...

    @staticmethod
    def from_yaml(filename: str) -> KineticsOptions:
        """
        Create KineticsOptions from a YAML file.

        Args:
            filename (str): Path to YAML file

        Returns:
            KineticsOptions: Configuration options
        """
        ...

    def reactions(self) -> List[Reaction]:
        """Get all kinetic reactions."""
        ...

    @overload
    def Tref(self) -> float:
        """Get reference temperature [K] for kinetics reactions."""
        ...

    @overload
    def Tref(self, value: float) -> KineticsOptions:
        """
        Set reference temperature [K] for kinetics reactions.

        Args:
            value (float): Reference temperature [K]

        Returns:
            KineticsOptions: class object for method chaining
        """
        ...

    @overload
    def Pref(self) -> float:
        """Get reference pressure [Pa] for kinetics reactions."""
        ...

    @overload
    def Pref(self, value: float) -> KineticsOptions:
        """
        Set reference pressure [Pa] for kinetics reactions.

        Args:
            value (float): Reference pressure [Pa]

        Returns:
            KineticsOptions: class object for method chaining
        """
        ...

    @overload
    def arrhenius(self) -> ArrheniusOptions:
        """Get options for Arrhenius rate constants."""
        ...

    @overload
    def arrhenius(self, value: ArrheniusOptions) -> KineticsOptions:
        """
        Set options for Arrhenius rate constants.

        Args:
            value (ArrheniusOptions): Arrhenius options

        Returns:
            KineticsOptions: class object for method chaining
        """
        ...

    @overload
    def coagulation(self) -> CoagulationOptions:
        """Get options for coagulation reactions."""
        ...

    @overload
    def coagulation(self, value: CoagulationOptions) -> KineticsOptions:
        """
        Set options for coagulation reactions.

        Args:
            value (CoagulationOptions): Coagulation options

        Returns:
            KineticsOptions: class object for method chaining
        """
        ...

    @overload
    def evaporation(self) -> EvaporationOptions:
        """Get options for evaporation reactions."""
        ...

    @overload
    def evaporation(self, value: EvaporationOptions) -> KineticsOptions:
        """
        Set options for evaporation reactions.

        Args:
            value (EvaporationOptions): Evaporation options

        Returns:
            KineticsOptions: class object for method chaining
        """
        ...

    @overload
    def evolve_temperature(self) -> bool:
        """Check if temperature evolution is enabled."""
        ...

    @overload
    def evolve_temperature(self, value: bool) -> KineticsOptions:
        """
        Enable/disable temperature evolution during kinetics.

        Args:
            value (bool): Whether to evolve temperature (default: False)

        Returns:
            KineticsOptions: class object for method chaining
        """
        ...

class Kinetics:
    """
    Chemical kinetics model for evolving reaction systems.

    This class integrates chemical kinetic equations to evolve
    species concentrations over time.
    """

    def __init__(self, options: KineticsOptions) -> None:
        """
        Initialize Kinetics with kinetics options.

        Args:
            options (KineticsOptions): Configuration options
        """
        ...

    def forward(
        self,
        temp: torch.Tensor,
        pres: torch.Tensor,
        conc: torch.Tensor
    ) -> torch.Tensor:
        """
        Evolve kinetic reactions.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            pres (torch.Tensor): Pressure tensor [Pa]
            conc (torch.Tensor): Concentration tensor [mol/m^3]

        Returns:
            torch.Tensor: Updated concentrations
        """
        ...

    def forward_nogil(
        self,
        temp: torch.Tensor,
        pres: torch.Tensor,
        conc: torch.Tensor
    ) -> torch.Tensor:
        """
        Evolve kinetic reactions without Python GIL.

        This method releases the Global Interpreter Lock for better
        parallel performance.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            pres (torch.Tensor): Pressure tensor [Pa]
            conc (torch.Tensor): Concentration tensor [mol/m^3]

        Returns:
            torch.Tensor: Updated concentrations
        """
        ...

    def jacobian(
        self,
        temp: torch.Tensor,
        conc: torch.Tensor,
        cvol: torch.Tensor,
        rate: torch.Tensor,
        rc_ddC: torch.Tensor,
        rc_ddT: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the Jacobian matrix for kinetics.

        Args:
            temp (torch.Tensor): Temperature tensor [K]
            conc (torch.Tensor): Concentration tensor [mol/m^3]
            cvol (torch.Tensor): Volumetric concentration tensor
            rate (torch.Tensor): Reaction rates
            rc_ddC (torch.Tensor): Rate derivatives w.r.t. concentration
            rc_ddT (torch.Tensor, optional): Rate derivatives w.r.t. temperature

        Returns:
            torch.Tensor: Jacobian matrix
        """
        ...

# Module-level functions
def species_names() -> List[str]:
    """
    Retrieve the list of species names.

    Returns:
        list[str]: List of species names
    """
    ...

def set_species_names(names: List[str]) -> List[str]:
    """
    Set the list of species names.

    Args:
        names (list[str]): List of species names

    Returns:
        list[str]: Updated list of species names
    """
    ...

def species_weights() -> List[float]:
    """
    Retrieve the list of species molecular weights.

    Returns:
        list[float]: List of species weights [kg/mol]
    """
    ...

def set_species_weights(weights: List[float]) -> List[float]:
    """
    Set the list of species molecular weights.

    Args:
        weights (list[float]): List of species weights [kg/mol]

    Returns:
        list[float]: Updated list of species weights
    """
    ...

def species_cref_R() -> List[float]:
    """
    Retrieve specific heat ratios for the reference state.

    Returns:
        list[float]: List of specific heat ratios
    """
    ...

def set_species_cref_R(cref_R: List[float]) -> List[float]:
    """
    Set specific heat ratios for the reference state.

    Args:
        cref_R (list[float]): List of specific heat ratios

    Returns:
        list[float]: Updated list of specific heat ratios
    """
    ...

def species_uref_R() -> List[float]:
    """
    Retrieve internal energies for the reference state.

    Returns:
        list[float]: List of internal energies
    """
    ...

def set_species_uref_R(uref_R: List[float]) -> List[float]:
    """
    Set internal energies for the reference state.

    Args:
        uref_R (list[float]): List of internal energies

    Returns:
        list[float]: Updated list of internal energies
    """
    ...

def species_sref_R() -> List[float]:
    """
    Retrieve entropies for the reference state.

    Returns:
        list[float]: List of entropies
    """
    ...

def set_species_sref_R(sref_R: List[float]) -> List[float]:
    """
    Set entropies for the reference state.

    Args:
        sref_R (list[float]): List of entropies

    Returns:
        list[float]: Updated list of entropies
    """
    ...

def set_search_paths(path: str) -> str:
    """
    Set the search paths for resource files.

    Args:
        path (str): The search paths

    Returns:
        str: The updated search paths

    Example:
        >>> import kintera
        >>> kintera.set_search_paths("/path/to/resource/files")
    """
    ...

def get_search_paths() -> str:
    """
    Get the search paths for resource files.

    Returns:
        str: The search paths

    Example:
        >>> import kintera
        >>> kintera.get_search_paths()
    """
    ...

def add_resource_directory(path: str, prepend: bool = True) -> str:
    """
    Add a resource directory to the search paths.

    Args:
        path (str): The resource directory to add
        prepend (bool): If True, prepend the directory. If False, append it (default: True)

    Returns:
        str: The updated search paths

    Example:
        >>> import kintera
        >>> kintera.add_resource_directory("/path/to/resource/files")
    """
    ...

def find_resource(filename: str) -> str:
    """
    Find a resource file from the search paths.

    Args:
        filename (str): The name of the resource file

    Returns:
        str: The full path to the resource file

    Example:
        >>> import kintera
        >>> path = kintera.find_resource("example.txt")
        >>> print(path)  # /path/to/resource/files/example.txt
    """
    ...

def evolve_implicit(
    rate: torch.Tensor,
    stoich: torch.Tensor,
    jacobian: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    Evolve the kinetics model via implicit integration.

    This function performs implicit time integration of chemical kinetics
    equations using the given reaction rates, stoichiometry, and Jacobian.

    Args:
        rate (torch.Tensor): The reaction rates
        stoich (torch.Tensor): The stoichiometric matrix
        jacobian (torch.Tensor): The Jacobian matrix
        dt (float): The time step for the evolution

    Returns:
        torch.Tensor: The concentration differences
    """
    ...

def relative_humidity(
    temp: torch.Tensor,
    conc: torch.Tensor,
    stoich: torch.Tensor,
    op: NucleationOptions
) -> torch.Tensor:
    """
    Calculate the relative humidity.

    Args:
        temp (torch.Tensor): Temperature tensor [K]
        conc (torch.Tensor): Concentration tensor [mol/m^3]
        stoich (torch.Tensor): Stoichiometric coefficients tensor
        op (NucleationOptions): Nucleation options

    Returns:
        torch.Tensor: Relative humidity tensor

    Examples:
        >>> from kintera import relative_humidity, ThermoOptions
        >>> temp = torch.tensor([300.0, 310.0, 320.0])
        >>> conc = torch.tensor([1.e-3, 2.e-3, 3.e-3])
        >>> stoich = thermo.get_buffer("stoich")
        >>> rh = relative_humidity(temp, conc, stoich, op.nucleation())
    """
    ...

# Constants submodule
class constants:
    """Physical constants used in kintera calculations."""

    Rgas: float  # Universal gas constant [J/(mol*K)]
    Avogadro: float  # Avogadro's number [1/mol]
