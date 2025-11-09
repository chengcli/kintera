"""
Test script to demonstrate stub file type hints.

This script can be used with IDEs and type checkers to verify
that the kintera.pyi stub file provides proper type information.
"""

# This file will be type-checked but not executed
# It demonstrates that IDEs/type checkers can understand the API

def test_species_thermo() -> None:
    """Test SpeciesThermo type hints."""
    from kintera import SpeciesThermo
    
    # Create instance
    st = SpeciesThermo()
    
    # Method chaining with proper types
    st2 = st.vapor_ids([1, 2, 3]).cloud_ids([4, 5]).cref_R([2.5, 2.7])
    
    # Get methods return correct types
    species_list: list[str] = st.species()
    vapor_list: list[int] = st.vapor_ids()
    cref_list: list[float] = st.cref_R()

def test_reaction() -> None:
    """Test Reaction type hints."""
    from kintera import Reaction
    
    # Create from equation
    rxn = Reaction("H2 + O2 => H2O2")
    
    # Get equation
    eq: str = rxn.equation()
    
    # Get reactants and products
    reactants: dict[str, float] = rxn.reactants()
    products: dict[str, float] = rxn.products()

def test_thermo_options() -> None:
    """Test ThermoOptions type hints."""
    from kintera import ThermoOptions, NucleationOptions
    
    # Create and configure
    opt = ThermoOptions()
    opt2 = opt.Tref(300.0).Pref(1.e5).max_iter(10).ftol(1.e-6)
    
    # Get values with correct types
    tref: float = opt.Tref()
    pref: float = opt.Pref()
    max_it: int = opt.max_iter()
    tolerance: float = opt.ftol()
    
    # Load from YAML
    opt3: ThermoOptions = ThermoOptions.from_yaml("config.yaml")

def test_thermox() -> None:
    """Test ThermoX type hints."""
    import torch
    from kintera import ThermoX, ThermoOptions
    
    options = ThermoOptions().Tref(300.0).Pref(1.e5)
    thermo = ThermoX(options)
    
    # Create tensors
    temp = torch.tensor([300.0, 310.0, 320.0])
    pres = torch.tensor([1.e5, 1.e6, 1.e7])
    xfrac = torch.tensor([0.1, 0.2, 0.3])
    
    # Forward pass returns tensor
    result: torch.Tensor = thermo.forward(temp, pres, xfrac)
    
    # Compute transformations
    result2: torch.Tensor = thermo.compute("X->Y", [xfrac])

def test_kinetics() -> None:
    """Test Kinetics type hints."""
    import torch
    from kintera import Kinetics, KineticsOptions
    
    options = KineticsOptions.from_yaml("kinetics.yaml")
    kinetics = Kinetics(options)
    
    temp = torch.tensor([300.0])
    pres = torch.tensor([1.e5])
    conc = torch.tensor([1.0, 2.0, 3.0])
    
    # Forward pass
    result: torch.Tensor = kinetics.forward(temp, pres, conc)
    
    # Forward without GIL
    result2: torch.Tensor = kinetics.forward_nogil(temp, pres, conc)

def test_module_functions() -> None:
    """Test module-level functions."""
    from kintera import (
        species_names, set_species_names,
        species_weights, set_species_weights,
        find_resource, add_resource_directory
    )
    
    # Get and set species
    names: list[str] = species_names()
    names2: list[str] = set_species_names(["H2", "O2", "N2"])
    
    # Get and set weights
    weights: list[float] = species_weights()
    weights2: list[float] = set_species_weights([2.0, 32.0, 28.0])
    
    # Resource paths
    path: str = find_resource("data.txt")
    updated_paths: str = add_resource_directory("/path/to/resources")

def test_constants() -> None:
    """Test constants submodule."""
    from kintera import constants
    
    # Access physical constants
    R: float = constants.Rgas
    Na: float = constants.Avogadro

if __name__ == "__main__":
    print("This script is for type checking demonstration only.")
    print("Run with: mypy examples/test_type_hints.py")
