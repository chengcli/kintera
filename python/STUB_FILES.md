# Stub Files for Type Hints and Documentation

This directory contains Python stub files (`.pyi`) that provide type hints and API documentation for the `kintera` package, which is implemented using pybind11 C++ bindings.

## What are Stub Files?

Stub files are special Python files with a `.pyi` extension that contain type signatures and documentation for modules without including the actual implementation. They are used by:

- **Type checkers** (mypy, pyright, etc.) to verify type correctness
- **IDEs** (VS Code, PyCharm, etc.) to provide autocomplete and documentation
- **Documentation generators** to extract API information

## Structure

- `kintera.pyi` - Main stub file containing all type signatures for the compiled pybind11 module
- `py.typed` - Marker file indicating this package supports type hints (PEP 561)

## Benefits of Separation

By separating the pybind11 implementation (`python/csrc/*.cpp`) from the API documentation (stub files), we achieve:

1. **Clear Documentation**: Type hints and docstrings are in pure Python syntax
2. **Better IDE Support**: IDEs can parse stub files to provide autocomplete
3. **Type Safety**: Static type checkers can verify correct usage
4. **Maintainability**: Documentation is separate from C++ binding code
5. **PEP 561 Compliance**: Package properly declares type information

## Usage

### For Users

The stub files are automatically included when you install the package. Your IDE will automatically use them for autocomplete and type checking:

```python
from kintera import ThermoOptions, ThermoX
import torch

# IDE will show autocomplete and parameter hints
options = ThermoOptions().Tref(300.0).Pref(1.e5)
thermo = ThermoX(options)

# Type checkers will verify correct types
temp = torch.tensor([300.0, 310.0])
pres = torch.tensor([1.e5, 1.e6])
xfrac = torch.tensor([0.1, 0.2])
result = thermo.forward(temp, pres, xfrac)  # result is torch.Tensor
```

### For Type Checking

Run mypy to verify type correctness:

```bash
mypy your_script.py
```

### For Developers

When modifying the C++ bindings in `python/csrc/`, remember to update the corresponding type signatures in `kintera.pyi`:

1. Add new classes or functions to the stub file
2. Update parameter types and return types
3. Include docstrings with examples
4. Test with `mypy` to ensure validity

## Type Checking Example

See `examples/test_type_hints.py` for a demonstration of how the stub file enables type checking:

```bash
cd examples
mypy test_type_hints.py
```

## Implementation Details

### C++ Bindings Location

The actual pybind11 implementation is in:
- `python/csrc/kintera.cpp` - Main module, SpeciesThermo, Reaction
- `python/csrc/pythermo.cpp` - ThermoOptions, ThermoX, ThermoY
- `python/csrc/pykinetics.cpp` - KineticsOptions, Kinetics, Arrhenius
- `python/csrc/pyconstants.cpp` - Physical constants

### Stub File Contents

The `kintera.pyi` stub file includes:

- All public classes with their methods
- Type signatures using Python's `typing` module
- Overloaded methods using `@overload` decorator
- Comprehensive docstrings with examples
- Module-level functions
- Constants submodule

### Package Configuration

The stub files are included in the package via `pyproject.toml`:

```toml
[tool.setuptools.package-data]
"kintera" = ["*.pyi", "py.typed", ...]
```

## References

- [PEP 561 - Distributing and Packaging Type Information](https://www.python.org/dev/peps/pep-0561/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [mypy - Optional Static Typing for Python](http://mypy-lang.org/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

## Contributing

When adding new features to the C++ bindings:

1. Implement the binding in the appropriate `python/csrc/*.cpp` file
2. Add the corresponding type signature to `kintera.pyi`
3. Include docstrings with parameter descriptions and examples
4. Run `mypy` to validate the stub file
5. Test that IDEs can properly autocomplete the new features

This separation ensures that the API remains well-documented and type-safe while maintaining the performance benefits of C++ implementation.
