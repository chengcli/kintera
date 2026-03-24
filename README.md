# JINKELA: Joint INtegrated Kinetics for Exoplanetary and pLanetary Atmospheres

*A fork of [KINTERA](https://github.com/chengcli/kintera)*

**Atmospheric Chemistry and Thermodynamics Library**

KINTERA is a library for atmospheric chemistry and equation of state calculations, combining C++ performance with Python accessibility through pybind11 bindings.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Photochemistry Module](#photochemistry-module)
- [Testing](#testing)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## Overview

KINTERA provides efficient implementations of:
- Chemical kinetics calculations (Arrhenius, coagulation, evaporation)
- Photochemistry and photolysis reactions
- Thermodynamic equation of state
- Phase equilibrium computations
- Atmospheric chemistry models

The library is written in C++17 with Python bindings, leveraging PyTorch for tensor operations and providing GPU acceleration support via CUDA.

## Features

- **High Performance**: C++17 core with optional CUDA support
- **Python Interface**: Full Python API via pybind11
- **PyTorch Integration**: Native tensor operations using PyTorch
- **Chemical Kinetics**: Comprehensive reaction mechanism support
- **Photochemistry**: Wavelength-dependent photolysis with multi-branch products
- **Thermodynamics**: Advanced equation of state calculations
- **Cloud Physics**: Nucleation and condensation modeling

## Prerequisites

### System Requirements

- **C++ Compiler**: Support for C++17 (GCC 7+, Clang 5+, or MSVC 2017+)
- **CMake**: Version 3.18 or higher
- **Python**: Version 3.9 or higher
- **NetCDF**: NetCDF C library

### Python Dependencies

- `numpy`
- `torch` (version 2.7.0-2.7.1)
- `pyharp` (version 1.7.2+)
- `pytest` (for testing)

### Platform-Specific Setup

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libnetcdf-dev
```

#### macOS

```bash
brew update
brew install cmake netcdf
```

## Installation

### Quick Start

```bash
# 1. Install Python dependencies
pip install numpy 'torch==2.7.1' 'pyharp>=1.7.1'

# 2. Clone the repository
git clone https://github.com/chengcli/kintera.git
cd kintera

# 3. Configure and build the C++ library
cmake -B build
cmake --build build --parallel

# 4. Install the Python toolkit
pip install .
```

## Photochemistry Module

KINTERA includes a complete photochemistry module for modeling photolysis reactions in planetary atmospheres.

### Architecture

```
src/kinetics/
├── photolysis.hpp        # PhotolysisOptions and PhotolysisImpl definitions
├── photolysis.cpp        # Implementation with YAML parsing and rate computation
├── actinic_flux.hpp      # ActinicFluxData structure and helper functions
└── jacobian_photolysis.cpp  # Jacobian for implicit time integration
```

### Key Components

| Component | Description |
|-----------|-------------|
| `PhotolysisOptions` | Configuration: wavelength grid, cross-sections, branches |
| `Photolysis` | PyTorch module computing rates via wavelength integration |
| `ActinicFluxData` | Wavelength/flux tensor storage with interpolation |
| `jacobian_photolysis()` | Jacobian computation for implicit solvers |

### Rate Calculation

Photolysis rates are computed by integrating cross-sections weighted by actinic flux:

```
k = ∫ σ(λ,T) · F(λ) dλ
```

where σ is the cross-section [cm² molecule⁻¹], F is the actinic flux [photons cm⁻² s⁻¹ nm⁻¹], and λ is wavelength [nm].

### YAML Configuration

Photolysis reactions are defined in YAML format:

```yaml
reactions:
- equation: CH4 => CH3 + H + (1)CH2 + H2
  type: photolysis
  branches:
    - "CH4:1"           # photoabsorption
    - "CH3:1 H:1"       # CH3 + H branch
    - "(1)CH2:1 H2:1"   # singlet CH2 + H2 branch
  cross-section:
    - format: KINETICS7
      filename: "CH4.dat2"
    # Or inline YAML format:
    - format: YAML
      temperature-range: [0., 300.]
      data:
        - [100., 1.e-18, 0.5e-18]
        - [150., 2.e-18, 1.0e-18]
```

### C++ Usage

```cpp
#include <kintera/kinetics/photolysis.hpp>
#include <kintera/kinetics/actinic_flux.hpp>

// Create options
auto opts = PhotolysisOptionsImpl::create();
opts->wavelength() = {100., 150., 200.};
opts->reactions().push_back(Reaction("N2 => N2"));
opts->cross_section() = {1.e-18, 2.e-18, 1.e-18};

// Create module and move to GPU
Photolysis module(opts);
module->to(torch::kCUDA, torch::kFloat64);

// Create actinic flux
auto flux = create_solar_flux(100., 200., 11, 1.e14);

// Compute photolysis rates
auto rate = module->forward(temp, pres, conc, flux.to_map());
```

### Python Usage

```python
from kintera import (
    PhotolysisOptions, Photolysis, Reaction,
    create_solar_flux, set_species_names
)
import torch

# Initialize species list
set_species_names(["N2", "O2", "CH4"])

# Configure photolysis
opts = PhotolysisOptions()
opts.wavelength([100., 150., 200.])
opts.reactions([Reaction("N2 => N2")])
opts.cross_section([1e-18, 2e-18, 1e-18])

# Create module
module = Photolysis(opts)

# Create flux and compute rates
flux = create_solar_flux(100., 200., 11, 1e14)
rate = module.forward(temp, pres, conc, flux.to_map())
```

### Cross-Section File Formats

The module supports multiple cross-section formats:

| Format | Description |
|--------|-------------|
| `YAML` | Inline wavelength/cross-section data |
| `KINETICS7` | NCAR KINETICS7 format files |
| `VULCAN` | VULCAN photochemistry format |

## Testing

KINTERA includes comprehensive C++ and Python tests.

### Running All Tests

```bash
cd build/tests
ctest
```

### Photochemistry Tests

Run photochemistry-specific tests:

```bash
# C++ tests
cd build/tests
ctest -R photolysis

# Python tests
pytest tests/test_photolysis.py -v
```

### Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_photolysis_options.cpp` | YAML parsing, cross-section loading |
| `test_photolysis_kinetics.cpp` | Kinetics integration, stoichiometry |
| `test_actinic_flux.cpp` | Flux interpolation, tensor shapes |
| `test_ch4_photolysis.cpp` | End-to-end CH4 photolysis, Jacobian |
| `test_photolysis.py` | Python bindings integration |

## Documentation

Full documentation is available at: [https://kintera.readthedocs.io](https://kintera.readthedocs.io)

To build documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

### Dependency Cache

A successful build saves cache files in `.cache/`. To force a clean rebuild:

```bash
rm -rf .cache build
```

## Development

### Project Structure

```
kintera/
├── src/
│   ├── kinetics/       # Kinetics modules (Arrhenius, photolysis, etc.)
│   ├── thermo/         # Thermodynamics
│   ├── xsection/       # Cross-section loading
│   └── math/           # Interpolation utilities
├── python/
│   ├── csrc/           # pybind11 bindings
│   ├── kintera.pyi     # Type stubs
│   └── py.typed        # PEP 561 marker
├── tests/              # C++ and Python tests
├── examples/           # Usage examples
└── data/               # Test data (cross-sections, YAML configs)
```

### Code Style

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Type Hints

KINTERA provides full type hint support through Python stub files:
- IDE autocomplete in VS Code, PyCharm
- Type checking with `mypy` or `pyright`

See [`python/STUB_FILES.md`](python/STUB_FILES.md) for details.

### Continuous Integration

GitHub Actions CI pipeline:
1. Pre-commit checks (formatting, linting)
2. Build on Linux and macOS
3. Run all C++ and Python tests

## License

See [LICENSE](LICENSE) file for details.

## Authors

- **Cheng Li** - [chengcli@umich.edu](mailto:chengcli@umich.edu)
