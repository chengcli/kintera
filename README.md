# KINTERA

**Atmospheric Chemistry and Thermodynamics Library**

KINTERA is a library for atmospheric chemistry and equation of state calculations, combining C++ performance with Python accessibility through pybind11 bindings.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Detailed Build Instructions](#detailed-build-instructions)
- [Testing](#testing)
- [Documentation](#documentation)
- [Dependencies](#dependencies)
- [Development](#development)
- [License](#license)

## Overview

KINTERA provides efficient implementations of:
- Chemical kinetics calculations
- Thermodynamic equation of state
- Phase equilibrium computations
- Atmospheric chemistry models

The library is written in C++17 with Python bindings, leveraging PyTorch for tensor operations and providing GPU acceleration support via CUDA.

## Features

- **High Performance**: C++17 core with optional CUDA support
- **Python Interface**: Full Python API via pybind11
- **PyTorch Integration**: Native tensor operations using PyTorch
- **Chemical Kinetics**: Comprehensive reaction mechanism support
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

The simplest way to build and install KINTERA:

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

## Testing

KINTERA includes both C++ and Python tests.

### Running All Tests

After building and installing:

```bash
cd build/tests
ctest
```

## Documentation

Full documentation is available at: [https://kintera.readthedocs.io](https://kintera.readthedocs.io)

To build documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

Documentation will be generated in `docs/_build/html/`.

### Dependency Cache

A successful build saves cache files for each dependency in the `.cache` directory. These cache files:
- Can be safely deleted at any time
- Allow offline builds after the first successful build
- Are automatically populated on first build

To force a clean rebuild:

```bash
rm -rf .cache build
```

## Development

### Code Style

KINTERA uses pre-commit hooks for code formatting and linting:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Project Structure

```
kintera/
├── src/              # C++ source code
├── python/           # Python bindings and API
│   ├── csrc/        # pybind11 binding code
│   └── api/         # Pure Python API
├── tests/            # C++ and Python tests
├── examples/         # Example usage
├── docs/             # Documentation source
├── cmake/            # CMake modules and macros
└── data/             # Test data and examples
```

### Continuous Integration

The project uses GitHub Actions for continuous integration. The CI pipeline:
1. Runs pre-commit checks (formatting, linting)
2. Builds on Linux and macOS
3. Runs all C++ and Python tests

### Continuous Deployment

Releases are automatically built and published to PyPI using GitHub Actions when a new release is created on GitHub.

## Staying Updated

If you have forked this repository, please enable notifications or watch for updates to stay current with the latest developments.

## License

See [LICENSE](LICENSE) file for details.

## Authors

- **Cheng Li** - [chengcli@umich.edu](mailto:chengcli@umich.edu)
