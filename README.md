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

# 2. Configure and build the C++ library
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build --parallel

# 3. Install the Python package
pip install .
```

### Detailed Build Instructions

#### Step 1: Clone the Repository

```bash
git clone https://github.com/chengcli/kintera.git
cd kintera
```

#### Step 2: Install Python Dependencies

```bash
pip install numpy 'torch==2.7.1' 'pyharp>=1.7.1'
```

#### Step 3: Configure CMake

```bash
cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=OFF
```

**Configuration Options:**

- `-DCMAKE_BUILD_TYPE`: Build type (`Release`, `Debug`, `RelWithDebInfo`)
- `-DBUILD_TESTS`: Enable/disable test building (`ON`/`OFF`)
- `-DBUILD_EXAMPLES`: Enable/disable examples (`ON`/`OFF`)
- `-DCUDA`: Enable CUDA support (`ON`/`OFF`, default: `OFF`)

#### Step 4: Build the C++ Library

```bash
cmake --build build --config Release --parallel
```

The `--parallel` flag uses all available CPU cores for faster compilation.

#### Step 5: Install the Python Package

```bash
pip install .
```

This command builds the Python extension and installs the `kintera` package in your Python environment.

**Alternative: Development Installation**

For development work where you want changes to take effect immediately:

```bash
pip install -e .
```

**Alternative: Build Wheel**

To create a distributable wheel file:

```bash
python -m build --wheel
pip install dist/*.whl
```

## Testing

KINTERA includes both C++ and Python tests.

### Running All Tests

After building and installing:

```bash
cd build/tests
ctest --output-on-failure
```

### Running C++ Tests Only

```bash
cd build/tests
ctest -R test_thermo --output-on-failure
ctest -R test_kinetics --output-on-failure
```

### Running Python Tests Only

```bash
cd build/tests
pytest test_earth.py -v
```

Or from the repository root:

```bash
pytest tests/ -v
```

### Test Coverage

The test suite includes:
- **C++ Unit Tests**: Core library functionality (thermodynamics, kinetics, reactions)
- **Python Integration Tests**: Python bindings and API functionality
- **Example Tests**: Verification of example calculations

## Documentation

Full documentation is available at: [https://kintera.readthedocs.io](https://kintera.readthedocs.io)

To build documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

Documentation will be generated in `docs/_build/html/`.

## Dependencies

KINTERA automatically downloads and builds most dependencies during the CMake configuration. Internet connection is required for the first build.

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [fmt](https://github.com/fmtlib/fmt) | 11.1.2 | String formatting |
| [yaml-cpp](https://github.com/jbeder/yaml-cpp) | 0.8.0 | YAML parsing |
| [googletest](https://github.com/google/googletest) | 1.13.0 | C++ testing framework |
| [pyharp](https://github.com/chengcli/pyharp) | 1.3.1+ | Atmospheric data handling |
| [torch](https://pytorch.org/) | 2.7.0-2.7.1 | Tensor operations |

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

### Building Documentation

Documentation is built using Sphinx:

```bash
cd docs
pip install -r requirements.txt
make html
```

### Continuous Integration

The project uses GitHub Actions for continuous integration. The CI pipeline:
1. Runs pre-commit checks (formatting, linting)
2. Builds on Linux and macOS
3. Runs all C++ and Python tests

## Troubleshooting

### Common Issues

#### NetCDF Not Found

If CMake cannot find NetCDF:

```bash
# Linux
export NC_HOME=/usr

# macOS (Homebrew)
export NC_HOME=/opt/homebrew
```

#### PyTorch ABI Compatibility

KINTERA automatically detects the PyTorch ABI setting. If you encounter linking errors, ensure your PyTorch installation is compatible with your system's C++ standard library.

#### Build Failures

For build issues:

1. Clear the build cache:
   ```bash
   rm -rf build .cache
   ```

2. Verify all prerequisites are installed

3. Check the [GitHub Issues](https://github.com/chengcli/kintera/issues) for similar problems

## Staying Updated

If you have forked this repository, please enable notifications or watch for updates to stay current with the latest developments.

## License

See [LICENSE](LICENSE) file for details.

## Authors

- **Cheng Li** - [chengcli@umich.edu](mailto:chengcli@umich.edu)
- **Sihe Chen** - [sihechen@caltech.edu](mailto:sihechen@caltech.edu)

## Citation

If you use KINTERA in your research, please cite:

```bibtex
@software{kintera,
  title = {KINTERA: Atmospheric Chemistry and Thermodynamics Library},
  author = {Li, Cheng and Chen, Sihe},
  url = {https://github.com/chengcli/kintera},
  year = {2024}
}
```

## Acknowledgments

This project builds upon several open-source libraries and acknowledges the contributions of the atmospheric science and computational chemistry communities.
