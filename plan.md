# Photochemistry Module Implementation Plan

## Overview

This plan outlines the implementation of a photochemistry module for JINKELA, following the established patterns from the kinetics module. The module computes photolysis rates based on wavelength-dependent cross-sections and actinic flux, with full PyTorch tensor support.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Photochemistry Module                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐ │
│   │ PhotolysisOpts  │───▶│  PhotolysisImpl │───▶│ Rate Constants │ │
│   └─────────────────┘    └─────────────────┘    └────────────────┘ │
│           │                      │                                   │
│           ▼                      ▼                                   │
│   ┌─────────────────┐    ┌─────────────────┐                        │
│   │ YAML Cross-Sec  │    │ Xsec Interpolate│                        │
│   │    Data         │    │                 │                        │
│   └─────────────────┘    └─────────────────┘                        │
│                                  │                                   │
│                                  ▼                                   │
│                          ┌─────────────────┐                        │
│                          │  Actinic Flux   │                        │
│                          │      Data       │                        │
│                          └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Phases with Checkpoint Tests

### Phase 1: Core Photolysis Module ✅

**Files Created:**
- `src/kinetics/photolysis.hpp` - Options and module header
- `src/kinetics/photolysis.cpp` - Implementation

**Key Components:**
- `PhotolysisOptionsImpl`: Configuration with wavelength grid, cross-sections, branches
- `PhotolysisImpl`: PyTorch module computing rates via wavelength integration
- YAML parsing for cross-section data (KINETICS7, VULCAN, inline formats)

**Checkpoint Test 1:** `tests/test_photolysis_options.cpp`
- Verify YAML parsing
- Validate cross-section loading
- Check wavelength grid setup

### Phase 2: Kinetics Integration ✅

**Files Modified:**
- `src/kinetics/kinetics.hpp` - Added `PhotolysisOptions photolysis` member
- `src/kinetics/kinetics_options.cpp` - Parse photolysis from YAML
- `src/kinetics/kinetics.cpp` - Register photolysis evaluator

**Key Integration Points:**
- `KineticsOptionsImpl::create()` initializes photolysis options
- `reactions()` includes photolysis reactions
- `reset()` registers photolysis rate evaluator

**Checkpoint Test 2:** `tests/test_photolysis_kinetics.cpp`
- Verify photolysis reactions register correctly
- Test stoichiometry matrix includes photolysis
- Validate total reaction count

### Phase 3: Actinic Flux Interface ✅

**Files Created:**
- `src/kinetics/actinic_flux.hpp` - Flux data structure and helpers

**Key Components:**
- `ActinicFluxData`: Wavelength/flux tensor storage
- `ActinicFluxOptions`: Configuration options
- Helper functions: `create_uniform_flux()`, `create_solar_flux()`

**Checkpoint Test 3:** `tests/test_actinic_flux.cpp`
- Test flux interpolation
- Verify tensor shapes
- Test edge cases (wavelength bounds)

### Phase 4: Python Bindings ✅

**Files Created:**
- `python/csrc/pyphotolysis.cpp` - pybind11 bindings

**Files Modified:**
- `python/csrc/kintera.cpp` - Register `bind_photolysis()`
- `python/kintera.pyi` - Type hints for photolysis classes

**Exposed Classes:**
- `PhotolysisOptions`
- `Photolysis`
- `ActinicFluxOptions`
- `ActinicFluxData`

**Checkpoint Test 4:** `tests/test_photolysis.py`
- Python-level integration test
- Verify options parsing
- Test forward pass from Python

### Phase 5: Jacobian and End-to-End Validation ✅

**Files Created:**
- `src/kinetics/jacobian_photolysis.cpp` - Jacobian computation

**Files Modified:**
- `src/kinetics/jacobian.hpp` - Added `jacobian_photolysis()` declaration

**Checkpoint Test 5:** `tests/test_ch4_photolysis.cpp`
- End-to-end CH4 photolysis test
- Jacobian computation and validation
- Multi-branch photolysis
- Cross-section interpolation
- Integration with Kinetics module

## File Summary

### New Files Created
| File | Description |
|------|-------------|
| `src/kinetics/photolysis.hpp` | Photolysis options and module header |
| `src/kinetics/photolysis.cpp` | Photolysis implementation |
| `src/kinetics/actinic_flux.hpp` | Actinic flux data structure |
| `src/kinetics/jacobian_photolysis.cpp` | Jacobian for photolysis |
| `python/csrc/pyphotolysis.cpp` | Python bindings |
| `tests/test_photolysis_options.cpp` | Phase 1 checkpoint test |
| `tests/test_photolysis_kinetics.cpp` | Phase 2 checkpoint test |
| `tests/test_actinic_flux.cpp` | Phase 3 checkpoint test |
| `tests/test_photolysis.py` | Phase 4 checkpoint test |
| `tests/test_ch4_photolysis.cpp` | Phase 5 checkpoint test |

### Modified Files
| File | Changes |
|------|---------|
| `src/kinetics/kinetics.hpp` | Added photolysis include and member |
| `src/kinetics/kinetics_options.cpp` | Parse photolysis from YAML |
| `src/kinetics/kinetics.cpp` | Register photolysis evaluator |
| `src/kinetics/jacobian.hpp` | Added jacobian_photolysis declaration |
| `python/csrc/kintera.cpp` | Call bind_photolysis() |
| `python/kintera.pyi` | Added photolysis type hints |
| `tests/CMakeLists.txt` | Added new test targets |

## Usage Example

### C++ Usage

```cpp
#include <kintera/kinetics/photolysis.hpp>
#include <kintera/kinetics/actinic_flux.hpp>

// Create options
auto opts = PhotolysisOptionsImpl::create();
opts->wavelength() = {100., 150., 200.};
opts->temperature() = {200., 300.};
opts->reactions().push_back(Reaction("CH4 => CH3 + H"));
opts->cross_section() = {1.e-18, 2.e-18, 1.e-18};
opts->branches().push_back({parse_comp_string("CH3:1 H:1")});

// Create module
Photolysis module(opts);
module->to(torch::kCUDA, torch::kFloat64);

// Create actinic flux
auto flux = create_solar_flux(100., 200., 11, 1.e14);

// Compute rates
auto rate = module->forward(temp, pres, conc, flux.to_map());
```

### Python Usage

```python
from kintera import (
    PhotolysisOptions, Photolysis, Reaction,
    create_solar_flux, set_species_names
)
import torch

# Initialize species
set_species_names(["CH4", "CH3", "H"])

# Create options
opts = PhotolysisOptions()
opts.wavelength([100., 150., 200.])
opts.reactions([Reaction("CH4 => CH3 + H")])
opts.cross_section([1e-18, 2e-18, 1e-18])

# Create module
module = Photolysis(opts)

# Create flux and compute rates
flux = create_solar_flux(100., 200., 11, 1e14)
rate = module.forward(temp, pres, conc, flux.to_map())
```

## Rate Calculation Formula

The photolysis rate constant is computed as:

$$k = \int_{\lambda_1}^{\lambda_2} \sigma(\lambda, T) \cdot F(\lambda) \, d\lambda$$

where:
- $\sigma(\lambda, T)$ is the temperature-dependent cross-section [cm² molecule⁻¹]
- $F(\lambda)$ is the actinic flux [photons cm⁻² s⁻¹ nm⁻¹]
- $\lambda$ is the wavelength [nm]

For multi-branch reactions, the effective stoichiometry is computed as the weighted average based on branch rates.

## Building and Testing

```bash
# Build
cmake -B build
cmake --build build --parallel

# Run C++ tests
cd build/tests
ctest -R photolysis

# Run Python tests
pytest tests/test_photolysis.py
```

