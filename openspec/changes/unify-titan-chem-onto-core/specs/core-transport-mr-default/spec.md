## ADDED Requirements

### Requirement: mixing-ratio transport is the core default
The core vertical transport operator SHALL default to the mixing-ratio (mr) form
`∂z(K·n·∂z(c/n))`; the concentration form `c_diffusion` SHALL remain selectable
explicitly.

#### Scenario: Default with no form specified
- **WHEN** a transport matrix is built without an explicit `form`
- **THEN** the mr-form is used

#### Scenario: c_diffusion still selectable
- **WHEN** `form="c_diffusion"` is requested explicitly
- **THEN** the operator uses the concentration form unchanged

#### Scenario: Non-Titan regression set unaffected by the flip
- **WHEN** the existing core/test cases (e.g. `tests/test_atm2d.py`, earth/jupiter mechanisms) are run after the default flip
- **THEN** they pass (the flip is validated before it is made the default)

### Requirement: Gravitational settling discretization owned by core
The molecular-diffusion gravitational-settling discretization SHALL live in the
core transport module, consuming a binary-diffusion coefficient field and molecular
weights supplied by the case layer.

#### Scenario: Coefficient provided as a field
- **WHEN** the core transport operator is built with a `binary_diffusion` field and `molecular_weights`
- **THEN** it computes the settling contribution without embedding any case-specific (Cheng/Moses) coefficient formula

#### Scenario: Diagnostic probe removed
- **WHEN** the refactor is complete
- **THEN** the `KINTERA_GRAV_MASS` diagnostic env-var probe no longer exists in the transport module
