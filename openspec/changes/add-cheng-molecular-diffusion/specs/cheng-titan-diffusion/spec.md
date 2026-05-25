## ADDED Requirements

### Requirement: Cheng Titan binary-diffusion tensor builder

The system SHALL provide
`kinetics_base_titan_cheng_diffusion(state, masses, *, n_total=None,
temperature=None)` returning a tensor of shape
`(ncol, nlyr, nspecies, nspecies)` whose only non-zero entries are on
the diagonal. The diagonal entry `D[c, l, i, i]` MUST equal the Cheng
2013 Titan formula

```
D_i(T, n) = 7.3e16 × T^0.75 / n × sqrt((1 + 28/m_i) / (1 + 28/16))
```

where `T` and `n` are the cell-centered temperature and total density
read from the state (or overridden by the optional `n_total` /
`temperature` kwargs for test injection), and `m_i` is the molecular
mass of species `i` in atomic mass units from the `masses` vector.

#### Scenario: diagonal binary-diffusion at face-relevant values

- **WHEN** `kinetics_base_titan_cheng_diffusion(state, masses)` is
  called on a state with `temperature[0, 18] = 67.0 K` and
  `density[0, 18] = 1.89e+12 cm^-3`, with `masses[hcn_idx] = 27.0`
- **THEN** the returned tensor satisfies
  `D[0, 18, hcn_idx, hcn_idx] = 7.3e16 × 67.0^0.75 / 1.89e+12 ×
  sqrt((1 + 28/27) / (1 + 28/16))` (to within `1e-6` relative tol)
- **AND** all off-diagonal entries `D[0, 18, i, j]` with `i != j` are
  exactly zero

#### Scenario: per-species mass extraction

- **WHEN** `masses` is constructed from `pun_metadata` via
  `_kinetics_base_species_mass_amu` for each species in the Titan
  network
- **THEN** the resulting vector MUST have shape `(nspecies,)`,
  positive entries for every gas-phase species, and explicit known
  values for canonical species: `m[H] = 1.0`, `m[H2] = 2.0`,
  `m[CH4] = 16.0`, `m[C2H2] = 26.0`, `m[N2] = 28.0`

### Requirement: Driver wires molecular diffusion behind env var

The system SHALL respect a `KINTERA_TITAN_MOLECULAR_DIFFUSION`
environment variable in `diagnostics/no_grain_stability.py`. When
set to `1`, the driver MUST pass `binary_diffusion` (from the Cheng
helper above) and `molecular_weights` to `build_transport_matrix`.
When unset or set to `0`, the driver MUST call
`build_transport_matrix` with eddy diffusion only, preserving the
current behavior.

#### Scenario: off by default

- **WHEN** the driver runs with `KINTERA_TITAN_MOLECULAR_DIFFUSION`
  unset
- **THEN** the `build_transport_matrix` call MUST have
  `binary_diffusion=None` and `molecular_weights=None`, and a
  startup-time log line MUST report
  `[setup] molecular diffusion: OFF (set KINTERA_TITAN_MOLECULAR_DIFFUSION=1 to enable)`

#### Scenario: on via env var

- **WHEN** `KINTERA_TITAN_MOLECULAR_DIFFUSION=1` is set
- **THEN** the `build_transport_matrix` call MUST be invoked with
  `binary_diffusion = kinetics_base_titan_cheng_diffusion(state,
  masses)` and the matching `molecular_weights` vector
- **AND** a startup-time log line MUST report
  `[setup] molecular diffusion: ON (Cheng Titan formula)`

### Requirement: Gravity-separation term remains active

The system SHALL keep the gravity-separation term from
`build_binary_diffusion_matrix(..., include_gravity=True)` enabled
whenever the binary-diffusion tensor is provided to
`build_transport_matrix`. The driver MUST NOT disable the gravity
term when molecular diffusion is on.

#### Scenario: heavier species drift downward

- **WHEN** molecular diffusion is enabled and a long-enough
  integration is run (NT=44 or more) on KB-Titan neutrals_only
- **THEN** the C2H6/CH4 mixing-ratio profiles at L20-L39 MUST show
  C2H6 (m=30) at a lower mixing ratio than CH4 (m=16) at the same
  altitudes, AND the slope of `log(mr)` vs altitude for each
  species MUST scale roughly with `1/m_i` (heavier species fall
  off faster with altitude)
