## ADDED Requirements

### Requirement: Core and Titan config objects
Behavior switches currently driven by `KINTERA_*` environment variables SHALL be
expressed as explicit configuration objects: `CoreConfig` (e.g. transport form,
chemistry solver) and `TitanConfig` (e.g. network mode, photo-activation policy,
chemb-override toggle, EI scales).

#### Scenario: Behavior set via config object
- **WHEN** a run is configured with a `CoreConfig`/`TitanConfig` instance
- **THEN** transport form, solver, network mode, photo policy, chemb overrides, and EI scales are taken from the config without reading environment variables

#### Scenario: Photo policy fields replace env vars
- **WHEN** the photo activation policy (allow-radicals, include-XSCN) is set on `TitanConfig`
- **THEN** it controls photolysis-branch activation, replacing `KINTERA_TITAN_PHOTO_ALLOW_RADICALS` / `KINTERA_TITAN_PHOTO_INCLUDE_XSCN`

### Requirement: Thin env→config loader retained for diagnostics
A thin loader SHALL map the existing `KINTERA_*` environment variables onto the
config objects so diagnostic scripts keep working without code changes.

#### Scenario: Env var maps to config
- **WHEN** a known `KINTERA_*` variable is set and the loader runs
- **THEN** the corresponding config field is populated with the same effective behavior as before

#### Scenario: Default config reproduces the baseline
- **WHEN** no env vars are set and default `CoreConfig`/`TitanConfig` are used for moses00
- **THEN** the run reproduces the documented baseline SS ratios
