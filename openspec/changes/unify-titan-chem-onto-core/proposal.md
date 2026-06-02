## Why

The Titan/KINETICS-base (KB) case hand-rolls its own reaction-rate evaluator
(`kinetics_base/titan/atm2d_sources.linearize` + `atm2d/sources/combine.build_source_linearization`)
and never invokes the compiled core `kintera.Kinetics`/`Photolysis` engine. The
core engine is therefore dead weight for the flagship case — "we're doing nothing
kintera if we are not using our core engine." This refactor makes the Titan
chemistry run *through* the core engine, so kintera's kinetics/photochemistry code
is actually exercised and the Titan case stops being a parallel reimplementation.

## What Changes

- Add a **multi-range Arrhenius** rate option to core to represent KB's
  3-temperature-range constants (AK/AK2/AK3).
- Add a **KB → `KineticsOptions` translator** in Titan: `KBPunReaction` → core
  `Reaction` + Arrhenius/ThreeBody/Troe options, with `UPDATE_CHEMB` overrides
  baked into the translated parameters.
- Route **photolysis through core `Photolysis`**: Cheng catalog →
  `PhotolysisOptions` (wavelength, cross-sections, branches), folding in the
  validated c-/l- isomer strip + XSCN exclusion + IPHOTO policy; Titan supplies the
  attenuated actinic-flux field.
- Add a **thin Titan layer** for electron-impact and ion charge-balance on top of
  core `Kinetics` outputs (core stays neutral; ion/EI specifics stay in Titan).
- **BREAKING (internal):** delete the hand-rolled Python rate path
  (`atm2d_sources.linearize`, the titan `build_*_source_terms` rate evaluation);
  the atm2d implicit step calls `Kinetics`+`Photolysis`+gap-layer instead.
- Flip the core transport default `c_diffusion` → `mr_diffusion` (heavy-species
  settling discretization owned by core), after validating the non-Titan regression
  set; keep `c_diffusion` as an explicit option.
- Consolidate ~11 `KINTERA_*` env vars into `CoreConfig` + `TitanConfig` objects
  (thin env→config loader retained for diagnostics).

## Capabilities

### New Capabilities
- `core-multirange-arrhenius`: core Arrhenius rate model supporting multiple
  temperature ranges (KB AK/AK2/AK3), validated against KB's `ZK1/ZK2`.
- `titan-core-chemistry`: Titan thermal + photo chemistry translated to core
  `KineticsOptions`/`PhotolysisOptions` and evaluated via core `Kinetics`/
  `Photolysis`; the hand-rolled Python rate path is removed.
- `titan-ion-ei-layer`: thin Titan-side layer adding electron-impact source terms
  and ion charge-balance on top of the core engine's tendency/Jacobian.
- `core-transport-mr-default`: core vertical transport defaults to the mixing-ratio
  (mr) form, with the gravitational-settling discretization owned by core.
- `kintera-config-consolidation`: `CoreConfig` + `TitanConfig` replace scattered
  `KINTERA_*` environment-variable behavior switches.

### Modified Capabilities
- (none — `openspec/specs/` is currently empty; all are new.)

## Impact

- **Core (`python/atm2d/`, compiled kinetics):** new multi-range Arrhenius option;
  transport default flip + settling ownership; `Kinetics`/`Photolysis` become the
  evaluation path for Titan.
- **Titan (`python/kinetics_base/titan/`):** `source_terms`, `atm2d_sources`,
  `photochemistry`, `parsing`, `electron_impact`, `ion_chemistry`,
  `transport_diffusion` refactored into translator + gap-layer + coefficient
  provider; hand-rolled rate evaluator deleted.
- **Config:** `CoreConfig`/`TitanConfig` objects; env vars become a thin loader.
- **Regression gate (every stage):** per-reaction rate match KB 1.000 + moses00
  steady-state recipe ratios (CH3 L40 0.64, C6H6 1.80, C3H3 0.69). Baseline =
  the validated photo fix (tag `refactor-base`).
- **Risk:** transport default flip affects non-Titan cases (earth/jupiter YAML);
  the translator must preserve the validated KB rate match.
