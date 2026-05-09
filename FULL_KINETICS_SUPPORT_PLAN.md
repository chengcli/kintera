# Full KINETICS-base Titan Network Support Plan

## Goal

Adapt Kintera so it can import and run the full Titan chemistry network from
`cshsgy/KINETICS-base` and numerically match the upstream Fortran model for a
single fresh-start Titan step under the same input files.

The target upstream data set is:

- `examples/titan/kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun`
- `examples/titan/kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.special`
- `examples/titan/ions_c6h7+_H2CN.inp-1`
- `examples/titan/titan_Cheng_N_ions_H2CN.bc_save`
- `examples/titan/kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz`
- `examples/titan/Cheng_catalog_v4.dat`
- `examples/titan/Cheng_cross/`
- radiation auxiliaries used through Fortran units `fort.20`, `fort.21`,
  `fort.27`, `fort.45`, `fort.46`, and `fort.47`

The immediate milestone is not "full convergence" or long Titan evolution. It
is one-step equivalence:

1. Run upstream `titan.release` from a fresh-start patched run input
   (`ISTART=0`).
2. Build the same selected Kintera state and operators from the same files.
3. Advance exactly one comparable step.
4. Compare selected species number-density profiles against
   `kintitan.out.pun`.

## Current Status

### Completed

- A dedicated branch, `full-kinetics-support`, exists and has been rebased on
  current `main`.
- Optional external KINETICS-base tests are gated by:
  - `KINTERA_KINETICS_BASE_ROOT`
  - `KINTERA_KINETICS_BASE_EXECUTABLE`
- The external Fortran Titan executable can be run from tests with the correct
  scratch `fort.*` wiring and a patched fresh-start run input.
- The `.pun` smoke parser can load the full Titan selected network:
  - `NATOM=8`
  - `NMOL=268`
  - `NREACT=2139`
  - `NPART=517`
- The run-input selection parser can read:
  - `NFIX=8`
  - `NVARYS=0`
  - `NVARYF=120`
  - `NPHOTO=9`
  - selected photolysis ids: `221 222 223 224 225 226 227 228 245`
- The catalog/cross-section smoke path resolves all 522 entries in
  `Cheng_catalog_v4.dat`.
- The atmosphere/profile parser can load the Titan atmosphere file:
  - 50 altitude levels
  - density, temperature, pressure, eddy diffusion, and wind profiles
  - 128 selected species profiles
- Python exposes `parse_kinetics_base_atmosphere`.
- A Python one-step equivalence test exists:
  `test_external_titan_one_step_equivalence_if_available`.
  It runs Fortran and then constructs the matching Kintera Titan state from the
  same atmosphere and boundary files.
- The first one-step output target now hard-passes against `kintitan.out.pun`
  within the printed precision of the upstream `.pun` output.

Current accepted one-step tolerance:

```text
rtol=5.0e-4
atol=1.0e-6
```

This tolerance is set by the limited significant digits written in the Fortran
`.pun` output. The large previous mismatch was traced to state-conversion bugs:
special fixed profiles such as `U` and `SGA` were incorrectly treated as mixing
ratios, `ILOWER=5` lower-boundary mixing ratios were not applied, and
zero-density top layers were not zeroed.

### Important Correction

The previous `KBPunReaction::reaction_type` field was misnamed. In the upstream
Fortran read format, the first two integers after the reaction id are:

```text
NOREACT, NOPROD
```

They are the number of reactants and products, not a kinetic class. For the
Titan `.pun`, the observed distribution is:

```text
NOREACT=1: 319 reactions
NOREACT=2: 1698 reactions
NOREACT=3: 122 reactions
```

Photolysis membership comes from `IPHOTO,IPHOTS,IPHOTR,IPHOTD` in the run input,
not from this field. The branch now stores this value as `n_reactants` and also
parses participant slots plus KINETICS-base rate blocks, but classification into
Kintera reaction option classes is still pending.

## How Far We Are From One-Step Match

### Already usable for matching

- External Fortran oracle execution.
- Full Titan file discovery and scratch runtime wiring.
- Parser-level counts for `.pun`, run input, catalog, cross sections, and
  atmosphere profiles.
- Kintera 1D transport matrix construction from the parsed atmosphere.
- A diagnostic comparison harness that can become a hard assertion later.

### Partially usable

- `.pun` reactions: species ids and expanded stoichiometry are parsed, but rate
  blocks, coefficient chars, temperature ranges, source ids, and semantics are
  not structured.
- Atmosphere profiles: numeric sections are parsed, but conversion between
  KINETICS-base input values, internal number densities, output number
  densities, and fixed/background species is still ad hoc.
- Photolysis catalog: files resolve and parse, but selected Titan photolysis
  reactions are not assembled into `PhotoChemOptions` from `.pun` + run input.
- Python API: enough exists for the diagnostic, but there is no high-level
  `from_kinetics_base_titan` importer.

### Not implemented yet

- Species initialization from `.pun` species and thermo blocks.
- Proper `KBPunReaction` structure for the Fortran line format:
  - `NOREACT`
  - `NOPROD`
  - seven participant slots `(ICOFT, INDXT, CHART)`
  - three kinetic rate blocks
  - source traceability
- Mapping `.pun` thermal reactions into Kintera reaction option classes.
- Reading and applying `.special` reaction behavior.
- Building selected-only versus all-species Kintera state consistently.
- Fixed species and grouped species semantics.
- Boundary-condition import from `titan_Cheng_N_ions_H2CN.bc_save`.
- Radiation/actinic-flux equivalence with KINETICS-base.
- A Kintera time-step path that matches Fortran `NTIME=1`, `DELTIM`, internal
  chemistry solve, transport update, output timing, and floor/clamp policy.
- A hard one-step equality/close assertion.

## Main Technical Gaps

### 1. `.pun` Reaction Records Are Underparsed

The upstream Fortran reads each `.pun` reaction with:

```fortran
NOREACT(I), NOPRODT(I),
(ICOFT(I,J), INDXT(I,J), CHART(I,J), J=1,7),
AKT(I), BKT(I), CKT(I), DKT(I), EKT(I), FKT(I), TLT(I), THT(I), FCT(I),
TINT1(I), TOUTT1(I),
AKT2(I), BKT2(I), ...
AKT3(I), BKT3(I), ...
```

Kintera now keeps:

- reaction id
- `n_reactants`
- product count
- structured participant slots
- parsed rate blocks
- expanded reactant ids
- expanded product ids
- raw line

This is enough to start reaction classification, but not enough yet to compute
rates because KINETICS-base rate semantics and special reaction behavior are not
mapped into Kintera option classes.

Required fix:

- Keep `n_reactants` and `n_products` aligned with the Fortran `NOREACT` and
  `NOPROD` fields.
- Use participant records:

```cpp
struct KBPunParticipant {
  int coefficient = 1;
  int species_id = 0;
  char marker = ' ';
};
```

- Use rate block records:

```cpp
struct KBPunRateBlock {
  double A = 0.0;
  double b = 0.0;
  double C = 0.0;
  double D = 0.0;
  double E = 0.0;
  double F = 0.0;
  double Tmin = 0.0;
  double Tmax = 0.0;
  double Fc = 1.0;
  double Tin = 0.0;
  double Tout = 0.0;
};
```

- Preserve source line and original KINETICS-base id.
- Add more tests against one representative reaction of each observed full
  Titan form.

Acceptance criteria:

- Full Titan `.pun` still loads 2139 reactions.
- Rate block 1/2/3 values round-trip from raw lines for at least three known
  reaction ids.
- Selected photolysis ids are recognized as photolysis by run input selection,
  not by `NOREACT`.

### 2. Kinetic Classification Is Missing

Kintera must classify reactions based on the selected run input plus `.pun`
semantics:

- selected photolysis ids become `PhotoChemOptions` reactions
- normal thermal reactions become `ArrheniusOptions`
- third-body forms become `ThreeBodyOptions`
- pressure-dependent low/high forms become `LindemannFalloffOptions` first
- unsupported `.special`, particle, surface, or grain reactions must be
  reported explicitly

The first pass should be conservative. Do not silently skip anything required
for one-step match.

Acceptance criteria:

- An import report prints counts:
  - total `.pun` reactions
  - selected photolysis reactions
  - thermal reactions converted by class
  - reactions skipped with reason
  - reactions blocked by unsupported special behavior
- Full Titan selected network can be converted in "report-only" mode without
  throwing.
- Strict mode throws if any reaction needed by the selected one-step comparison
  is unsupported.

### 3. Species Ordering And Fixed Species Need A Policy

The run input selects 128 profiles:

- 8 fixed species:
  `JDUST`, `N2`, `E`, `PROD`, `U`, `RAYEAR`, `SGA`, `M`
- 120 varying-fast species

The `.pun` contains 268 species. Kintera needs a consistent state ordering for:

- chemistry modules
- transport solver
- output comparison
- fixed/background species
- species used in reactions but not evolved

Recommended policy for the first matching milestone:

1. Build a Kintera chemistry option set with all species required by converted
   reactions.
2. Build the one-step state with the 128 run-selected species in KINETICS-base
   order.
3. Keep fixed species in the state tensor but mark them as fixed for the update
   and comparison.
4. Do not remove species from reaction stoichiometry until an explicit
   selected-only reduction exists.

Acceptance criteria:

- Import report exposes:
  - `.pun` id to name
  - `.pun` id to Kintera state index
  - fixed/varying flags
  - output comparison order
- The diagnostic can compare only the species that Fortran writes to
  `kintitan.out.pun`.
- Fixed species differences are either excluded from evolved-species tolerance
  or checked against the Fortran fixed-species policy explicitly.

### 4. Atmosphere Unit Conversion Is Now Partially Owned By The Importer

The input atmosphere file mixes quantities that behave like number densities,
mixing ratios, and special placeholders. The Fortran output writes number
densities. The Python importer now owns the initial Titan conversion policy used
by the one-step equivalence test.

Required fix:

- Continue hardening `KBTitanState`, which currently produces:
  - `temperature`
  - `pressure`
  - number-density concentration tensor
  - density/background profile
  - eddy diffusion profile
  - wind profile
  - per-species input-unit metadata
- Cross-check this conversion against the Fortran first output before chemistry
  by using species where output is known to be direct density conversion.
- Decide and document how to handle special profiles using `.pun` species
  metadata rather than name-only Titan special cases:
  - fixed species with `molecular_weight <= 0`
  - empty elemental composition
  - names ending in `*`, such as excited/pseudo species
  - Titan placeholders such as `M`, `U`, `RAYEAR`, `PROD`, `JDUST`, and grain
    placeholders such as `GH`, `GCH4`, `GC2H2`, while avoiding a blanket
    `molecular_weight <= 0` rule for normal gas species such as `CH4`

Acceptance criteria:

- Done: conversion decisions are in `build_kinetics_base_titan_state`, not in the
  test body.
- Done: Titan state construction can take the `.pun` file and derive
  number-density/special profile handling from species metadata. This covers
  `U`, `SGA`, `M`, `RAYEAR`, `JDUST`, `PROD`, and `*` species without relying
  only on a hardcoded name list.
- Done: lower `ILOWER=5` mixing-ratio boundaries are applied for selected
  species such as `CH4`.
- Done: zero-density top layers are zeroed to match Fortran output behavior.
- Remaining: generalize boundary conversion beyond the current lower
  mixing-ratio case.

### 5. Boundary Conditions Are Partially Imported

The Fortran run uses `fort.15`:

```text
examples/titan/titan_Cheng_N_ions_H2CN.bc_save
```

Kintera now applies the lower `ILOWER=5` mixing-ratio boundary needed by the
current one-step output comparison. Full boundary-condition semantics are not
implemented yet.

Required fix:

- Parse the boundary file for the selected transport species.
- Map lower/upper boundary kinds and values into
  `SpeciesBoundaryConditions2D`.
- Match KINETICS-base semantics for fixed flux, fixed concentration, no-flux,
  deposition/escape if present.

Acceptance criteria:

- Done for current one-step output target: lower `ILOWER=5` boundary values are
  applied during Titan state construction.
- Remaining: parse all lower/upper boundary kinds into
  `SpeciesBoundaryConditions2D`.
- Remaining: quantify transport-only mismatch with full boundary semantics.

### 6. Photolysis And Radiation Are Not Equivalent

The selected Titan run has 9 photolysis reactions, but Kintera currently does
not build selected Titan `PhotoChemOptions` from the `.pun` network and full
catalog. The Fortran executable also computes radiation using its wavelength,
flux, aerosol, and diffusion-radiation files.

Required fix:

- Build selected photolysis reactions from the run input ids.
- Map each selected `.pun` photolysis id to catalog branch/cross-section data.
- Combine absorption and branching ratios on the same wavelength grid.
- Produce an actinic-flux tensor consistent with the Fortran first step.
- Initially support a "Fortran-provided radiation oracle" mode if direct
  radiation reproduction is too large for the first match.

Acceptance criteria:

- `PhotoChemOptions` contains exactly the 9 selected photolysis reactions.
- Cross-section arrays are non-empty and finite for every selected reaction.
- A photolysis-only diagnostic reports rates from Kintera and Fortran-side
  `prod+loss`/debug outputs when available.

### 7. Time-Stepping Semantics Are Not Matched

The current diagnostic uses `dt=1.0e-15` only as a placeholder. The Fortran run
input includes:

```text
DELTIM=-1.0E-15
NTIME=1
ITRY=5
ISTART=0
```

Kintera must reproduce the same meaning, not just use the same absolute value.

Required fix:

- Parse run timing fields into structured data.
- Identify whether negative `DELTIM` triggers internal auto-step behavior in
  KINETICS-base.
- Match the Fortran one-step update order:
  - read/convert atmosphere
  - update radiation
  - chemistry rates
  - transport/boundary update
  - output/flooring/clamping
- Decide whether the first hard match compares:
  - rates/tendencies before stepping
  - one implicit chemistry step
  - one full chemistry + transport step

Recommended first hard target:

1. Match parsed initial number-density state.
2. Match thermal chemistry rate vector for one altitude level with photolysis
   disabled.
3. Match photolysis rate vector for one altitude level with chemistry disabled.
4. Match full one-step output.

## Updated Implementation Roadmap

### Phase A: Lock The Oracle And Diagnostics

Status: mostly complete, with one improvement needed.

Tasks:

1. Done: replace the previous xfail diagnostic with
   `test_external_titan_one_step_equivalence_if_available`.
2. Done: store diagnostic metadata in assertion messages:
   - max absolute difference
   - max relative difference on nonzero reference entries
   - number of changed entries
   - top 10 species by max difference
3. Done: ensure the C++ and Python external tests use the same `fort.*` wiring.

Acceptance criteria:

- Default tests pass without external data.
- External one-step equivalence test hard-passes within `.pun` output precision.
- The diagnostic can identify which species dominate the mismatch.

### Phase B: Rebuild `.pun` Reaction Model

Status: partially implemented.

Tasks:

1. Done: rename `KBPunReaction::reaction_type` to `n_reactants`.
2. Done: replace expanded-only participant parsing with structured participant
   slots.
3. Done: parse rate blocks from the Fortran line format.
4. Done: preserve expanded reactant/product ids as derived convenience fields.
5. Next: add `classify_kinetics_base_pun_reaction(...)` with report-only output.
6. Next: add broader full-Titan fixture assertions for representative reaction
   ids and nonzero multi-block rates.

Acceptance criteria:

- Parser tests cover one `NOREACT=1`, one `NOREACT=2`, and one `NOREACT=3`
  Titan line.
- Existing full Titan parser count test still passes.
- Import report shows a non-empty classification summary.

### Phase C: Build Titan Species And State Import

Status: partially implemented and used by the hard one-step output test.

Tasks:

1. Done: add `KBTitanState` with selected species names, flags, profiles, and
   tensors.
2. Done: move `_initial_concentration_guess` out of the test and into the
   importer as `build_kinetics_base_titan_state`.
3. Partially done: implement explicit conversion rules for mixing-ratio versus number-density
   profile sections.
4. Done: expose Python API for constructing an `AtmState2D` from Titan files.
5. Remaining: add fixed-species mask support for future chemistry/transport
   diagnostics.

Acceptance criteria:

- Done: the 128 selected species are emitted in KINETICS-base order.
- Done: the state builder creates finite `AtmState2D` tensors with shape
  `(1, 50, 128)`.
- Done: conversion decisions are visible in the returned state object.

### Phase D: Convert Thermal Chemistry

Status: not implemented for `.pun`.

Tasks:

1. Initialize species and thermo data from `.pun`, not from the master fixture.
2. Convert ordinary thermal reactions to `ArrheniusOptions`.
3. Convert third-body reactions to `ThreeBodyOptions`.
4. Convert low/high pressure forms to `LindemannFalloffOptions`.
5. Add unsupported-reaction diagnostics for special/grain/particle forms.
6. Validate units against current master importer conventions.

Acceptance criteria:

- `KineticsOptions.from_kinetics_base_titan(..., photolysis=False)` constructs.
- `Kinetics` can run a finite forward pass for all selected altitude levels.
- A thermal-only rate diagnostic can compare at least one altitude level against
  Fortran-derived rates or a trusted local reimplementation.

### Phase E: Convert Selected Photolysis

Status: catalog smoke parsing exists, selected photolysis import does not.

Tasks:

1. Build photolysis reactions from selected ids `221-228,245`.
2. Match selected reactions to catalog entries and cross-section files.
3. Interpolate onto Kintera's wavelength grid or import the KINETICS-base grid.
4. Add branch handling for product channels.
5. Add an option to feed Fortran-equivalent actinic flux into Kintera.

Acceptance criteria:

- `PhotoChemOptions.from_kinetics_base_titan(...)` constructs with 9 reactions.
- Photolysis rates are finite for the Titan atmosphere.
- Selected photolysis rate diagnostics are available separately from thermal
  chemistry.

### Phase F: Import Boundary And Transport Semantics

Status: not implemented.

Tasks:

1. Parse `titan_Cheng_N_ions_H2CN.bc_save`.
2. Map lower/upper boundary conditions to Kintera boundary condition objects.
3. Use parsed eddy diffusion and wind consistently.
4. Decide whether Kintera's 1D transport operator needs a KINETICS-base
   compatibility mode for grid staggering or flux definitions.

Acceptance criteria:

- Transport-only diagnostic runs with KINETICS-base boundary conditions.
- Transport-only mismatch is quantified separately from chemistry mismatch.

### Phase G: Assemble One-Step Compatibility Driver

Status: not implemented.

Tasks:

1. Add high-level Python API:

```python
state, kinetics, photo, report = kt.from_kinetics_base_titan(
    root="...",
    fresh_start=True,
)
```

2. Add an explicit one-step function:

```python
next_state, step_report = kt.step_kinetics_base_titan_once(
    state,
    kinetics=kinetics,
    photo_chem=photo,
    report=report,
)
```

3. Done for state construction: the current one-step test uses
   `build_kinetics_base_titan_state` instead of constructing profiles inline.
4. Tighten the remaining chemistry/radiation work into staged assertions:
   - parsed-state assertion
   - thermal-rate assertion
   - photolysis-rate assertion
   - transport assertion
   - full output assertion

Acceptance criteria:

- The external diagnostic no longer contains importer policy.
- Every mismatch belongs to a named report section.
- The first state/boundary one-step output test is non-xfail.

## Testing Strategy

### Default Tests

These must not require the external KINETICS-base checkout:

- minimal master importer tests
- minimal `.pun` and run-input fixture tests
- minimal atmosphere fixture tests
- Python package import and binding tests
- Kzz diffusion smoke test using small fixture species

### Optional External Tests

These run only with:

```text
KINTERA_KINETICS_BASE_ROOT=/path/to/KINETICS-base
KINTERA_KINETICS_BASE_EXECUTABLE=/path/to/titan.release
```

They should include:

- parser count smoke test
- Fortran first-step execution test
- hard state/boundary one-step equivalence test
- future chemistry/radiation one-step diagnostics as those modules are imported

### Match Tolerances

Use staged tolerances:

1. Parser/state identity:
   - exact counts and species names
   - exact selected ids
2. Initial concentration conversion:
   - exact or near-exact for finite non-updated profiles after documented unit
     conversion
3. Rate-level comparison:
   - relative tolerance first, with absolute floor for tiny radical/ion entries
4. Full one-step output:
   - start with per-species diagnostics
   - only set global hard tolerances after rate and transport components are
     independently understood

## Open Design Decisions

1. Should the production importer build all 268 `.pun` species, or a selected
   128-species state plus fixed/background extras?
2. Should fixed species remain in the state tensor or enter `Kinetics.forward`
   through `extra`?
3. How should `PROD`, `U`, `RAYEAR`, `JDUST`, `SGA`, grain species, and aerosol
   placeholders be represented in Kintera?
4. Do we need a KINETICS-base compatibility transport mode, or can existing
   `atm2d` operators match after boundary/grid conversion?
5. Should radiation be matched by reproducing KINETICS-base internally, or by
   allowing an oracle actinic-flux input for the first one-step chemistry match?
6. The first non-xfail target is now the Fortran fresh-start one-step
   `kintitan.out.pun` profile output after KINETICS-base state conversion and
   lower-boundary application. The next non-xfail targets should be
   thermal-rate and selected-photolysis-rate comparisons.

## Recommended Next PR Scope

The next PR should now focus on converting the structured `.pun` records into a
reportable importer:

1. Add an import report with classification/skipped counts.
2. Implement report-only reaction classification from structured `.pun`
   records plus run-input photolysis ids.
3. Add representative full-Titan reaction assertions for rate blocks and
   participant markers.
4. Move Titan atmosphere concentration conversion out of the test into a
   reusable importer helper.
5. Start `KBTitanState` construction for the 128 selected species.

This gives the next implementation phase a reliable foundation before adding
large chemistry and photolysis conversion code.
