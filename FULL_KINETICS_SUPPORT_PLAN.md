# Full KINETICS-base Titan Network Support Plan

## Goal

Adapt Kintera so it can import and run the full Titan chemistry network from
`cshsgy/KINETICS-base`, not just the small curated KINETICS-base fixture already
covered by `tests/kinetics_base`.

The target upstream data set is:

- `general/KINDATA/kindata8master_Cheng.inp`
- `examples/titan/Cheng_catalog_v4.dat`
- `examples/titan/Cheng_cross/`
- `examples/titan/kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun`
- `examples/titan/ions_c6h7+_H2CN.inp-1`

The practical milestone is to load the Titan selected network into
`KineticsOptions`, construct `Kinetics`, and execute a finite forward pass with
temperature, pressure, concentrations, and actinic flux tensors.

## Current State

Kintera currently supports a small KINETICS-base-like input path:

- `KineticsOptions::from_kinetics_base(master, catalog, cross_dir)`
- master parser in `src/photolysis/kinetics_base_reader.cpp`
- catalog parser in `parse_kinetics_base_catalog`
- cross-section parser in `parse_kinetics_base_cross_section`
- Python binding in `python/csrc/pykinetics.cpp`
- focused tests in `tests/test_kinetics_base.cpp` and
  `tests/test_kinetics_base.py`

This works for `tests/kinetics_base/data/test_master.inp`, which is a small
network with 10 species and 20 reactions.

The full upstream Titan data is much larger and uses the native KINETICS-base
workflow:

- `kindata8master_Cheng.inp` is a broad source database, not the selected Titan
  model alone.
- `Cheng_ions_c6h7+_v3_H2CN.pun` is the generated selected network. Its header
  reports `NMOL=268`, `NREACT=2139`, and `NPART=517`.
- `ions_c6h7+_H2CN.inp-1` selects active species and photolysis reactions for
  the Titan run. It reports `NFIX=8`, `NVARYF=120`, and `NPHOTO=9`.
- `Cheng_catalog_v4.dat` maps 522 photolysis branches to 522 cross-section
  files under `Cheng_cross/`.

## Main Gaps

### 1. Master Parser Is Too Permissive

The current master parser treats any non-species line with `=` as a reaction.
In the full master file this misclassifies lines such as:

- continuation-only lines like `HF = -223.0`
- pseudo/aerosol species without elemental composition, such as `SALUMINA`,
  `SCARBON`, `SH2OS`, `SH2OL`, `VHAZE1`, and `JDUST`
- some special KINETICS-base metadata lines

This produces false photolysis reactions and missing species references.

### 2. The Selected Titan Network Is In `.pun`, Not Just Master

The full Titan model is not simply "all reactions in `kindata8master_Cheng.inp`".
The generated `.pun` file contains the selected species, reaction list, indices,
thermo blocks, and kinetic coefficients used by the Titan example.

Kintera needs either:

- a `.pun` importer, or
- a faithful reimplementation of the KINETICS-base selection/generation step.

The `.pun` importer is the smaller and more reliable first target.

### 3. Run Input Selection Is Not Represented

`ions_c6h7+_H2CN.inp-1` contains run-specific active sets:

- fixed species
- varying species
- photolysis reaction indices
- wavelength bins
- boundary/radiation controls

Kintera does not need every transport/radiation field immediately, but it does
need enough selection metadata to build the same chemistry state vector and
photolysis subset.

### 4. Full Format Includes Special Kinetics Syntax

The upstream files include rate formats and semantics not covered by the small
fixture:

- rate constants like `1.803-09`, missing the `E`
- reactions with one bare rate number rather than full Arrhenius triples
- pressure-dependent reactions with low/high coefficients
- ions and electrons
- pseudo species and particles
- grain/aerosol or surface reactions
- `PROD`, `U`, `M`, and other special placeholders

Some of these should become supported features. Others should initially be
explicitly skipped with diagnostics if they are outside Kintera's chemistry
scope.

### 5. Photolysis Matching Needs To Scale To Full Catalog

The current catalog/cross-section parser is close to useful, but the full
catalog has more naming variants:

- excited states, such as `O(1D)`
- charged species, such as `TI++E`
- special branch files and absorption files
- branching ratio files that must be combined with parent absorption

We need comprehensive matching diagnostics: loaded, missing, skipped, and
branching-ratio-without-parent cases.

## Proposed Architecture

### Add A Dedicated Import Layer

Create a new import module instead of making `kinetics_base_reader.cpp` larger:

- `src/photolysis/kinetics_base_master_reader.*`
- `src/photolysis/kinetics_base_pun_reader.*`
- `src/photolysis/kinetics_base_run_input_reader.*`
- `src/photolysis/kinetics_base_importer.*`

Keep `kinetics_base_reader.*` as a compatibility facade while moving detailed
format logic into testable units.

### Introduce Structured Data Types

Add intermediate structures before constructing `KineticsOptions`:

- `KBElement`
- `KBSpeciesRecord`
- `KBReactionRecord`
- `KBRateRecord`
- `KBPunNetwork`
- `KBRunSelection`
- `KBPhotolysisCatalog`
- `KBImportReport`

The importer should produce both:

- `KineticsOptions`
- a report object with counts, skipped entries, warnings, and source mappings

### Prefer `.pun` For The Full Titan Milestone

Initial full-network path:

```text
.pun selected network
+ run input selection
+ catalog/cross-section directory
=> KineticsOptions
=> Kinetics
```

Master parsing remains useful for metadata, fallback, and tests, but it should
not be the only path for full Titan support.

## Implementation Phases

### Phase 1: Baseline Fixtures And Diagnostics

1. Add a small script or CMake option to stage external KINETICS-base data from
   a user-provided path, without vendoring the full upstream data.
2. Add lightweight parser tests using tiny fixture slices copied into
   `tests/kinetics_base_full/`.
3. Add an import report type and expose basic counts:
   species, thermal reactions, photolysis reactions, cross-section files,
   skipped reactions, and warnings.
4. Add robust errors that include source file and line number.

Acceptance criteria:

- Existing `tests/kinetics_base` still pass.
- New parser tests cover continuation lines, pseudo species, ions, and `1.803-09`
  style rates.

### Phase 2: Harden Master Parsing

1. Require reaction candidates to have a valid left-hand and right-hand side,
   not just any `=`.
2. Recognize species lines even when elemental composition is absent but the
   line is in the species section.
3. Track source sections instead of guessing from each line independently.
4. Normalize species names consistently:
   `^1CH2`, `O^+`, `C2H^-`, `E`, excited states, and charged names.
5. Parse Fortran-style rates without explicit `E`, such as `1.803-09`.
6. Preserve line numbers in parsed records.

Acceptance criteria:

- `kindata8master_Cheng.inp` can be scanned without false photolysis entries
  from species metadata.
- The parser reports meaningful counts and no bogus species like `-223.0`.

### Phase 3: Add `.pun` Reader

1. Parse `.pun` header: `NATOM`, `NMOL`, `NREACT`, `NPART`, `VER`.
2. Parse element table.
3. Parse species blocks:
   species id, name, thermo coefficients, composition, vapor metadata, and
   reaction index lists.
4. Parse reaction records and map them to `KBReactionRecord`.
5. Identify reaction classes:
   photolysis, Arrhenius, three-body, falloff, particle/surface, special.
6. Preserve original KINETICS-base ids for traceability.

Acceptance criteria:

- `Cheng_ions_c6h7+_v3_H2CN.pun` loads to an intermediate `KBPunNetwork` with
  268 species and 2139 reactions.
- The importer can list the 9 Titan photolysis ids referenced by
  `ions_c6h7+_H2CN.inp-1`.

### Phase 4: Add Run Input Selection Reader

1. Parse dimensions from `ions_c6h7+_H2CN.inp-1`.
2. Parse `IFIX`, `IVARYS`, `IVARYF`, and photolysis index sections.
3. Map KINETICS-base species ids to Kintera species ordering.
4. Decide how fixed species enter the Kintera state:
   include all species in `KineticsOptions`, but allow caller to evolve a subset,
   or build a selected-only network with fixed species supplied through `extra`.
5. Document unsupported transport/radiation fields for the chemistry-only
   importer.

Acceptance criteria:

- The importer can build the same species ordering used by the Titan run input.
- Fixed/varying species are available in the import report.

### Phase 5: Convert To Kintera Reaction Options

1. Convert elementary thermal reactions to `ArrheniusOptions`.
2. Convert three-body reactions to `ThreeBodyOptions`.
3. Convert low/high pressure reactions to `LindemannFalloffOptions` first.
4. Defer unsupported Troe/SRI or special forms unless they are required by the
   Titan selected network.
5. Represent photolysis reactions through `PhotolysisOptions`.
6. Decide policy for unsupported particle/surface reactions:
   fail by default, optional skip with report, or add a minimal reaction class.

Acceptance criteria:

- A `KineticsOptions` can be constructed from the selected Titan `.pun` network.
- Unsupported reactions are counted and explain why they are unsupported.

### Phase 6: Full Photolysis Catalog Integration

1. Reuse and harden `parse_kinetics_base_catalog`.
2. Load all catalog entries and verify all referenced files exist.
3. Parse absorption and branching ratio datasets.
4. Match parent absorption to branch datasets by parent species and temperature.
5. Interpolate all cross-sections onto a shared wavelength grid.
6. Expose diagnostics for unmatched photolysis reactions.

Acceptance criteria:

- All 522 `Cheng_catalog_v4.dat` entries resolve to files.
- The 9 selected Titan photolysis reactions have non-empty cross-section data.

### Phase 7: Runtime Validation

1. Build `Kinetics` from the imported options.
2. Run `forward(temp, pres, conc, extra)` on CPU with float64.
3. Use simple positive concentrations and a synthetic actinic flux first.
4. Verify finite rates and finite species tendencies.
5. Add a regression fixture with counts and a smoke-test forward pass.

Acceptance criteria:

- Full Titan selected network constructs successfully.
- Forward pass returns finite tensors with expected shapes.
- Test can be skipped unless `KINTERA_KINETICS_BASE_ROOT` is set, avoiding
  vendoring the large external data.

### Phase 8: Python API And Example

Add a high-level Python API:

```python
opts, report = kt.KineticsOptions.from_kinetics_base_titan(
    pun_path=".../Cheng_ions_c6h7+_v3_H2CN.pun",
    run_input_path=".../ions_c6h7+_H2CN.inp-1",
    catalog_path=".../Cheng_catalog_v4.dat",
    cross_dir=".../Cheng_cross",
)
```

Also add an example:

- `examples/example_kinetics_base_titan.py`

The example should print:

- species count
- reaction count by type
- selected photolysis count
- skipped/unsupported count
- forward tensor shapes

Acceptance criteria:

- Example runs against a user-provided external data root.
- No large KINETICS-base data is committed to this repository.

## Testing Strategy

### Unit Tests

- master line classification
- species normalization
- Fortran numeric parsing
- `.pun` header/species/reaction parsing
- run input dimension/selection parsing
- catalog parsing
- cross-section parsing

### Integration Tests

- existing small KINETICS-base fixture
- tiny synthetic `.pun` fixture
- optional full Titan test gated by `KINTERA_KINETICS_BASE_ROOT`

### Regression Checks

Expected external Titan counts:

- `.pun`: `NMOL=268`, `NREACT=2139`, `NPART=517`
- run input: `NFIX=8`, `NVARYF=120`, `NPHOTO=9`
- catalog entries: `522`
- cross-section files resolved: `522`

## Open Design Questions

1. Should Kintera import all 268 `.pun` species, or only the fixed/varying
   subset required by the run input?
2. How should fixed species be supplied during `forward`?
3. Should particle/aerosol reactions be implemented in Kintera now, skipped, or
   represented as inert diagnostics?
4. Should reactions with only one rate coefficient be interpreted as
   temperature-independent Arrhenius rates with `b=0` and `Ea_R=0`?
5. How much of KINETICS-base radiation/transport input belongs in Kintera versus
   the caller?

## Recommended First PR

The first implementation PR should be intentionally narrow:

1. Add parser utilities for KINETICS-base numbers and species normalization.
2. Harden master line classification.
3. Add `KBImportReport`.
4. Add fixture tests for the exact failure modes found in
   `kindata8master_Cheng.inp`.

This reduces risk before adding the larger `.pun` importer.
