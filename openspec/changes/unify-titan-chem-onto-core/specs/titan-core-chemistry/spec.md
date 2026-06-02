## ADDED Requirements

### Requirement: KB network translated to core kinetics options
The Titan case SHALL translate a parsed KB PUN network into core
`KineticsOptions` — mapping each `KBPunReaction` to a core `Reaction` with the
appropriate rate model (multi-range Arrhenius, ThreeBody, or falloff) — rather than
evaluating rates in a hand-rolled Python path.

#### Scenario: Every floating-species reaction maps to a core reaction
- **WHEN** the moses00 network is translated
- **THEN** each KB reaction over the floating+fixed species set has a corresponding core `Reaction` with matching reactants, products, and stoichiometry

#### Scenario: UPDATE_CHEMB overrides baked into parameters
- **WHEN** a reaction has a KB `UPDATE_CHEMB` runtime override
- **THEN** the translated core reaction's rate parameters reproduce the override value (not the unmodified catalog value)

### Requirement: Thermal chemistry evaluated through core Kinetics
Titan thermal-chemistry tendencies and Jacobians SHALL be produced by core
`Kinetics.forward`/`jacobian`, not by `atm2d_sources.linearize`.

#### Scenario: Per-reaction rate match against KB
- **WHEN** the translated `Kinetics` is evaluated at the moses00 fort.50 state
- **THEN** each reaction's rate matches the KB reference to within 1e-3 relative (rate-match 1.000)

#### Scenario: Hand-rolled rate path removed
- **WHEN** the refactor is complete
- **THEN** `atm2d_sources.linearize` and the titan-specific `build_source_linearization` rate evaluation no longer exist and nothing imports them

### Requirement: Photolysis evaluated through core Photolysis
Titan photolysis SHALL be expressed as core `PhotolysisOptions` (wavelength grid,
cross-sections, branches) and evaluated by core `Photolysis.forward(actinic_flux)`,
with Titan supplying the attenuated actinic-flux field.

#### Scenario: Cheng catalog maps to PhotolysisOptions with validated policy
- **WHEN** the Cheng catalog is loaded
- **THEN** the c-/l- isomer normalization, XSCN-block exclusion, and IPHOTO activation policy are applied while building `PhotolysisOptions`

#### Scenario: Photo-rate parity with the prior path
- **WHEN** core `Photolysis` is evaluated with the Titan-supplied actinic flux at fort.50
- **THEN** photolysis rates match the prior titan photo path to within 1e-3 relative

### Requirement: moses00 steady state preserved
The translated, core-evaluated pipeline SHALL reproduce the documented moses00
steady-state benchmark.

#### Scenario: SS ratios reproduced
- **WHEN** the moses00-ss-recipe is run on the unified pipeline
- **THEN** the kt/KB ratios match the baseline within tolerance (CH3 L40 ≈ 0.64, C6H6 L40 ≈ 1.80, C3H3 L40 ≈ 0.69)
