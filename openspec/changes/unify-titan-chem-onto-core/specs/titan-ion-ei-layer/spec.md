## ADDED Requirements

### Requirement: Electron-impact as a thin Titan layer
Electron-impact (EI) source terms SHALL be applied as a Titan-side layer added on
top of the core `Kinetics` tendency/Jacobian; core SHALL NOT contain EI-specific
rate code.

#### Scenario: EI tendency added on top of core
- **WHEN** the network includes electron-impact reactions
- **THEN** their tendency contribution is computed by the Titan EI layer and summed with the core `Kinetics` tendency, leaving core unchanged

#### Scenario: EI scales honored from config
- **WHEN** EI scale factors are set in `TitanConfig`
- **THEN** the EI layer applies them (replacing the former `KINTERA_EI_SCALE*` env vars)

### Requirement: Ion charge-balance as a thin Titan layer
Ion charge-balance SHALL be folded into the assembled Jacobian by the Titan layer
(`fold_charge_balance_into_jacobian`) on top of the core engine output; core SHALL
remain neutral-gas general.

#### Scenario: Charge balance applied after core assembly
- **WHEN** the network contains charged species (ions, electrons)
- **THEN** the Titan layer folds charge balance into the Jacobian after the core kinetics+EI contributions are assembled

#### Scenario: Neutral-only network bypasses the ion layer
- **WHEN** the network has no charged species
- **THEN** the ion charge-balance layer is a no-op and results are identical to the core engine output
