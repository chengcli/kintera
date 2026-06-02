## ADDED Requirements

### Requirement: Multi-range Arrhenius rate option
The core Arrhenius rate model SHALL support multiple temperature ranges per
reaction, each with its own (A, B, C) coefficients, selecting the active range by
the local temperature, to represent KINETICS-base (KB) AK/AK2/AK3 constants.

#### Scenario: Single-range reaction is unchanged
- **WHEN** a reaction defines only one temperature range
- **THEN** the computed rate equals the existing single-range Arrhenius result for all temperatures

#### Scenario: Range selection by temperature
- **WHEN** a reaction defines ranges with bounds [TL, TH] and the local temperature falls within a given range
- **THEN** the rate is computed from that range's (A, B, C) coefficients

#### Scenario: KB ZK1/ZK2 form parity
- **WHEN** B > 0 the rate uses ZK1 form A·(T/T0)^B·exp(C/T), and when B < 0 it uses ZK2 form A·(T0/T)^|B|·exp(C/T)
- **THEN** the core rate matches KB's `ZKT` value to within 1e-3 relative across the reaction's temperature ranges

### Requirement: Multi-range Arrhenius is configurable from options
The multi-range Arrhenius parameters SHALL be expressible through the core kinetics
options so a translated KB network can be loaded without bespoke code per reaction.

#### Scenario: Build kinetics with multi-range reactions
- **WHEN** kinetics options include reactions with multi-range Arrhenius parameters
- **THEN** constructing `Kinetics` succeeds and `forward(temp, pres, conc)` returns finite rates for all reactions
