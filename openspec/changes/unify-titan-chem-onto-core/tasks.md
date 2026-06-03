# Tasks

Regression gate for every stage: per-reaction rate match KB 1.000 + moses00 SS
recipe ratios (CH3 L40 ≈ 0.64, C6H6 L40 ≈ 1.80, C3H3 L40 ≈ 0.69).

## 0. Lock the baseline

- [x] 0.1 Remove the `KINTERA_GRAV_MASS` diagnostic probe from `python/atm2d/transport.py`
- [x] 0.2 Commit the validated photo fix (parsing.py c-/l- strip + photochemistry.py ALLOW_RADICALS + XSCN exclusion) on a refactor branch
- [x] 0.3 Tag the commit `refactor-base`
- [x] 0.4 Capture the current moses00 SS ratios + a per-reaction rate-match snapshot as the regression fixture

## 1. Core: multi-range Arrhenius

- [x] 1.1 Determine scope — RESOLVED: COMPILED C++ change. Arrhenius is `src/kinetics/rate_constant.{hpp,cpp}` (`ArrheniusOptionsImpl` A/b/Ea_R single-range lists + `Arrhenius::forward(T,other)`); needs per-range params + T-range selection, pybind in `python/csrc/pykinetics.cpp`, and a CMake rebuild (build/ is configured; torch cpp_extension).
- [x] 1.2 Add multi-range (A,B,C per temperature range) support to the core Arrhenius option + model
- [x] 1.3 Unit-test single-range parity (identical to current single-range result)
- [x] 1.4 Unit-test KB ZK1 (B>0) / ZK2 (B<0) parity across ranges to 1e-3 relative

## 2. KB → KineticsOptions translator (thermal)

- [x] 2.1 Probe `Kinetics.forward` input shapes/units against the 2-D atm state (resolve design open questions on shape + cm³ vs mol/m³)
- [x] 2.2 Translate `KBPunReaction` → core `Reaction` (reactants/products/stoichiometry) for the moses00 network
- [x] 2.3 Assign rate models: single-range Arrhenius (moses00 has no multi-range) + KB falloff (new `kb_falloff` core model). ThreeBody not used by moses00 (M-dependence folded into the falloff form).
- [x] 2.4 UPDATE_CHEMB overrides: kept as a Titan special layer (`core_chemb.py::ChembOverrideLayer`, matched by reaction signature) applied on top of core forward — the formulas are bespoke (piecewise/variable-Fc/`rk2-zk·d`), not core parameters, per the general→C++ / special→outside split. 21 moses00 reactions matched; reproduces the validated chemb callable exactly (rel 0.0); provenance recorded per column.
- [x] 2.5 Build `KineticsOptions` and evaluate via core `Kinetics.forward`/`jacobian`
- [ ] 2.6 GATE: per-reaction rate match KB 1.000 at fort.50 (vs the Stage 0 snapshot) — thermal (341 rxns) bit-exact (1e-14); photolysis pending Stage 3

## 3. Photolysis onto core Photolysis

- [ ] 3.1 Map Cheng catalog → `PhotolysisOptions` (wavelength grid, cross-sections, branches)
- [ ] 3.2 Apply the validated policy while building options: c-/l- isomer strip, XSCN exclusion, IPHOTO activation
- [ ] 3.3 Wire Titan's attenuated actinic-flux field into `Photolysis.forward(actinic_flux)`
- [ ] 3.4 GATE: photo rates match the prior titan photo path at fort.50 to 1e-3 relative

## 4. Thin Titan EI + ion gap-layer

- [ ] 4.1 Implement electron-impact source terms as a Titan layer added on top of core tendency/Jacobian
- [ ] 4.2 Keep `ion_chemistry.py` classifiers + `fold_charge_balance_into_jacobian` as the Titan ion layer on core output
- [ ] 4.3 Verify neutral-only networks make the ion layer a no-op (identical to core output)

## 5. Wire into atm2d implicit step + transport default

- [ ] 5.1 Replace `build_source_linearization` in `build_implicit_step_system` with core `Kinetics` + `Photolysis` + Titan EI/ion layer
- [ ] 5.2 Validate net dC/dt + a 1-step solve against the baseline before removing the old path
- [ ] 5.3 Validate the mr-form default flip on the non-Titan regression set (earth/jupiter YAML, `tests/test_atm2d.py`)
- [ ] 5.4 Flip core transport default `c_diffusion` → `mr_diffusion`; keep `c_diffusion` selectable
- [ ] 5.5 GATE: moses00 SS ratios reproduced through the unified pipeline

## 6. Config consolidation + delete dead path

- [ ] 6.1 Introduce `CoreConfig` (transport form, chem solver) and `TitanConfig` (network mode, photo policy, chemb overrides, EI scales)
- [ ] 6.2 Replace `KINTERA_*` reads in the library with config fields
- [ ] 6.3 Add a thin env→config loader so diagnostics keep working
- [ ] 6.4 Delete the hand-rolled rate path (`atm2d_sources.linearize`, titan `build_source_linearization` rate evaluation) and any now-dead imports

## 7. Validate

- [ ] 7.1 Run moses00-ss-recipe; confirm baseline SS ratios
- [ ] 7.2 Run the full kintera test suite green (especially post mr-default flip)
- [ ] 7.3 Update kt-kb-matching / moses00-ss-recipe skills + memory to the unified pipeline
