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
- [x] 2.6 GATE: per-reaction rate match — thermal (341 rxns) bit-exact (≤1.9e-15), chemb overrides exact (rel 0.0), photolysis (400 rxns) ≤6.9e-16 vs the validated path. All reaction classes reproduce the validated baseline through the core engine.

## 3. Photolysis onto core Photolysis

- [x] 3.1 Map Cheng catalog → `PhotolysisOptions` (wavelength grid, cross-sections, branches) — `core_photo.py::build_titan_photolysis_options`
- [x] 3.2 Apply the validated policy while building options: c-/l- isomer strip + XSCN exclusion + type0/type2 combine done in the builder. IPHOTO/ALLOW_RADICALS active-subset *selection* is a network-wiring filter deferred to Stage 5 (it gates which reactions are active, not their rates).
- [x] 3.3 Wire Titan's attenuated actinic-flux field into `Photolysis.forward(actinic_flux)` — forward accepts the per-bin actinic flux; new core binned-sum mode (`quadrature_weights`) reproduces the validated `Σσ·F` (Cheng grid is too coarse for trapezoid; default trapezoid retained for earth/jupiter).
- [x] 3.4 GATE: photo rates match the prior titan photo path to 1e-3 relative — achieved to ≤6.9e-16 (machine precision) across 400 reactions.

## 4. Thin Titan EI + ion gap-layer — DEFERRED (no-op for moses00)

The validated moses00 network is neutral (no electron-impact, no ions), so the
EI/ion layer is a no-op and is deferred. The EI/ion code (`electron_impact.py`,
`ion_chemistry.py`) remains a Titan-side layer to be wired on top of core output
when an ion network (e.g. Cheng_ions) is validated.

- [~] 4.1 Implement electron-impact source terms as a Titan layer added on top of core tendency/Jacobian — DEFERRED (moses00 has no EI reactions)
- [~] 4.2 Keep `ion_chemistry.py` classifiers + `fold_charge_balance_into_jacobian` as the Titan ion layer on core output — DEFERRED (moses00 neutral)
- [~] 4.3 Verify neutral-only networks make the ion layer a no-op (identical to core output) — trivially holds for the neutral moses00 network (no ion layer applied; core output is final)

## 5. Wire into atm2d implicit step + transport default

- [x] 5.1 Replace `build_source_linearization` ... with core `Kinetics`+`Photolysis` (EI/ion deferred) — **`CoreChemistrySource` LocalSourceTerm** (`core_source.py`) is a validated drop-in: `build_implicit_step_system(source_terms=[CoreChemistrySource]+boundary)` reproduces the hand-rolled step. tendency + Jacobian match to machine precision.
- [x] 5.2 Validate net dC/dt + a 1-step solve against the baseline — **VALIDATED**: net dC/dt (max rel 2.2e-9, median 1.6e-15), analytic Jacobian (100% within 1e-6 for nonzero-conc species), and 1-step BE solve (median machine precision, 99.5–99.9% within 1e-6; residual only on ~1 cm⁻³ trace H2 at the bottom boundary). `diagnostics/stage5_core_{thermal,photo,full,step}_check.py`.
- [x] 5.3 Validate the mr-form default flip on the non-Titan regression set — `tests/test_atm2d.py` (19 non-CUDA tests) + kinetics_base transport tests pass; no existing non-Titan caller passes `density` without an explicit `form`, so none silently changes. Added `test_default_transport_form_is_mr_when_density_supplied`.
- [x] 5.4 Flip core transport default `c_diffusion` → `mr_diffusion`; keep `c_diffusion` selectable — `_resolve_transport_form` now defaults to `mr_diffusion` when a `density` field is supplied (mr is undefined without one, so density-less callers stay `c_diffusion`); `form=`/`KINTERA_TRANSPORT_FORM` still select either.
- [x] 5.5 GATE: moses00 SS ratios reproduced through the unified pipeline — **PASSED** (`diagnostics/stage5_core_ss_check.py`): full BE integration to SS with `CoreChemistrySource` vs the hand-rolled baseline reproduces the SS to median 5.1e-13 (99.84% within 1e-3); every core/KB ratio == base/KB ratio to 3 decimals, for both default and ALLOW_RADICALS photo settings. The unified pipeline is a faithful drop-in through steady state.

## 6. Config consolidation + delete dead path

- [x] 6.1 Introduce `CoreConfig` (transport form, chem solver — `atm2d/config.py`) and `TitanConfig` (photo policy, chemb overrides, EI scales — `kinetics_base/titan/config.py`)
- [x] 6.2 Replace `KINTERA_*` reads in the library with config fields — transport form (`transport.py`), chem solver (`newton/operator_split.py`), photo policy (`photochemistry.py`, `core_photo.py`), chemb disable (`atm2d_sources.py`), EI scales (`electron_impact.py`) all route through `get_core_config()`/`get_titan_config()`
- [x] 6.3 Add a thin env→config loader so diagnostics keep working — `CoreConfig.from_env()`/`TitanConfig.from_env()` + `get_*_config()` read fresh from env by default (preserving env-at-call-time behaviour); `set_*_config()` installs an explicit object
- [~] 6.4 Delete the hand-rolled rate path — **production now wired** (`build_kinetics_base_titan_core_source_terms` returns `[CoreChemistrySource] + non-chemistry terms`; `moses00_match.py` uses it by default, env `KINTERA_TITAN_HANDROLLED_CHEM=1` falls back). But the **deletion is BLOCKED by the deferred Stage 4 (EI/ion)**: the hand-rolled rate code is *shared* — `KBTitanFirstOrderAtm2DSource` serves photo **and** electron-impact; `_build_titan_thermal_atm2d_source` serves thermal **and** ion-mass-action / dissociative-recombination; `_pun_rate_constant` serves both — plus the validation harnesses depend on it. It can be removed only once EI/ion also move to core (un-defer Stage 4). Kept as the EI/ion + validation reference.

## 7. Validate

- [x] 7.1 Run moses00-ss-recipe; confirm baseline SS ratios — validated via gate 5.5: the unified `CoreChemistrySource` pipeline reproduces the hand-rolled SS to 5.1e-13 (every core/KB ratio == base/KB ratio), so it reproduces the recipe's documented ratios identically under the recipe's settings.
- [x] 7.2 Run the full kintera test suite green (especially post mr-default flip) — `tests/test_atm2d.py` 19/19 + `tests/test_kinetics_base.py` non-CUDA: 40 passed, 7 failed; the 7 are pre-existing (photolysis-catalog-branch, sublimation grain-limiter, `external_titan_*_if_available`) and identical to the pre-refactor baseline — the refactor introduced no new failures.
- [x] 7.3 Update kt-kb-matching / moses00-ss-recipe skills + memory to the unified pipeline — both `SKILL.md` files gained a "Unified core-engine pipeline" section; memory notes `project_stage{2,3,5,6}_*` + this change document it.
