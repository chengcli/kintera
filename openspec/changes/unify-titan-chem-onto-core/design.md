## Context

The Titan/KB case parses KB PUN networks (`parse_kinetics_base_pun` → `KBPunReaction`,
compiled) but evaluates reaction rates in a hand-rolled Python/torch path
(`atm2d_sources.linearize` builds tendency+Jacobian per `KBTitanSourceTerm`;
`build_source_linearization` sums them). The compiled core `kintera.Kinetics`
(`forward(temp,pres,conc)->tendency`, `jacobian(...)`) and `kintera.Photolysis`
(`forward(actinic_flux)->rates`) are never used by Titan. Core already supports
Arrhenius, ThreeBody, Lindemann/Troe/SRI falloff, Evaporation, and Photolysis with
wavelength/cross-section/branch data. The validated baseline is the photo fix
(c-/l- strip + ALLOW_RADICALS + XSCN exclusion), reproducing the moses00 SS recipe
ratios. The known open transport issue (heavy-species over-settling at dz/H≈1) is a
core concern, out of scope for behavior here but the settling code lands in core.

## Goals / Non-Goals

**Goals:**
- Titan thermal + photo chemistry evaluated through core `Kinetics`/`Photolysis`.
- Titan reduced to: a KB→`KineticsOptions`/`PhotolysisOptions` translator, a thin
  EI+ion gap-layer, and a binary-diffusion coefficient provider.
- Core gains a multi-range Arrhenius option (KB AK/AK2/AK3).
- Core transport defaults to mr-form; settling discretization owned by core.
- `KINTERA_*` env-var sprawl → `CoreConfig` + `TitanConfig`.
- Per-reaction rate match KB 1.000 and moses00 SS ratios preserved throughout.

**Non-Goals:**
- Fixing the heavy-species L60+ over-settling physics (separate effort; the code
  moves to core but behavior is unchanged here).
- Moving electron-impact or ion charge-balance *into* core (they stay a Titan layer).
- Reworking the radiation/actinic-flux attenuation (Titan keeps supplying the flux
  field; only the σ×F integration moves to core `Photolysis`).
- Changing KB input formats or the KB Fortran oracle.

## Decisions

- **Titan = translator + thin layer, not evaluator.** `KBPunReaction` → core
  `Reaction` + the matching core rate option. Rationale: core is the rate engine;
  Titan owns only KB-specific mapping. Alternative (keep Python evaluator) rejected
  per the unification mandate.
- **EI + ion charge-balance as a thin Titan layer on core outputs.** Add EI source
  tendency and `fold_charge_balance_into_jacobian` on top of `Kinetics.forward`/
  `jacobian`. Rationale: keeps ion/EI specifics out of core; core stays neutral-gas
  general. Alternative (extend core with EI/charge) rejected — too case-specific.
- **Multi-range Arrhenius added to core** (not emulated by N reactions). Rationale:
  one reaction, correct identity, cleaner Jacobian; KB semantics (ZK1 for B>0, ZK2
  for B<0, per-range AK/BK/CK). Alternative (split into multiple core reactions)
  rejected — pollutes the species/reaction graph and stoichiometry.
- **`UPDATE_CHEMB` overrides baked into translated parameters.** Rationale: the
  override *is* the rate; expressing it as the reaction's core parameters keeps a
  single source of truth. Provenance recorded in the translator.
- **Binary-diffusion coefficient stays Titan, passed as a field.** Core transport
  operator consumes `binary_diffusion` + `molecular_weights`; Titan computes Cheng/
  Moses. Rationale: Titan owns the physics-formula choice; core stays general.
- **mr-form becomes the core default** after the non-Titan regression set passes;
  `c_diffusion` retained as an explicit option. Rationale: mr-form is the correct
  variable-density discretization and what KB uses.
- **Config objects** `CoreConfig`(transport_form, chem_solver) + `TitanConfig`
  (network_mode, photo policy, chemb-overrides, EI scales); a thin env→config loader
  preserves diagnostic entry points.

## moses00 rate-form findings (Stage 2.1/2.2 scoping)

Measured against the actual moses00 `.pun` (521 reactions, 87 species) and the
validated Python evaluator `kinetics_base/titan/physics.py::_pun_rate_constant`:

- **Single-range, not multi-range.** Every moses00 reaction has exactly one
  `rate_block`, and `Tmin=1` for all of them, so the validated path uses only
  `rate_blocks[0]` with `t0 = Tmin = 1`. The Stage-1 multi-range Arrhenius is
  correct and general but is *not exercised* by moses00; moses00 maps onto the
  single-range Arrhenius fast path (bit-exact).
- **Plain thermal (291 reactions):** `k = A·T^b·exp(C/T)` (t0=1, T-clip is a
  no-op for atmospheric T). Maps to core Arrhenius with `Tref=1`, `b=b`,
  `Ea_R=-C`, **A taken raw (CGS-native, no unit conversion)**, built as an
  **irreversible** reaction so the core SI Kc reverse path is skipped. This is
  the chosen CGS-native+irreversible strategy and reproduces `_pun_rate_constant`
  exactly.
- **Falloff (50 reactions, block `D>0`):** KB form
  `ratio = k_low·n/k_high`, `k = (k_low/(1+ratio))·0.6^(1/(1+log10(ratio)^2))`
  with `k_low=A·T^b·exp(C/T)`, `k_high=D·T^E·exp(F/T)`, `n`=total number
  density, `fc=0.6` hardcoded. This is **not** core `LindemannFalloff`
  (`k0·M_eff/(1+Pr)`) — it differs by a factor of `n` (KB's is an effective
  *bimolecular* rate constant) and by the `fc=0.6` Troe broadening. Requires a
  faithful KB-falloff rate model.
- **Zero-A (180 reactions):** `block.A==0` — photolysis / special placeholders;
  handled in Stage 3 (photolysis) and Stage 4 (EI/ion), not as thermal Arrhenius.

Decision: the general translation (`.pun` → core `Reaction` + single-range
Arrhenius, CGS-native, irreversible) lands in C++ alongside the existing
`kinetics_options_from_kinetics_base` (which reads the *master* format; the
moses00 path reads the `.pun` `KBPunNetwork`). The KB falloff form is general
(one `fc=0.6` form across all 50) so it also belongs in core; ion/EI/CHEMB
specifics stay outside per the unification mandate.

## Risks / Trade-offs

- [Translated rates drift from KB] → per-reaction rate-match harness (reuse existing
  validation) gates Stage 2/3; fail the stage if any reaction ≠ 1.000.
- [mr-form default flip breaks non-Titan cases] → run earth/jupiter YAML +
  `tests/test_atm2d.py` before flipping; keep `c_diffusion` selectable; flip is its
  own gated step.
- [Photolysis flux/units mismatch core `Photolysis`] → validate photo rates against
  the current titan path at fort.50 before deleting the old path.
- [Deleting the Python rate path loses a working reference] → keep it until Stages
  2–4 reproduce moses00 SS; delete only in Stage 6 behind a green gate.
- [Jacobian/units mismatch between core `Kinetics` and atm2d implicit step] →
  validate net dC/dt + a 1-step solve against the baseline before wiring in.

## Migration Plan

Staged, each gated by the rate-match + moses00 SS regression:
0. Lock baseline (commit photo fix, tag `refactor-base`, drop `KINTERA_GRAV_MASS`).
1. Core multi-range Arrhenius (+ unit tests vs KB ZK1/ZK2).
2. KB→`KineticsOptions` translator (thermal); rate-match gate.
3. Photolysis → core `Photolysis`; photo-rate gate.
4. Thin Titan EI + ion charge-balance layer.
5. Wire `Kinetics`+`Photolysis`+layer into atm2d implicit step; flip mr-form default.
6. Config consolidation; delete hand-rolled rate path.
7. Full validation (moses00 SS + suite).

Rollback: each stage is a separate commit on a branch off `refactor-base`; revert to
the last green stage. The Python rate path stays available until Stage 6.

## Open Questions

- ~~Does core `Kinetics.forward` consume the 2-D atm state `(ncol,nlyr,nsp)` directly,
  or need reshaping?~~ **RESOLVED (Stage 2.1 probe):** consumes it directly, no
  reshaping. `forward(temp(ncol,nlyr), pres(ncol,nlyr), conc(ncol,nlyr,nsp))` returns
  `rate (ncol,nlyr,nrxn_aug)`, `rc_ddC (ncol,nlyr,nsp,nrxn_aug)`, `rc_ddT` optional
  (None unless `evolve_temperature`). This matches `AtmState2D` shapes exactly
  (`atm_state2d.py`: temp/pres `(ncol,nlyr)`, conc `(ncol,nlyr,nsp)`; Titan ncol=1).
  Verified empirically against `KineticsOptions.from_kinetics_base(test_master.inp)`:
  rate `(1,5,24)`, rc_ddC `(1,5,10,24)`. The atm2d step's `build_source_linearization`
  returns species tendency `(ncol,nlyr,nsp)` + Jacobian `(ncol,nlyr,nsp,nsp)`; the core
  outputs are per-reaction, so Stage 5 contracts them with the stoichiometry matrix
  (`species_rate` + `jacobian`) to get the species-space tendency/Jacobian.
- ~~Are KB rate units (cm³-based) vs core (`mol/m³`) reconciled by a single scaling, or
  per-reaction?~~ **RESOLVED (Stage 2.1 probe):** per-reaction — the conversion factor
  depends on the reaction order (sum of reactant stoichiometric coefficients), via
  `UnitSystem::convert_from(A, "molecule^(1-Σν) * cm^(-3(1-Σν)) * s^-1")`. This is
  already implemented in `kinetics_options_from_kinetics_base` (master format) and is
  the model for the `.pun` translator. The Titan atm2d state is CGS throughout
  (conc in molecule cm⁻³, k in cm³ⁿ⁻¹/(molecule·s), tendency molecule cm⁻³ s⁻¹).
  **Two consistent strategies** for the translator: (A) SI — convert A per-reaction and
  feed conc as mol/m³ (`n_cgs · 1e6 / N_A`); (B) CGS-native — put raw KB A straight into
  the Arrhenius `A_ranges`, keep conc in cm⁻³, and build **irreversible** reactions
  (`=>`) so the core's SI thermodynamic reverse/Kc path is skipped. Strategy (B) matches
  KB's forward-only `.pun` rates with no unit conversion, so the per-reaction k and net
  dC/dt match the hand-rolled path exactly — preferred for the rate-match gate (2.6).
  NOTE: the existing `from_kinetics_base` reads the *master* format (single-range
  `KBReaction`); the moses00 network is a `.pun` (`KBPunReaction.rate_blocks` carry
  per-range A/b/C + Tmin/Tmax — the multi-range data Stage 1 added support for), so the
  Stage 2 translator operates on `KBPunReaction`, not the master path.
- ~~Multi-range Arrhenius: is the core change Python-only or does it touch the compiled
  extension?~~ **RESOLVED (Stage 1.1):** compiled C++ change (`arrhenius.{hpp,cpp}` +
  pybind in `pykinetics.cpp` + rebuild). Done in Stage 1.
