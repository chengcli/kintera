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

- Does core `Kinetics.forward` consume the 2-D atm state `(ncol,nlyr,nsp)` directly,
  or need reshaping? (Verify in Stage 2 with a shape probe before wiring.)
- Are KB rate units (cm³-based) vs core (`mol/m³`) reconciled by a single scaling, or
  per-reaction? (Resolve in Stage 2 against the rate-match harness.)
- Multi-range Arrhenius: is the core change Python-only or does it touch the compiled
  extension? (Determines Stage 1 build effort.)
