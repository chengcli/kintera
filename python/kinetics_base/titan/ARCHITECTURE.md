# Titan chemistry model — architecture

This package evaluates the **Titan photochemistry** (the moses00 and related
KINETICS-base PUN networks) and is validated to reproduce the original
KINETICS-base (KB Fortran) results. After the `unify-titan-chem-onto-core`
refactor, **all numerics run through the compiled core `kintera` engine**; the
Titan layer is reduced to a *translator + thin gap-layer + input provider*.

---

## 1. Design principle

> **Core owns all numerics.** **Titan owns only KB-specific mapping, policy, and
> the fields core consumes.**

The entire chemistry tendency `dc/dt` and its Jacobian — thermal mass action and
photolysis — are assembled by core C++ modules. Titan never evaluates a reaction
rate or a mass-action product itself anymore.

---

## 2. What runs where

| Physics | Where | Module / entry point |
|---|---|---|
| Thermal rate constants `k(T)` | **core** | `Arrhenius` (multi-range), `KBFalloff` |
| Thermal mass action `k·∏Cᵛ` + Jacobian | **core** | `Kinetics.forward` / `Kinetics.jacobian` |
| Photolysis rate `J = Σλ σ(λ)·F(λ)` | **core** | `Photolysis` (binned-sum `quadrature_weights`) |
| Photolysis source `J·[parent]` + Jacobian | **core** | `PhotoChem.forward` / `PhotoChem.jacobian` |
| Vertical transport (eddy + molecular diffusion) | **core** | `atm2d/transport` (mr/c diffusion) |
| Implicit step assembly + sparse solve | **core** | `build_implicit_step_system`, `solve_sparse_system` |
| KB `.pun` → core options translation | Titan | `KineticsOptions.from_kinetics_base_pun` (C++) |
| KB `UPDATE_CHEMB` rate overrides (the *values*) | Titan | `core_chemb.ChembOverrideLayer` → injected via `extra["kf_override"]` |
| Photo cross-section build + policy | Titan | `core_photo.build_titan_photolysis_options` |
| Photo activation / isomer / XSCN policy | Titan | `photochemistry.py` (`ALLOW_RADICALS`, c-/l- strip, `_XSCN_` exclusion) |
| Photo per-reaction multipliers / min-altitude | Titan | thin correction in `CoreChemistrySource` |
| Attenuated actinic-flux field `F_att(z,λ)` | Titan | `radiation.py` (DISORT / direct beam) |
| Binary-diffusion coefficient (Cheng/Moses) | Titan | `transport_diffusion.py` (a field for the core operator) |
| Electron-impact + ion charge-balance | Titan | `electron_impact.py`, `ion_chemistry.py` — **deferred** (no-op for neutral moses00) |
| Boundary conditions, time-stepping, pins | Titan/atm2d driver | `atm2d_sources.py` boundary terms, `diagnostics/moses00_match.py` |

---

## 3. The unified entry point: `CoreChemistrySource`

`core_source.CoreChemistrySource` is a single `LocalSourceTerm` (it has
`.linearize(state) -> (tendency, jacobian)`) that replaces the old hand-rolled
per-reaction Titan source terms. Drop it into the atm2d implicit step:

```python
from kintera.kinetics_base.titan.core_source import CoreChemistrySource

# `filtered` = the KB source terms restricted to the active species subset
core_chem = CoreChemistrySource(titan_state, pun_path, filtered)

# moses00 has only chemistry + boundary terms (no condensation/global ops):
boundary = [t for t in filtered if t.kind in
            ("upper_boundary_flux", "upper_boundary_velocity", "lower_boundary_flux")]
sources = [core_chem] + build_kinetics_base_titan_atm2d_source_terms(titan_state, boundary)

system, rhs = kt.build_implicit_step_system(
    state, kzz, dt, density=density, transport_form="mr_diffusion",
    source_terms=sources, binary_diffusion=binary_diffusion,
    molecular_weights=molecular_weights)
```

### What `linearize` does internally

1. **Thermal** — permute the concentration into core (`.pun` id-sorted) species
   order; build the `kf_override` tensor (KB `UPDATE_CHEMB` values on the matched
   reaction columns, `NaN` elsewhere); call
   `kin.forward(T, P, conc_core, {"number_density": n, "kf_override": kf})` and
   `kin.jacobian(...)`; contract with the stoichiometry matrix; permute back.
   - The `kf_override` is a **general** core feature: finite entries replace the
     computed rate constant and zero that reaction's rate-constant derivative
     (treated constant-in-C). This lets the bespoke KB overrides ride through
     core's mass action while still matching the baseline's frozen-`k` Jacobian.
   - `KBFalloff`'s `k` depends on `number_density` (supplied via `extra`, **not**
     `sum(C)`), so its Jacobian contribution is zero — matching KB, which treats
     the falloff density as a frozen parameter.

2. **Photolysis** — compute the attenuated actinic flux `F_att` (`radiation.py`),
   prime the cross-sections, call `PhotoChem.forward(T, conc_core, F_att)`
   (= `J·[parent]`) and `PhotoChem.jacobian(...)`; apply the thin Titan
   per-reaction multiplier / min-altitude correction; contract with stoich;
   permute back.

---

## 4. Configuration

The scattered `KINTERA_*` environment switches are consolidated into two
dataclasses (env vars are still read by the loaders, so existing recipes work):

- `atm2d/config.py::CoreConfig` — `transport_form`, `chem_solver`.
- `kinetics_base/titan/config.py::TitanConfig` — `photo_allow_radicals`,
  `photo_include_xscn`, `disable_chemb_overrides`, `ei_scale`,
  `ei_channel_scales`.

`get_core_config()` / `get_titan_config()` read the environment fresh on each
call; `set_core_config(cfg)` / `set_titan_config(cfg)` install an explicit
object (a `None` argument restores env-driven loading).

The core **transport default is now `mr_diffusion`** when a `density` field is
supplied (the correct variable-density discretization, and what KB uses); without
a density it falls back to `c_diffusion`. Both stay selectable via `form=` or
`KINTERA_TRANSPORT_FORM`.

---

## 5. moses00 network specifics

The moses00 `.pun` network (521 reactions, 87 species, neutral) translates as:

- **Thermal (341):** every reaction is *single-range* (`Tmin = 1`), so
  `k = A·T^b·exp(C/T)` → core `Arrhenius` with `Tref = 1`, `Ea_R = −C`, raw CGS
  `A` (no unit conversion), built **irreversible** (`=>`) to skip the SI
  thermodynamic-reverse path.
  - 291 plain Arrhenius (`block.D ≤ 0`).
  - 50 falloff (`block.D > 0`) → core `KBFalloff` (the KB effective-bimolecular
    `fc = 0.6` blend, total number density `n`).
- **Photolysis (180 zero-A in the `.pun`):** built from the Cheng catalog +
  cross-sections; ~68 are *active* in moses00 after the IPHOTO/`ALLOW_RADICALS`
  policy; 1 has a non-trivial multiplier (C2H2→C2H+H ×4).
- **UPDATE_CHEMB overrides:** 21 of the ~24 KB hand-coded rate overrides match
  moses00 reactions by signature and are injected via `kf_override`.
- **No electron-impact / ions** — the EI/ion layer is a no-op here.

Units are **CGS-native** throughout (concentration `molecule cm⁻³`, rate constant
`cm³ⁿ⁻¹/(molecule·s)`, tendency `molecule cm⁻³ s⁻¹`), matching the atm2d state, so
no unit conversion is needed between Titan and core.

---

## 6. Validation

The unified pipeline reproduces the hand-rolled baseline (and KB) to machine
precision at every level — see `diagnostics/stage5_core_*.py`:

| Level | core vs hand-rolled baseline |
|---|---|
| per-reaction rate constant | ≤ 6.9e-16 |
| net `dc/dt` | median 1.6e-15, 100% within 1e-6 (physical cells) |
| analytic Jacobian | 100% within 1e-6 (nonzero-conc species) |
| 1-step BE solve | median machine precision |
| full steady state | median ≈ 5e-13 (every `core/KB` ratio == `baseline/KB`) |

The refactor changes the moses00 steady state by **nothing**.

---

## 7. Deferred / not-yet-wired

- **Electron-impact + ion charge-balance layer** (`electron_impact.py`,
  `ion_chemistry.py`) — for ion networks (e.g. Cheng_ions); a no-op for the
  neutral moses00.
- **Deleting the hand-rolled rate path (task 6.4)** — production is now wired
  (`build_kinetics_base_titan_core_source_terms` is the default chemistry source
  in `moses00_match.py`), but the hand-rolled rate code **cannot yet be deleted**
  because it is *shared with the deferred EI/ion layer*:
  `KBTitanFirstOrderAtm2DSource` serves photo **and** electron-impact,
  `_build_titan_thermal_atm2d_source` serves thermal **and**
  ion-mass-action / dissociative-recombination, and `_pun_rate_constant` serves
  both (the validation harnesses also depend on it). It can be removed only once
  EI/ion are moved to core as well. It is retained as the EI/ion + validation
  reference. (`KINTERA_TITAN_HANDROLLED_CHEM=1` selects the old path.)

See `openspec/changes/unify-titan-chem-onto-core/` for the full design,
capability specs, and staged task list.
