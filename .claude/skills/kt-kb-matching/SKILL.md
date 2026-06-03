---
name: kt-kb-matching
description: Protocol for matching kintera (Python/PyTorch) against KINETICS-base (KB Fortran) on a given Titan PUN network. Use whenever there is a non-trivial discrepancy between a kintera result and the corresponding KB run, including running KB from binaries in `~/dev/titan_chem/paper/moses*/`, comparing steady-state or step-N states, and isolating where the discrepancy comes from. The protocol is structured as a fixed sequence of bisection tests (chemistry → transport → solver) that quickly localises any gap.
---

# kt-kb-matching

When a kintera SS or step-N state doesn't match KB's corresponding state, follow this protocol. **Don't speculate which layer is responsible** — bisect.

## Setup: running KB

KB binaries ship pre-compiled in `~/dev/titan_chem/paper/moses*/kinetics`.
Runtime files come from the kintera repo. See
[[project-kb-binary-runnable]] for the recipe.

Minimum smoke test:
```bash
cd /tmp/kb_test && rm -rf *
cp ~/dev/titan_chem/paper/<case>/kinetics .
cp ~/dev/titan_chem/paper/<case>/fort.{1,3,4,15,21,30,50} .
cp ~/dev/titan_chem/paper/<case>/kinetics.inp .
ln -sf ~/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan/Cheng_cross crossfilepath
ln -sf ~/dev/kintera/diagnostics/KINETICS-base-compare/general/WAVELENGTH-LORES-001.DAT fort.20
touch OPEN_INC SPECIAL-INDICES-COMMON
echo "" > kinetics.res && ln -sf kinetics.res fort.10
./kinetics < kinetics.inp > kinetics.out 2>&1
# exit 0 = ran. Check fort.7 / kinetics.res for SS state.
```

For longer runs use `kinetics.inp-100` after setting `ISTART=0` in the
SPECIFIC RUN PARAMETERS block (otherwise KB tries to read fort.10 and
crashes on empty restart).

KB writes the final state to `fort.7` (concentrations) when it completes.
That is the right reference for comparing against kintera's SS.

## The protocol — perturbation experiments at a fixed state

The core idea is **inject the SAME state on both sides, take a tiny dt
step, compare the after-state**. Frozen-state per-reaction prod/loss
tells you what kintera's chemistry _alone_ computes; the perturbation
experiment tells you what kintera's _full pipeline_ (chemistry +
transport + RT + escape + solver) actually does. These give different
answers if the non-chemistry contributions are non-zero, and that's
the whole point — by toggling components on/off you isolate which
one owns the gap.

### The perturbation experiment

For a given (state, species, layer):

1. Inject KB's converged state (fort.7) into kintera. See
   [[project-titan-simplify-workflow]] for the MR-to-concentration
   conversion recipe. Use `SKIP_INJECT = {"M","JDUST"}` so
   placeholders in fort.7 don't overwrite build-time values.

   **CRITICAL: KB log layer offset.** KB output (REACTION RATES /
   MIXING RATIOS / fort.7 species blocks) is 1-indexed and skips
   layers below `NALT1`. For moses00 NALT1=10, so:
   **KB row N (1-indexed) = kintera 0-indexed L(N + NALT1 - 2)**.
   For NALT1=10 the offset is +8: KB row 1 → kintera L9, KB row 21 →
   kintera L29. Mis-aligning by integer layer count produces phantom
   ~600× rate-constant mismatches via density² scaling — see
   [[project-moses00-kb-layer-offset]]. Always verify by comparing
   pressures: at the same layer index, KB and kintera should agree
   within interpolation.
2. Take **one small step** of size `dt = 1e-3` to `1e-1` s. Small
   enough that:
   - The tendency stays linear (Δc ≈ dt × dc/dt)
   - The solver does not exercise Newton convergence (any reasonable
     dt gives a correct answer at this scale)
   - Accumulation of multiple sub-step errors is negligible
3. Compute Δc per species per layer = `c_after - c_before`. The same
   step on KB gives `Δc_KB`. The difference `Δc_kt - Δc_KB` is the
   total pipeline disagreement at this state.
4. Compare per species (especially the discrepancy target). Large
   `|Δc_kt - Δc_KB|/dt` ≫ plausible tendency magnitude → kintera's
   pipeline differs from KB's at this state.

**Why this is better than frozen prod/loss alone:** prod/loss assumes
chemistry-only and misses transport-divergence, photolysis, escape,
condensation. A tiny-dt step lumps all contributions together, then
the subtraction experiments below split them apart cleanly.

### Subtraction: isolate which component owns the gap

Repeat the experiment with components disabled, _on both sides_:

| Experiment | Kzz | photochem | escape | What is left in Δc |
|---|---|---|---|---|
| A. Baseline | on | on | on | chemistry + transport + RT + escape |
| B. Kzz=0 | **0** | on | on | chemistry + RT + escape |
| C. Kzz=0, RT off | **0** | **0** | on | chemistry + escape |
| D. Chemistry only | **0** | **0** | **0** | chemistry only |

If `Δc_kt − Δc_KB`:
- shrinks substantially in **B vs A** → the gap was **transport**
  ([[project-kintera-transport-form]] documents the c-form vs
  MR-form divergence)
- shrinks in **C vs B** → the gap was **radiative transfer / photolysis**
  (cross-section reading, opacity model, freeze_actinic_flux, IPHOTO
  filter, etc.)
- shrinks in **D vs C** → the gap was **upper-boundary escape**
  (escape velocity for H/H2, hot-atom flux, etc.)
- persists in **D** → the gap is **pure chemistry**: either a
  rate-constant mismatch (compare against [[kb-fortran-map]]
  UPDATE_CHEMA/B), a missing reaction in the kintera subset filter,
  or stoichiometry/coefficient bugs

In code:

- kintera: pass `kzz=torch.zeros_like(ts.kzz)`, drop photo source
  terms from the source-term list, and zero the upper-boundary flux
  in the boundary conditions.
- KB: set `IDIFUS=0` (or zero `fort.30` eddy column), `RAD=0` plus
  `NPHOTO=0`, and zero the upper escape flux in `fort.15` /
  boundary file.

The goal is not numerical perfection (small numerical mismatches in
solver discretisation are fine) — it's localisation. Each
"shrinks in X" tells you which subsystem to fix next.

### State-mismatch clarification: fort.7 IS KB's run-end state

A red herring that came up several times: I worried that KB's fort.7
(written via IPUN=1 dump) wasn't the same state as KB's final
"REACTION RATES" / "MIXING RATIOS" printout. After fixing the layer
offset (above), they DO match — kintera fort.7 CH3@L20 (1.71e+6) and
KB log row 12 CH3 (1.145e-11 MR × 1.5e+17 density = 1.72e+6) are
identical. **Use fort.7 as the injection target with confidence,
provided the parser correctly handles the NALT1 offset.**

The earlier apparent state mismatch (factor 20×) was an artifact of
indexing fort.7's CH3 array directly — fort.7 stores 91 layer values
but only L9-L90 (0-indexed) are real; values L0-L8 are filler.
Always use kintera's `parse_kinetics_base_atmosphere` for fort.7,
not raw indexing.

### Chemistry-only deep-dive (after experiment D points at chemistry)

When experiment D shows pure-chemistry disagreement, drill in:

- **Frozen prod/loss probe**: `diagnostics/moses00_prodloss.py` and
  `moses05_c2_prodloss.py` annotate every top-contributing reaction
  with k(T,ρ), the .pun (A,b,C,D,E,F) parameters, and the override
  source label ("chemb(id=N)" / "chemb(sig)" / "pun_cat"). Read the
  output and verify each large contributor's k(T) against KB's
  UPDATE_CHEMA/B formulas (grep `kinetgen1X.F:3952-7384`).
- **Override dispatch by signature**: chemb overrides are keyed by
  reactant/product signature so they fire across networks (Cheng
  reaction_id differs from moses00). See
  [[project-moses00-rate-match-state]] and [[project-moses00-M-id-87]].
- **Mass conservation**: confirm filter doesn't drop reactions when
  only some products are missing — that silently destroys reactants.

If KB's `prod+loss/<species>_<dir>.dat` is available (post-2022
binary), compare per-reaction rates directly. If not, kintera's
annotated probe + KB source-code inspection is the substitute.

### Known kintera-side fixes/knobs for moses00-class runs

These come up repeatedly and are necessary before any meaningful
comparison:

1. **`KINTERA_DISABLE_CHEMB_OVERRIDES=1`** — the 2012-era paper KB
   binaries were compiled without `__TITAN`, so UPDATE_CHEMB Moses-2005
   Troe overrides don't fire (KB uses pure .pun catalog rates).
   kintera's `chemb_overrides.py` applies those formulas anyway. For
   moses00-class runs set this env var to fall back to pure
   `_pun_rate_constant`. Verifies rate-constant ratio = 1.000 across
   all altitudes for all non-photo reactions.

2. **`force_zero = ("JDUST",)` in the atm shim** — moses00 atm file
   ships a Cheng-era aerosol concentration profile (~21500 cm⁻³ at L60).
   Kintera's per-unit σ=1e-8 cm² applied to this gives τ≈2800 across
   the column, killing photolysis. For simplified PUNs without aerosol
   chemistry, force JDUST=0 in the atm shim. See
   [[project-moses00-jdust-photolysis]].

3. **Photolysis 4× scaling** (radiation.py:158, 246) — DISORT's
   actinic flux convention is 4π × J̄ (photochemistry standard) but
   KB-2012's cross sections were tabulated against π × F_down. Both
   the DISORT path (line 158) and the direct-attenuation path (line
   246) now scale by 1/4 to match KB. If you regenerate cross
   sections from raw photoabsorption data using the 4π convention,
   remove these factors. See
   [[project-moses00-kb-layer-offset]] for the trace.

4. **`SKIP_INJECT = {"M","JDUST"}` in inject_state** — placeholders
   in KB fort.7 that should not overwrite build-time values.

### Iron rule: prod/loss must match before transport / solver tests

Don't move past chemistry until **prod/loss agrees at the fixed KB SS
state**. Transport and solver tests give meaningless answers when
chemistry rates are wrong — a Kzz=0 disagreement could be either a
real transport gap _or_ a chemistry rate residual; you cannot tell.
Close every identified chemistry-rate mismatch (Troe falloff,
UPDATE_CHEMB overrides, condensation rates) first. See
[[feedback-match-rates-before-transport]].

### Final escalation

If experiments A-D don't localise the gap (e.g. all experiments show
the same residual size), raise it back to the user. Document which
experiment was run, the per-species `|Δc_kt − Δc_KB|` table, and
which subsystems were toggled.

### KB-side rate verification: per-reaction k from KB's own log

The 2012 paper-era KB binary doesn't write `prod+loss/<species>.dat`
files (that code is post-2022). However the `kinetics.out` log
contains a **"REACTION RATES:" section** (around line 119241 for
moses00) with k×∏[reactants] for every reaction × every altitude.
Parser: `diagnostics/moses00_kb_rates_parse.py` → `.npz`.

To get the rate CONSTANT k (not the rate × conc), back-out from KB's
own concentrations parsed from the "MIXING RATIOS :" section
immediately above REACTION RATES. Apply the layer offset (KB row N →
kintera L(N + NALT1 − 2)) when aligning the arrays. Tool:
`diagnostics/moses00_kb_rate_constants.py`.

With the chemb_overrides disabled and layer offset corrected,
non-photo k values match KB at **ratio = 1.000 across all
altitudes**. This is the cleanest possible model-level verification.

## Common pitfalls

- **fort.50 vs fort.7**: `fort.50` is KB's per-step restart dump, often
  written mid-Newton iteration; values are not the true SS. fort.7 is
  the converged SS. Use fort.7 as the reference. The moses05.C2 paper
  shipped fort.50 (not fort.7) and we mistakenly took it as the SS —
  see [[project-titan-simplify-workflow]].
- **Mixing ratio vs concentration storage**: KB's fort.7/fort.50 stores
  species in two different conventions depending on the case (and on
  the species). moses05.C2 stores H/H2 as concentration but C2H6/CH4
  as mixing ratio; moses00 stores everything as MR. Verify by
  multiplying raw values by total density and checking which way gives
  physically plausible concentrations. The inject recipe in
  [[project-titan-simplify-workflow]] documents the CONCENTRATION_SPECIES
  set per case.
- **Fixed species need pinning**: in BE integration without pin
  postprocess, fixed species like H2O / OH / CO can autocatalytically
  grow from 0 (numerical noise → finite values via chemistry coupling).
  Apply `kt.apply_kinetics_base_titan_boundary_pins` after every solve
  step.
- **freeze_actinic_flux + MR-storage atm**: kintera's
  `freeze_actinic_flux` uses `titan_state.concentration` for opacity;
  if that's the raw atm-file value (mixing ratio for many species),
  opacity is essentially 0 and photolysis is unattenuated through
  the column. **Force `t.parameters['freeze_actinic_flux'] = False`**
  on all photo source terms. See
  [[project-moses05-c2-photo-attenuation-fix]] (merged into
  [[project-titan-simplify-workflow]]).
- **Skipping bisection**: every time you've tried to skip the protocol
  and guess the bug source, the guess was wrong. Bisect.

## Fast first pass: single-sided physics isolation (no KB run)

Before the full both-sides A–D subtraction (which needs KB re-runs), run the
**kintera-only isolation harness** `diagnostics/perturbation_isolate.py`. It
injects KB's SS (fort.50) and decomposes kintera's tendency per species/layer
into components that can be toggled WITHOUT touching KB:

```bash
python3.10 diagnostics/perturbation_isolate.py HC3N C6H6 C4H2
# columns: c | chem_full chem_thermal chem_photo | trans_full trans_eddy trans_mol | NET net/c
```

- **chem-only** (transport off) = `build_source_linearization` over filtered
  source terms; **transport-only** (chem off) = `build_transport_matrix(...).matvec(c)`;
  eddy vs molecular split = mr_hybrid full minus mr_diffusion (eddy-only).
- **actinic flux = 0**: rebuild source terms WITHOUT the photo reactions
  (`kind != "pun_photo_rate_reaction"`) → `chem_thermal`; `chem_photo =
  chem_full − chem_thermal`. Isolates photochemistry from thermal chemistry
  **with no KB run** — decisive for "is the chem gap photo or thermal?".

**Why FIRST:** at KB's SS, KB net = 0, so kintera's NET is the pipeline
disagreement. The split says which process owns it; chem_photo says whether the
chem piece is photo. Only escalate to the both-sides A–D protocol (or a KB
prod+loss re-run for chem_KB) if this can't localise it.

**Worked result (moses00 L60+ heavy-species collapse, 2026-06-02):** at L70–L80
chem_full is tiny, chem_photo is a negligible fraction of it, and trans_mol
(molecular diffusion) is 5–15× larger and negative → the collapse is molecular-
diffusion transport, NOT photochemistry. The actinic-flux=0 split also closed a
real hole: completeness + per-rxn-k match does NOT prove chem_kt = chem_KB
(kintera computes photo rates independently), but photo being negligible at
altitude means it can't be the gap. See [[project_moses00_moldiff_drain_confirmed]].

## Output

After completing the protocol on a discrepancy, write a one-paragraph
summary covering:
- the species and layer with the discrepancy,
- where the bisection isolated the gap (chemistry / transport / solver),
- the specific cause (rate constant, missing reaction, transport-form,
  solver convergence),
- and what was changed to close it (or what is escalated).

Save the summary to a project memory entry if the finding is reusable;
otherwise keep it in the conversation.

## Unified core-engine pipeline (2026-06-03)

The `unify-titan-chem-onto-core` refactor routes kintera's Titan chemistry
through the compiled core `kintera.Kinetics`/`Photolysis` engine via
`CoreChemistrySource` (`kinetics_base/titan/core_source.py`) — a `LocalSourceTerm`
that is a validated drop-in for the hand-rolled per-reaction source terms (it
reproduces the hand-rolled tendency/Jacobian/SS to machine precision; see
[[project_stage5_atm2d_wiring]]).

Implications for the bisection protocol:
- **Chemistry layer**: rates now come from core `Arrhenius` + `KBFalloff`
  (`from_kinetics_base_pun`) + `ChembOverrideLayer` + core `Photolysis`. When a
  per-reaction rate gap appears, compare the core module output against the
  validated `_pun_rate_constant` / `_photo_rate_profile` with the harnesses
  `diagnostics/stage5_core_{thermal,photo,full,step}_check.py` (these compare
  core-vs-hand-rolled at fort.50 to machine precision — a regression there means
  the translator/options changed).
- **Transport layer**: the core default is now `mr_diffusion` when a density
  field is supplied (`atm2d/config.py`); `c_diffusion` stays selectable via
  `form=`/`KINTERA_TRANSPORT_FORM` (now mirrored by `CoreConfig`).
- **Config**: `KINTERA_*` switches are mirrored by `CoreConfig`/`TitanConfig`
  ([[project_stage6_config]]); `get_*_config()` reads env fresh so existing
  env-var bisection still works.

The hand-rolled rate path is retained as the reference (Stage 6.4 deletion
deferred), so both paths remain comparable during bisection.
