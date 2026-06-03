---
name: moses00-ss-recipe
description: Recipe for reproducing the moses00 stable steady-state (SS) comparison against KB fort.50. Use whenever you need to (1) re-run the moses00 SS benchmark, (2) iterate on transport / chemistry / photo changes and see how they shift the SS, or (3) verify that a tag or commit still produces the documented baseline ratios. The recipe pins specific env vars, NT, dt_max, and transport_form because off-defaults produce blow-up at L80+ (validated 2026-06-02 — many alternatives tried, this is the only reproducible stable SS).
---

# moses00-ss-recipe

## What this gets you

A stable kintera SS for moses00 vs KB fort.50 at L40-L80. **New baseline as of
2026-06-02** — the validated photo fix (ALLOW_RADICALS + c-/l- strip + XSCN
exclusion) is now part of the standard run. Reference ratios (kt/KB):

| species | L40 | L60 | L70 | L80 |
|---|---|---|---|---|
| CH3 | 0.64 | 0.61 | 0.87 | 0.31 |
| C2H2 | 0.96 | 0.27 | 0.21 | 0.21 |
| C2H6 | 0.97 | 0.64 | 0.70 | 0.58 |
| HCN | 0.99 | 1.35 | 1.32 | 1.29 |
| HC3N | 1.15 | 0.27 | 0.10 | 0.02 |
| C6H6 | 1.80 | 0.00 | 0.00 | 0.00 |
| C3H3 | 0.69 | 0.05 | 0.01 | 0.00 |

The photo fix lifted C3H3 (L40 0.00→0.69) and C6H6 (L40 4.33→1.80) without
destabilizing the bulk; C3/C3H trace overshoot is gone via the XSCN exclusion.
**Remaining gap = L60+ collapse of C6H6/C4H2/HC3N/C3H3** (→0 at L70-L80) — this
is the documented transport drift, NOT photochemistry (see
[[project_moses00_L60_transport]], [[project_moses00_photo_fix_validated]]). The
canonical SS dump is `/tmp/kt_moses00_ss_baseline.npz` (the pre-photo-fix state
is archived as `/tmp/kt_moses00_ss_prephoto.npz`).

Old pre-photo-fix reference (for context, tag `moses00-ss-repro`): CH3 L40 0.56,
C6H6 L40 4.3, C3H3 L40 0.00.

## Quick run

```bash
cd /home/sam2/dev/kintera
git checkout moses00-ss-repro       # or any descendant that keeps the plumbing

KINTERA_DISABLE_CHEMB_OVERRIDES=1 \
KINTERA_TITAN_NETWORK_MODE=full \
KINTERA_TITAN_NTIME=60 \
KINTERA_TITAN_MAX_DT=1.0e+7 \
KINTERA_TITAN_TRANSPORT=mr_hybrid \
KINTERA_TITAN_PHOTO_ALLOW_RADICALS=1 \
python3.10 -u diagnostics/moses00_match.py 2>&1 | tee /tmp/moses00_ss.log
```

(XSCN radical-block exclusion is on by default; `KINTERA_TITAN_PHOTO_INCLUDE_XSCN`
would re-enable it. The photo fix lives in `parsing.py` + `photochemistry.py`.)

The script saves the SS to `/tmp/kt_moses00_ss.npz`. Total runtime
~3 min on the dev box.

## What every env var does

| var | required value | why |
|---|---|---|
| `KINTERA_DISABLE_CHEMB_OVERRIDES` | `1` | moses00 KB-2012 binary was compiled without `__TITAN`, so UPDATE_CHEMB Troe overrides do not fire. Without this kintera applies overrides KB does not — rxn 157 (CH3+CH3+M) ratio shifts from 1.000 to 605×. See [[project_moses00_validation_milestone]] |
| `KINTERA_TITAN_NETWORK_MODE` | `full` | Without this, the script defaults to `no_grain` which is apples-to-oranges with KB's grain-on oracle. See [[feedback_default_grain_mode]] |
| `KINTERA_TITAN_NTIME` | `60` (do **NOT** raise) | Each BE step at large dt is structurally unstable for moses00. NT=60 with the geomspace schedule hits dt_max at the right point — partial relaxation, not full SS. NT=100 with `MAX_DT=3e+8` *blows up* (C2H6 reaches 8e+9 at L80, C6H6 reaches 1e+7); NT=120 with `MAX_DT=1e+10` *blows up worse*. The "1-yr partial relax" produced here is the most converged stable state available |
| `KINTERA_TITAN_MAX_DT` | `1.0e+7` | dt_max=1e+7 sec ≈ 0.3 yr. Combined with NT=60, total simulated time = 0.98 yr. Sweet spot between partial relaxation and BE blow-up |
| `KINTERA_TITAN_TRANSPORT` | `mr_hybrid` | `mr_hybrid` = light species (H, H2) use exponential differencing for gravitational separation, heavy species use centered FV. This is what KB-2012's COEFF1.f90 effectively does. `mr_diffusion` gives slightly worse L60+ match, `mr_exp` blows up |
| `KINTERA_TITAN_MOLDIFF` | default = `cheng` | Cheng-2013 molecular diffusion formula. KB-2012 binary unconditionally overwrites the ADIFH2/SDIFH2 path with Cheng in COEFF1.f90:60-63. See [[project_moses00_moldiff]] |
| `KINTERA_TITAN_PHOTO_ALLOW_RADICALS` | `1` (now recommended) | The real photo loader bug fix (kintera misses ~60 photo channels KB activates via `Cheng_cross/CROSS_*.DAT`). **Validated stable & beneficial 2026-06-02** with the c-/l- isomer-strip in `parsing.py` and the XSCN exclusion (below) staged: NT=60 run does NOT blow up (C2H6 max identical to baseline 7.52e13). A/B: C3H3 L40 0.00→0.69, C6H6 L40 4.33→1.80, small gains in CH3/C2H2/C2H6/C4H2/CH3C2H, no regressions. The earlier "blows up to 1e10" warning predates the c-/l- strip (ALLOW_RADICALS alone, isomer channels mis-routed, was the unstable case). See [[project_moses00_photo_fix_validated]], [[project_moses00_photo_450x_artifact]] |
| `KINTERA_TITAN_PHOTO_INCLUDE_XSCN` | leave UNSET | `_kinetics_base_photo_rates` now drops the short-λ/X-ray `_XSCN_` radical cross-file block by default (KB-2012 moses00 doesn't load it; its ZK rates are exactly 0). Leaving this unset is correct. Setting it re-enables the block, which under ALLOW_RADICALS over-produces C3/C3H by 20-900× at L60-L80 (PUN rxn 23 C3H2→C3+H2). Only needed for ion runs that genuinely use the X-ray channels |

## What NOT to set

| env var | why not |
|---|---|
| `KINTERA_TITAN_NTIME>60` with default `MAX_DT=1e+7` | Adds more late steps at the same dt_max, can drift slightly but mostly redundant. NT=100 with same MAX_DT still works |
| `KINTERA_TITAN_MAX_DT>1e+7` at any NT | Triggers blow-up at L80+ for C2H6, C4H2, HC3N. The BE solver does not handle moses00's stiffness at large dt without line search or coupled Newton with strong damping (KB-mimic mode tested, oscillates and diverges) |
| `KINTERA_TITAN_KB_MIMIC=1` | Attempted algebraic SS via 1-step large-dt implicit. Median residual plateaus at 1e-6, max never drops below 1e+6, output values non-physical (C2N2 L20 ratio 3586, NH 1e+11). The implementation has no line search — Newton iterates diverge for stiff systems |

## Verifying the run

After the script finishes, the final block prints `Species ... ratio kt/KB`
at L20/L40/L60. Cross-check those values against the table above. If
CH3 L40 is not 0.56 ±0.05, *something has regressed* — start by checking
what env vars or transport changes were applied.

For deeper inspection (L70, L80, etc.):

```python
import numpy as np
data = np.load("/tmp/kt_moses00_ss.npz", allow_pickle=True)
species = list(data["species"])
c_kt = np.array(data["c_kintera_ss"])
c_kb = np.array(data["c_fort50"])
i = species.index("CH3")
for L in [40, 60, 70, 80, 88]:
    print(f"CH3 L{L}: kt={c_kt[L,i]:.2e} KB={c_kb[L,i]:.2e} ratio={c_kt[L,i]/c_kb[L,i]:.3f}")
```

## Investigating gaps

The remaining gaps at this baseline are documented:

1. **C6H6 / C4H2 / HC3N collapse at L60+**: photo loader bug ([[project_moses00_photo_loader_bug]]) — kintera misses 60 photo channels including 7 producing CH3, 5 producing C3H3, 3 producing CH3C2H. Cheng cross-section files exist; kintera's `_is_active_kinetics_base_photo_branch` gates them by IPHOTO opacity list. KB does not.

2. **L60+ transport drift** for HC3N, C2H2, C4H2: not yet localized. mr_hybrid is the best of {mr_diffusion, mr_exp, mr_hybrid} but residual gap remains. See [[project_moses00_L60_transport]], [[project_moses00_transport_audit]], [[project_moses00_face_flux]].

3. **NH at L40** orders of magnitude over KB: not yet investigated. Probably N(2D) chain coupling.

When working a new gap, **don't change the baseline knobs above**. Iterate
on the chemistry or transport code, re-run with the same env vars, and
compare to the table to see if the gap closes without introducing new ones.

## Unified core-engine pipeline (2026-06-03)

As of the `unify-titan-chem-onto-core` refactor, kintera's Titan chemistry can
be evaluated **through the compiled core `kintera.Kinetics`/`Photolysis`
engine** instead of the hand-rolled Python rate path. The drop-in is
`CoreChemistrySource` (`kinetics_base/titan/core_source.py`), a `LocalSourceTerm`:

```python
from kintera.kinetics_base.titan.core_source import CoreChemistrySource
core_chem = CoreChemistrySource(ts, pun_path, filtered_source_terms)
# moses00 has only chemistry + boundary terms (no condensation):
sources = [core_chem] + build_kinetics_base_titan_atm2d_source_terms(ts, boundary_terms)
sys, rhs = kt.build_implicit_step_system(ts.state, ts.kzz, dt, density=ts.density,
                                         transport_form="mr_diffusion", source_terms=sources, ...)
```

It builds `KineticsOptions.from_kinetics_base_pun(pun)` (291 plain Arrhenius +
50 `KBFalloff`), applies `ChembOverrideLayer` (21 UPDATE_CHEMB overrides), and a
core `Photolysis` over the active photo terms (unit `quadrature_weights` → the KB
per-bin `Σσ·F`). **Validated to reproduce the hand-rolled baseline to machine
precision** at every level: per-reaction rate ≤6.9e-16, net dC/dt 1.6e-15,
Jacobian 1e-6, 1-step BE solve, and full SS 5.1e-13 — so switching to the core
engine changes the moses00 SS by nothing. Harnesses: `diagnostics/stage5_core_*.py`.

Config: the `KINTERA_*` env vars are now mirrored by `CoreConfig`
(`atm2d/config.py`: transport_form, chem_solver) and `TitanConfig`
(`kinetics_base/titan/config.py`: photo_allow_radicals, photo_include_xscn,
disable_chemb_overrides, ei scales). `get_*_config()` still reads env fresh, so
the env-var recipe above is unchanged; `set_*_config(cfg)` installs an explicit
object. Transport default is now `mr_diffusion` when a density field is supplied.
See [[project_stage5_atm2d_wiring]], [[project_stage6_config]],
[[project_kintera_core_unification_refactor]].
