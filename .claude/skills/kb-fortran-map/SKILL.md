---
name: kb-fortran-map
description: Navigation map for the KINETICS-base Fortran source at diagnostics/KINETICS-base-compare/src/KINETGENX/ (~100k lines). Use whenever investigating WHY KB computes a specific rate, photo rate, transport, or boundary value differently from kintera. Reading KB source is the GROUND TRUTH for resolving any per-reaction or per-mechanism gap; reverse-engineering from outputs alone usually fails because KB has many runtime overrides in UPDATE_CHEMA/B that bypass the catalog rate constants.
---

# kb-fortran-map

KINETICS-base (KB) is a ~100k-line Fortran code with 300+ subroutines across 8 files. Many gaps between kintera and KB come from **runtime overrides** that replace the catalog (.pun-file) rate constants with hand-coded formulas (e.g. Moses et al. 2005 for Titan, Cravens et al. for ions). These overrides live in `UPDATE_CHEMA` and `UPDATE_CHEMB` and are GUARDED by `#ifdef __TITAN`, `__CHENG`, `__TITANTEST*`, `__EARTH`, etc.

**Iron rule**: when a kintera rate doesn't match KB's `prod+loss/<species>_*.dat`, grep the UPDATE_CHEMB subroutine for the reaction name BEFORE assuming it's a parser or formula bug.

## File inventory

All paths relative to `/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/src/KINETGENX/`.

| File | Lines | Purpose |
|------|-------|---------|
| `kinetgen2X.F` | 1.13M | Main driver + most subroutines (~250). Reads .pun, computes rates, integrates. |
| `kinetgen2X_merge.F` | 1.17M | Merge variant — usually ignore unless explicitly diffing. |
| `kinetgen1X.F` | 423k | Earth/Titan/Mars-specific subroutines including UPDATE_CHEMA, UPDATE_CHEMB. |
| `kinetgen1X_rl.F` | 429k | Reduced-Lagrangian variant — usually ignore. |
| `kinetics.F` | 124k | DISORT-related radiative transfer. |
| `kinetics_main.F` | 21k | Top-level main program. |
| `rad.F` | 28k | Radiation subroutine bodies. |
| `Karen_rad.F` | 9k | Auxiliary radiation. |

When in doubt about which file owns a routine: `grep -n "^      SUBROUTINE <NAME>\|^      FUNCTION <NAME>" *.F`

## Key subroutines by question

### "What rate constant does KB use for reaction X?"

The rate constant `ZK(I,IL,IN,J)` for reaction `I` at cell (IL,IN,J) is computed in `kinetgen2X.F:15190-15300` (`SUBROUTINE ZKT`). Path:

1. **Base formula** (line 15224-15225):
   - `ZK1(A,B,C,T,T0) = A*((T/T0)**B)*EXP(C/T)` — used when B > 0
   - `ZK2(A,B,C,T,T0) = A*((T0/T)**ABS(B))*EXP(C/T)` — used when B < 0
   - Called via `AKA(I)`, where `AKA(I) = TBEFF*AK(I)` (line 15238). `TBEFF` defaults to 1.0 (set in `ZKTB` DATA statement, `kinetgen1X.F:7459`).

2. **Three-temperature-range overrides** (`kinetgen1X.F:5692-5740`):
   - If `AK2(I) ≠ 0`, KB has a 2nd-range or 3rd-range rate constant.
   - .pun file stores 3 blocks per reaction (`AK,BK,CK / AK2,BK2,CK2 / AK3,BK3,CK3`); kintera's parser currently only reads block 1.

3. **Hand-coded overrides in `UPDATE_CHEMB`** (`kinetgen1X.F:6803-7384`):
   - Specific reactions get formulas like Moses et al. 2005 that REPLACE the catalog rate.
   - **Example found 2026-05-23**: `H+C2H3 → C2H2+H2` (ISP(468)) at line 7077-7079: `zk = 7e-11*T^0.18 - zk(313)*density`. This is 10× the catalog rate.
   - **To find an override**: `grep -n "<species_A>+<species_B>\|<formal name>" kinetgen1X.F` (uses Fortran comments). Then `grep -n "isp(<rxn_index>)" kinetgen1X.F`.

4. **Special branches**:
   - `kinetgen1X.F:5027` etc.: per-reaction multipliers (e.g. `ZK(ISP(107)) = 10. * ZK(ISP(107))`).
   - `kinetgen1X.F:9185-9210`: rate cloning across reactions (`zk(j) = zk(j1)`, sometimes `* 10` or `/ 300`).

### "What photo rate / photolysis cross-section is used?"

- `kinetgen2X.F:5580` `SUBROUTINE CROST` — main cross-section loading.
- `kinetgen2X.F:5608` `SUBROUTINE CROST_INTER` — temperature-interpolating cross-section.
- `kinetgen2X.F:6930` `SUBROUTINE JPHOTO` — photo-rate per altitude / wavelength.
- `kinetgen2X.F:11512` `SUBROUTINE RX` — rate × cross × flux integration.
- `kinetgen2X.F:11791` `SUBROUTINE SLANT_CAL` — slant-column attenuation.
- `kinetgen1X.F:8067` `SUBROUTINE CROSS_SRB` — Schumann-Runge band (Earth-only).

### "How does KB do radiation / actinic flux?"

- `kinetgen2X.F:3171` `ATTEN1` — attenuation main entry.
- `kinetgen2X.F:3262` `ATTEN1IO` — IO variant.
- `kinetgen2X.F:3405` `FUNCTION ATT0(APBF,T,COFF0)` — the **Cogley-Borucki daily-average attenuation function** (line 3420: `ATT0 = (EXP(-CF)/PI)*ACOS(ZF)`).
- `kinetgen2X.F:10424` `RAD` — main radiation entry.
- `kinetgen2X.F:10181` `RAD_CHENG` — Cheng-variant radiation (relevant for Titan ions network).
- `kinetgen2X.F:11960` `slant_z_ge_0` — slant calculation for above-horizon.
- `kinetgen2X.F:12077` `SOLAR` — solar source term for DISORT.
- `kinetics.F` — DISORT radiative-transfer library (multi-stream scattering, used when `NZEN!=0` or `NDISORT!=0`).

### "Where do prod+loss/.dat files get written?"

`kinetgen1X.F:10891-10977` (look for `'prod+loss/'//trim(amol(...))//'_prod.dat'`).

Format: `srate(rxn_id, lat, lon, alt)` is written directly. `srate` is computed in `kinetgen2X.F:10780-10809` as `ZK(I) × Π[reactant_concentrations]^stoichiometry`. **Output is REACTION RATE (not species tendency)** — no stoichiometry multiplication for the species column.

### "How does KB read the .pun file?"

`kinetgen2X.F:741-789` `SUBROUTINE READTRUNCB`:
- Line 770-776: per-reaction reads of `NOREAC, NOPROD, ICOF/INDX/CHAR (7 species), AK, BK, CK, DK, EK, FK, TL, TH`. Format `11004` (FORMAT 1004 with extra 5X leading skip).
- For the THREE rate-block variant (`AK2, BK2, CK2, AK3, BK3, CK3, ...`), additional reads happen elsewhere (search for `READ.*AK2\|READ.*AK3`).

### "What about boundary conditions?"

- `kinetgen2X.F:3605` `BND` — main BC entry.
- `kinetgen2X.F:4090` `BNDRY1` — implicit BC matrix (Type-2 deposition/escape velocities applied to Jacobian diagonal).
- `kinetgen1X.F:3402` `BNDA` — auxiliary BC.

### "How does the integration step work?"

- `kinetgen2X.F:13120` `STEP` — time step driver.
- `kinetgen2X.F:7175` `MARCH` — Newton iteration step.
- `kinetgen2X.F:6664` `JACOB` — Jacobian assembly.
- `kinetgen2X.F:14012` `UPDATE` — per-step variable updates.

## Compile-time guards (ifdef macros)

The KB code is one source with many compile targets. Key macros:

| Macro | Meaning | Where to look |
|-------|---------|---------------|
| `__TITAN` | Titan-specific (default for our oracle) | grep for `#ifdef __TITAN` blocks |
| `__CHENG` | Cheng/Yelle network mods (excludes Lavvas aerosols) | combined `__TITAN && !__CHENG` for Lavvas |
| `__TITANTEST`, `__TITANTEST1` | Experimental Titan changes (often zero-out reactions) | These often disable reactions; check before assuming a reaction is active |
| `__EARTH` | Earth-specific (irrelevant for Titan) | safe to ignore |
| `__MARS`, `__ISM`, `__DISK` | Mars/ISM/protoplanetary disk | safe to ignore for Titan |
| `__ALL` | Enables all code paths | search for `#ifdef __ALL` in subroutines you care about |
| `__MPI` | Parallel MPI variant (output and gather) | usually safe to ignore for physics questions |
| `__DOPRINT` | Debug print statements | always safe to ignore |

**Always check `__TITAN`/`__CHENG` blocks first** for any species or reaction. The default for Titan oracle runs is `__TITAN __CHENG` (no Lavvas aerosols).

## Quick recipe: "Why does KB compute reaction X differently?"

1. Find the reaction in `Reactions.dat` (the catalog) — note KB's ID and reactants/products.
2. Find the .pun reaction line — note ISP species index (column 4-5 of the formatted line).
3. `grep -n "isp(<index>)\|ISP(<index>)" kinetgen1X.F` — looks for overrides.
4. If found in `UPDATE_CHEMA` (line 3952) or `UPDATE_CHEMB` (line 6803), READ THE OVERRIDE — that's the KB rate, not the catalog.
5. Also check `kinetgen1X.F:9100-9210` for cross-reaction rate cloning blocks.
6. Reproduce the formula in our `_pun_rate_constant` if applicable, OR register a special override in `source_terms.py`.

## Important: don't reverse-engineer KB

Repeated lesson during the 2026-05-23 work: trying to back out KB's formula from `prod+loss` rate ratios is unreliable because of UPDATE_CHEMB overrides. **Open the source code, grep for the reaction, READ the override.** Total time: 5-15 minutes per gap, vs hours of guess-and-check.

## What's NOT yet mapped (TODO if needed)

- `UPDATE_CHEMA` content (line 3952-6802, ~3000 lines). Likely has more overrides like UPDATE_CHEMB.
- `kinetgen2X.F` advection/dynamics subroutines (lines 16000-22000) — only relevant if transport gaps emerge.
- Falloff/Troe formulas: `zkcalcx`, `ZKCALC`, `ZKINT` — search by name when investigating 3-body reactions.
- The exact format of three-rate-block storage in the .pun file (do they overflow lines? multi-line records?).
