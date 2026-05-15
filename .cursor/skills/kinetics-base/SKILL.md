---
name: kinetics-base
description: Run and debug KINETICS-base Titan oracle comparisons for kintera. Use when the user mentions KINETICS-base, Titan oracle tests, external Titan runs, kintitan outputs, prod+loss budgets, or long-time neutral/ion chemistry traces.
---

# KINETICS-base Titan Oracle

## Paths

Default external checkout used by this repo:

```bash
export KINTERA_KINETICS_BASE_ROOT=/Users/sihechen/Dev/kintera/diagnostics/KINETICS-base-compare
export KINTERA_KINETICS_BASE_EXECUTABLE=/Users/sihechen/Dev/kintera/diagnostics/KINETICS-base-compare/build/bin/titan.release
```

Titan input paths under the checkout:

```text
examples/titan/kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun
examples/titan/kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.special
examples/titan/ions_c6h7+_H2CN.inp-1
examples/titan/kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz
examples/titan/titan_Cheng_N_ions_H2CN.bc_save
```

## Quick Checks

Run focused external oracle tests from the repo root:

```bash
KINTERA_KINETICS_BASE_ROOT=/Users/sihechen/Dev/kintera/diagnostics/KINETICS-base-compare \
KINTERA_KINETICS_BASE_EXECUTABLE=/Users/sihechen/Dev/kintera/diagnostics/KINETICS-base-compare/build/bin/titan.release \
python -m pytest tests/test_kinetics_base.py::test_external_titan_multi_step_equivalence_gap_if_available -q -p no:randomly
```

If the executable fails on the first `TIME STEP = 1.0E-15 SEC`, retry once before treating it as a kintera regression. This KINETICS-base build has intermittent startup failures.

## Manual Run Pattern

Use a stable work directory under `diagnostics/kinetics_base_oracle_runs/`. Create `prod+loss/`, patch `fort.81` by setting `NTIME` and `ISTART=0`, then symlink the Fortran unit files:

```text
fort.1   -> Cheng_ions_c6h7+_v3_H2CN.pun
fort.3   -> kintitan.truncate
fort.4   -> Cheng_ions_c6h7+_v3_H2CN.special
fort.15  -> titan_Cheng_N_ions_H2CN.bc_save
fort.20  -> Cheng_wavel.dat
fort.21  -> flare_kin_oct2003.inp
fort.27  -> kintitan-difrad-2.inp
fort.30  -> Cheng_catalog_v4.dat
fort.45  -> kintitan_aerosol_interp_albedo.inp
fort.46  -> kintitan_aerosol_interp_gr.inp
fort.47  -> kintitan_aerosol_interp_asymm.inp
fort.50  -> kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz
fort.81  -> patched run input
crossfilepath -> Cheng_cross
fort.7   -> kintitan.out.pun
fort.10  -> kintitan.res
fort.11  -> titandebug.dat
```

The helper functions in `tests/test_kinetics_base.py` already implement this pattern: `_external_titan_paths()`, `_write_fresh_start_run_input()`, and `_run_titan_steps()`.

## Comparing kintera

For repeated kintera steps, use `_solve_titan_transport_steps()` from `tests/test_kinetics_base.py` as the reference pattern. It applies the Titan source terms, solves the implicit system, clamps nonnegative, and reapplies KINETICS-base boundary pins each step.

KINETICS-base grows the negative-`DELTIM` startup sequence by half decades in this Titan run:

```text
1e-15, 3.2e-15, 1e-14, ...
```

## Budget Trace Notes

Use `Reactions.dat` to map operational reaction ids in `prod+loss/*.dat` back to equations. When parsing equations, split reactants/products only on spaced separators like `" + "`; plain `split("+")` breaks charged species such as `CH+` and `C+`.

Generated trace outputs belong under `diagnostics/kinetics_base_titan_budget_trace/` or `diagnostics/kinetics_base_oracle_runs/` and should remain ignored by git.
