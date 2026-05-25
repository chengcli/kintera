## 1. Coupled advance implementation

- [x] 1.1 Audit `python/atm2d/newton/coupled.py` — found
  `newton_implicit_step` already implements the coupled solve. The
  driver `diagnostics/no_grain_stability.py` already routes through
  it via `KINTERA_SOLVER_MODE=coupled` (existed pre-change).
- [x] 1.2 No new `coupled_advance` function needed — the existing
  `newton_implicit_step` + driver's `step_fn_coupled` cover the
  contract. Verified by running NT=44 KB-Titan and dumping
  `/tmp/kt_traj_44_coupled_loose.npz`.
- [x] 1.3 Hooks wired: `system_postprocess` and
  `concentration_postprocess` already accepted by
  `newton_implicit_step`; driver maps `_apply_dirichlet` and
  `_apply_pins` to them.
- [x] 1.4 NaN-on-non-convergence sentinel applied in driver
  `step_fn` (line 388 of no_grain_stability.py), uniformly for both
  split and coupled modes. Confirmed working at NT=44 step 44 with
  `acc=6 rej=5 reasons=[reject_non_finite=5]`.
- [-] 1.5 No new exports needed: `kt.newton_implicit_step` is
  already top-level via `python/atm2d/__init__.py`. Wrapping it in a
  new `coupled_advance` helper deferred — drivers can call directly.

## 2. Selector and driver integration

- [x] 2.1 `KINTERA_SOLVER_MODE` env var already exists in
  `no_grain_stability.py:267`; values `split` and `coupled` already
  honored. Validated by running both modes back-to-back.
- [ ] 2.2 Add an `[setup] solver mode = ...` print at driver
  startup (currently only printed inside step_fn doc); cosmetic.
- [x] 2.3 `KINTERA_CHEM_SOLVER` only kicks in inside operator-split
  path (`operator_split.py:_select_chem_step`); coupled mode uses
  `newton_implicit_step` directly. Behavior is already correct;
  documenting in tasks 5.x.

## 3. Validation

- [x] 3.1 Ran `KINTERA_SOLVER_MODE=coupled
  KINTERA_TITAN_NETWORK_MODE=neutrals_only KINTERA_TITAN_NTIME=44`
  with tight `NEWTON_TOL=1e-4` → cascading rejections (step 39 took
  481 sub-iterations). KB-style loose tol settings made it
  practical: `NEWTON_TOL=5e-2 NEWTON_DAMP_FACTOR=0.1
  NEWTON_DAMP_TRIGGER=0.1 NEWTON_MAX_ITER=80`. Dump at
  `/tmp/kt_traj_44_coupled_loose.npz`. Final stats: `acc=49 rej=5
  non_converged=5 avg_iter=15.4 max_iter=80`. Rejections only at the
  last macro step (dt=3.16e+6).
- [x] 3.2 Counted zero-between-non-zero cells:
  - op-split + Newton (the original buggy path): 68
  - op-split + BDF chemistry-only: 0
  - **coupled + loose Newton: 21** (down from 68, 69% reduction)
  - The 21 are all trace radicals (C2H5, C3H7, C4H9, C6H, C6H3-5,
    C4H10, C3H8) at concentrations 1e-12 to 1e-3 cm^-3. Major
    species are all smooth.
- [x] 3.3 C2H6 at the user-flagged L26 dip (~600 km):
  - op-split + Newton: 0.0 (the dip)
  - op-split + BDF: 90.7 (smooth)
  - **coupled + loose: 32.3 (smooth)** — fix achieved
  - Both BDF and coupled paths produce monotonic profiles around
    L26. The coupled path matches BDF within 3× on C2H6 absolute
    values across L22-L32.
- [-] 3.4 Comparison vs KB fort.7: deferred to a follow-up change.
  Both BDF and coupled paths are 2-4 orders of magnitude below KB
  because the NT=44 schedule reaches only 53 simulated days vs KB's
  1660-year integration. This is a simulated-time gap, not a
  solver gap.

## 4. Tests

- [ ] 4.1 Add a regression test in `tests/test_atm2d.py` (or new
  `test_coupled_advance.py`) that runs `newton_implicit_step` on a
  small fixture (5-layer column, 3-species network) and asserts the
  returned state matches a reference within machine precision
- [ ] 4.2 Add a regression test asserting that, when given a
  pathologically large `dt`, the coupled advance returns
  `NewtonResult.converged=False` (and the driver maps it to NaN
  via the step_fn sentinel) rather than silently emitting a corrupt
  state
- [ ] 4.3 Add a regression test asserting that `KINTERA_SOLVER_MODE`
  values `split` and `coupled` both produce finite output on the
  same fixture and that the coupled path's zero-between-non-zero
  count is strictly lower than split's at NT=44

## 5. Documentation and report integration

- [ ] 5.1 Update `STATUS_REPORT.html` (via
  `diagnostic_tools/build_report.py`) to add a coupled-Newton variant
  column alongside the existing Newton and BDF columns in section 3
- [ ] 5.2 Update the section-3 callout to describe the coupled-solver
  win and the operator-split failure mode it addresses
- [x] 5.3 Memory entry created:
  `/home/sam2/.claude/projects/-home-sam2-dev-kintera/memory/project_coupled_solver_settings.md`
  documents the recommended loose-tol + heavy-damping env-var
  configuration for the coupled path.
- [x] 5.4 Updated memory `MEMORY.md` index to cross-reference the
  coupled-solver settings entry.

## 6. Follow-up (deferred to next change)

- [ ] 6.1 Reduce the 21 trace-radical zeros further by either
  tightening tolerance in a dt-adaptive way or adding KB-style
  positivity-clipping with damping (KB's CONVRG has a per-cell
  positivity override that re-enters Newton with smaller step).
- [ ] 6.2 Implement a coupled BDF path (`scipy.integrate.solve_ivp`
  over the full `dn/dt = -T·n + S(n)` system) for the cases where
  even the loose-tol coupled Newton struggles. Block-banded
  Jacobian sparsity pattern.
- [ ] 6.3 Add a true `coupled_advance` wrapper around
  `newton_implicit_step` that supports `n_subcycles` (like
  `operator_split_advance`) for cases where the schedule's outer
  dt is larger than the coupled Newton's stability limit.
- [ ] 6.4 Investigate KB's outer iteration loop (`DIFFUS+MARCH`
  alternation) and whether adding it improves the trace-radical
  accuracy without going to full BDF.
