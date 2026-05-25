## 1. Coupled advance implementation

- [ ] 1.1 Audit `python/atm2d/newton/coupled.py` to understand the
  existing coupled-step plumbing and identify what's missing for a
  full `coupled_advance` analogous to `operator_split_advance`
- [ ] 1.2 Implement `coupled_advance(state, dt_target, *, kzz,
  source_terms, …)` in `python/atm2d/newton/coupled.py` that builds
  the implicit operator via `build_implicit_step_system`, iterates
  Newton with rollback, and supports the same hook contract as
  `operator_split_advance`
- [ ] 1.3 Ensure boundary-condition `_apply_boundary_conditions` and
  hook callbacks (`transport_system_postprocess`,
  `transport_concentration_postprocess`, `chemistry_postprocess`,
  `between_substeps_postprocess`) are wired the same way as
  `operator_split_advance` so KB-Titan pins keep working
- [ ] 1.4 Add the NaN-on-non-convergence sentinel so
  `adaptive_advance`'s `default_accept` rejects-and-retries (matches
  the BDF chemistry-only pattern from the previous change)
- [ ] 1.5 Export `coupled_advance` from
  `python/atm2d/newton/__init__.py` and re-export through
  `python/atm2d/__init__.py` so drivers can call
  `kt.atm2d.newton.coupled_advance` directly

## 2. Selector and driver integration

- [ ] 2.1 Add `KINTERA_SOLVER` env-var selector to
  `python/atm2d/newton/operator_split.py` (or a new
  `dispatch.py`) that routes between `operator_split_advance` (default)
  and `coupled_advance` based on the env var
- [ ] 2.2 Update `diagnostics/no_grain_stability.py` to print the
  active solver (`split` vs `coupled`) at startup so log output is
  unambiguous
- [ ] 2.3 Verify the existing `KINTERA_CHEM_SOLVER` env var (newton /
  bdf / lsoda / radau) is ignored when `KINTERA_SOLVER=coupled`
  since chemistry is no longer a standalone step; document this in
  the selector code

## 3. Validation

- [ ] 3.1 Run `KINTERA_SOLVER=coupled
  KINTERA_TITAN_NETWORK_MODE=neutrals_only KINTERA_TITAN_NTIME=44` on
  the KB-Titan example and dump to `/tmp/kt_traj_44_coupled.npz`;
  verify the run completes with `non_converged=0` and `acc=44,
  rej=0`
- [ ] 3.2 Count zero-between-non-zero cells in the coupled dump;
  this number MUST be 0 (matching the BDF chemistry-only run)
- [ ] 3.3 Compare the coupled-Newton concentration profiles to the
  BDF chemistry-only run at matched simulated time; verify the
  agreement on major species (CH4, H2, HCN, C2H2, C2H6) is within
  10% across all 40 active layers
- [ ] 3.4 Compare coupled-Newton against KB `/tmp/kb_run_xport/fort.7`
  for major species; document any persistent divergences (these are
  separate from the operator-split issue and out of scope for this
  change)

## 4. Tests

- [ ] 4.1 Add a regression test in `tests/test_atm2d.py` (or new
  `test_coupled_advance.py`) that runs `coupled_advance` on a small
  fixture (5-layer column, 3-species network) and asserts the
  returned state matches a reference within machine precision
- [ ] 4.2 Add a regression test asserting that, when given a
  pathologically large `dt`, the coupled advance returns
  `NewtonResult.converged=False` (and matching NaN concentration)
  rather than silently emitting a corrupt state
- [ ] 4.3 Add a regression test asserting that the
  `KINTERA_SOLVER=split` env var preserves the exact behavior of the
  current `operator_split_advance` on the same fixture (numerical
  parity to <1e-10 relative)

## 5. Documentation and report integration

- [ ] 5.1 Update `STATUS_REPORT.html` (via
  `diagnostic_tools/build_report.py`) to add a coupled-Newton variant
  column alongside the existing Newton and BDF columns in section 3
- [ ] 5.2 Update the section-3 callout to describe the coupled-solver
  win and the operator-split failure mode it addresses
- [ ] 5.3 Add a memory entry under
  `/home/sam2/.claude/projects/-home-sam2-dev-kintera/memory/`
  documenting the coupled-advance selector and the KB
  `DIFFUS`-equivalent semantics
- [ ] 5.4 Update existing memory `project_kintera_newton_silent_unconverge.md`
  to cross-reference the coupled-advance change (the deeper fix on
  top of the BDF chemistry-only workaround)
