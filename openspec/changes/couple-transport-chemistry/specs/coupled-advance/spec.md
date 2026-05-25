## ADDED Requirements

### Requirement: Coupled transport+chemistry advance API

The system SHALL provide a `coupled_advance` function (or method) that
advances `state.concentration` by `dt_target` by solving the combined
implicit operator `(I/dt − T − ∂S/∂c) Δc = − F(c_k)` per Newton
substep, where `T` is the transport operator and `S(c)` is the
chemistry source term. The function MUST accept the same hook kwargs
as `operator_split_advance` (`kzz`, `source_terms`,
`species_diffusion_scale`, `transport_system_postprocess`,
`transport_concentration_postprocess`, `chemistry_postprocess`,
`between_substeps_postprocess`, `newton_kwargs`).

#### Scenario: Coupled advance solves transport and chemistry in one Newton system

- **WHEN** `coupled_advance(state, dt, kzz=…, source_terms=…)` is called
  with `dt = 1.0e+5 s` on the Titan neutrals network at the initial
  atmosphere
- **THEN** the returned concentration MUST satisfy
  `|F(c_new)| < convergence_tol · max(|c_new|)` and the returned
  `NewtonResult.converged` MUST be `True`
- **AND** the per-cell residual MUST account for both
  `(I/dt − T) c_new − c0/dt` (transport+identity terms) and the
  chemistry source `S(c_new)`

#### Scenario: Coupled advance uses the same hook contract as operator_split

- **WHEN** `coupled_advance(...)` is invoked with
  `transport_concentration_postprocess` and `chemistry_postprocess`
  callbacks provided
- **THEN** both callbacks MUST be applied at the same logical points
  as in `operator_split_advance`:
  `transport_concentration_postprocess` is called once after the
  initial coupled Newton solve assembles its operator;
  `chemistry_postprocess` is called after each accepted Newton
  iteration (matching the iteration-level pin re-application used by
  `chemistry_only_newton_step`)

### Requirement: Selector via KINTERA_SOLVER env var

The system SHALL select between the existing operator-split path and
the new coupled path via the `KINTERA_SOLVER` environment variable.
Values `split` (default) and `coupled` MUST be accepted. Any other
value MUST raise `ValueError` at startup with a message naming the
allowed values.

#### Scenario: Default is operator-split for backward compatibility

- **WHEN** `KINTERA_SOLVER` is unset and a driver calls
  `operator_split_advance` (or the dispatch wrapper that replaces it)
- **THEN** the implementation MUST use the existing operator-split
  path (BE transport followed by chemistry-only Newton or BDF as
  selected by `KINTERA_CHEM_SOLVER`)

#### Scenario: Coupled path selected by env var

- **WHEN** `KINTERA_SOLVER=coupled` and a driver calls the dispatch
  wrapper
- **THEN** the implementation MUST use `coupled_advance` instead of
  `operator_split_advance`, and the returned `NewtonResult` MUST be
  the result of the coupled Newton solve

#### Scenario: Unknown selector raises

- **WHEN** `KINTERA_SOLVER=fancy_made_up_method`
- **THEN** the dispatch wrapper MUST raise `ValueError` whose message
  includes `KINTERA_SOLVER`, the offending value, and the allowed
  list `split, coupled`

### Requirement: Per-substep rollback on Newton non-convergence

The system SHALL signal Newton non-convergence to the
`adaptive_advance` controller by returning a non-finite concentration
tensor (NaN-filled), causing `default_accept` to reject the step.
The framework MUST then halve `dt` and retry without committing the
failed state.

#### Scenario: Non-converged coupled Newton triggers rollback

- **WHEN** `coupled_advance` is called with a `dt` so large that the
  coupled Newton fails to converge within `max_iterations` (default
  100)
- **THEN** the returned `NewtonResult.converged` MUST be `False`
- **AND** the returned `NewtonResult.concentration` MUST be a
  `torch.full_like(c0, float("nan"))` sentinel
- **AND** `state.concentration` MUST be restored to the pre-call
  value (no partial commit)
- **AND** when this is invoked through `adaptive_advance` the
  controller MUST record a `reject_non_finite` entry and re-attempt
  with `dt_attempt = 0.5 · dt_attempt`

### Requirement: Coupled advance preserves boundary pins and Dirichlet rows

The coupled-Newton solve SHALL apply Dirichlet row replacements
(KB-Titan fixed species and lower-boundary pinned mixing ratios)
through `transport_system_postprocess` exactly as the operator-split
path does. Concentration pins (boundary species held to a value at
specific levels) SHALL be re-applied after each accepted Newton
iteration via `chemistry_postprocess`.

#### Scenario: Lower-boundary CH4 pin holds across the coupled solve

- **WHEN** `coupled_advance` runs on KB Titan with CH4 lower-boundary
  pinned at mixing ratio `4.0e-4` (`CONC[L0] = 4e-4 · n_atm[L0]`) and
  `dt = 1e+6 s`
- **THEN** after the coupled solve the returned
  `concentration[0, 0, ch4_idx]` MUST equal `4.0e-4 · n_atm[L0]`
  to within `1e-6` relative tolerance

#### Scenario: Upper-boundary H escape velocity remains active

- **WHEN** `coupled_advance` runs with `H` upper-boundary velocity
  `v_esc = 1.44e+5 cm/s` at `L39`
- **THEN** the chemistry source-term list MUST still include the
  `upper_boundary_velocity` term for `H` at `L39`, and the resulting
  Newton residual MUST include the `−v_esc · n[H] / dz[L39]` loss
  term in the chemistry Jacobian
