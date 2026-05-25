## Why

Kintera's chemistry-only Newton solver fails silently for the Titan
photochemistry network at `dt ≳ 10³ s` because the operator-split
architecture decouples transport from chemistry. At each macro step the
transport BE solve and chemistry Newton fight each other: the
chemistry-Newton's starting point (post-transport concentration) is far
from any sensible chemistry steady state, so Newton diverges without
the transport gradient to anchor it. KB's `DIFFUS` solves transport
and chemistry **simultaneously** in one Newton system, which is why it
sustains `dt` up to `10⁹ s` with the same chemistry network and the
same initial conditions.

The previous session swapped scipy BDF in as a drop-in replacement for
the chemistry-only Newton step. BDF eliminated the
zero-between-non-zero artifacts (Newton 68 cells → BDF 0) but
inherited the operator-split limitation: it integrates chemistry over
`dt` *after* transport has produced an unphysical intermediate state,
adding numerical cost without addressing the underlying split-error
accumulation.

## What Changes

- Add a new **coupled transport+chemistry** advance path that builds a
  single implicit operator over both transport (sparse vertical
  diffusion + species coupling) and chemistry (cell-local source
  Jacobian), and solves it with one Newton iteration per substep.
- **BREAKING (default flip later)**: behind an env-var/kwarg flag,
  reroute `operator_split_advance` callers to the new coupled path.
  Initial release keeps the operator-split path available
  (`KINTERA_SOLVER=split` falls back) so existing tests and other
  consumers don't break in one shot.
- Add per-substep rollback semantics matching KB's `RETRY` machinery:
  on Newton non-convergence, restore the pre-step state, halve `dt`,
  retry. This is what the `step_fn → NaN` patch tried to do but with
  the cost amortized over a single coupled solve instead of cascading
  through operator-split retries.
- Optional: replace the single coupled Newton with a coupled BDF
  integrator (`scipy.integrate.solve_ivp` over `dn/dt = -T·n + S(n)`)
  for the same robustness guarantees BDF gave for chemistry-only.

## Capabilities

### New Capabilities

- `coupled-advance`: One-shot transport+chemistry advance over a macro
  `dt`. Builds the implicit operator `(I/dt - T - J_S(c))`, applies
  boundary conditions / pins, solves with Newton (and optionally
  internal substepping), and commits the result. Includes
  per-substep rollback on non-convergence.

### Modified Capabilities

- (none — operator_split remains available alongside the new coupled
  path during the transition.)

## Impact

- **Code**: new module `python/atm2d/newton/coupled_advance.py` (or
  extension of `python/atm2d/newton/coupled.py` which currently only
  has the chemistry+transport coupled-step plumbing). Update
  `python/atm2d/newton/operator_split.py` only to add a selector
  routing to the new path when `KINTERA_SOLVER=coupled`.
- **Drivers**: `diagnostics/no_grain_stability.py` gains a
  `KINTERA_SOLVER` env var that selects `split` (current) / `coupled`
  (new). All other env vars (`KINTERA_CHEM_SOLVER`,
  `KINTERA_REJECT_NON_CONV`, atomic-projection, etc.) keep working.
- **Tests**: add a regression test running NT=44 with coupled solver
  on a small fixture and asserting zero "zero-between-non-zero" cells
  (the C2H6-dip fingerprint).
- **Diagnostics**: the existing `STATUS_REPORT.html` gains a fourth
  variant column (operator-split-Newton, operator-split-BDF,
  coupled-Newton, coupled-BDF) once the new path is wired.
- **Dependencies**: none new for the Newton coupled path; scipy
  already present for the BDF variant.
