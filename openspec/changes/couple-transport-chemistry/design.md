## Context

Kintera's current architecture (`python/atm2d/newton/operator_split.py`)
runs one BE transport solve and one chemistry-only Newton per macro
`dt`. For Titan-like networks with reaction timescales spanning
microseconds to years, the chemistry-Newton starting from a
post-transport intermediate diverges at `dt ≳ 10³ s`. KB
(`kinetgen1X.F:DIFFUS`) avoids this by writing the implicit Newton
system over the combined transport+chemistry residual:

```
F(c_new) = (I/dt - T) c_new − c0/dt − S(c_new)
```

where `T` is the (sparse) transport operator and `S(c)` is the
cell-local chemistry source. Linearizing gives

```
J(c_k) = I/dt − T − ∂S/∂c |_{c_k}
J(c_k) · Δc = −F(c_k)
```

which is a sparse system the size of the operator-split transport
matrix (block-tridiagonal in 1D × dense per cell from chemistry
Jacobian), solved with one tridiagonal-blockwise direct solver per
Newton iteration. KB does this with a hand-coded tridiagonal solver
(`TRIGO1`); we already have `kintera.atm2d.solver.solve_sparse_system`
that can solve the same structure via CSR LU.

This proposal does not invent the coupled formulation; both
`python/atm2d/assembly.py:build_implicit_step_system` and the
existing-but-unused `python/atm2d/newton/coupled.py` already build the
coupled operator. The change is to (a) make this the default advance
path, (b) wrap it with Newton-with-rollback semantics matching KB, and
(c) expose it under a `KINTERA_SOLVER=coupled` selector.

## Goals / Non-Goals

**Goals:**

- One-shot coupled transport+chemistry advance: a single Newton
  iteration solves a system that sees both the transport gradient and
  the chemistry source, avoiding the operator-split disagreement that
  drives Newton divergence at large `dt`.
- Restore correctness for `dt` up to KB's `1e+9 s` schedule without
  hidden non-convergence (matching the new BDF chemistry-only path,
  but coupled with transport in the same solve).
- Preserve the existing operator-split path behind
  `KINTERA_SOLVER=split` so existing tests, scripts, and other
  consumers can opt in to the change gradually.
- Provide per-substep rollback on non-convergence (mirroring KB's
  `RETRY`): if the coupled Newton fails after max_iter, restore the
  pre-step concentration, halve `dt`, and retry. Surface this as
  `acc/rej` counts in the diagnostic logs (already supported by
  `adaptive_advance`).

**Non-Goals:**

- Match KB's `DIFFUS` numerically bit-for-bit. The coupled formulation
  is shared, but discretization details (tridiagonal Thomas vs sparse
  LU, damping schedule, atom-projection placement) will differ.
- Replace operator-split for the GPU path. The initial coupled advance
  is CPU-only via `solve_sparse_system` (which already lives there).
  Operator-split remains available for GPU work.
- Implement coupled BDF (`solve_ivp` over `dn/dt = -T n + S(n)` for
  the full state) in this change. The Newton-coupled path is the
  scope; a coupled BDF can layer on top once the coupled Jacobian and
  RHS construction is in place.

## Decisions

### Decision 1: build on `coupled.py`, not new file

`python/atm2d/newton/coupled.py` already exists and contains plumbing
for coupled chemistry+transport (currently unused by drivers). The
new advance lives there. Alternative: new `coupled_advance.py`
sibling. Rejected because the coupled-step pieces shouldn't fragment.

### Decision 2: Newton-with-rollback, not coupled BDF first

Coupled BDF would require wrapping the full `(ncol·nlyr·nspecies)`
state through scipy callbacks plus a sparse Jacobian builder for the
combined operator. That's strictly more work than the existing
Newton, and the coupled-Newton's failure mode is much milder than
chemistry-only-Newton's (since transport anchors the initial guess).
Defer coupled-BDF to a follow-up change once the coupled-Newton path
is validated.

### Decision 3: per-substep rollback in `adaptive_advance`, not in the step

The existing `kt.adaptive_advance` already has rollback semantics —
it just needs to see `non_finite` from the step on failure. Mirror
the chemistry-only BDF pattern: if coupled Newton returns
`converged=False`, the wrapper returns
`torch.full_like(c0, float("nan"))`, which `default_accept` rejects.
Alternative considered: extend `default_accept` to take a
`converged` flag from a richer return type. Rejected because it's
more invasive and the NaN-return pattern already works.

### Decision 4: env-var selector, not config-file flag

Match the existing `KINTERA_CHEM_SOLVER` and
`KINTERA_TITAN_NETWORK_MODE` patterns. Add `KINTERA_SOLVER` with
values `split` (default) or `coupled`. After the coupled path is
validated, a follow-up change flips the default.

### Decision 5: keep `chemistry_postprocess` and Dirichlet hooks intact

The new coupled advance accepts the same hook kwargs that
`operator_split_step` does (`transport_system_postprocess`,
`transport_concentration_postprocess`, `chemistry_postprocess`). KB
Titan boundary pins and atomic-budget projection still apply in the
same places. Alternative: redesign the hook contract. Rejected
because every consumer relies on the current contract.

## Risks / Trade-offs

- **Performance**: coupled Newton solves a system larger than
  operator-split's chemistry-only system (rougly `ncol·nlyr·nspecies`
  vs `nspecies` per cell). For Titan that's ~6400 × 6400 vs 128 × 128.
  Mitigation: the system is block-tridiagonal in vertical structure,
  so sparse LU cost is `O(ncol·nlyr·nspecies³)`, comparable to running
  the per-cell solve `ncol·nlyr` times. Initial benchmark expected on
  the same order as operator-split.
- **Numerical conditioning at very large dt**: at `dt = 1e+9 s`, the
  `(I/dt − T − J)` matrix is dominated by `−T − J`, so conditioning
  is set by the chemistry Jacobian's worst eigenvalues (typically the
  fast ion-recombination rates). Mitigation: the chemistry-only BDF
  path remains available and is the right tool for ultra-large dt;
  the coupled-Newton path covers the regime where transport anchors
  the chemistry.
- **Loss of operator-split as the validated default**: the existing
  diagnostic dumps (`/tmp/kt_traj_*.npz`) were generated with
  operator-split. Comparisons against KB will need fresh dumps with
  the coupled path. Mitigation: keep operator-split available, run
  both during validation.
- **Pin / Dirichlet interaction**: coupled solves enforce boundary
  pins via the operator's row-replacement (already implemented in
  `_apply_boundary_conditions`). Need to verify pins commute through
  the coupled Newton iteration the same way they do through
  operator_split's sequential apply. Mitigation: regression test on a
  fixture that exercises both lower-boundary deposition velocity and
  upper-boundary escape velocity.

## Migration Plan

1. Land `coupled_advance` behind `KINTERA_SOLVER=coupled` env var
   (default remains `split`).
2. Update `STATUS_REPORT.html` to support a coupled-Newton column
   alongside the existing Newton/BDF columns.
3. Run NT=44 with `KINTERA_SOLVER=coupled` on the Titan neutrals
   network; verify zero-between-non-zero count is 0 across all
   species (the C2H6 dip fingerprint).
4. Compare concentrations against the BDF chemistry-only run at
   matched simulated time. Within 10% on major species would be
   acceptable; deeper analysis if not.
5. Once validated, file a follow-up change flipping the default.
6. Rollback: revert this change's commit; `KINTERA_SOLVER=split`
   already keeps the operator-split path alive in the meantime.

## Open Questions

- Should we accept the `non_finite-to-reject` pattern as the official
  contract between `step_fn` and `adaptive_advance`, or design a
  richer return type? Defer until the coupled path is in.
- Coupled BDF: which sparse pattern does scipy support for the
  combined transport+chemistry Jacobian? The transport sparsity is
  block-tridiagonal across layers (with cross-species coupling per
  cell); chemistry is dense per cell. Combined: block-banded with
  block-size `nspecies`. Could be a follow-up change.
