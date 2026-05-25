## Context

KB's COEFF1 (`kinetgen2X.F:5020-5141`) computes for each face between
layers I and I+1:

```fortran
TEMPP = (T[I] + T[I+1]) / 2
DENN  = (n[I] + n[I+1]) / 2
DIFF  = 7.3e16 * TEMPP^0.75 / DENN * sqrt((1 + 28/m_i) / (1 + 28/16))
EEK   = (Kzz[I] + Kzz[I+1]) / 2
D(i, face) = DIFF + EEK
SCALE = dz / log(n[I]/n[I+1])
ATMMAS = avg-mass(species mix at face)
ZMM   = m_i / ATMMAS
FA    = 1 + (1 + α - ZMM) × SCALE/ZMM × dT/dz / T_avg    (α = 0 for now)
H(i, face) = D / ((EEK + DIFF × ZMM × FA) / SCALE)
```

The flux at face is `F = -D × (∂_z c + c/H)`. Without the molecular
term, only the eddy contribution matters and the species effective
scale height collapses to the atmospheric scale height — every species
sees the same diffusion, no gravitational separation.

kintera's `python/atm2d/transport.py:build_binary_diffusion_matrix`
already accepts a binary-diffusion tensor (cell-centered, shape
`(ncol, nlyr, nspecies, nspecies)`) and `molecular_weights` (shape
`(nspecies,)`). When `include_gravity=True` (default), it adds the
mass-weighted gravity term via `_interface_thermo`. So the gravity
separation comes for free once binary_diffusion is populated.

The Cheng formula is a *binary* diffusion of species `i` in a bath
(N2 + CH4 mix). For kintera's binary-diffusion-matrix shape
`(nspecies, nspecies)`, this means a diagonal matrix per cell: each
species' self-diagonal entry is `D_i(T, n)` (treated as if every
species diffuses against the same bath, which is what KB does too).

## Goals / Non-Goals

**Goals:**

- Implement the Cheng formula in a single Titan-specific helper
  (`kinetics_base_titan_cheng_diffusion`) returning a
  `(ncol, nlyr, nspecies, nspecies)` diagonal tensor.
- Provide per-species molecular masses for the Titan network via the
  existing `_kinetics_base_species_mass_amu` helper in
  `python/kinetics_base/titan/physics.py`.
- Wire into the no_grain_stability driver behind
  `KINTERA_TITAN_MOLECULAR_DIFFUSION` env var (default `0` initially,
  flip to `1` after validation).
- Verify on NT=44 KB-Titan that turning on molecular diffusion
  brings HCN/NH3/HC3N/CH3CN to non-trivial values at L5-L15.

**Non-Goals:**

- Match KB's `FA` thermal-diffusion factor exactly. With α=0 and
  near-isothermal atmosphere (T varies only 38-103 K across our
  altitude range), `FA ≈ 1` everywhere; we'll set `FA=1` in this
  pass and add the thermal correction later if needed.
- Replace eddy-only transport as the default. The change is
  opt-in via env var; flipping the default is a separate change.
- Refactor to mixing-ratio-form eddy diffusion. That's the second
  half of the KB transport story but separate work. Cheng molecular
  diffusion on its own is a significant correction.

## Decisions

### Decision 1: diagonal binary-diffusion matrix

KB treats each species as diffusing in a homogeneous bath (effective
N2-dominated mix). The natural representation is a diagonal
`(nspecies, nspecies)` matrix per cell where entry `(i, i)` is the
Cheng `D_i(T_cell, n_cell)`. Off-diagonal cross-species terms are
zero (no Stefan-Maxwell coupling, consistent with KB).

Alternative: provide as a per-species 1D vector that
`build_binary_diffusion_matrix` then promotes internally. Rejected
because the existing API expects the full matrix; adding a vector
overload doubles the surface area for no benefit at this scope.

### Decision 2: Cheng formula in Titan namespace, not core kintera

The `7.3e16 × T^0.75 / n × sqrt((1+28/m_i)/(1+28/16))` formula is
Titan-specific (N2-dominated bath). Putting it in
`python/kinetics_base/titan/transport_diffusion.py` keeps the core
kintera transport module bath-agnostic.

### Decision 3: molecular_weights from pun_metadata

Reuse `_kinetics_base_species_mass_amu` from
`python/kinetics_base/titan/physics.py` (already used by other
photochemistry code) to extract per-species masses. The driver
constructs the `(nspecies,)` vector once and passes it through.

### Decision 4: env-var off-by-default during initial validation

Initial release ships with
`KINTERA_TITAN_MOLECULAR_DIFFUSION` defaulting to `0` so the
existing tests keep passing and existing comparisons stay
reproducible. Flip to default-on in a follow-up change after
verifying the HCN/NH3 fix and checking no regression on simpler
species.

## Risks / Trade-offs

- **Compute cost**: molecular diffusion adds `O(ncol·nlyr·nspecies)`
  work per Newton iteration for the per-species D_i evaluation,
  plus the `build_binary_diffusion_matrix` cost (which is roughly
  the same as `build_eddy_diffusion_matrix` since the per-cell
  matrix is diagonal). Mitigation: cache the `D_i` tensor when
  state doesn't change between Newton iterates — but actually
  `T` and `n_tot` are fixed during the chemistry+transport solve
  (they're part of the state, not the Newton variable), so the
  tensor can be built once per outer step.
- **Numerical conditioning at high altitudes**: molecular D
  grows as `T^0.75 / n_tot`, so at L39 where `n_tot ≈ 10⁸` we get
  `D ≈ 10⁹` cm²/s for light species — many orders larger than
  Kzz. The implicit `(I/dt - T)` matrix may become ill-conditioned.
  Mitigation: clamp `D` to a maximum (KB has no clamp visible; if
  this becomes an issue we add one).
- **Heavy species gravitational settling**: with the gravity term
  enabled, heavy species (m_i > m_avg) drift downward. For Titan
  species (mostly hydrocarbons with m ≤ 78), gravity separation
  is mild but real. Mitigation: this is the correct physics; if
  it produces unphysical results that points to a bug, not a
  trade-off.

## Migration Plan

1. Land the new helper + driver wiring behind env var (off by
   default).
2. Run `KINTERA_TITAN_MOLECULAR_DIFFUSION=1
   KINTERA_SOLVER_MODE=coupled` NT=44 KB-Titan neutrals_only and
   compare HCN/NH3 profiles at L5-L15 to the molecular-off baseline.
3. If HCN/NH3 reach >1 cm⁻³ at L5 (vs current 10⁻¹²) and major
   species are unchanged, document the win and propose a follow-up
   change to flip the default.
4. Rollback: revert this change; the env-var off-by-default means
   existing consumers see no effect until they opt in.

## Open Questions

- Does kintera's `_interface_thermo` (the gravity-term helper) read
  from a `state.gravity` attribute that's set sensibly for Titan?
  Need to verify; if not, add `state.gravity = 1.352e2 cm/s²` (Titan
  surface gravity) in the Titan state builder.
- Should we add the thermal diffusion factor `α` for H/H2 (KB has
  `COEFF1C` setting non-zero α for light species)? Likely small
  effect at Titan temperatures, but matters for upper atm. Defer
  to follow-up.
- Should the gravitational separation use the WHOLE species composition
  (sum over species of m_s × x_s) or the **fixed-mass** atmospheric
  bath? KB uses AVGMAS which is the full mix. Same here.
