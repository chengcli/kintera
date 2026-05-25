## Why

Kintera's default transport operator uses only eddy diffusion (Kzz) on
the **concentration** state. KB uses both eddy AND species-specific
molecular diffusion (Cheng Titan formula `7.3e16 T^0.75/n × √((1+28/m_i)/(1+28/16))`)
on the **mixing ratio**, with gravity separation via the species
effective scale height.

For HCN at L18 (337 km, T=67 K, [M]=1.89×10¹² cm⁻³):
- D_mol(HCN) ≈ 1.87×10⁴ cm²/s (Cheng formula)
- K_zz       = 3×10³ cm²/s (constant in this run)
- Molecular dominates by 6×; the gap grows with altitude as n_tot
  falls.

Without the molecular term, kintera can't deliver photochemically-
produced HCN, NH3, HC3N, CH3CN, etc. down from their production zone
(L15-L25) to the lower atmosphere. The result, visible in the
NT=44 dumps for both BDF and coupled-Newton: HCN at low altitudes is
10⁻¹² cm⁻³ noise-floor vs KB's 10⁹ — eighteen orders of magnitude
off.

## What Changes

- Add `kinetics_base_titan_cheng_diffusion(state, masses)` helper that
  returns the diagonal binary-diffusion tensor at cell centers using
  the Cheng formula.
- Add a `molecular_weights` accessor for the Titan species set
  (derive from pun_metadata or species names).
- Wire both into `diagnostics/no_grain_stability.py` so the
  `build_transport_matrix` call gets `binary_diffusion=` and
  `molecular_weights=` populated. The existing
  `build_binary_diffusion_matrix` infrastructure handles the
  gravity-separation term once those are passed.
- Guard with an env var `KINTERA_TITAN_MOLECULAR_DIFFUSION` (default
  `1` once validated, off-by-default initially to allow A/B compare).

## Capabilities

### New Capabilities

- `cheng-titan-diffusion`: Compute the species-dependent molecular
  diffusion coefficient `D_i(T, n)` for Titan-relevant species using
  the Cheng 2013 formula, return it as a kintera-compatible binary-
  diffusion tensor, and provide a driver hook to wire it into the
  transport operator alongside Kzz.

### Modified Capabilities

- (none — `build_binary_diffusion_matrix` already exists with the
  right shape; this change just supplies its inputs for the Titan
  case.)

## Impact

- **New module**: `python/kinetics_base/titan/transport_diffusion.py`
  (or extension of `atmosphere.py`) housing the Cheng formula and a
  per-species mass-extraction helper.
- **Driver**: `diagnostics/no_grain_stability.py` gains the env-var
  flag and the wiring through `build_transport_matrix`.
- **Tests**: regression test that turning on molecular diffusion
  produces non-trivial HCN/NH3 concentrations at L5-L10 after a short
  integration (vs the current 10⁻¹² noise).
- **Report**: update `STATUS_REPORT.html` to include a
  molecular-diffusion variant comparison if results are useful.
- **No removal**: existing eddy-only behaviour kept as the default
  when `KINTERA_TITAN_MOLECULAR_DIFFUSION=0`.
- **Dependencies**: none (formula is pure Python/torch).
