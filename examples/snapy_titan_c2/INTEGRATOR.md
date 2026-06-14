# Chemistry integrator & tuning (Titan C2)

How `TitanC2Chemistry.advance()` integrates the stiff photochemistry inside the
operator-split GCM step, and how to tune it. The chemistry is advanced in place
once per hydro step, after the RK3 dynamics stages, from the post-transport
state (Lie operator splitting).

## What it is

A **per-cell adaptive, compacting ROS2 integrator**:

- **ROS2** — a 2-stage, 2nd-order, L-stable linearly-implicit Rosenbrock method
  (the atmospheric-chemistry standard; Sandu & Verwer 1997, *Atm. Env.* 31; the
  KPP solvers). One Jacobian factorization shared across both stages, two rate
  evaluations, an embedded 1st-order solution for the local-error estimate. No
  Newton iteration. Hand-rolled in torch (`_ros2_substep`) because kintera's
  compiled kernel can't be driven per-cell here; the tableau matches
  `kintera.evolve_ros2` to ~1e-12. **Element-conserving for any step size** —
  the stoichiometry annihilates the conserved-atom vectors, so each stage solve
  preserves them exactly (the only atom drift is from the `>=0` clamp).

- **Per-cell adaptive marching with compaction** — each cell integrates `0 → dt`
  with its own step size from its own embedded error; once a cell reaches `dt`
  it is **removed from the active batch**, so the batched solve shrinks to only
  the still-stiff cells. This is the key to GPU efficiency: the stiff cells are
  a small minority (the jet-perturbed terminator strip + the dense lower
  boundary) that need ~100+ sub-steps, while ~90% of cells finish in a handful
  and drop out instead of being re-solved alongside them.

- **Step floor** — the sub-step is floored at `dt / max_substeps` and
  force-accepted there, so every cell covers the full `dt` (no silent
  under-integration) even on a radical building from exactly zero.

Controller: order-2 method with an order-1 embedded estimate → step exponent
`1/2`; safety factor `1.4` (Frey et al. 2025, *GMD* 18 — the textbook `0.9`
systematically over-shrinks). `atol = 1e-15 mol/m³`: the reactive species here
span 1e-12..1e-16 while the bath is ~1e-5, so this keeps the species that matter
(CH3/H/C2Hx) in the rtol-controlled regime without chasing negligible trace
abundances to zero.

## Knobs

| knob | where | default | effect |
|---|---|---|---|
| `chem_rtol` | case yaml / `advance(rtol=)` | `0.1` (yaml), `0.05` (lib) | relative error target; governs the **bulk** cells |
| `max_substeps` | `advance(max_substeps=)` | `400` | sub-step floor `dt/max_substeps`; the cost/accuracy lever for the **stiffest** cells |
| `atol` | `advance(atol=)` | `1e-15` | absolute floor [mol/m³]; rarely touched |

`TITAN_TIMING=1` makes the UM-TITAN runner print per-cycle `dyn`/`rt`/`chem`
wall time and the max chemistry sub-step count.

## Cost in the 3-D GCM

Per hydro cycle, superrotation case, 64²×40 ×6 blocks on 2× H100 (`dyn≈0.18 s`,
`rt≤0.05 s` throughout — chemistry dominates):

| scheme | chem/cycle | nsub_max | vs first cut | 0-D err |
|---|---|---|---|---|
| block-synchronized, rtol=0.05 | 36–48 s | 105 | 1× | — |
| + compaction, rtol=0.05 | 11–15 s | 322 | ~3.2× | <0.01% |
| + compaction, **rtol=0.1** (default) | 7–10 s | 205 | ~5× | <0.02% |
| + compaction, rtol=0.2 | 3–5 s | 99 | ~10× | <0.06% |

Compaction also *improves* accuracy: per-cell, the stiff cells resolve to their
own tolerance instead of being force-accepted at a block-wide floor. Mass is
conserved to ~15 digits; the 1-D init relax takes ~45 s; gates 7/7.

> Note: block-vs-global sub-step *synchronization* is a wash here (a dense
> batched solve touches every cell, and the GCM barrier-syncs ranks each step).
> What pays is **compaction** — dropping converged cells across the ~100
> sub-steps. (Strang operator-splitting was considered and dropped: over a long
> marched run it telescopes to the existing Lie split bar O(dt) end-caps.)

## The stiffness cliff — and how tuning changes with grid resolution

Chemistry cost/accuracy is governed by where the hydro `dt` sits relative to a
per-cell **stiffness cliff**, not by a smooth scaling. Above the cliff the
controller can't meet tolerance above the floor, so it saturates at
`max_substeps` and force-accepts (inaccurate). Below it the same cell needs only
~15–20 sub-steps and is essentially exact.

0-D scan, near-SS day parcels, `nsub` / H-error at rtol=0.05 (rtol barely moved
either column — see below):

| dt [s] | ~120 km (n=2e13) | ~100 km (n=1e14) |
|---|---|---|
| 66 | 404 / 7.2% | 404 / 13% |
| 33 | 404 / 1.1% | 404 / 3.3% |
| 16 | **22 / ~0** | 404 / 0.7% |
| 8  | 15 / ~0 | 404 / 0.12% |
| 4  | 14 / ~0 | **407 / 0.004%** |

Reading it:

- **The cliff scales with stiffness (≈ density/altitude).** The 120 km cell
  clears it by dt≈16 s; the 100 km cell is so stiff (τ_radical ~0.01 s) it stays
  floor-bound at *every* dt ≥ 4 s — only its accuracy improves, never its cost.
- **rtol is the wrong knob for stiff cells.** Across 0.05 / 0.1 / 0.2 the
  stiff-cell `nsub` barely moves (404 when floored; 22 vs 16 when free). rtol
  only modulates the *bulk* cells. The stiff cells are governed by **dt**
  (accuracy) and **max_substeps** (cost).
- **The current 64²×40 grid (dt≈66 s) sits above the cliff** for the dense
  bottom cells → floor-saturated (the persistent nsub≈322 seen in the GCM) with
  ~7–13% error there. That error is localized **below the 150–300 km science
  region**. *(The absolute error %s are upper bounds — the 0-D relax used floored
  dt=66 steps — but the cliff and dt-trend are confirmed by the GCM's nsub.)*

Because a finer dynamics grid means a smaller dt (CFL), the recommendation
**flips with resolution**:

| grid / dt | bottom cells | recommendation |
|---|---|---|
| coarse (current, dt ≳ 30 s) | floor-saturated, 7–13% err | `chem_rtol = 0.1` for bulk cost; raise `max_substeps` if bottom accuracy matters |
| medium (~4× finer, dt ~ 16 s) | mid-stiff cells escape floor (cheap); bottom accurate (<1%) but densest still cost-floored | **tighten `chem_rtol` back to 0.05** — chemistry is now cheap *and* accurate |
| fine (dt ≲ 8 s) | nearly all escape the floor | `chem_rtol = 0.05` or tighter; **dynamics becomes the bottleneck** (scales ~N⁴ vs chemistry ~N³) |

**Practical guidance**

- The looser-rtol default (0.1) is a *coarse-grid* mitigation. Refine the grid
  and you should revert to 0.05 — chemistry stops being the bottleneck on its
  own.
- **Refine vertically near the dense bottom**: doubly good — it resolves those
  cells *and* lowers global dt below their cliff.
- The one knob that doesn't care about grid is **`max_substeps`**: the
  cost/accuracy lever for the irreducibly-stiff bottom layer. Lower it for
  cheaper/looser, raise it for accurate/slower. (Currently a function argument,
  not yet a yaml knob.)
