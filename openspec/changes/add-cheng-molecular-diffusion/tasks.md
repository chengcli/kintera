## 1. Cheng formula implementation

- [ ] 1.1 Create `python/kinetics_base/titan/transport_diffusion.py`
  with `kinetics_base_titan_cheng_diffusion(state, masses, *,
  temperature=None, density=None)` returning the diagonal
  `(ncol, nlyr, nspecies, nspecies)` binary-diffusion tensor
- [ ] 1.2 Add a helper to extract the per-species masses vector from
  pun_metadata using `_kinetics_base_species_mass_amu`; expose as
  `kinetics_base_titan_species_masses(species, pun_metadata)`
- [ ] 1.3 Export both helpers via `python/kinetics_base/titan/__init__.py`
  and add to the top-level `kintera` re-exports if appropriate

## 2. Driver wiring

- [ ] 2.1 Add `MOLECULAR_DIFFUSION = os.environ.get("KINTERA_TITAN_MOLECULAR_DIFFUSION", "0") == "1"`
  flag in `diagnostics/no_grain_stability.py`
- [ ] 2.2 When the flag is on, compute the binary-diffusion tensor and
  masses once at setup (T and n_tot are fixed for the run) and stash
  them on a local closure so they're reused across all
  `step_fn_split` / `step_fn_coupled` invocations
- [ ] 2.3 Pass `binary_diffusion=` and `molecular_weights=` through to
  `operator_split_advance` / `newton_implicit_step`; both already
  accept these kwargs and forward them to `build_implicit_step_system`
- [ ] 2.4 Print the molecular-diffusion status at driver startup so
  the log is unambiguous

## 3. Validation

- [ ] 3.1 Run NT=44 KB-Titan neutrals_only coupled+loose with molecular
  diffusion OFF (baseline) and ON (test); dump to
  `/tmp/kt_traj_44_coupled_moldiff_on.npz` and
  `/tmp/kt_traj_44_coupled_moldiff_off.npz`
- [ ] 3.2 Compare HCN, NH3, HC3N, CH3CN profiles at L5-L15
  - HCN at L5 should rise from ~10⁻⁹ (off) to >10² (on)
  - HCN at L10 should rise from ~10⁻⁵ to >10⁴
  - Major species (CH4, H2, C2H6) should remain within 2× of the
    OFF baseline
- [ ] 3.3 Compare to KB fort.7: with molecular diffusion on, kintera
  HCN/NH3 at L5 should be within 4 orders of magnitude of KB (vs
  currently 18 orders of magnitude off). Document the remaining gap
  (likely the mixing-ratio-form eddy diffusion, separate change).
- [ ] 3.4 Sanity check: turn molecular diffusion off again and verify
  the resulting dump is bit-identical to the baseline (rules out
  accidental state mutation)

## 4. Tests

- [ ] 4.1 Unit test in `tests/` or `python/kinetics_base/titan/` for
  the Cheng formula at canonical (T, n, m_i) values: assert the
  returned `D` matches hand calculation to `1e-10` relative
- [ ] 4.2 Unit test for the per-species masses helper: assert known
  values (H=1, H2=2, CH4=16, etc.)
- [ ] 4.3 Integration test: build a small (5-layer, 3-species)
  fixture and verify that turning on molecular diffusion produces a
  different transport divergence than turning it off

## 5. Documentation

- [ ] 5.1 Add memory entry
  `/home/sam2/.claude/projects/-home-sam2-dev-kintera/memory/project_titan_molecular_diffusion.md`
  documenting the formula source, the env var, and the validation
  results
- [ ] 5.2 Update `STATUS_REPORT.html` to add a moldiff-on column
  alongside the existing solver variants if the validation shows a
  meaningful improvement
- [ ] 5.3 Update the existing `project_kintera_transport_form.md`
  memory to cross-reference this change (it documents the broader
  transport-formulation gap; this change closes one of the two
  identified pieces)
