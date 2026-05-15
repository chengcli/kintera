# Ion Chemistry Support Plan

当前事实来源见 `KINETICS_BASE_TITAN_GAP_STATUS.md`。这份文件只保留 ion chemistry 相关的设计记录，已删除早期的 stale baseline 和过时阶段计划。

## 当前状态

Ion chemistry 的第一层 KINETICS-base 兼容已经实现：

- charged species 与 charge balance 可分类。
- `E` 和 cations/anions 会作为 number-density state species 参与演化。
- `E` 不再被当作 fixed/background profile。
- ion-neutral mass-action reactions 和 dissociative recombination 已接入 `atm2d` source linearization。
- Cheng `.special` ISP mapping 已用于选择 runtime-referenced electron-impact branches。
- electron-impact ionization 已作为独立 source category，而不是混在普通 photolysis 里。

最近验证状态见：

```bash
python -m pytest tests/test_kinetics_base.py -q
```

## 仍然成立的设计原则

Ion chemistry 在 kintera 中应作为普通 chemistry/source 类型进入统一 state vector，而不是作为 Titan-only 后处理：

- charged species 和 neutrals 使用同一个 concentration tensor。
- mass-action Jacobian 应包含 electron/ion reactants。
- electron-impact ionization 应暴露 altitude-dependent rate profile，便于诊断。
- dissociative recombination 可以复用 mass-action kernel，但保留独立分类用于统计和验证。
- `.special` entries 应解析为明确的 supported/no-op/unsupported 类别，不应静默跳过。

## 当前剩余问题

Ion chemistry 本身不再是当前最大 blocker。主要剩余 gap 已转移到：

- baseline stiff-network timestep acceptance；
- full grain feedback；
- full RHS oracle。

这些内容统一维护在 `KINETICS_BASE_TITAN_GAP_STATUS.md`。

## 后续 Ion 相关工作

后续如果继续完善 ion chemistry，建议按以下顺序：

1. 把 electron-impact empirical scale 从 compatibility helper 升级成明确的 rate-profile source metadata。
2. 增加 selected species 的 ion/electron RHS oracle 对照。
3. 如果未来打开 `AMBIDIFF`，再实现 ambipolar diffusion / ion transport 特性。
4. 将当前 string-based `KBTitanSourceTerm.kind` 逐步替换为 typed descriptors。
# Ion Chemistry Support Plan

## Goal

Add first-class support for Titan ion chemistry in kintera so steady-state runs
can match KINETICS-base beyond the current neutral-only agreement window.

The target architecture is:

- kintera owns the executable physics categories and their numerical
  linearization.
- Python KINETICS-base helpers parse `.pun`, `.special`, `.inp`, `.bc_save`,
  catalog, cross-section, and atmosphere files.
- Python maps parsed KINETICS-base records into typed kintera reaction/source
  categories.
- Python does not remain the long-term implementation site for ion reaction
  physics.

This keeps the KINETICS-base importer thin and makes ion chemistry usable by
other kintera frontends, not just the Titan compatibility path.

## Current Baseline

The neutral Titan compatibility path is in good shape:

- One-step KINETICS-base oracle comparison passes.
- Ten-step neutral/source comparison passes.
- Boundary-condition mismatches have been fixed:
  - type-2 grain deposition at lower boundary
  - type-2 H/H2 escape at upper boundary
  - CH4 `__CHENG` cold-trap pinning at Fortran level 24
- `NBOT=24` is understood to apply only to CH4 in the `__CHENG` branch, not to
  all species.

Steady-state agreement is still blocked by missing ion chemistry.  The current
neutral-only long run diverges in ion-sensitive species such as `N(2D)`, `H2`,
`GH`, `HCN`, `HC3N`, `HC5N`, and heavy hydrocarbons/nitriles.

## Design Principle

Represent every physical contribution as a kintera-supported category:

| Category | Long-term home | Python role |
| --- | --- | --- |
| Thermal mass-action reactions | kintera core / atm2d source layer | parse `.pun` rate blocks and stoichiometry |
| Photolysis | kintera photochemistry/radiation/source layer | parse selected photo ids, catalog entries, cross sections |
| Electron-impact ionization | kintera source/reaction category | parse ionization branches and rate/cross-section metadata |
| Ion-neutral reactions | kintera mass-action category with charged species support | parse `.pun` reactions and classify charge-transfer/proton-transfer behavior |
| Dissociative recombination | kintera ion/electron reaction category | parse electron reactant reactions and products |
| Attachment / detachment, if present | kintera ion/electron reaction category | classify from charge balance and `.special` metadata |
| Grain condensation/sublimation | kintera source category | parse `X + SGA <-> GX + U` patterns and vapor coefficients |
| Boundary flux/velocity conditions | kintera boundary source category | parse `.bc_save` |
| KINETICS-base special formulas | kintera special physics categories | parse `.special` ids and select implemented formulas |

The importer may initially emit compatibility `KBTitanSourceTerm` objects, but
those objects should be treated as an adapter layer.  The implementation target
is typed kintera categories, not stringly typed Python-side physics.

## Proposed kintera Categories

### 1. Ion Mass-Action Reactions

Support ordinary mass-action reactions where any reactant/product may be charged:

- cations: species ending in `+`
- anions: species ending in `-`
- electrons: `E`

Required behavior:

- preserve stoichiometric coefficients from `.pun`
- support one-, two-, and three-body reactions already covered by the thermal
  rate machinery
- keep charged species in the same concentration state vector as neutrals
- compute local Jacobian entries including electron/ion reactants
- maintain non-negativity under stiff implicit stepping

This category covers ion-neutral charge transfer, proton transfer, clustering,
and many neutralization routes if they are encoded as normal `.pun` thermal
reactions.

### 2. Dissociative Recombination

Dedicated classification for reactions containing `E` as a reactant and a
positive ion as another reactant, for example:

```text
E + X+ -> products
```

These can use the same numerical machinery as mass action, but should be
classified separately because they are a major steady-state control knob for:

- `N(2D)`
- `H`
- `HCN`
- nitrile production chains
- electron density closure

Implementation may share the mass-action kernel; the category is still useful
for diagnostics, validation counts, and feature gating.

### 3. Electron-Impact Ionization

Electron-impact reactions should be first-class, not special-cased as ordinary
photolysis.  KINETICS-base currently marks some electron/ionizing catalog
branches, but the Titan `.inp` active photo ids do not fully activate all ion
branches used by the long-run oracle behavior.

Required support:

- classify catalog/cross-section branches that produce `E` and ions
- distinguish solar photolysis from electron-impact ionization
- support altitude-dependent rate profiles or flux-driven profiles
- expose rate profiles for diagnostics
- participate in the implicit source/Jacobian path

Open question:

- whether KINETICS-base Titan uses a fixed empirical electron-impact profile,
  catalog-derived cross sections, `.special` hooks, or a combination.  This must
  be confirmed against the Fortran source before implementation.

### 4. Attachment / Detachment

If the Titan network includes anion chemistry or electron attachment, classify
these separately:

```text
E + X -> X-
X- -> X + E
X- + Y+ -> products
```

This may initially be a no-op category if the current Titan selected network has
no active anion branch, but the charge bookkeeping should not assume cations
only.

### 5. KINETICS-base Special Physics

`.special` entries should map to explicit categories.  Known examples already
encountered:

- disabled placeholders: recognized no-op
- Cheng-specific special thermal branch for reaction 642
- special photolysis branches

Ion branch work should audit `.special` for formulas that modify:

- electron-impact rates
- recombination branching
- ion pair production
- altitude gates
- empirical Titan profiles

Each supported formula should become a named category with a test.  Unsupported
formulas should fail loudly or be reported in a classification summary, not
silently skipped.

## Data Model Plan

Replace or wrap string `kind` values with typed source/reaction descriptors.

Short-term compatibility:

```python
KBTitanSourceTerm(kind="pun_ion_mass_action_reaction", ...)
KBTitanSourceTerm(kind="pun_dissociative_recombination", ...)
KBTitanSourceTerm(kind="pun_electron_impact_reaction", ...)
```

Long-term target:

```python
IonMassActionReaction(...)
DissociativeRecombinationReaction(...)
ElectronImpactIonization(...)
BoundarySource(...)
CondensationSublimationPair(...)
SpecialKineticsBaseFormula(...)
```

The typed representation should include:

- source reaction id
- source file/category (`pun`, `special`, `catalog`, `bc`)
- reactants/products with coefficients
- charge balance summary
- rate formula id
- rate parameters
- optional altitude/radiation/electron-impact profile metadata
- validation status (`supported`, `recognized_noop`, `unsupported`)

## Implementation Phases

### Phase 1: Classification Audit

Build a complete report for the Titan selected network:

- total charged species count
- ion/electron reactions by class
- reactions with `E` as reactant
- reactions producing `E`
- reactions consuming/producing cations
- charge-balanced vs charge-imbalanced records
- `.special` entries that target charged reactions
- catalog branches that produce ions/electrons
- currently unsupported records

Deliverables:

- `classify_kinetics_base_titan_reactions()` extended with ion class counts
- tests asserting counts for the Cheng Titan dataset
- a machine-readable summary useful for debugging

### Phase 2: Enable Ion Mass Action and Recombination

Map ordinary charged `.pun` thermal reactions into kintera local source terms:

- classify ion-neutral reactions separately from neutral thermal reactions
- classify `E + ion -> products` as recombination
- reuse the implicit mass-action kernel where possible
- add focused tests with synthetic charged networks
- add Titan count tests proving the expected number of ion reactions is active

Expected effect:

- electron and ion densities become non-zero through reaction coupling
- radical and nitrile steady-state profiles move closer to KINETICS-base

### Phase 3: Electron-Impact Ionization

Implement the electron-impact source category after confirming the Fortran path:

- identify the exact KINETICS-base routines and flags controlling Titan
  electron-impact ionization
- reproduce altitude/rate profile construction
- add kintera category and atm2d linearization
- activate relevant catalog branches
- compare source-rate profiles against Fortran diagnostics where possible

Expected effect:

- primary ion production (`N2+`, `N+`, hydrocarbon ions, `E`) appears at the
  correct altitudes
- long-run electron/ion budgets become physically meaningful

### Phase 4: Special Ion Physics and Branching

Implement any special formulas discovered in Phase 1/3:

- recombination branching overrides
- empirical altitude gates
- ion pair yields
- special ion-neutral rates not represented by ordinary `.pun` rates

Each formula should have:

- a named category
- a parser/classifier test
- a source/Jacobian test
- a Titan oracle comparison checkpoint

### Phase 5: Steady-State Validation

After ion categories are active, compare against KINETICS-base at increasing
time horizons:

1. `NTIME=10`: must remain passing.
2. `NTIME=30`: radical and first ion profiles should agree within order unity.
3. `NTIME=50`: neutral steady-state profiles should broadly match in shape and
   peak altitude.
4. Longer runs: only after timestep and convergence controls are audited.

Validation should be split by chemistry family:

- fixed/background: `N2`, `M`, `JDUST`, `SGA`, `U`
- primary neutrals: `CH4`, `H2`
- radicals: `H`, `N`, `N(2D)`, `CH3`
- nitriles: `HCN`, `HC3N`, `HC5N`, `C2N2`
- hydrocarbons: `C2H2`, `C2H4`, `C2H6`, `C4H2`, `C6H2`
- grains: `GCH4`, `GC2H2`, `GH`
- ions/electrons: `E`, `N2+`, `N+`, `CH3+`, major hydrocarbon ions

## Testing Strategy

### Unit Tests

- charged species name classification
- charge-balance classification
- reaction class assignment
- electron/reactant coefficient handling
- ion mass-action Jacobian entries
- recombination Jacobian entries
- electron-impact altitude profile construction

### Integration Tests

- synthetic charged `.pun` network import
- Titan ion classification counts
- Titan source-term category counts
- focused oracle comparisons for selected ions after one step
- long-run neutral profile smoke comparisons after ion source categories are
  active

### Diagnostics

Keep separate diagnostics for:

- per-reaction source rates
- per-species production/loss budgets
- altitude profiles of ions/electrons
- neutral steady-state comparison panels

The most useful next diagnostic is a production/loss budget for:

- `E`
- `N2+`
- `N+`
- `N(2D)`
- `H`
- `HCN`
- `GH`

## Acceptance Criteria

Ion support is considered usable when:

- charged thermal reactions are no longer silently classified as generic neutral
  reactions without charge diagnostics
- recombination and electron-impact branches have explicit categories
- unsupported ion/special physics is listed in a report with reaction ids
- the 10-step oracle comparison still passes
- the 30-step comparison improves for `N`, `N(2D)`, `H`, `H2`, `HCN`, and `GH`
- the 50-step neutral steady-state comparison no longer shows order-of-magnitude
  failures caused by missing ion production/loss channels

## Immediate Next Steps

1. Add ion classification to `classify_kinetics_base_titan_reactions()`.
2. Add tests that lock down Titan charged reaction counts.
3. Audit KINETICS-base Fortran for electron-impact ionization and recombination
   handling.
4. Decide the first executable category to implement:
   - lowest risk: ion mass-action + recombination using existing local source
     kernels
   - highest steady-state impact: electron-impact ionization profile
5. Add the first category behind explicit tests and re-run the `NTIME=10`,
   `NTIME=30`, and `NTIME=50` comparison diagnostics.
