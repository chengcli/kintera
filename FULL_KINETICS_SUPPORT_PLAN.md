# Full KINETICS-base Titan Network Support Plan

当前事实来源见 `KINETICS_BASE_TITAN_GAP_STATUS.md`。这份文件保留 full-network support 的目标和分层路线，已删除早期 one-step / NTIME=10 旧结论。

## 目标

让 kintera 能导入并运行 KINETICS-base Titan Cheng ion network，并在同一输入、同一边界、同一 source 语义下与 Fortran oracle 对齐。

主要输入：

- `Cheng_ions_c6h7+_v3_H2CN.pun`
- `Cheng_ions_c6h7+_v3_H2CN.special`
- `ions_c6h7+_H2CN.inp-1`
- `titan_Cheng_N_ions_H2CN.bc_save`
- `kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz`
- `Cheng_catalog_v4.dat`
- `Cheng_cross/`
- radiation auxiliary files used by the Titan example

稳定 checkout 路径：

```text
diagnostics/KINETICS-base-compare
```

## 已完成的 Full-Network 支撑

- Full Titan `.pun` / `.special` / `.inp` / boundary / atmosphere / catalog 输入可解析。
- `kintitan.truncate` active network 可用于 source-term construction。
- source terms 覆盖 thermal, ion mass-action, dissociative recombination, photolysis, electron-impact, boundary, condensation, sublimation。
- fixed/boundary cells 可作为 Dirichlet rows 进入 implicit matrix。
- Titan `G*` grain species diffusion scale 已接入。
- CH4 grain special source 和 sublimation `NTOT` limiter 已实现。
- 诊断 runner 支持 `KINTERA_TITAN_NETWORK_MODE=no_grain`。

## 当前分层策略

full network 不应继续直接作为唯一调试目标。当前推荐顺序：

1. `no_grain` baseline：关闭 KB `FREEZE/SUBLIM`，过滤 kintera grain terms。
2. baseline stiff-network timestep control：确保大步不会接受 bad solve。
3. condensation only。
4. sublimation only。
5. paired reversible condensation/sublimation。
6. CH4 shifted release。
7. full grain network。

运行 no-grain diagnostic：

```bash
KINTERA_TITAN_NETWORK_MODE=no_grain python diagnostics/compare_gch4.py
```

## 当前 Blocker

full network 的早期 CH4/GCH4 runaway 已被定位为 grain feedback 相关；`no_grain` 能移除这类早期 runaway。

但 `no_grain` 在 `NTIME=50` 仍会在超大 timestep 触发 baseline species 的不稳定。这说明在继续 grain 工作前，需要先实现可靠的 timestep acceptance/retry policy。

详细 gap 见 `KINETICS_BASE_TITAN_GAP_STATUS.md`。

## 后续 Engineering Plan

### 1. Timestep Controller

独立实现，不混入当前 checkpoint：

- reject non-finite solve；
- reject severe negative concentrations；
- retry smaller `dt`；
- step-doubling 或 embedded error estimate；
- cap growth/shrink factor；
- 明确记录 accepted/rejected steps。

### 2. RHS Oracle

为 selected species 建立完整 per-reaction RHS oracle：

- KB instrumentation 输出完整 contribution table；
- kintera mirror 输出同 schema；
- 支持 selected timestep/species diff。

### 3. Typed Source Model

当前 `KBTitanSourceTerm(kind=...)` 是兼容层。后续可以逐步替换为 typed descriptors：

- `IonMassActionReaction`
- `DissociativeRecombination`
- `ElectronImpactIonization`
- `PhotolysisBranch`
- `BoundarySource`
- `CondensationSource`
- `SublimationSource`
- `SpecialKineticsBaseFormula`

这不是当前 blocker，可等数值稳定后再整理。
