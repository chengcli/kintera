# KINETICS-base Titan 当前状态与 Gap

这份文档是 Titan/KINETICS-base 兼容工作的当前唯一状态总览。旧的实现计划和阶段性分析文档只保留索引，不再作为事实来源。

## 当前结论

我们已经把大部分“输入语义”和“source 语义”对齐到 KINETICS-base，但长时间积分还没有完全闭合。当前最重要的判断是：

- `no_grain` baseline 可以去掉早期 `CH4/GCH4` runaway，说明之前最明显的 CH4 爆炸主要来自 gas-grain feedback。
- `no_grain` 在固定 KB startup step sequence 下可稳定到 `NTIME=45`。
- `no_grain` 到 `NTIME=50` 仍可能在超大 timestep (`dt ~ 3e9 s`) 产生非 grain species 的巨大值或 NaN，主导物种包括 `C6N2`、`H2`、`C2H3CN`。
- 因此下一步应先解决 baseline stiff-network 的 timestep acceptance/retry 问题，再继续叠 grain chemistry。

## 稳定路径

KINETICS-base checkout 和输出都已移到 `diagnostics/` 下，避免 `/tmp` 被清掉：

- KB root: `diagnostics/KINETICS-base-compare`
- KB executable: `diagnostics/KINETICS-base-compare/build/bin/titan.release`
- Scratch outputs: `diagnostics/kinetics_base_oracle_runs`

常用诊断：

```bash
python diagnostics/compare_gch4.py
KINTERA_TITAN_NETWORK_MODE=no_grain python diagnostics/compare_gch4.py
```

`KINTERA_TITAN_NETWORK_MODE=no_grain` 会同时：

- 在 KB run input 中设置 `FREEZE=0`、`SUBLIM=0`。
- 在 kintera source terms 中过滤涉及 `SGA`、`U`、`G*` 的 grain terms。

这样可以先验证 neutral/ion/photolysis/transport baseline，而不是被 grain feedback 干扰。

## 已实现内容

### KINETICS-base 输入与网络

- 默认外部 KB 路径改为 `diagnostics/KINETICS-base-compare`。
- `.gitignore` 忽略 KB checkout/build/run artifacts。
- source-term 构建支持 `kintitan.truncate` active network。
- `.special` 的 ISP mapping 已解析成可查询结构。
- 诊断脚本和测试不再依赖 `/tmp`。

### Ion Chemistry

- charged species、charge balance、ion reaction counts 已可分类。
- `E` 和离子不再被当作固定背景物种。
- ion mass-action 与 dissociative recombination 接入 `atm2d` source linearization。
- electron-impact ionization 只启用 Cheng runtime 实际引用的分支。
- electron-impact scale 已按 KB behavior 做过经验对齐，但仍应视作兼容层，不是最终物理 API。

### Photolysis / Electron Impact

- A=0 unary branches 按 KB catalog/cross-section/flux 语义接成 first-order source。
- Cheng CH4 photolysis branch 语义已修正：
  - product branches 共享/clone rate。
  - 部分 branch suppress parent loss。
  - special multiplier 按 Fortran runtime 逻辑处理。
- actinic flux 默认路径已对齐到 KB 输入文件。

### Boundary / Transport

- Titan fixed/boundary cells 现在可作为 implicit matrix 的 Dirichlet rows，而不是只在 solve 后 patch。
- CH4 cold trap 保留 KB 初始 atmosphere 中已经准备好的 number density，不再硬编码 `0.0157` VMR。
- CH4 cold-trap pinning 只作用在 Fortran level 24 对应的 level。
- Titan `G*` mantle species 的 diffusion scale 已按 grain species 处理。

### Grain Semantics

已实现的 KB-style grain 特殊语义：

- `CH4 + SGA -> GCH4` 是 product-only loading，不显式消费 CH4 reservoir。
- `GCH4 + U -> CH4` 是 shifted release：在 level `i` loss `GCH4`，在 level `i+1` produce `CH4`。
- shifted release 提供 global sparse operator，可进入 implicit system。
- sublimation rate 现在使用 KB 的 surface coverage limiter：

```text
NTOT = max(total G* ice abundance, 4 * SGA * nsite)
```

注意：这些修复是必要的，但还不足以让 full grain network 长时间稳定。

## 当前验证状态

最近一次轻量验证：

```bash
python -m pytest tests/test_atm2d.py tests/test_kinetics_base.py -q
```

结果：

```text
43 passed, 4 skipped
```

诊断结论：

- Full network 仍会在长步区间出现 `CH4/GCH4` runaway。
- `no_grain` 去掉了这一类早期 grain runaway。
- `no_grain` 后的剩余失败出现在更晚、更大的 timestep，且主导物种不再是 grain species。

## 剩余 Gap

### 1. Baseline Stiff-Network Timestep Control

这是当前优先级最高的 gap。

问题：

- kintera 诊断目前仍按 KB negative-`DELTIM` startup sequence 固定推进。
- 如果某个大步导致 bad solve，当前流程可能在 clamp 后继续推进坏状态。
- KB/VULCAN 类模型通常会 reject bad step、缩小 timestep、重新求解，而不是接受坏状态。

建议：

- 先实现一个清晰、可测试的 timestep acceptance controller。
- 最小版本可以包在现有 backward-Euler implicit solve 外层：
  - reject non-finite solve；
  - reject 严重负值；
  - step doubling 或 embedded estimate 估计误差；
  - retry with smaller `dt`；
  - 限制 `dt` 增长。
- 不建议把半成品 adaptive controller 混进当前 commit；应该作为独立 PR/commit 做。

### 2. Full Grain Feedback

full network 的 grain runaway 仍未关闭。

已排除/改善：

- CH4 product-only loading 已实现。
- CH4 shifted release 已实现。
- `G*` diffusion scaling 已实现。
- `NTOT` limiter 已实现。

仍可能缺：

- KB `HETEROK` 中更多 gas-grain runtime overrides。
- freezeout/sublimation 与 nonlinear iteration/timestep rejection 的耦合语义。
- 部分 grain reactions 的 active network pairing 或 source category 仍不完全。

建议重新引入 grain 的顺序：

1. `no_grain` baseline。
2. condensation only。
3. sublimation only。
4. paired reversible source。
5. CH4 shifted release。
6. full grain network。

### 3. Full RHS Oracle

KB 现有 `prod+loss` 输出通常是 top-N diagnostic，不是完整 RHS oracle。

已有：

- kintera 侧 `diagnostics/ch4_full_rhs_mirror.py`。
- KB instrumentation 曾用于定位问题，但还不是稳定工具链。

建议：

- 正式做 selected species / selected step 的 per-reaction RHS oracle。
- 固定 CSV schema。
- 将比较脚本和输出目录规范化到 `diagnostics/`。

### 4. Solver Semantics

kintera 当前 implicit step 是一次 frozen linearization：

```text
S(c_new) ~= S(c0) + J(c_new - c0)
(I - dt * (T + J)) c_new = c0 + dt * (S(c0) - J c0)
```

KB 更接近 residual/correction solve，并有自己的 convergence/retry 语义。两者在线性、小步时接近，但在大 `dt` 和强 nonlinear coupling 下会分叉。

这个问题不应再用更多 physical shim 掩盖。应先给 baseline network 一个不会接受 bad step 的 timestep policy。

## 建议 Commit Scope

建议这次 commit 作为一个大的 Titan compatibility checkpoint：

```text
Improve KINETICS-base Titan source semantics and diagnostics
```

包含：

- ion chemistry classification/source support；
- Cheng `.special` mapping；
- photolysis/electron-impact alignment；
- Dirichlet boundary rows/cold trap；
- grain special source semantics；
- `NTOT` limiter；
- stable KB diagnostics paths；
- `no_grain` diagnostic mode；
- 当前状态文档。

不包含：

- 未验证完成的 adaptive timestep controller。
# KINETICS-base Titan Gap Status

This note summarizes the current state of the KINETICS-base Titan comparison work after the ion chemistry, boundary, photolysis, and grain-source passes.

## Current Baseline

The KINETICS-base checkout and oracle outputs now live under stable `diagnostics/` paths instead of `/tmp`.

- KINETICS-base root: `diagnostics/KINETICS-base-compare`
- KINETICS-base executable: `diagnostics/KINETICS-base-compare/build/bin/titan.release`
- Scratch oracle outputs: `diagnostics/kinetics_base_oracle_runs`

The Titan diagnostics can be run in two network modes:

```bash
python diagnostics/compare_gch4.py
KINTERA_TITAN_NETWORK_MODE=no_grain python diagnostics/compare_gch4.py
```

`no_grain` disables `FREEZE/SUBLIM` in the KINETICS-base input and filters kintera source terms involving `SGA`, `U`, or `G*` species. This gives a cleaner baseline for debugging neutral and ion chemistry without gas-grain feedback.

## Implemented Alignment Work

- Ion chemistry parsing/classification now identifies charged species, dissociative recombination, ion mass-action reactions, and charge balance.
- Titan source construction now uses the active `kintitan.truncate` network and Cheng `.special` ISP mappings.
- Electron-impact ionization branches are enabled only for the Cheng runtime-referenced branches.
- CH4 photolysis branches follow KINETICS-base semantics: cloned branch rates, special multipliers, and parent-loss suppression for Cheng product branches.
- Actinic-flux/catalog photolysis is connected as first-order source terms.
- Titan boundary handling now uses formal Dirichlet rows in the implicit matrix, including fixed species and the CH4 cold-trap level.
- CH4 cold-trap pinning now preserves the prepared KB number density at level 24 instead of overwriting it with a hardcoded VMR.
- Grain diffusion scaling treats Titan `G*` mantle species as non-diffusive, matching the KB gas-grain branch more closely.
- CH4 grain loading/release special semantics are implemented:
  - `CH4 + SGA -> GCH4` is product-only for CH4.
  - `GCH4 + U -> CH4` releases gas one vertical level above the grain loss.
- Sublimation rates now include the KB-style `NTOT = max(total G* ice, 4 * SGA * nsite)` surface coverage limiter.

## Validation So Far

Passing tests at the latest checked point:

```bash
python -m pytest tests/test_kinetics_base.py -q
python -m pytest tests/test_atm2d.py -q
```

Observed diagnostic behavior:

- Full network still develops a CH4/GCH4 runaway around the long-step region near `NTIME ~= 45-50`.
- `no_grain` removes the earlier CH4/GCH4 grain runaway. This strongly suggests the previous CH4 explosion is localized to gas-grain feedback rather than the basic neutral/ion chemistry network.
- `no_grain` is stable through `NTIME=45` under the fixed KINETICS-base startup step sequence.
- `no_grain` at `NTIME=50` can still produce very large non-grain species values / NaNs at the very large final timestep (`dt ~ 3e9 s`). The leading species in that failure are baseline network species such as `C6N2`, `H2`, and `C2H3CN`, not `GCH4`.

## Remaining Gaps

### 1. Baseline Long-Step Stability

The no-grain network still fails at very large timesteps. This is now the highest-priority numerical gap.

Likely issue:

- kintera currently follows a fixed KINETICS-base startup step sequence in the diagnostics.
- If a step is too aggressive, kintera can accept a bad implicit solve after clamping, allowing large errors to enter the next state.
- KINETICS-base appears to have more mature retry/fallback behavior when convergence fails.

Recommended next step:

- Implement an adaptive timestep controller around the existing implicit solve, but do it deliberately as a separate task:
  - reject non-finite solves,
  - reject severe negative concentrations,
  - estimate error using step doubling or a Rosenbrock-style embedded estimate,
  - reduce `dt` and retry before committing the state,
  - cap `dt` growth.

This should be done after deciding whether to build a lightweight backward-Euler controller or a more VULCAN-like Rosenbrock/Ros2 solver.

### 2. Grain Feedback Semantics

The full network runaway is still not closed.

Known improvements already made:

- product-only CH4 loading,
- shifted CH4 release,
- non-diffusive grain species,
- `NTOT` sublimation limiter.

Remaining possibilities:

- CH4 loading/release may need additional KB iteration semantics, not just source-term semantics.
- Sublimation/freezing may need to be coupled with timestep rejection to avoid large-step overshoot.
- Some gas-grain reactions may still be incorrectly paired or included compared with the active KB runtime network.
- The Fortran `HETEROK` branch may have more special rate overrides beyond the current `NTOT` limiter and CH4-specific behavior.

Recommended next step:

- Keep using `KINTERA_TITAN_NETWORK_MODE=no_grain` for baseline work.
- Reintroduce grain behavior in layers:
  1. condensation only,
  2. sublimation only,
  3. paired reversible sources,
  4. CH4 shifted release,
  5. full grain network.

### 3. Diagnostic Oracle Completeness

The existing KB diagnostic files are top-N style and not complete RHS oracles.

Status:

- kintera has a CH4 full RHS mirror diagnostic.
- KB full RHS instrumentation was started and used for insight, but it is not yet a polished, reusable oracle pipeline.

Recommended next step:

- Formalize per-reaction RHS oracle generation for selected species and timesteps.
- Store output schemas and comparison scripts under `diagnostics/`.

### 4. Solver Semantics vs. KINETICS-base

kintera uses a linearized backward-Euler step for transport/source terms. KINETICS-base has its own implicit iteration and convergence behavior.

Current gap:

- The matrix/operator semantics are now closer, especially with Dirichlet rows.
- The timestep acceptance/retry semantics are not yet equivalent.

Recommended next step:

- Treat timestep control as the next standalone engineering item before continuing grain chemistry debugging.
- Avoid adding more physical shims until the baseline network cannot accept non-finite or wildly unstable long steps.

## Suggested Commit Scope

This branch is currently a large but coherent Titan-support checkpoint. A reasonable commit message would be:

```text
Improve KINETICS-base Titan source semantics and diagnostics
```

High-level commit contents:

- ion chemistry classification and source-term support,
- Cheng Titan special mappings,
- photolysis/electron-impact alignment,
- Dirichlet boundary rows and cold-trap handling,
- Titan grain special source semantics,
- stable diagnostics paths and no-grain comparison mode,
- tests covering these semantics.

Do not treat adaptive timestep work as complete in this commit; it should be a follow-up.
