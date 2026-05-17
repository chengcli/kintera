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

#### 第一层：subdivision-on-rejection controller（已实现）

`python/atm2d/timestep.py` 提供 `adaptive_advance(state, dt_target, step_fn, ...)`，行为：

- 调用 `step_fn(state, dt)` 拿到 candidate concentration；
- `default_accept` 检查 non-finite / 严重负值（per-species threshold）/ 绝对量级 cap；
- reject 则 `dt *= shrink_factor` 重试，accept 则下一次最多 `grow_factor` 放大；
- 超过 `max_subdivisions` 仍 reject 就抛 `RuntimeError`。

`diagnostics/compare_gch4.py` 已切到该 controller；`NTIME` 通过 `KINTERA_TITAN_NTIME` 配置，
`max_subdivisions` 通过 `KINTERA_TITAN_MAX_SUBDIV` 配置。Unit tests 在
`tests/test_adaptive_timestep.py`（10/10 passed 纯 python 验证）。

验收 (TODO 用户在 macOS 或 docker 环境跑)：
```bash
KINTERA_TITAN_NETWORK_MODE=no_grain KINTERA_TITAN_NTIME=50 \
    python diagnostics/compare_gch4.py
```
应无 NaN/Inf，trace 显示 subdivision 增多但不 collapse。

#### KB Fortran timestep 行为 audit（重要发现）

针对 `diagnostics/KINETICS-base-compare/src/KINETGENX/` 的审计揭示：

1. **KB 真实 dt schedule 不是 `1e-15 × √10`**，是 stage-based:
   - branch 1 (initial): 60s → 600s → 3600s → 7200s → 10800s
   - branch 2 (subsequent): 600s → 3600s → 7200s → 10800s
   - 每个 stage 推进固定 `TELAPSE` 累计时间，之后进入下一 stage dt
   - kintera diagnostic 现有 `_fixed_timestep_sequence`（`1e-15 × 10^0.5`）是诊断脚本自创，不是 KB 等价
   - 这意味着 `NTIME=50` 那个 `dt~3e9 s` 是 KB **从来不会尝试**的步长

2. **KB `MARCH` 是 true Newton with full Jacobian re-evaluation**:
   - `HETEROK` 每个 inner iteration 重算 rate coefficient
   - `BMATRM` 每个 iteration 重组 `B = I/DELT - dF/dC`
   - 收敛准则: `|10^(log|new|−log|old|) − 1| ≤ CONV`（≈ 1e-4），per species
   - kintera 是 single frozen linearization — 这是真正的物理 gap

3. **KB 有 step rejection + state rewind**:
   - `CONVRG` 返回 `IFLAG ∈ {1,2,3,4}`
   - `IFLAG=2,4` (含负值 or 不收敛) → `RETRY`
   - `RETRY` 把 `DELT /= 2^(NTRYS−1)`，从 backup `CONCP` 回滚 state
   - 上限通常 ~8 层 subdivision
   - 我们 controller 的 reject/subdivide 是同一思路，但没有 Newton 内层

4. **KB 不 clamp**: 负值由 `CONVRG` 检测后触发 reject，而不是 post-hoc clamp。

#### 第二层：true Newton inner iteration（已实现）

`python/atm2d/newton.py` 提供 `newton_implicit_step(state, dt, ...)`，行为：

- 入口：`state.concentration = c_0`（BE 起点）。
- 每个 iter 重新 build `(I − dt·(T+J(c_k)))` 和 RHS = `c_0 + dt·(S(c_k) − J(c_k) c_k)`
  （`build_implicit_step_system` 加了可选 `c_0` 参数，让 RHS 用固定 c_0 而不是当前 iterate）。
- solve → `c_proposed`；可选 damping 把 step scale 下来（KB IDAMP=1 等价）。
- 收敛判据：per-species fractional change `max |c_{k+1}−c_k| / max(|c_k|, floor)` < `convergence_tol`。
- **Out-of-basin guard**：iter 1 max_rel > `out_of_basin_threshold`（默认 1.0）→ 立即退出，返回 iter-1 结果（等价 frozen-J）。这避免在 Newton 出 basin 时继续 re-linearize 把状态弄得更糟。
- **Best-iterate fallback**：max_iter 用完仍未收敛时，返回所有 iter 中 max_rel 最小的那个，而不是最后一个。
- **Divergence guard**：max_rel 单 iter 涨 10×，或绝对值超 1e6 → 退出。
- 退出前把 `state.concentration` 恢复成 c_0，让 controller 的 `accept_fn` 用正确的 reference state 算 species scale。

在 `compare_gch4.py` / `no_grain_stability.py` 里，step_fn 把
`build_implicit_step_system + apply_dirichlet + solve + apply_pins` 整个 BE 单步包成 Newton 内层。
Controller 的外层 accept_fn 仍负责 reject non-finite / severe negative / magnitude cap。

#### 第三层：Controller rejection memoization（已实现）

`adaptive_advance` 现在 track `min_rejected_dt`，在同一次 advance 里不会再尝试 ≥ 该 dt 的步长。
这避免了在 stiff zone 反复 "尝试 5e-4 → 被拒 → 降到 2.5e-4 → 接受 → 再尝试 5e-4..." 的浪费。

#### 当前验证 (NTIME=27, no_grain)

`KINTERA_TITAN_NTIME=27 python diagnostics/no_grain_stability.py` 端到端跑通：

```text
[done] totals: accepted=87 rejected=5
[done] newton: runs=92 total_iters=159 non_converged=10 avg_iter_per_run=1.73 max_iters_in_step=6
[done] final max abs concentration: 2.174e+15
[done] final min concentration:    -1.932e-07
```

各 step 行为：

| step | dt_target | newton 行为 | 结果 |
|------|-----------|-------------|------|
| 1–23 | 1e-15 .. 1e-4 | 1 iter 收敛 | 单步 accept |
| 24 | 3.16e-4 | iter 1 max_rel=3.3e4 > basin threshold → 返回 iter-1（= frozen-J），min(c)=-1.8e-33 | 单步 accept |
| 25 | 1e-3 | 在 dt=2.5e-4 处 Newton 收敛；5e-4 reject 1 次后 memo | 4 accept + 2 reject, last_dt=2.5e-4 |
| 26 | 3.16e-3 | 同上，max safe dt ≈ 4.1e-4 | 12 accept + 1 reject |
| 27 | 1e-2 | safe dt ~ 3.1e-4 | 47 accept + 2 reject |

stability 达成：无 NaN，无 floor hit，final min(c)=-1.93e-7（trace ion noise，在 tolerance 内）。
wall time ~5 min。

NTIME ≥ 30 的 dt_target 走到 0.1 s 以上，子步数随 √10 增长几何放大，单跑 wall time
不可接受。但 NTIME=50 让 dt_target 达 3.16e9 s（积分总时间 4.6 billion s = 146 年），
本身就不是 KB 等价 schedule — KB 实际 schedule 上限 10800 s。Step 2 换成 KB 的
stage schedule 后 wall time 才会回到合理量级。

#### 第四层：KB DELTIM ramp + stage schedule（已实现）

`python/kinetics_base_titan/schedule.py` 提供 `kinetics_base_titan_dt_schedule(ntime, deltim=-1e-15, ncycle=10, branch=1)`，精确复刻 KB Fortran 的 schedule（`kinetgen2X.F:14911-14918`）：

- 负 DELTIM 触发 `10^(1/NCYCLE) = 10^0.1 ≈ 1.26×/step` 指数 ramp 从 `|DELTIM|` 起
- 配合 stage 表（branch 1: `60s → 600s → 3600s → 7200s → 10800s`，TELAPSE 各阶段 `1hr → 1hr → 6hr → 16hr → ∞`）
- 一旦 ramp 达到 stage TSTEP，DELT 被 cap 到该 stage 值；ESTAGE 累计直到 TELAPSE 后切下一 stage

诊断里 `KINTERA_TITAN_SCHEDULE=kb` (default) / `legacy_sqrt10` 切换。NTIME=50 在 KB ramp 下 dt 从 1e-15 → 7.94e-11 s（与 KB 完全等价）。NTIME=167 才到 stage 1 cap 60s。

#### 第五层：Chemistry-only Newton + operator-split transport（已实现）

audit 揭示我们之前 Newton 把 transport+chemistry 耦合到一个 inner iter，而 KB 是 chemistry-only Newton + 单独的 transport step (`FLOW2D` before/after `MARCH`)。在 stiff dt 下耦合 Newton 跳出 basin，KB 不会。

`python/atm2d/newton.py` 加了 `chemistry_only_newton_step(state, dt, source_terms, ...)`：

- 每个 iter 把 `state.concentration` 设到当前 iterate，build `S(c_k)` 和 `J(c_k)`（per-cell local 已经在 `build_source_linearization`）
- 求解 per-cell `(I − dt·J(c_k)) Δc = −(c_k − c0 − dt·S(c_k))`
- batched `torch.linalg.solve(A, rhs)` 对 ncol×nlyr 个 nspecies×nspecies 系统并行
- damping / out-of-basin guard / best-iterate fallback 都保留

`diagnostics/no_grain_stability.py` 的 step_fn 做 Lie splitting：

1. transport-only BE: `build_implicit_step_system(..., source_terms=None)` 解 `(I − dt·T) c = c0` + Dirichlet
2. chemistry-only Newton: 从 c_after_transport 开始 per-cell Newton
3. 都加 boundary pins

`KINTERA_SOLVER_MODE=split` (default) / `coupled` 切换。

**效果对比** (NTIME=200, no_grain, KB schedule, step 117-130 区间):

| 模式 | step 118 (dt=5e-4) | step 124 (dt=2e-3) | step 130 (dt=8e-3) |
|------|----|----|----|
| coupled | 2 acc + 1 rej | 12 acc + 1 rej (cascade 起点) | 47 acc + 2 rej |
| split | 1 acc + 0 rej | 1 acc + 0 rej | 1 acc + 0 rej |

stiff zone 完全消失。dt 直接跟着 KB ramp 走。

#### 后续

- BC type-4 prescribed flux (H/H2 lower-bound upward flux) 当前未 parse — 影响 budget 不影响稳定性
- Full network (ion + grain) 还需测试
- KB IFLAG=2 partial-convergence 报告我们没有；KB ICNV=2 negative→abs handling 也没有 — 当前 default_accept 用 severe-negative threshold 替代

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
