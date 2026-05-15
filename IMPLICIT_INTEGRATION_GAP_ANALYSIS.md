# 隐式积分 Gap 分析

当前事实来源见 `KINETICS_BASE_TITAN_GAP_STATUS.md`。这份文件只保留仍然成立的数值结论，已删除早期针对 CH4/NH3 collapse 的过时 trace。

## 当前判断

当前最大的数值 gap 是 timestep acceptance/retry，而不只是单个 reaction rate 或单个 source term。

在 `no_grain` baseline 下：

- 早期 CH4/GCH4 grain runaway 被移除。
- 固定 KINETICS-base startup timestep sequence 可以稳定到 `NTIME=45`。
- 到 `NTIME=50` 时，超大 timestep (`dt ~ 3e9 s`) 仍可能让 baseline stiff network 产生巨大值或 NaN。
- 主导失败物种已不是 grain species，而是 `C6N2`、`H2`、`C2H3CN` 等 baseline network species。

因此，在继续修 full grain network 前，应先避免 kintera 接受 bad implicit step。

## kintera 当前 Step 形式

当前 `atm2d` implicit step 使用一次 frozen linearization：

```text
S(c_new) ~= S(c0) + J(c_new - c0)

(I - dt * (T + J)) c_new = c0 + dt * (S(c0) - J c0)
```

其中：

- `S` 是 local chemistry/source tendency；
- `J = dS/dc`；
- `T` 是 transport operator；
- source/boundary/grain terms 可以进入同一个 implicit matrix；
- Titan fixed/boundary cells 可以通过 Dirichlet rows 强制。

这个结构对小步和近线性问题是合理的，但在大 `dt`、强 nonlinear coupling 下，单次 frozen linearization 可能给出非物理解。

## KINETICS-base / VULCAN 风格的关键差异

KINETICS-base 更接近 residual/correction solve，并有自己的 convergence/fallback 语义。即使我们还没有完整复刻它，也不应该接受明显坏掉的 step。

VULCAN 的思路也不是“崩了之后继续固定步长”，而是：

- 每步检查 positivity / truncation error / conservation；
- 失败则 reject step，缩小 `dt` 重新算；
- 成功后根据误差调大或调小下一步 `dt`；
- 不把 bad state 推进时间序列。

## 建议实现方向

下一步应作为独立任务实现 timestep controller。建议先做轻量版本，不急着重写成 Rosenbrock：

1. 用现有 `build_implicit_step_system` 和 `solve_sparse_system` 作为 primitive。
2. 每个 proposed `dt` 先尝试 solve。
3. 若出现 non-finite、严重负值、或误差过大，则 reject，缩小 `dt` retry。
4. 用 step-doubling 估计误差：

```text
full = step(dt)
half = step(dt / 2) twice
error = norm(half - full)
```

5. 接受更可信的 half-step result。
6. 限制 `dt` 增长，例如最多 `2x`。
7. 记录 accepted/rejected step，方便与 KB timestep trace 对照。

## 不建议现在做的事

- 不要继续用更多 physical shim 掩盖 baseline timestep failure。
- 不要把未验证的 adaptive controller 混入当前 checkpoint commit。
- 不要只靠 solve 后 `clamp(min=0)`，因为 bad state 已经进入系统。
- 不要简单删除 off-diagonal Jacobian；full Jacobian 是必要的，问题在 step acceptance 和 nonlinear update 语义。

## 与当前状态文档的关系

更完整的已实现内容、验证结果和下一步排序统一维护在：

```text
KINETICS_BASE_TITAN_GAP_STATUS.md
```
