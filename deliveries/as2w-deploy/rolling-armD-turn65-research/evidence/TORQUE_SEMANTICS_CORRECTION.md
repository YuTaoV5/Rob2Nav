# 🔴 更正：`applied_torque` 是**被限幅的遥测估算**，不是求解器驱动力矩 —— 饱和率必须 `not_measured`

操作员指出：`20 → 13.5` 只验证了**读出字段经过限幅**；Isaac Lab 的 `ImplicitActuator` 用当前关节状态估算 PD+前馈并限幅该**估算值**，而 PhysX 在求解器内部**另行**施加驱动力矩。两条路径都能出现 13.5，故截断不能识别求解器实际施力。

本轮复核了**服务器上已安装版本**的源码，**该纠正成立，而且源码把这件事写得很直白**。

## 1. 已安装版本源码原文

`/workspace/isaaclab/source/isaaclab/isaaclab/actuators/actuator_pd.py`，`actuator_effort_limit` 的 docstring：

```
The actuator's rated force/torque reflected at the joint when configured through
ActuatorBaseCfg.actuator_effort_limit; it clips the effort telemetry but is not pushed to the solver.
```

**关键句："it clips the effort telemetry but is not pushed to the solver"**
- **clips the effort telemetry** —— 限幅的是**遥测/估算值**
- **is not pushed to the solver** —— **不推送到求解器**

**→ 我上一轮判定 `applied_torque` 为「限幅后的真实施力」是错的。它是被限幅的估算遥测。**

## 2. 为什么「20 → 13.5」不足以判定

| 路径 | 是否会出现 13.5 |
| --- | --- |
| **估算路径**：`ImplicitActuator.compute()` 用当前 q/dq 估算 PD+前馈 → 对该估算限幅 → 写入遥测 | ✅ 会 |
| **求解器路径**：PhysX 隐式求解内部自行施加驱动力矩 | ✅ 也会（但**不经过**同一个限幅代码） |

**→ 两条路径都能给出 13.5，截断不可区分来源。** 我把它当作「决定性证据」是过度推断。

## 3. 正确的标记

| 量 | 标记 |
| --- | --- |
| **Isaac 隐式驱动的实际饱和率** | **`not_measured`** |
| 此前报的 **0.0%** | 改标为**「估算力矩触顶率」**（`estimated_torque_ceiling_rate`），**不是**求解器饱和率 |
| MuJoCo 侧的饱和率 | ✅ 仍可用 —— MuJoCo 的 `d.ctrl` 就是施加力矩，`actuator_force` 是求解器输出，路径单一可验证 |

## 4. 这对已有结论的影响

| 结论 | 状态 |
| --- | --- |
| 「两端轮端回路在无载时等价」 | ✅ **不受影响** —— 该结论基于轮速（B 臂收敛到 0.5000），不依赖力矩字段 |
| 「Isaac 悬空测试饱和占比 0.0 是真实值」 | ❌ **改为 `not_measured`**；0.0 只说明**估算力矩**没触顶 |
| 「接地工况可用 applied_torque 判饱和」 | ❌ **不可** —— 必须另找可验证的求解器驱动力矩，或改用操作员第 3 条的显式力矩对照 |

## 5. 按操作员要求采用的统计口径（下一轮）

| 字段 | 记录方式 |
| --- | --- |
| 原始 effort 命令 | 保存 |
| **限幅前估算力矩** | 保存（`computed_torque`） |
| **限幅后估算力矩** | 保存（`applied_torque`） |
| **可验证的求解器驱动力矩** | 待找；若接口只提供净关节力/反力，**单独标记来源** |

**触顶统计口径：`abs(tau) >= 0.999 × 13.5`，覆盖正反两侧。**

## 6. 方法论教训（第二次同类）

| # | 我把什么当成了什么 |
| --- | --- |
| 1 | 把**行进距离**当漂移 |
| 2 | 把**估算遥测算力**当成**求解器施力** |

**共同点：都把「一个看起来对的数」当成了「它名字所声称的量」，而没有核对产生它的路径。****→ 规则：任何从仿真接口读出的量，必须能说清它由哪条代码路径产生、是否经过限幅/滤波/估算；说不清就标 `not_measured`。**
