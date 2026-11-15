# 双仿真四象限对照（同一权重、同一判据）

- 权重：`weights/armE_support2p0_final.pt`，sha256 `4bdf4d26b076d7c7…`
- 判据：**每个用例的所有站态（MuJoCo 5 态）/ 所有种子（Isaac 7393+311）都 PASS 才算 PASS**
- 包装：**base**（原包装）与 **gyro-1**（`yaw_kp 60 / ff_cap 4`），两端**同一套参数**
- MuJoCo 站态 2/3/4/5/6 s；Isaac 种子 7393 / 311
- 网关与容差与冻结矩阵完全一致（未改动）

## 逐例结果

| 用例 | MuJoCo base | MuJoCo gyro-1 | Isaac base | Isaac gyro-1 |
| --- | --- | --- | --- | --- |
| hold | PASS | PASS | PASS | PASS |
| turn_left_0.05 | fail | fail | fail | fail |
| turn_right_0.05 | fail | fail | fail | fail |
| turn_left_0.10 | fail | fail | fail | fail |
| turn_right_0.10 | fail | fail | fail | fail |
| straight_fwd_0.25 | PASS | PASS | PASS | PASS |
| straight_bwd_0.25 | PASS | PASS | PASS | PASS |
| arc_left_0.25 | PASS | PASS | fail | fail |
| arc_right_0.25 | PASS | PASS | fail | PASS |
| brake_fwd_0.25 | PASS | PASS | PASS | PASS |
| brake_bwd_0.25 | PASS | PASS | PASS | PASS |

| **合计** | **7/11** | **7/11** | **5/11** | **6/11** |

## 直接回答「哪些用例两端都通过」

| 包装 | 两端同时通过 | 项数 |
| --- | --- | --- |
| **base** | hold、straight_fwd_0.25、straight_bwd_0.25、brake_fwd_0.25、brake_bwd_0.25 | **5** |
| **gyro-1** | 上述 5 项 + **arc_right_0.25** | **6** |

## 三项关键事实

1. **唯一「仅单端通过」的用例是 `arc_left_0.25`**：MuJoCo 两套包装都 PASS，Isaac 两套包装都 fail。
2. **`arc_right_0.25` 是 gyro-1 带来的唯一跨端收益**：Isaac base `fail` → gyro-1 `PASS`。
3. **gyro-1 在 MuJoCo 上不改变任何判定（0 项变化）**：说明该包装在不需要它的被控对象上**是惰性的**，
   不会干扰已经工作的端 —— 这对可部署性是一个正面信号。

## 包装一致性核对（两端逐项对齐）

| 项 | MuJoCo 实现 | Isaac 实现 | 是否一致 |
| --- | --- | --- | --- |
| 差速符号 `S` | `(-1, +1, -1, +1)` for FL,FR,RL,RR | 同一向量 | ✅ |
| 驱动量 | `wz_cmd` 与 `wz_meas`（指令与实测偏航角速度） | 同 | ✅ |
| `wz_meas` 来源 | 自由关节 `qvel[5]`（**机体系**） | `root_ang_vel_b[2]`（**机体系**） | ✅ |
| `wz_meas` 采样 | **每 20 ms 策略步读一次**，32 个子步内保持不变 | 每策略步读一次，4 个子步内保持不变 | ✅ 语义一致 |
| 前馈限幅 | `clip(yaw_kp*err, ±4 N·m)` | 同 | ✅ |
| 力矩总限幅 | 前馈**加到** kd 律上，和值再 clamp ±13.5 N·m | 同 | ✅ |
| 积分/滤波状态 | 无 | 无 | ✅ |

## 边界

1. MuJoCo 的 `wz_meas` 取 `qvel[5]`，Isaac 取 `root_ang_vel_b[2]`；两者都是机体系偏航角速度，
   但**来自不同的状态估计通路**，数值不保证逐帧相同。
2. 仅 1 个权重候选；未在 m799/m1999/m2800 上重复。
3. 未测 `arc_left` 在 Isaac 上失败的**首个环节**（见 `evidence/PURE_TURN_FIRST_FAILURE.md`）。
