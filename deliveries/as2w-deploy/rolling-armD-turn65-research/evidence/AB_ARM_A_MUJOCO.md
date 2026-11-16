# A 组终点的 MuJoCo 矩阵（双仿真对照的另一半）

A 组（`4bdf4d26` 续训 1,000 轮、gyro 在闭环内）的**终点权重**在 MuJoCo 上跑同一冻结矩阵（5 站态、11 用例、gyro-1 `yaw_kp 60 / ff_cap 4`）。

- 终点权重 sha256 `da8c16747590d67b…`
- 原始数据：`evidence/mujoco/abA/abA_final.json`

## 1. 结果：**7/11，与对照同分，但构成变了**

| 用例 | 对照 `4bdf4d26`（MuJoCo） | **A 组终点（MuJoCo）** | 变化 |
| --- | --- | --- | --- |
| hold | PASS | PASS | 保持 |
| **turn_right_0.05** | **FAIL**（0.17–0.72） | **PASS（1.02–1.04）** | **转 PASS** |
| turn_left_0.05 | FAIL | FAIL（1.08–1.13 超调） | 不变 |
| turn_left_0.10 | FAIL | FAIL（yaw 0.84–0.87 在带内，**支撑不达标**） | 不变 |
| turn_right_0.10 | FAIL | FAIL（1.45–1.63 超调，比对照更远） | 更差 |
| straight_fwd 0.25 | PASS | PASS（跟踪 1.07） | 保持 |
| straight_bwd 0.25 | PASS | PASS（跟踪 1.06） | 保持 |
| arc_left 0.25 | FAIL | FAIL（0.65–0.67，略低于 0.70） | 略好但仍不过 |
| arc_right 0.25 | PASS | PASS（0.74–0.76） | 保持 |
| brake_fwd / brake_bwd | PASS | PASS | 保持 |
| **合计** | **7/11** | **7/11** | **不变** |

## 2. 双仿真并置

| 端 | 对照 | A 组终点 |
| --- | --- | --- |
| MuJoCo | 7/11 | **7/11**（`turn_right_0.05` 转 PASS，`turn_right_0.10` 更差） |
| Isaac | 6/11 | **6/11**（转向普遍退化，见 `AB_ARM_A_SCREEN.md`） |
| **两端共同通过** | hold、straight_fwd、straight_bwd、arc_right、brake_fwd、brake_bwd = **6** | **同样 6 项** |

**→ A 组把 `turn_right_0.05` 在 MuJoCo 上转过来了，但 Isaac 上对应工况反而退化，**两端共同通过集合没有扩大**。

## 3. 结论

- A 组**未扩大**双仿真共同通过集合（仍 6 项）
- MuJoCo 的 `turn_right_0.05` 改善**没有迁移到 Isaac**，这与此前「MuJoCo 排名不能预测 Isaac」的结论一致
- 与 goal 的硬约束一致：**A 组不具备扩到 4,000 轮的资格**

## 4. 边界

1. 仅 1 个权重、5 个站态、gyro-1 包装
2. 未测 A 组各中间检查点的 MuJoCo 表现
3. MuJoCo 侧判定沿用冻结门槛（yaw ∈[0.70,1.30]、支撑 ≥0.95、无终止、upright ≥0.85）
