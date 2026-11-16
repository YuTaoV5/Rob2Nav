# 候选筛选：MuJoCo 表现能否预测 Isaac 表现？

同一 gyro-1 包装（`--yaw-kp 60 --ff-cap 4`）、两种子（7393 / 311）、同一冻结矩阵。

| 候选 | sha256 | MuJoCo | **Isaac** | 结论 |
| --- | --- | --- | --- | --- |
| **armE final** | `4bdf4d26` | **7/11**（端点，5 站态复现） | ✅ **6/11** | **双端最好** |
| armA m4200 | `d3e6cdf5` | 7/11（中途点，按 8 项口径最佳） | 🔴 **5/11** | 中途优势不迁移 |
| armE m4400 | `5952d90e` | 6/11（`turn_right_0.05` PASS） | 🔴 **5/11** | MuJoCo 特有优势不迁移 |

## 逐例（Isaac，seed 7393）

| 用例 | armE final | armA m4200 | armE m4400 |
| --- | --- | --- | --- |
| hold | PASS | PASS | PASS |
| straight_fwd | PASS | PASS | PASS |
| **arc_right** | 🎯 **PASS（0.78）** | FAIL（0.62） | FAIL（0.18） |
| arc_left | FAIL（0.50） | FAIL（0.38） | FAIL（0.27） |
| brake_fwd | PASS | PASS | PASS |

## 结论

1. **MuJoCo 表现不能预测 Isaac 表现。** 两个在 MuJoCo 上各有所长的中途检查点，在 Isaac 上都只有 5/11，
   且两个弧线都不通过；而 armE 终点在两端都是最高分。
2. **armE final 是唯一同时为两端最佳的候选**，因此续训起点与交付权重都应固定为它 —— 候选筛选问题已闭合。
3. 这也再次说明「MuJoCo 的 7/11 不能外推」：不同的 7/11 在 Isaac 上会落到 5/11 或 6/11。

## 边界

仅仿真、两种子、gyro-1 包装；未在其他候选（m799/m1999/m2800）上重复该对照。
