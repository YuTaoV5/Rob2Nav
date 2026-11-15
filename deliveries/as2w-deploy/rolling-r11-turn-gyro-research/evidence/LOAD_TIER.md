# 固定载荷档：0 kg 与 +6 kg 对照（仿真）

脚本：`runtime/r11_acceptance_ff.py --payload-kg <kg>`（gyro-1 包装 `--yaw-kp 60 --ff-cap 4`）
权重：`weights/armE_support2p0_final.pt`（sha `4bdf4d26`），种子 7393，Isaac
原始数据：`load/payload_0kg.json`、`load/payload_6kg.json`

## 质量读回（按刚体名，非假定）

| 载荷 | base_link 质量 | 总质量 |
| --- | --- | --- |
| 0 kg | —（基线） | **26.00 kg**（第 77 节实测） |
| **+6 kg** | **16.580 kg** | **32.000 kg** |

载荷通过 startup 质量事件施加到 `base_link`（`operation=add`），读回值与请求值一致。

## 结果

| 载荷 | 通过数 |
| --- | --- |
| **0 kg** | **6/11** |
| **+6 kg** | **4/11** |

6 kg 下的逐例变化：

| 用例 | 0 kg | +6 kg | 失效原因 |
| --- | --- | --- | --- |
| hold | PASS | PASS | vx 0.0173（逼近 0.02 门槛） |
| straight_fwd_0.25 | PASS | **FAIL** | 四轮支撑 **0.593**、1 次终止 |
| arc_right_0.25 | PASS | **FAIL** | 四轮支撑 **0.720**、1 次终止 |
| arc_left_0.25 | FAIL | **PASS** | — |
| brake_fwd_0.25 | PASS | **FAIL** | 末段 vx 0.0722 |
| turn_right_0.05 | FAIL | FAIL | 支撑 **0.907**，偏航 0.70 已在带内 |
| 其余纯转 | FAIL | FAIL | — |

## 结论

**实际支持的载荷范围：0 kg（无额外负载）。+6 kg 档不通过**（6/11 → 4/11，支撑掉到 0.59–0.91 并出现终止）。
最终报告的载荷范围应写为「**无载通过；+6 kg 未通过**」。

## 边界

1. **仅仿真**，单种子
2. 载荷加在 `base_link` 质心处，**未验证真实负载的质心/惯量分布**
3. 未与摩擦、时延、推扰组合
4. 未做 3 kg 或 9 kg 等中间档
