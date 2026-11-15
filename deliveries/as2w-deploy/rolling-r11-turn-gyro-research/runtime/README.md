# runtime

| 文件 | 用途 |
| --- | --- |
| `r11_acceptance_ff.py` | Isaac 11 用例验收，**含 gyro-1 包装**（`--ff-gain`/`--ff-cap`/`--yaw-kp`/`--leg-kp-scale`） |
| `r11_acceptance_v2.py` | 同一矩阵的**未改包装**基线（对照用） |
| `r11_mj_matrix_v2.py` | MuJoCo 11 用例矩阵（多站态） |
| `r11_isaac_exec_probe.py` | 逐轮逐帧执行链探针（目标/实测/力矩/触顶/载荷） |
| `r11_caliber.py` | 双口径汇总（raw11 与数据驱动可达集） |
| `export_policy_onnx.py` | 从 checkpoint 的 `actor_state_dict` 导出 `policy.onnx` 并做 ORT 一致性校验 |
| `r11_c8_dump.py` | 捕获真实 rollout 的逐帧 obs / 原始输出 / 最终轮目标 / 实测 dq / 下一帧 obs |
| `r11_c8_verify.py` | 用 `policy.onnx` 重放捕获的 obs，报告最大误差、最差维度与首次分歧帧 |

## gyro-1 包装（版本 gyro-1）

```
tau_ff = S * clamp( yaw_kp * (wz_cmd - wz_meas), +-ff_cap )
S = (-1, +1, -1, +1)   # FL,FR,RL,RR → 左右差速
sp    = dq + tau_ff / kd          # 执行器仍按 kd 律并在 +-13.5 N·m 处 clamp
```

- **驱动量是指令**的 `wz` 与实测 `wz`：`wz_cmd = 0` 时前馈严格为 0，因此驻停、直线、刹停按构造不受影响。
- **无状态**：无积分器、无滤波器，每个策略步 4 次子步调用不会复合。
- **仅四轮**，腿不动。
- **90% 力矩硬限位不变**：执行器仍 clamp ±13.5 N·m；前馈自身另有 ±4 N·m 上限。
- 已验证最优点：`--yaw-kp 60 --ff-cap 4`（Isaac 6/11，两种子一致）。`--yaw-kp 0` 即回到未改包装。

## 已知缺陷（包装层，非本目录问题）

1. **符号**必须为 `S = (-1,+1,-1,+1)`。曾误用 `(-1,-1,+1,+1)`，那是**前后**力偶，不产生偏航。
2. 早退守卫必须同时检查 `ff_gain` 与 `yaw_kp`，否则陀螺项被静默跳过（`ff_max` 会显示 0.00）。
3. **每次新臂必须验证「声称会变化的量确实非零」**（`ff_max>0`、`caps>0`），通过数与打印值都曾误导。
