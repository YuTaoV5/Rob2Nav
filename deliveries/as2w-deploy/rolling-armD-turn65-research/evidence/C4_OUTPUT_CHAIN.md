# C4 完整输出链路

## 策略输出 → 最终目标

```
raw = policy(obs)                        # 16 维，float32，policy.onnx 即这一段
a   = clip(raw, -6.0, 6.0)               # clip_actions = 6.0
leg = a[:12] * 0.35                      # 腿：KP 55, KD 2, scale 0.35
wheel_tgt = a[12:] * 16.0                # 轮：KP 0, KD 6, scale 16, 单位 rad/s
```

| 组 | KP | KD | scale | 单元 |
| --- | --- | --- | --- | --- |
| hip / thigh / calf | 55 | 2 | 0.35 | 位置 PD |
| foot（轮） | **0** | **6** | **16** | 速度 PD |

## gyro-1 包装（本包新增，`runtime/r11_acceptance_ff.py`）

```
D      = S * clamp( yaw_kp*(wz_cmd - wz_meas) + ff_gain*wz_cmd , ±ff_cap )
S      = (-1, +1, -1, +1)                # FL,FR,RL,RR → 左右差速
tau    = KD*(wheel_tgt - dq) + D          # 仅四轮
sp     = dq + tau / KD                    # 交给执行器，仍在其 ±13.5 N·m 处 clamp
```

参数：`yaw_kp = 60`、`ff_cap = 4.0 N·m`、`ff_gain = 0`（已验证最优点）。

## 力矩保护

| 层 | 限值 |
| --- | --- |
| 位置/速度 PD + 前馈合计 | 每轴 **90%** 硬限位（hip/thigh 54、calf 81、wheel 13.5 N·m） |
| 动作 clip | ±6.0（非约束：6×16 = 96 > 98.47 的 90% 轮上限，故轮力矩实际由 13.5 N·m 约束） |
| 前馈自身 | ±4.0 N·m（在 13.5 之内） |
| 无积分器 | 是（gyro-1 无状态，4 次/策略步不复合） |

## 训练 / 评估 / 部署差异

| 项 | 训练 | 评估 | 部署 |
| --- | --- | --- | --- |
| 观测 65 维 | 同 | 同 | 需实机重建，**not_verified** |
| 动作延迟 DR | 已关闭 | 关闭 | **not_measured** |
| gyro-1 | 无 | 启用（cap4/k60） | 建议启用，**实机未验证** |
| 腿滤波 | 5 Hz | 同 | 同 |
| DDS 槽位 / 零位 / 方向 | — | — | **not_run**（R3 项） |

## 缺口

- **未提供实机 DDS 映射表**（这是 R3 的前置，需要现场固件资料与授权）
- 观测在实机的重建（时间戳、数据龄、缺测）为 `not_measured`
