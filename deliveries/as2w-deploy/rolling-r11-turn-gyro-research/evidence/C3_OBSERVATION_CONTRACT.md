# C3 逐维观测合同（65 维）

来源：`training/as2w-backup/as2w-current/code/as2w_isaac_flat_env_all.py::observation()`
（sha256 `8d6438705e3bf911…`）。下表按代码逐段核对，不是凭记忆。

| 索引 | 维度 | 名称 | 单位 / 形式 | 坐标系 | 缩放 / 处理 | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| 0:3 | 3 | 速度指令偏移 | m/s, rad/s | 机体系 | `command − (0.4, 0, 0)` | `base_velocity` 指令（经 0.12 斜率 lerp） |
| 3:6 | 3 | 机体角速度 | rad/s | 机体系 | 不缩放 | `root_ang_vel_b` |
| 6:9 | 3 | 投影重力 | 单位向量 | 机体系 | 不缩放 | `projected_gravity_b` |
| 9 | 1 | 常值 0.4 | — | — | `full_like(clock, 0.4)` | 代码常量 |
| 10:26 | 16 | 关节位置误差 | rad | 关节 | `q − default_joint_pos`；**轮轴（12:16）回绕到 [−2π, 2π]** | `joint_pos` |
| 26:42 | 16 | 关节速度 | rad/s | 关节 | 不缩放 | `joint_vel` |
| 42:58 | 16 | 上一帧动作 | 策略输出单位 | — | 原始动作（裁剪前） | `action_manager.action` |
| 58 | 1 | 相位 sin | — | — | `sin(clock)`，`clock = 2π·ep_len·dt/0.62` | 回合内相位 |
| 59 | 1 | 相位 cos | — | — | `cos(clock)` | 同上 |
| 60:64 | 4 | 零指令轮角漂移 | rad | 关节 | `(Δwheel / 0.20).clamp(−3, 3)` **× 零指令标志** | 轮角相对「进入零指令时」的参考 |
| 64 | 1 | 零指令标志 | 0/1 | — | 三轴指令全为 0 时为 1 | 指令 |

## 关节顺序（16 轴，canonical）

`JOINTS = (FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, ..., FL_foot, FR_foot, RL_foot, RR_foot)`，
即 **按 part 分组、组内按腿**。`env.as2w_joint_ids` 给出该顺序，轮轴在 **12:16**。

## 内部状态 / 复位

| 项 | 说明 |
| --- | --- |
| `_wheel_reference` | 进入零指令（或回合第 0 帧）时锁存轮角；**离开零指令时该参考不更新** |
| `_was_zero` | 上一帧零指令标志，用于检测「进入」沿 |
| 相位 `clock` | 由 `episode_length_buf` 与 `step_dt` 导出，**无独立状态**，reset 即归零 |
| 动作历史 42:58 | 仅上一帧（1 帧），不是多帧队列 |

## 缺失 / 滤波

| 项 | 现状 |
| --- | --- |
| 观测滤波 | **无**（腿滤波作用在动作侧，不在观测侧） |
| 时间戳 / 数据龄 | **not_measured** —— 训练与仿真内不建模观测龄；实机侧需在部署包装中补 |
| 缺测处理 | 仿真内不存在缺测；实机 `not_measured` |

## 与本包 ONNX 的关系

`weights/policy.onnx` 的输入就是这 65 维、**顺序与本表完全一致**；C8 的 600 帧真实回放
（max_abs_err 5.96e-07）是在这条合同上验证的。
