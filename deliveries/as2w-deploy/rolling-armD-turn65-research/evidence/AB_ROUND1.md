# A/B 第一轮：A 已开训，B 课程已实现并验证采样

## 1. A 组（等预算对照）—— 在训

| 项 | 值 |
| --- | --- |
| 热启动 | `4bdf4d26`（iter 4597）✅ `R11_RESUMED_FROM` |
| gyro 闭环 | ✅ `AS2W_GYRO_TRAIN_LOOP ON yaw_kp=60.0 ff_cap=4.00` |
| 预算 | 1,000 轮（4597 → 5597） |
| 保存间隔 | 100 轮（已落盘 `model_4600.pt`） |
| 环境数 | 4096 |
| 采样 | **与 armE 完全相同**（等预算对照的定义） |
| 启动时 robust_env | `857f8edd`（见第 3 节溯源） |

## 2. B 组（长转课程）—— 课程已验证，**尚未开训**

新增 `CurriculumFlatCommands(FlatCommands)`，**只改采样与命令驻留**：

| 类别 | 目标占比 | 命令配置 |
| --- | --- | --- |
| 纯转 | 0.35 | vx=0, vy=0, wz = ±0.05 或 ±0.10 |
| 弧线 | 0.25 | vx=+0.25, wz = ±0.05 |
| 直线 | 0.15 | vx 在 [−0.25, +0.25] 连续变化, vy=0 |
| 刹停 | 0.15 | 前进/倒车各半, vx = ±0.25 |
| 驻停 | 0.10 | 全零 |

**驻留**：`resampling_time_range = (8, 12)` 秒（`AS2W_AB_TURN_DWELL_MIN_S/MAX_S` 控制）。

### 采样数学的 CPU 单元测试（20 万样本，不占 GPU）

| 类别 | 实测占比 | 目标 | 命令检查 |
| --- | --- | --- | --- |
| pure_turn | 0.3497 | 0.35 | vx=0.000，wz 仅取 0.05/0.10 ✅ |
| arc | 0.2505 | 0.25 | vx=0.250 ✅ |
| straight | 0.1496 | 0.15 | vx∈[−0.25,0.25]，vy=0 ✅ |
| brake | 0.1498 | 0.15 | 前进占比 **0.500** ✅ |
| hold | 0.1003 | 0.10 | 全零 ✅ |

**结论：`CURRICULUM_OK`**

### 开关与防「配了但无效」

- 课程类**只在 `AS2W_AB_CURRICULUM=1` 时启用**（配方 B 分支显式设置），A 组保持 armE 原采样
- 实例化时打印 `AS2W_AB_CURRICULUM ON  fractions=[...]  dwell=8-12 s`
- **开训 B 前必须先确认该行出现**，否则判为未生效 —— 本项目已发生 7 次「看起来配了但没生效」

## 3. 溯源说明（重要）

A 组启动时的 `as2w_isaac_robust_env.py` 是 `857f8eddf7ddebb926796569586da7bf96c8f4154e21417e45e8165d6285728e`；此后为接入课程改为 `a8b0e3f12da259e5c34a54235ec3aea15ad43a86ba2930e6e3e23954142e6bfd`。**新增代码块在 `AS2W_AB_CURRICULUM` 未设时完全惰性**，因此用当前文件重跑 A 组在功能上等同；但严格溯源时 A 组的实际输入哈希应记为 `857f8edd…`。配方现 pin 新哈希。

| 文件 | 当前 sha256 |
| --- | --- |
| `as2w_isaac_robust_env.py` | `a8b0e3f12da259e5c34a54235ec3aea15ad43a86ba2930e6e3e23954142e6bfd` |
| `as2w_ab_curriculum.py` | `f969e454a1b4096eee2a52ddcb89fa1d6c63c977bdf6cfca10e3d1bdbfd51347` |
| `as2w_gyro_wheel_action.py` | `8532a884c808b462dea70c8f48368bb768864b33cd0dc6177338e299570e0c9c` |

## 4. 待办

1. A 组跑满 1,000 轮后：每 200 轮快筛 + 完整 11 例双仿真评测
2. 开训 B 组（先确认 `AS2W_AB_CURRICULUM ON` 打印）
3. 连续手柄序列回归用例
4. 两端接触参数读回
5. 盯 `arc_right` 支撑（新闭环 0.913）

## 5. 边界（如实记录）

1. 驻留 8–12 s 是**全局** resampling 范围，非仅纯转
2. `brake` 类只采驱动段，松杆到零由既有 lerp/stop 逻辑产生，**未单独采「停稳后回 fixed stand」段**
3. 单元测试证明采样数学与类别定义，**不证明已接入活环境**；接入由启动打印行断言
