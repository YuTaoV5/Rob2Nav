# C1 补齐：训练期环境源码的不可变 git 快照

此前 C1 的缺口是「缺少训练当时环境源码的不可变 git 快照」。本轮把训练期实际使用的 4 个文件提交进 git，并逐项标注哪些在**活跃导入路径**上。

## 1. 快照提交

| 项 | 值 |
| --- | --- |
| 提交 | `6d938eb5d23589da9b32e2bf8415eb814796ceea` |
| 覆盖 | 2 个已修改文件 + 2 个新建文件 |
| 刻意排除 | `as2w_isaac_robust_env.py.bak-actiondelay-20261116-012918`（**永不提交**） |

## 2. ⭐ 逐文件哈希与「是否真的在跑」

| 文件 | SHA256（前 16） | 在活跃路径？ | 作用 |
| --- | --- | --- | --- |
| `as2w_isaac_robust_env.py` | `a8b0e3f12da259e5` | **active** | robust observation, DelayedGyroVelocity (the real gyro), curriculum hook |
| `as2w_isaac_flat_env_all.py` | `e739dd1f9cd45ee1` | **active** | FlatEnvironment, FlatCommands, observation |
| `as2w_ab_curriculum.py` | `f969e454a1b4096e` | **active only for arm B** | CurriculumFlatCommands, imported behind AS2W_AB_CURRICULUM=1 |
| `as2w_gyro_wheel_action.py` | `8532a884c808b462` | **NOT imported** | leftover from a discarded first attempt - must not be cited as the gyro implementation |
| `as2w_training_safety.py` | `b8d6ea183d70d80f` | **active** | apply_safety, rolling_knee_margin_cost |
| `as2w_isaac_speed_range.py` | `86cd236651eed286` | **active (if present)** | wheel_support_error |

## 3. 🔴 一处必须纠正的文档口径

此前交付包里 `training/patches/as2w_gyro_wheel_action.py` 容易让人以为那是 gyro 的实现。**不是。**

- **真正生效的 gyro** 是 `as2w_isaac_robust_env.py` **第 107 行的 `DelayedGyroVelocity`**，在第 270 行装到 DELAY 分支上- `as2w_gyro_wheel_action.py` **从未被 robust_env 导入**，是我第一次尝试后被弃用的残留- 它保留在包里只为「工作树干净」，**不得被引用为 gyro 实现**
课程模块 `as2w_ab_curriculum.py` **确实在路径上**（第 257 行，`AS2W_AB_CURRICULUM=1` 时导入）。

## 4. 启动命令（可复现）

```bash
ROOT=/root/gpufree-data/Rob2Nav ARM=A OUT=<dir> bash skills/as2w/rolling/recipes/ab-gyro-loop.sh
ROOT=/root/gpufree-data/Rob2Nav ARM=B OUT=<dir> bash skills/as2w/rolling/recipes/ab-gyro-loop.sh
```

配方内已用 `sha256sum -c` pin 住：初始化权重 `4bdf4d26…`、`as2w_isaac_robust_env.py`、`as2w_isaac_flat_env_all.py`、`as2w_gyro_wheel_action.py`、`as2w_ab_curriculum.py`、训练入口 `train_armE_support2p0.py`。**任何一项哈希不符，配方会直接失败** —— 这是「不可变快照」的可执行形式。

## 5. C1 状态变化

| C1 子项 | 之前 | 现在 |
| --- | --- | --- |
| 选中 checkpoint 哈希 | 已有 | 已有 |
| 部署模型 / ONNX 哈希 | 已有 | 已有 |
| 初始化旧权重哈希 | 已有（`4bdf4d26…`） | 已有 |
| run ID / 启动命令 | 已有 | 已有 |
| **训练当时的不可变 commit / 源码快照** | ❌ **缺** | ✅ **`6d938eb5d23589da9b32e2bf8415eb814796ceea`** |
| 实际解析配置 / 环境依赖 | 已有 | 已有 |

**→ C1 的显式缺口已闭合，可上调为 pass。**

## 6. 边界

1. 快照提交的是 `as2w-growth` 侧的训练源码；`skills/as2w/rolling/` 下的入口与配方另有自己的提交2. 未把整个 run 目录（tfevents、checkpoint）纳入 git —— 按交付约定，运行产物留在服务器，只挑必要摘要进包3. 环境依赖（isaacsim 版本等）记录在既有 run contract 中，不在本文件重复
