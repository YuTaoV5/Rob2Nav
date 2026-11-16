# B 组首启被守卫拦下：课程类被训练入口覆盖，未浪费训练

## 1. 发生了什么

按计划启动 B 组（`ARM=B`，配方已 `export AS2W_AB_CURRICULUM=1`）。启动后按**我自己设的硬要求**检查关键行：

```
AS2W_AB arm=B              OK
AS2W_GYRO_TRAIN_LOOP ON    OK
R11_RESUMED_FROM ... 4597  OK
AS2W_AB_CURRICULUM ON      ❌ 缺失
```

**→ 判为未生效，立即 `kill` 该作业（未进入有效训练），不记为 B 组。**

## 2. 根因

训练入口 `train_armE_support2p0.py` 在 `robust_config(...)` **之后**又覆盖了命令类：

```python
cfg = robust_config(SimpleNamespace(..., profile="pose-delay", ...))
cmd = cfg.commands.base_velocity
cmd.class_type = RollingCommands          # ← 覆盖了 robust_config 里设的 CurriculumFlatCommands
cmd.resampling_time_range = (10., 10.)    # ← 也覆盖了 8–12 s 的驻留设置
```

**→ 我把课程接在 `robust_config` 的 DELAY 分支上（与 gyro 同一层），但训练入口在更外层覆盖了它。**
**→ 这是本项目第 8 次「配置在错误的层被覆盖」类缺陷**；这次被启动打印行守卫当场拦住，未造成无效训练。

顺带说明：`resampling_time_range=(10,10)` 恰好落在要求的 8–12 s 内，所以驻留本身不是问题，**真正缺的只是 35/25/15/15/10 的类别配比**。

## 3. 正确做法（下一步，未执行）

课程必须实现**训练入口实际使用的那个类** —— 即 `train_armE_support2p0.py` 里的 `RollingCommands`：

1. 在 `RollingCommands` 内按 `AS2W_AB_*` 做类别抽样（35/25/15/15/10），或子类化它并在训练入口里按开关替换
2. **同样输出一行 `AS2W_AB_CURRICULUM ON ...`**，且必须在**覆盖之后**的类里输出
3. 启动后仍以该打印行为准，否则不记为 B 组

## 4. 与 A 组的关系

A 组已完成并快筛（6/11，转向普遍退化）。B 组**尚未产生任何训练数据**，本轮只有「被守卫拦下」这一项结果。

## 5. 边界与诚实声明

1. 被 kill 的作业**未进入有效训练步数**，不构成 B 组结果，**不得引用**
2. 配方与课程模块本身的 pin 校验均通过（`as2w_ab_curriculum.py: OK`），失败点在**接线层**
3. 未按「先改采样」要求之外叠加任何奖励/增益/摩擦改动
