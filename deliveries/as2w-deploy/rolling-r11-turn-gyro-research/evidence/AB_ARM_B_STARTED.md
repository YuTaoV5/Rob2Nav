# B 组已开训：课程实现在 RollingCommands（训练入口实际使用的类）

## 1. 修正（承接上一轮被守卫拦下的启动）

课程从上轮无效的位置（`robust_config` 的 DELAY 分支，被训练入口覆盖）**移到训练入口实际使用的类**`train_armE_support2p0.py` 里的 `RollingCommands`，并且**打印行从该类输出**（不可能由未运行的类误报）。

原 `RollingCommands` 的混合是：hold 20 / straight 15 / **pure turn 40** / arc 25，**无独立刹停类别**。新路径（`AS2W_AB_CURRICULUM=1`）改为操作员指定的 35/25/15/15/10，并新增刹停类别：

刹停的实现：该类在采样时对刹停回合设定 `target=±0.25`，并记录**回合内释放步**（3–6 s）；`_update_command` 到达该步时把目标置零，由既有 lerp 产生减速→停稳过程。

## 2. 启动验证（四条关键行全部出现）

```
AS2W_AB arm=B init=.../r11-final.pt gyro_kp=60 cap=4 save=100
AS2W_AB_CURRICULUM ON  class=RollingCommands  fractions=[0.35, 0.25, 0.15, 0.15, 0.1]  brake_release=3-6 s
AS2W_GYRO_TRAIN_LOOP ON  yaw_kp=60.0 ff_cap=4.00 ff_gain=0.0 kd=6.0
R11_RESUMED_FROM .../r11-final.pt iter 4597
Learning iteration 4612/5597
```

| 项 | 状态 |
| --- | --- |
| 热启动 `4bdf4d26` | ✅ |
| **课程生效** | ✅ `fractions=[0.35, 0.25, 0.15, 0.15, 0.1]` |
| gyro 在闭环 | ✅ |
| 预算 / 保存 | 1,000 轮 / 每 100 轮 |
| 环境数 | 4096 |
| 输出目录 | `results/train/ab-B-longturn-20261116-102312` |

## 3. 与 A 组的对照设计

两组同初始化、同种子、同奖励、同姿态参考、同增益、同 90% 包络、同 gyro 闭环；**唯一差异是命令采样与驻留**。

## 4. 边界

1. 刹停类别的释放时刻在 3–6 s 内随机，**未与「回 fixed stand」段绑定**（该类只控制命令，站姿由策略决定）
2. 驻留沿用既有 `resampling_time_range=(10,10)`，落在要求的 8–12 s 内
3. 本轮只启动并发起验证，**尚无训练结果**
