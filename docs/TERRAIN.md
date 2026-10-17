# 多地形导航与越障运控

新增水坑、台阶、垂直高台、真实深坑，以及串联这些要素的 `mixed` 场景。旧的空场与障碍物场景保持原入口；多地形使用独立的 `go2_nav.terrain_run`。

## 运行

```bash
# 在仓库根目录，使用已安装依赖的 Python 环境
export GO2_PYTHON=/root/gpufree-data/envs/go2-mapless/bin/python
export MUJOCO_GL=egl

bash run.sh -m go2_nav.terrain_run --scene water --episodes 3 --video --output outputs/water
bash run.sh -m go2_nav.terrain_run --scene stairs --height .08 --no-jump --episodes 3 --video --output outputs/stairs
bash run.sh -m go2_nav.terrain_run --scene platform --height .24 --no-jump --episodes 3 --video --output outputs/platform
bash run.sh -m go2_nav.terrain_run --scene gap --config configs/jump.json --gap-width .30 --episodes 3 --video --output outputs/gap
bash run.sh -m go2_nav.terrain_run --scene mixed --config configs/mixed.json --height .16 --max-seconds 60 --episodes 3 --video --output outputs/mixed
```

新机器先按根目录 README 安装，并将 `GO2_PYTHON` 指向自己的环境。`--height` 在 `stairs` 表示每级台阶高度，4 级累积升高；在 `platform` 表示单次垂直升高；在 `mixed` 表示前两级 8 cm 台阶之后的额外垂直升高。深坑宽度可用 `--gap-width` 调整，默认深度为 0.8 m。

大于已验证范围的尺寸可用于实验，不代表现有控制器能够通过。实际结果以随版本发布的多地形评估报告为准。

**已验证与实验边界**：四个单项场景分别通过 3/3；长距离串联 `mixed` 仍未稳定，当前归档测试 0/1，在跨坑落地阶段失败。Mixed 是研究入口，不是已通过的演示路线。

## 场景与感知

| 要素 | 物理与观测 | 控制目标 |
|---|---|---|
| 水坑 | 蓝色、低摩擦风险区域；用理想 RGB 颜色回波识别 | 足部不进入区域，绕行到目标 |
| 台阶 | 连续 4 级实体台阶，足部允许接触 | 逐级爬升 |
| 高台 | 无隐藏斜坡的垂直台面 | 跨越台沿并在台面上行走 |
| 深坑 | 移除地面、分离左右支撑面，坑底在下方 | 识别坑沿，起跳、腾空跨越并落地 |
| Mixed | 14 m 目标路线：绕水、深坑、两级台阶、高台串联 | 研究完整技能切换；当前尚未通过 |

地形感知保留地面回波，并利用 IMU 姿态对齐重力方向；足部接触与正运动学估计当前支撑高度。相机默认下俯 25°，分辨率 64×48；这些参数仍是假设安装配置。

水坑识别使用的是射线首次命中表面的理想颜色，不是直接读取场景中的 `water` 标签。它只适用于本仿真的颜色约定，不等于真实水面识别模型；水坑也不是流体动力学仿真。D435i 实机 RGB 需要另做感知与域适应。

导航只保存少量局部绕行目标和短期坑沿位置，不构建融合占据地图或 ESDF。坑沿记忆用于应对相机近距离盲区；起跳判断在 50 Hz 控制周期内执行。临时目标由当前颜色回波和里程计生成，场景几何元数据仅用于训练/评估。

## 底层运控

- `--locomotion mlp`：原 RoboGauge MLP 基线。
- `--locomotion moe`：RoboGauge MoE，作为多地形默认策略；仍使用上游观测、动作映射和关节 PD。
- 跳跃技能：准备蹲伏、蹬伸、收腿、落地四阶段关节目标。关节位置与电机力矩均受模型限制，不在控制期间改写根部位置/速度，不施加外部抬升力。
- `--no-jump`：禁用自动跳跃，用于爬台或原策略对照。
- `--straight`：不使用避障规划，保持前向速度并通过目标方向纠偏，用于本体运控对照；不能用它证明导航具备避障能力。

行走与腾空使用不同的姿态判定。跳跃中允许暂时较大的俯仰，但躯干撞击、倒置或跌到支撑面以下仍算失败；落地阶段根据足部接触和姿态切回原运动策略，恢复窗口有上限。到达目标后必须保持直立，并至少有两足接触支撑面才计成功。跨坑还必须记录到技能激活期间、坑上方至少 60 ms 的四足离地事件。

## 跳跃参数校准

```bash
# 全程从初始位置开始，使用传感回波触发；以物理结果选择参数
bash run.sh -m go2_nav.skill_search --sensor-trigger --trials 120 --width .30 --output outputs/jump_calibration

# 用校准结果跑独立种子；先查看 best.json 的 success 是否为 true
bash run.sh -m go2_nav.terrain_run --scene gap --config outputs/jump_calibration/best_config.json \
  --seed 200 --episodes 10 --video --output outputs/jump_eval
```

未加 `--sensor-trigger` 的搜索属于离线特权试验：使用已知起跳位置和近处初始重置以研究关节技能。其成功不能替代传感闭环导航验证；搜索结果必须再通过上面的独立评估。

## 残差运控训练

这里训练的是 **12 维关节目标修正**，与旧版输出 3 维速度的上层导航 PPO 不同。冻结 RoboGauge 的运动先验，增加有幅度限制的残差；观测包含本体状态、当前传感器形成的前向高度剖面与有效性掩码、上一残差，共 87 维。

```bash
bash run.sh -m go2_nav.residual_train --scene platform --height .24 --steps 100000 --envs 4 --output outputs/residual
bash run.sh -m go2_nav.terrain_run --scene platform --height .24 --no-jump \
  --residual-checkpoint outputs/residual/locomotion_residual.zip --seed 200 --episodes 10 --output outputs/residual_eval

# 从窄坑逐步增加到 30 cm；跳跃训练入口，不能据此认定已训练成功
bash run.sh -m go2_nav.residual_train --scene gap --steps 100000 --envs 4 --output outputs/residual_gap
```

平台/台阶训练逐步增加地形高度，坑训练逐步增加宽度。传感器约 10 Hz 更新，残差控制为 50 Hz，物理为 500 Hz。残差不允许改变仿真根部状态或施加额外外力。

32,768 步平台残差训练只是初始结果。与 MoE 相同成功率不表示优于 MoE；应对相同物理、相同传感噪声和独立种子分别比较通过率、接触、姿态和能耗。

已归档模型可这样复验：

```bash
bash run.sh -m go2_nav.terrain_run --scene platform --height .24 --straight --no-jump \
  --residual-checkpoint artifacts/terrain/residual_model/locomotion_residual.zip \
  --seed 200 --episodes 3 --output outputs/published_residual_eval
```
