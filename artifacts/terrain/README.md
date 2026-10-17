# 多地形实验归档

见根目录 [TERRAIN_VALIDATION.md](../../TERRAIN_VALIDATION.md) 和 [运行说明](../../docs/TERRAIN.md)。

- `water`：绕行蓝色低摩擦风险区，seed 0–2，3/3 成功、无入水。
- `stairs`：4 级、每级 8 cm，seed 200–202，3/3。
- `platform`：24 cm 垂直高台，seed 200–202，3/3。
- `gap`：30 cm 宽、80 cm 深坑，seed 200–202，3/3；每回合有腾空证据和终点支撑要求。
- `mixed`：14 m 串联研究路线，seed 0，0/1；落地时躯干撞击，未通过。使用 16 cm 额外高台、`jump_trigger_distance=0.4`。
- `mlp_platform24` / `moe_platform24` / `residual_platform24`：固定前向指令与方向纠偏的本体对照；不能将其解释为避障规划效果。
- `residual_model/locomotion_residual.zip`：初始 32,768 步的 12 维关节残差 PPO，不是旧版 3 维导航 PPO。
- `jump_calibration`：传感闭环参数校准记录；不使用特权起跳坐标重置。
- `tests.log` / `residual_train.log`：实际执行日志。
- `source_manifest.json`：远端归档时 Python 源文件的哈希；只读文档或报告脚本的后续整理不影响物理试验含义。

视频 `episode_N.mp4` 与 `summary.json` 对应；`trajectory_0.json` 提供 seed 首回合的逐步状态。`preview.png` 从实际仿真视频抽帧。

种子仅覆盖当前传感噪声，不代表已做实机、载荷或摩擦随机化验证。水坑是颜色约定的风险区，不是流体仿真。
