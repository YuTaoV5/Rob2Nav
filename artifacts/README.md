# 已执行实验的归档

这些文件来自 2026-10-17 的远端仿真实验。代码与实验结果的说明见仓库根目录 README 和 VALIDATION.md。

- `outputs/slalom_final/episode_0.mp4`：反应式导航通过交错障碍的 seed 0 视频。
- `outputs/*/summary.json`：各回合成功、碰撞、摔倒、超时等原始指标。
- `outputs/ppo_initial/navigation.zip`：32,768 步初始训练的上层 PPO，不是可靠导航模型。
- `outputs/sensor_preview.png` / `.npz`：渲染和传感器观测预览。
- `tests_final.log`、`train_initial.log`：实际执行的测试与训练日志。
- `environment.lock.txt`：原服务器环境记录，不是可跨平台安装的锁文件。
- `source_manifest.json`：原服务器验证时源码的哈希记录。仓库整理后的主 README、运行脚本与报告路径有所调整，因此它是原实验溯源记录，不是当前仓库全文件校验表。固定上游版本使用 `scripts/check_vendor.py` 校验。

PPO 评估与反应式基线分开记录；不要把反应式基线的成功率归因于 PPO。
