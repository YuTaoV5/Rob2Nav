# Go2 站立行走训练手册

本目录实现新的Go2后腿双足行走任务。训练重置含50%四足姿态、50%接近平衡的双足姿态；最终验收从四足站立起步，要求真实起立和行走。已完成32环境2次PPO更新及48维actor导出；短程模型MuJoCo未起立，0/1。4096环境5000更新完整训练正在运行。

GitHub核查：legged_rl_lab提交386ee24...注册Go2 Footstand但缺少对应文件；其Go1奖励是设计参考。当前适配使用本仓库task.py和固定Go2资产，不将缺失的Go2上游配置标为已复现。底层为原生IsaacLab PPO，非厂家SDK内置特技回放。

先按[侧空翻训练手册](../sideflip/TRAINING.md)的前两条clone/checkout段准备GO2_backflip提交954aa61ceb62c515689677f44a21e32109f6b62d和unitree_rl_lab提交4960b84732b0c2ec593dccbfe963fda1bcd7b1e3。本任务无需动作CSV或NPZ。

```bash
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh -m skills.upright_walk.train --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab --output outputs/upright_interface_new --num-envs 32 --iterations 2 --seed 44 --device cuda:0 --headless
ls outputs/upright_interface_new/model_*.pt outputs/upright_interface_new/completion.json
```

检查模型、事件文件及completion.json，不仅看退出0。完整训练：

```bash
/isaac-sim/python.sh -m skills.upright_walk.train --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab --output outputs/upright_full_new --num-envs 4096 --iterations 5000 --seed 44 --device cuda:0 --headless
```

本次同参数任务已在outputs/upright_walk_full_20261021运行，勿重复启动。日志capability-work/upright_walk/full.log。--resume检查点可继续，配合新输出目录；更新数是继续训练量。

合约：48维actor，包含机身线/角速度、投影重力、3维速度命令、关节状态与前一动作；12关节位置偏移，scale0.5，PD40/1，50Hz策略、200Hz物理。动作和传感缩放从导出policy.json逐项复用。双足站立时机身X轴指向上，水平行走前向取机身-Z在水平面的投影；奖励用水平世界速度与世界竖直yaw速率，避免把机身竖直轴误当平面前进方向。目标投影重力[-1,0,0]。

奖励包含姿态、前足抬高、后足支撑、速度/yaw跟踪、机身高度及力矩/动作变化/关节限位惩罚。混合重置仅用于训练探索，不构成起立成功证据。必须按[仿真手册](SIMULATION.md)从四足开始评价。

## 显式DC执行器迁移对照（实验）

旧5000轮结果保留，不覆盖。新增 `--transfer-motors` 将PhysX硬限速替换为显式线性力矩—速度包络，物理步长2ms、策略50Hz；两引擎使用相同包络。参数取URDF额定边界，仅为仿真模型，不代表已辨识的真实Go2电机。该对照不自动证明跨仿真成功。

```bash
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh -m skills.upright_walk.train --headless --device cuda:0 \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab \
  --resume outputs/upright_walk_full_20261021/model_4999.pt --transfer-motors \
  --num-envs 2048 --iterations 1000 --output outputs/upright_walk_dc_review
```

完整检查点含优化器和迭代数；恢复后文件编号继续沿用原编号，不应假定最终文件名为model_999.pt。按目录实际编号选择最高检查点，使用CPU导出器保存新目录；policy.json中的motor_model应为dc_linear_v1。评估时不应把DC模型权重套到旧理想PD执行器上。侧空翻适配同时将非足端接触惩罚设为−10，必须重新评估接触和落稳，不能借用旧模型成绩。


## 标准站姿首尾专项续训（2026-10-21）

新增[统一训练目标与操作说明](../FIXSTAND_ROUNDTRIP.md)。本次三项32环境短程接口均完成，但新首尾站姿验收未通过；1024环境、每项500次更新的专项续训正在独立目录执行。新候选不自动覆盖旧部署权重，具体状态和四种子独立验收以该说明中的status.json与evaluation.json为准。
