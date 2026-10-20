# Go2 导航训练手册

导航目录包含规则导航与可训练PPO速度头两种入口，底层均为RoboGauge MoE。当前网站已加载本次训练的PPO速度头；规则导航器保留为可复现对照。PPO训练与仿真使用相同273维观测、3维归一化速度动作及skills/navigation/config.json。

本次已实际执行512步训练，完成2次rollout、保存navigation.zip和TensorBoard事件；这是接口验证，不是收敛证明。随机障碍训练实际完成100352步，并在另设种子的空旷/绕障8回合中全部到达及稳定停止；网页加载同一SHA256策略的两次鼠标目标操作与断线停止也通过。详细判据见[仿真手册](SIMULATION.md)。底层MoE137k为上游预训练基座，本步骤训练导航头，不重训137k基座。

服务器准备：项目及vendor/RoboGauge已部署，独立环境为Python3.11、MuJoCo3.2.3、torch2.7CPU、stable-baselines3 2.6.0。

```bash
cd /root/gpufree-data/Rob2Nav
PY=/root/gpufree-data/envs/go2-mapless/bin/python
$PY -c 'import mujoco,torch,stable_baselines3; print(mujoco.__version__,torch.__version__,stable_baselines3.__version__)'
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY -m go2_nav.train --config skills/navigation/config.json --sensors lidar --scene empty --envs 1 --steps 512 --seed 42 --output outputs/nav_interface_new
ls -l outputs/nav_interface_new/navigation.zip outputs/nav_interface_new/manifest.json outputs/nav_interface_new/navigation_config.json
```

使用新输出目录；不要覆盖已有实验。manifest记录传感器、本体控制模式、配置哈希和维度，navigation_config.json是同一配置快照。PPO会按完整rollout取整，实际步数以日志为准。

```bash
$PY -m go2_nav.train --config skills/navigation/config.json --sensors lidar --scene random --envs 4 --steps 100000 --seed 42 --output outputs/nav_training_new
sha256sum outputs/nav_training_new/navigation.zip
```

断点继续可增加--resume outputs/nav_training_new/navigation.zip并选择新的--output。确认旧模型的273维观测、3维动作、MoE模式和传感器配置一致，不能混用原默认MLP底层实验。当前训练目标为场景默认(8,0)m；目标方向/距离进入观测，但任意网页目标的泛化必须单独验证，训练奖励上升不构成验证。

本次服务器任务日志位于capability-work/navigation-ppo-smoke.log与navigation-ppo-full.log。模型位于outputs/navigation_ppo_smoke_20261021及outputs/navigation_ppo_full_20261021。仅完整训练结束、文件存在并通过独立物理测试后，才能切换网页默认策略。

本次交付模型位于`skills/navigation/policy/navigation.zip`，SHA256为`078c784513ea7076b1d5b0306a2b609ce67a5670d8831107501497d780786f31`。配置快照和训练清单在同目录，原始训练日志在`evidence/training.log`。训练前种子42，独立评估种子1300–1303与1400–1403；这覆盖两个固定目标，不代表任意环境泛化。
