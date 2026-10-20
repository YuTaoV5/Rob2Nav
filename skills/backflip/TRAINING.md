# Go2 后空翻训练手册

状态：已执行32环境、2次PPO更新，生成model_0.pt、model_1.pt、TensorBoard事件。4096环境、5000更新的独立重训于2026-10-21启动，尚未完成；不能将接口验证称为收敛。

固定来源：[Robot-Nav/GO2_backflip](https://github.com/Robot-Nav/GO2_backflip/tree/954aa61ceb62c515689677f44a21e32109f6b62d)，MIT。分支PPO-backflip，提交954aa61ceb62c515689677f44a21e32109f6b62d。

以下命令在本次已部署服务器执行；Isaac Sim5.1、IsaacLab2.3.2、torch2.7 CUDA12.8、rsl-rl3.1.2。不要用MuJoCo CPU虚拟环境启动Isaac训练。

```bash
cd /root/gpufree-data
mkdir -p Rob2Nav-upstreams
cd Rob2Nav-upstreams
git clone --branch PPO-backflip https://github.com/Robot-Nav/GO2_backflip.git
git -C GO2_backflip checkout 954aa61ceb62c515689677f44a21e32109f6b62d
git -C GO2_backflip rev-parse HEAD
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh skills/backflip/train_isaac51.py --source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --headless --device cuda:0 --num_envs 32 --trainer rsl_rl --max_iterations 2 --seed 1 --run_name interface_check
```

已有源码目录时跳过clone，先检查提交与工作区。启动器只在独立进程中兼容URDF导入器的可选set_merge_fixed_ignore_inertia接口，不修改全局IsaacLab；运行时保留原导入器的固定关节惯量行为。此兼容差异需要在仿真对齐中核验，不能宣称与其他Isaac版本完全等价。

检查最新训练目录，必须同时存在模型与事件文件：

```bash
cd /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
RUN=$(find logs/rsl_rl/go2_backflip -maxdepth 1 -type d -name '*interface_check' | sort | tail -1)
find "$RUN" -maxdepth 2 -type f
```

完整重训命令（本次已启动同参数任务，不要重复启动占用GPU）：

```bash
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh skills/backflip/train_isaac51.py --source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --headless --device cuda:0 --num_envs 4096 --trainer rsl_rl --max_iterations 5000 --seed 1 --run_name reproduction
```

训练使用60维actor、165维critic、12维关节位置偏移、动作缩放0.5、50Hz策略、200Hz物理；Kp40/Kd1。先1000更新宽松保护、3000更新逐步收紧，最后1000更新完整约束。不能只训练两轮就跳过安全课程宣称技能成功。

本次任务观察：

```bash
ps -eo pid,etime,args | grep '[r]ob2nav_reproduction'
tail -40 /root/gpufree-data/capability-work/backflip-full.log
cat /root/gpufree-data/capability-work/backflip-full.exit
```

.exit文件在任务结束才创建；存在不等于成功，须核对值、最后更新编号和最终检查点。训练完成后按[仿真手册](SIMULATION.md)导出并验证本次模型；上游预训练模型的结果不得记为本次重训结果。
