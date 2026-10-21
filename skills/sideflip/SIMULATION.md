# Go2 侧空翻仿真手册

CPU环境固定为已执行的Python3.11依赖。首次建立环境时执行（已有环境不重复创建）：

```bash
cd /root/gpufree-data/Rob2Nav
python3.11 -m venv /root/gpufree-data/envs/go2-skills-cpu
/root/gpufree-data/envs/go2-skills-cpu/bin/python -m pip install -r skills/requirements-stunts-cpu.txt
```

本次服务器已具备相同版本的/root/gpufree-data/envs/go2-mapless；下文沿用它。新环境使用者把后续PY/解释器替换为go2-skills-cpu/bin/python。训练仍必须使用Isaac解释器，不能用这个CPU环境启动IsaacLab训练。

当前复现版本：原生5000轮后显式电机适配的model_5200，69维actor。MuJoCo独立种子3100–3107的动作通过8/8；最高关节速度比为1.08505，额定速度门限0/8，不能标为完整部署验收通过。场景固定，只随机初始关节±0.002rad，未做广泛摩擦/姿态泛化和实机验证。

先完整执行[训练手册](TRAINING.md)，获得outputs/sideflip_dc_reproduce_202/model_5200.pt及其训练所用reference_new.npz。以下输出目录必须不存在：

```bash
cd /root/gpufree-data/Rob2Nav
PY=/root/gpufree-data/envs/go2-mapless/bin/python
$PY -m rob2nav.export_actor \
  --checkpoint outputs/sideflip_dc_reproduce_202/model_5200.pt \
  --output outputs/sideflip_reproduced_5200_export
$PY -m skills.stunt_benchmark --kind sideflip \
  --policy outputs/sideflip_reproduced_5200_export/policy.pt \
  --motion /root/gpufree-data/capability-work/sideflip/reference_new.npz \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --physics-hz 1000 --seconds 6 --seeds 3100 3101 3102 3103 3104 3105 3106 3107 \
  --video-first --output outputs/sideflip_reproduced_5200_holdout
cat outputs/sideflip_reproduced_5200_holdout/benchmark.json
```

若仅复核本次服务器已有模型，policy改为outputs/sideflip_dc_adapt_20261021/probe_5200/policy.pt，motion改为/root/gpufree-data/capability-work/sideflip/reference.npz，输出仍使用新目录。不要混用其他参考文件；导出器及评估器校验参考和策略SHA。CPU导出支持已审查的ELU、identity normalization、12动作契约，带归一化状态时明确拒绝。

每个seed子目录包含真实summary.json、trace.json；第一个seed另有episode_0.mp4，视频只代表该回合。聚合benchmark.json包含每个回合的指标、同一策略/评估器哈希及complete标志。脚本检查每轮使用同一固定策略和评估器，返回码不能替代逐回合验收。

动作通过判据保持不变：roll解卷绕5.5–7.1rad，机身X轴角速度积分绝对值>4.5rad且占总角运动>75%；腾空≥80ms、最高>0.45m、途中upright<−0.8；无非足端/小腿机身接地；末2秒upright>0.90、机身>0.22m、水平速度<0.25m/s、角速度<1rad/s。额定约束独立要求每个物理步速度比≤1、关节越限≤0.001rad。

物理积分1000Hz、策略50Hz、PD40/1、scale0.5，执行policy.json所记dc_linear_v1力矩—速度包络。训练时物理500Hz；1ms评估用于提高积分精度，不能声称与PhysX逐步一致。电机参数来自URDF边界，是仿真模型。执行器未使用PhysX硬速度钳位，不能用钳位隐藏速度超限。

根状态仅在初始化设置；其后由真实关节力矩、重力与接触推动，未焊接机身、未施加稳定根的外力。参考结束后保持末帧，继续检验落地。机身角速度已从MuJoCo惯性轴转换到链接轴，并有独立刚体回归测试。

本次证据：[聚合结果](evidence/dc_5200_mujoco/summary.json)、[逐种子记录](evidence/dc_5200_mujoco/benchmark.json)、[代表性视频](evidence/dc_5200_mujoco/episode_0.mp4)。原始JSON在同目录raw/，策略SHA为7616e53e2c95434f824c2728621f8052e2d41a4e033fd00a25f316de211e07e1。

原引擎诊断命令（需改为自己的导出和参考文件）：

```bash
/isaac-sim/python.sh -m skills.sideflip.evaluate_isaac --headless --device cuda:0 \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab \
  --policy outputs/sideflip_reproduced_5200_export/policy.pt \
  --motion /root/gpufree-data/capability-work/sideflip/reference_new.npz \
  --output outputs/sideflip_reproduced_5200_native
```

历史边界：两轮短程模型未翻转；原生5000轮模型在PhysX完成约2π并落稳，但一次FR_thigh接触导致严格0/1。旧理想PD模型的MuJoCo迁移失败，修正坐标系和单独提高积分精度均未解决。它们与当前DC5200是不同模型，历史失败原件仍在evidence目录。

[训练手册](TRAINING.md) / [SDK部署方案](DEPLOYMENT.md)。
