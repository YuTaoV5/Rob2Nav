# Go2 侧空翻仿真手册

已实际执行两轮PPO模型的CPU导出和自由机身MuJoCo回合：未完成翻转，0/1。完整模型仍在训练。本页提供相同链路的复现命令，不把短程接口验证写成技能成功。

```bash
cd /root/gpufree-data/Rob2Nav
PY=/root/gpufree-data/envs/go2-mapless/bin/python
RUN=outputs/sideflip_native_full_20261021
test -f "$RUN/completion.json"
MODEL=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -f "$MODEL"
$PY -m rob2nav.export_actor --checkpoint "$MODEL" --output "$RUN/exported_final"
$PY -m rob2nav.mujoco_skill --kind sideflip --policy "$RUN/exported_final/policy.pt" --motion /root/gpufree-data/capability-work/sideflip/reference.npz --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --seconds 6 --seed 1000 --video --output outputs/sideflip_native_eval_new
```

使用[训练手册](TRAINING.md)自己的新实验时，将RUN和motion路径换为该实验值。输出目录必须未存在。CPU导出仅接受已审查的ELU、identity normalization、12动作模型；带归一化状态时明确拒绝，不猜测权重格式。导出保留原生关节顺序、默认姿态、参考哈希和策略哈希。

MuJoCo使用同一URDF来源的go2_isaac.xml，200Hz物理、50Hz策略、PD40/1与URDF力矩限幅。参考提供joint_pos/joint_vel及anchor方向，实际机器人由关节力矩推动；初始化后不写根位姿、不固定机身、不施加稳定根的外力。参考结束后保持最后参考，继续检验落地稳定。

任务通过判据：解卷绕roll旋转5.5–7.1rad；机身X轴角速度积分绝对值>4.5rad且占总角运动>75%，排除欧拉角奇异导致的假旋转；真实腾空至少80ms、机身最高>0.45m、途中upright<-0.8；无非足端/小腿机身触地；末2秒upright>0.90、机身>0.22m、水平速度<0.25m/s、角速度<1rad/s。单次通过后至少更换种子重复，并扩大初始姿态/摩擦扰动，不能用同一视频宣称泛化。

summary.json分别报告任务success和硬件模型约束；速度与关节越限每个物理步检查。硬件约束要求最大速度比≤1、越限≤0.001rad。MuJoCo未复刻PhysX的硬速度限幅与全部电机包络，需把差异作为迁移诊断，不能以任务通过掩盖约束失败。trace.json与可选episode_0.mp4用于复核。当前短程模型速度约束也失败，不能进入实机动作阶段。
