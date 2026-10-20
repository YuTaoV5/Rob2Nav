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

## 原训练引擎诊断与迁移差异（2026-10-21）

先固定最终模型，再从实际四足起始状态执行原训练引擎。下面只调用仿真，不发送SDK命令。输出目录不能已存在。

```bash
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh -m skills.sideflip.evaluate_isaac --headless --device cuda:0 \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab \
  --policy outputs/sideflip_native_full_20261021/exported_final/policy.pt --motion /root/gpufree-data/capability-work/sideflip/reference.npz \
  --output outputs/sideflip_native_review
cat outputs/sideflip_native_review/summary.json
```

评估器禁止策略回合中写入机身状态，动作参考只推进/保持端点，不允许通过重置机身完成特技。原引擎结果不能标成MuJoCo结果。MuJoCo评估现已将惯性轴角速度转换为机身轴，新增坐标系回归测试；旧坐标系版本的失败仅保留为历史诊断。

在本页MuJoCo命令中追加 `--physics-hz 1000` 可做1ms积分精度对照（策略仍50Hz），并使用不同输出目录。默认保留200Hz，以便对照训练步长。不得通过瞬时修改机身或焊接根节点改善分数。侧空翻可额外使用 `--initialization reference_state` 对照参考第0帧速度；此模式属于重置状态一致性诊断，不替代静止起步验收。

最终原生严格检查0/1：转轴−6.347rad、轴主导度0.872、腾空0.34秒、最高0.491米、落稳，但1.46秒出现一次FR_thigh接触。速度比1.000000049为限速边界的浮点量，原始门限未加容差，原始结果保留；不称为完整通过。修正坐标系后MuJoCo仍未通过，1ms积分对照也未解决。
