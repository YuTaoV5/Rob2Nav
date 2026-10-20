# Go2 站立行走仿真手册

两轮PPO短程模型已实际导出并执行8秒MuJoCo：保持四足、未起立，0/1；模型硬件约束通过不等于任务通过。完整模型仍训练中。

```bash
cd /root/gpufree-data/Rob2Nav
PY=/root/gpufree-data/envs/go2-mapless/bin/python
RUN=outputs/upright_walk_full_20261021
test -f "$RUN/completion.json"
MODEL=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -f "$MODEL"
$PY -m rob2nav.export_actor --checkpoint "$MODEL" --output "$RUN/exported_final"
$PY -m rob2nav.mujoco_skill --kind upright_walk --policy "$RUN/exported_final/policy.pt" --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --seconds 8 --command .2 0 0 --seed 1000 --video --output outputs/upright_walk_native_eval_new
```

自己的实验替换RUN。输出目录必须全新。导出读取训练配置中的真实观测项、缩放、clip、默认姿态及关节顺序；MuJoCo按这些字段构建48维观测，不套用其他45维Go2策略。

每回合从四足默认站姿和约2mm关节不确定性初始化，先站立0.5秒，再由策略自行起立。运行中仅关节力矩驱动，没有根状态纠正、固定机身或外部稳定力。不是将机器人初始化为双足后宣称完成起立。

通过条件：目标重力误差<0.45且机身>0.42m的时间至少2秒；末2秒始终维持姿态，期间实际水平位移>0.20m；末窗至少80%采样仅后足支撑、前足离地；无非足端/小腿机身触地。保留trace.json、summary.json和视频。需继续增加目标速度、不同种子、停止及方向变化测试后再声明完整能力；一次成功不代表任意地面泛化。

summary中任务判定与硬件约束分开。速度和关节限位逐物理步检查，MuJoCo与PhysX电机/接触差异单列诊断。[SDK部署仅方案](DEPLOYMENT.md)，未验证实机。

## 原训练引擎诊断与迁移差异（2026-10-21）

先固定最终模型，再从实际四足起始状态执行原训练引擎。下面只调用仿真，不发送SDK命令。输出目录不能已存在。

```bash
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh -m skills.upright_walk.evaluate_isaac --headless --device cuda:0 \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab \
  --policy outputs/upright_walk_full_20261021/exported_final/policy.pt \
  --output outputs/upright_walk_native_review
cat outputs/upright_walk_native_review/summary.json
```

评估器禁止策略回合中写入机身状态，动作参考只推进/保持端点，不允许通过重置机身完成特技。原引擎结果不能标成MuJoCo结果。MuJoCo评估现已将惯性轴角速度转换为机身轴，新增坐标系回归测试；旧坐标系版本的失败仅保留为历史诊断。

在本页MuJoCo命令中追加 `--physics-hz 1000` 可做1ms积分精度对照（策略仍50Hz），并使用不同输出目录。默认保留200Hz，以便对照训练步长。不得通过瞬时修改机身或焊接根节点改善分数。侧空翻可额外使用 `--initialization reference_state` 对照参考第0帧速度；此模式属于重置状态一致性诊断，不替代静止起步验收。

最终原生诊断从四足起立，直立7.64秒、末两秒位移0.402米、后足支撑比例1.0。当前只通过姿态/位移诊断，尚需逐足接触、滑移与重复测试，不能据此宣称鲁棒行走。修正坐标系后的MuJoCo最终模型0/1，保留失败摘要。
