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
