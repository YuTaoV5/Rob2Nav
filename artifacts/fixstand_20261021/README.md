# 标准站姿首尾训练记录（进行中，未通过交付）

目标与命令见[专项训练说明](../../skills/FIXSTAND_ROUNDTRIP.md)。本目录是记录时刻的快照，不是后台任务的实时状态。

后续证据：`evaluation_completed_v1.json`为三项首轮最终评估，每项6100–6103全部失败；`backflip_entry_v2_smoke/`保存延迟相位新合约、ONNX数值一致性与seed9200失败摘要；`upright_endpoint_v2_smoke/`保存seed9000失败摘要。最新源码12项单元测试通过，不能据此称能力通过。较早快照与来源源码保持原样。

- `smoke/`：侧翻69维、直立52维、后翻60维各32环境2更新后的真实MuJoCo验收。均未通过新首尾判据。
- `training_status_snapshot.json`、`evaluation_snapshot.json`：1024环境、每项500更新的串行任务快照。第一项侧翻model_5699已完成，6100–6103四个种子为0/4；直立和后翻当时尚未完成。实际后续状态在服务器同名训练目录中。
- `sideflip_v1_failed/`：第一轮失败侧翻冻结actor、完整新参考NPZ、参考清单、四种子摘要。该actor只用于复查失败，不是最佳/可部署权重。完整PPO检查点在服务器原目录，可供继续训练；没有覆盖旧DC5200。
- `sideflip_v2_smoke/`：有界参考残差试验的短程结果。旋转约一周，但存在身体触地及端点误差，未通过。该版本的1024环境500更新续训位于服务器 `outputs/fixstand_sideflip_residual_v2_20261021`，结束后自动评估7100–7103。
- `training_source_at_launch/`：首轮启动时的源文件快照，用于区分后续加入的动作参数化、完成标记和验收改动。它不是当前默认执行代码，不能覆盖工作区源码后冒充最新版本。

新的侧翻参考由固定上游CSV加站立段及平滑连接、再按Go2资产做FK生成；来源与原始CSV哈希见[侧翻训练手册](../../skills/sideflip/TRAINING.md)。FK参考不是物理动作通过证据。模型资产许可与来源沿用 [Go2资产说明](../../deployment/models/README.md)。

复查失败模型：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 python -m rob2nav.mujoco_skill \
  --kind sideflip --policy artifacts/fixstand_20261021/sideflip_v1_failed/policy.pt \
  --motion artifacts/fixstand_20261021/sideflip_v1_failed/sideflip.npz \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --seconds 8 --physics-hz 1000 --fixstand-roundtrip --seed 6100 --output outputs/fixstand_failed_review_new
```

新增目标和传输/物理帧约定共9项单元测试通过；后翻ONNX导出已执行数值一致性检查。上述工程验证不替代动作或首尾站姿验收，全部新候选仍未获实机资格。
