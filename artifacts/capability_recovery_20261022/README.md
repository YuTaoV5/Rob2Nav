# 全技能复验与恢复（2026-10-22）

本轮没有将所有技能标记为合格。网站仅升级已复验的固定地形演示；侧翻、后翻、直立与混合地形仍未通过完整验收。实机资格全部false。

- `outputs/capability_release_20261022/{stairs,platform}`：原始RoboGauge MoE固定几何四回合往返、登顶承重及终点稳定2秒，均4/4；summary固定模型SHA与实际源码SHA，完整轨迹及第一回合原速录像保留。高台为20cm，楼梯为4级×8cm上行+下行。
- `outputs/capability_navigation_recheck_20261022`：原生SCAN/真实MuJoCo空旷和绕障两目标2/2到达，含状态轨迹；仅两场景集成检查。
- `outputs/capability_candidates_20261022`：5份GO2_backflip公开ONNX及LeggedSkillDeploy的探索对比；diagnostic_only，不作独立验收。部分早期高台探索实际8cm，不能引用为20cm结果。
- `outputs/capability_recovery_20261022`：20cm单段、高台/楼梯往返早期复测、混合启跳仍失败、LSD频率对照、PIE原生深度策略运行记录。未取得相同协议的候选不替换默认。
- `outputs/fixstand_sideflip_quiet_bridge_v9_20261022`：最终9196检查点、导出、实际配置及四种子准备段通过/完整失败记录；仅用于v10恢复，不是网页默认合格权重。
- 其余正式/`_smoke`目录保存侧翻v10、后翻v7和直立v5的实际启动配置，烟测有completion；没有正式completion不代表训练已完成。后翻新actor的128样本数值一致性证据在 `outputs/backflip_upstream_initialization_20261022/provenance.json`。

本地及服务器55项控制/训练离线测试、服务器23项Studio相关测试通过。网站首选视频按显式资源排序，旧失败记录仍保留。SHA256SUMS校验原始字节；开源无许可权重留外部缓存，不在此再分发。

详见 [逐技能报告与复现步骤](../../docs/CAPABILITY_RECOVERY_20261022.md)。

`web_validation.json`记录网站真实实时仿真端到端检查：楼梯207帧、高台185帧，JPEG画面和相机操作正常，均登顶、抵达终点并停稳2秒，模型SHA与归档结果一致。
