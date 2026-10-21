# 网页技能编排验收记录

源：部署页 `UnifiedComposer` → `studio.live_skills` → `studio.live_runner` → `rob2nav.deployment.verify_mujoco`。两组试验均使用服务器上的原生MuJoCo自由机身模型、同一个 `SkillSequence` 与 `UnifiedRuntime`，并通过实时画面接口观察。序列和summary均保留原始JSON；此目录刻意不重复上传可再生的高频 `control_trace.npz` 与逐帧视频。原始运行记录仍在服务器 `outputs/live_skills/<session>/evaluation/`，重新运行命令见 `docs/UNIFIED_GO2_DEPLOYMENT.md`。

| 记录 | 会话 | 实际提交 | 结果 |
|---|---|---|---|
| `composer_short_*` | `8eebd60e1d78` | stand → walk_moe → stand | 通过；最大切换力矩步进0.30000000000037 Nm，峰值速度/额定0.3852 |
| `composer_packaged_*` | `1a996edb84c3` | stand → walk_moe → stand（仓库内模型） | 通过；从 `deployment/models/go2_isaac.xml` 及配套网格重新编译，验证全新克隆默认资源路径；峰值速度/额定0.2292 |
| `composer_repeat_*` | `fb513fb21371` | stand → MoE → sideflip → MoE → stand → MoE → sideflip → MoE → stand | 通过；两次侧翻、9次提交、92.821 s实时流、累计滚转-12.7132 rad；峰值速度/额定 **1.0850**，因此实机失败/隔离 |

`passed=true` 只表示该MuJoCo流程完成并满足仿真器自身的切换检查；不代表设备电机、连接时延、跌倒概率或安全资格通过。模型、策略、配置和源代码SHA位于各summary。运行时长、接触及动作过程可用同序列重放。所有随仓库发布的技能依然 `hardware_qualified=false`。
