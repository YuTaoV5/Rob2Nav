# 2026-10-21 资源与权重快照

此目录发布权重、导出模型、训练配置/日志、成功和失败的评估摘要、轨迹与视频。原始服务器快照597文件，共265,722,077字节；另附统一部署验证记录和索引。来源与每个文件SHA见[manifest.json](manifest.json)，模型选择依据见[model_index.json](model_index.json)。

| 技能 | 本次选择 | 边界 |
|---|---|---|
| 后空翻 | [5000轮ONNX](weights/backflip/selected/exported/policy.onnx)、同目录TorchScript和完整model_4999 | 代表复现模型；严格落地/电机约束未通过，不是可直接上机的最佳模型 |
| 侧空翻 | [固定5200 TorchScript](weights/sideflip/selected/exported/policy.pt)、完整model_5200与policy.json | 固定场景动作8/8；额定速度门限未通过；5998仅未评估候选 |
| 站立行走 | [5000轮代表基线](weights/upright_walk/representative_failed_baseline/exported/policy.pt) | 尚无通过真实迈步/MuJoCo验收的最佳权重；原生前移伴滑移 |
| 新增迈步奖励 | weights/upright_walk/stepping_snapshot_unvalidated/model_5700.pt | 容器重启中断，无完成标志，无技能验收，不自动续训 |
| 阶梯/高台/多地形 | weights/<技能>/manual_reproduction/model_199.pt | 独立手册复跑：阶梯32/32、高台18/32、多地形flat31/32+stairs32/32+platform32/32；不是全局最优声明 |

所有恢复训练检查点保留优化器等训练状态，不可直接当作TorchScript推理模型加载。各exported/policy.json记录输入项、顺序、缩放、默认姿态、SHA。后空翻使用原60维源码契约。新网站优先读取此目录的对应导出模型；参考动作NPZ仍按固定来源单独准备，未打包上游无许可声明的原始CSV或第三方缓存。

## 数据目录

- `evaluations/`：所有本次选择的特技诊断、独立侧翻8回合、导航/SCAN/托爪、地形评估、网页验收；包含失败与修复前结果。
- `logs/`：完整5000轮及DC适配日志、地形复跑和未完成迈步训练的日志快照。
- `weights/`：选定权重、原始5000轮、DC5998候选、5700中断快照及配置。
- `validation/`：统一控制核心/SDK离线检查、首轮真实MuJoCo切换失败。`switch_trial_9201_summary.json`中没有任何MLP/MoE提交，backend描述仅为测试计划，不能当作已运行这些模型。
- `source_patches/`：SCAN/Jazzy兼容补丁；原项目固定提交及许可说明见各技能手册和docs/GO2_SDK2_SOURCE_REVIEW.md。

## 校验与复现

```bash
python scripts/verify_release_resources.py
python -m rob2nav.deployment profiles
python -m rob2nav.deployment plan --from walk_moe --to backflip60
```

训练与推理环境要求见skills/<技能>/TRAINING.md、SIMULATION.md；统一部署仍在验证中，见[方案与失败边界](../../docs/UNIFIED_GO2_DEPLOYMENT.md)。本发布不包含凭据、虚拟环境、node_modules、外部源码缓存或全部中间检查点。原始第三方运动数据通过固定URL及SHA在训练手册中获取；本次训练得到的模型和本次产生的仿真轨迹均保留。
