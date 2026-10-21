# 2026-10-21 实验资源快照

本目录包含597份从实际服务器复制并逐项核对SHA256的原始资源，共265,722,077字节；另有本README与model_index.json选取索引。并非所有模型都通过了能力或实机验收。

- `weights/backflip/selected`：本地5000轮最终检查点、TorchScript/ONNX、训练参数；存在落地/电机约束失败。
- `weights/sideflip/selected`：固定5200检查点与导出器契约，对应MuJoCo动作8/8；额定速度仍失败。
- `weights/sideflip/native_5000`、`dc_final_unvalidated`：原生基线和适配最终候选，不覆盖选定模型。
- `weights/upright_walk/representative_failed_baseline`：可重现的失败基线，不是“行走最佳”。DC最终候选、迈步重训检查点另外保留，未验证者明确标注。
- `weights/{stairs,platform,multiterrain}/manual_reproduction`：公开手册新复跑的训练检查点、配置与逐轮数据；阶梯32/32，高台18/32，多地形flat31/32、stairs32/32、platform32/32。不可用历史其他模型分数替换。
- `evaluations`：仿真摘要、轨迹、视频、浏览器验证，以及相关失败对照。
- `logs`：完整训练日志与退出码，包含快照时尚在训练的记录。
- `source_patches`：SCAN Jazzy兼容补丁。
- `provenance`：参考动作FK对齐核对。

`manifest.json`记录597个原始文件的源路径、字节数及SHA256。`model_index.json`链接本目录与既有保留模型、导航权重，说明选择依据；“selected”只表示本次选取，不代表全局最优或允许实机运行。

```bash
python scripts/verify_release_resources.py
```

原始运动CSV、外部源码缓存和机器凭据没有打包；参考数据按各技能手册从固定提交获取和处理。统一SDK热切换说明见 `docs/UNIFIED_GO2_DEPLOYMENT.md`，发布时最新状态见 `handover.md`。
