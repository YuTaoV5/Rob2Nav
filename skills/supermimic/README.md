# SuperMimic 可复现资源

从[中文复现入口](../../docs/SuperMimic/00_复现入口.md)开始。这里是训练与评估资源包，不是 Codex 自动执行技能。

上游固定为 Zhengtianjin/go2-supermimic 的 `054a13c13285a1c94b2e69dc8ac7331f9a8c71cf`，遵守其 MIT 许可；兼容补丁保留原作者代码归属。机器人模型由上游依赖提供，遵守各模型资源自己的许可证。

检查点由本项目在服务器上从零训练，包含完整训练状态。当前仍为研究候选，不是合格盲楼梯或实机策略。原始证据不得通过修改报告字段来“晋级”。

```bash
python3 skills/supermimic/verify_resources.py
```

安装脚本只操作显式指定的独立上游目录，不安装依赖、不启动训练、不覆盖 Rob2Nav 服务。模型使用 PyTorch 完整检查点格式，仅加载可信来源的文件。
