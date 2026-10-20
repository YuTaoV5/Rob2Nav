# Go2 能力复现交付进度

更新于2026-10-21。本页记录已执行证据和剩余工作，不能替代逐项验收。

| 用户要求 | 当前证据 | 尚需完成 |
|---|---|---|
| Go2后空翻训练 | Robot-Nav/GO2_backflip固定954aa61；32环境2次PPO更新及检查点；4096环境5000更新重训运行中 | 最终模型、奖励记录、独立仿真与约束验收 |
| Go2后空翻仿真 | 上游60维ONNX在MuJoCo完整约2π翻转并落稳 | 速度39.10rad/s和越限0.1077rad失败；不可标为总体验收通过；新模型待验 |
| Go2侧空翻 | 既有LeggedSkillDeploy权重在本地推理落地失败，训练源码链不完整 | 继续补齐训练和仿真；不能用作者视频替代 |
| Go2站立行走 | legged_rl_lab固定386ee24bc62c8033641d65b940dc2cf1dec56545有Go2Footstand任务注册 | 对应footstand_env_cfg.py缺失；Go1任务不能冒充Go2；仍需补齐并执行 |
| 三份手册及研究报告链接 | 后空翻三份手册已建立；报告索引支持skills独立目录；网页跳转与刷新验收通过 | 扩展到所有最终保留能力；删除难以补齐的工作台能力，历史证据保留 |
| 网页实时目标点导航 | 历史MuJoCo批次导航与几何规划探针仍保留；SEA CBF桥接待验 | 真实持续物理仿真、页面选目标与闭环导航；不能用几何动画替代 |
| 实机SDK方案 | 后空翻及握手方案已有，无硬件连接或发送 | 为每个保留能力给出固定合约与逐步部署；按用户要求不做实机验证 |
| 网站去冗余、上传GitHub | 后空翻手册交互已发布到服务器，握手入口保留 | 完整工作台能力审计、导航、最终构建/验收和GitHub上传尚未完成 |

后空翻手册：[训练](../skills/backflip/TRAINING.md)、[仿真](../skills/backflip/SIMULATION.md)、[SDK部署方案](../skills/backflip/DEPLOYMENT.md)。

原始证据在服务器capability-work/backflip-smoke-v3.log、backflip-full.log、backflip-pretrained-mujoco.log；模型在Rob2Nav-upstreams/GO2_backflip/logs/rsl_rl/go2_backflip。网站手册浏览器验收在outputs/manuals_validation/browser.json，测试重现三份链接与刷新后的深链接。训练进程应通过ps与新日志/检查点核验，不依据本页推断仍在运行。

来源：[原生后空翻](https://github.com/Robot-Nav/GO2_backflip/tree/954aa61ceb62c515689677f44a21e32109f6b62d)、[站立行走候选源码](https://github.com/zihanwang0422/legged_rl_lab/tree/386ee24bc62c8033641d65b940dc2cf1dec56545)。
