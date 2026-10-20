# Go2 能力复现交付进度

更新于2026-10-21。本页记录已执行证据和剩余工作，不能替代逐项验收。

| 用户要求 | 当前证据 | 尚需完成 |
|---|---|---|
| Go2后空翻训练 | Robot-Nav/GO2_backflip固定954aa61；32环境2次PPO更新及检查点；4096环境5000更新重训已完成（exit 0）；最终4999检查点已导出 | 多回合与硬件约束验收 |
| Go2后空翻仿真 | 自身最终60维模型在MuJoCo翻转−6.151rad并落稳，末态倾角0.012rad | 任务PASS，硬件约束FAIL：速度77.96rad/s、越限0.5146rad；总体验收FAIL |
| Go2侧空翻 | 新Go2 mimic适配已完成FK、短程PPO、CPU导出和MuJoCo；4096环境5000更新已完成（exit 0）；最终模型已导出并进行独立MuJoCo验收 | 最终模型0/1，翻倒并超限，尚未复现成功；需定位跨仿真差异 |
| Go2站立行走 | 新48维Go2起立/双足PPO已完成短程训练、导出和MuJoCo；完整4096环境5000更新运行中 | 短程模型未起立；需从四足开始验证完整模型 |
| 三份手册及研究报告链接 | 后空翻、侧空翻、站立行走及实时导航均有三份手册；报告索引支持skills独立目录；网页跳转与刷新验收通过 | 扩展到所有最终保留能力；删除难以补齐的工作台能力，历史证据保留 |
| 网页实时目标点导航 | 已部署持续MuJoCo窗口，PPO100352步训练及8回合独立评估；网页两目标、断线停止、策略SHA一致性通过；三份手册已链接 | 扩大目标/布局覆盖；非SCAN/SEA原生策略，实机仅方案 |
| 实机SDK方案 | 四项技能及握手方案已有，无硬件连接或发送 | 为每个保留能力给出固定合约与逐步部署；按用户要求不做实机验证 |
| 网站去冗余、上传GitHub | 后空翻手册交互已发布到服务器，握手入口保留 | 完整工作台能力审计、导航、最终构建/验收和GitHub上传尚未完成 |

后空翻手册：[训练](../skills/backflip/TRAINING.md)、[仿真](../skills/backflip/SIMULATION.md)、[SDK部署方案](../skills/backflip/DEPLOYMENT.md)。

原始证据在服务器capability-work/backflip-smoke-v3.log、backflip-full.log、backflip-pretrained-mujoco.log；模型在Rob2Nav-upstreams/GO2_backflip/logs/rsl_rl/go2_backflip。网站手册浏览器验收在outputs/manuals_validation/browser.json，测试重现三份链接与刷新后的深链接。训练进程应通过ps与新日志/检查点核验，不依据本页推断仍在运行。

来源：[原生后空翻](https://github.com/Robot-Nav/GO2_backflip/tree/954aa61ceb62c515689677f44a21e32109f6b62d)、[站立行走候选源码](https://github.com/zihanwang0422/legged_rl_lab/tree/386ee24bc62c8033641d65b940dc2cf1dec56545)。

导航手册：[训练](../skills/navigation/TRAINING.md)、[仿真](../skills/navigation/SIMULATION.md)、[SDK方案](../skills/navigation/DEPLOYMENT.md)。默认模型078c7845…786f31；规则导航与PPO各有8回合原始摘要，不混用成绩。

已实现的源码适配来源：官方unitree_rl_lab提交4960b84732b0c2ec593dccbfe963fda1bcd7b1e3包含通用mimic/MDP与G1配置，actor结构可提供Go2动作模仿适配的基础，但没有直接可执行的Go2侧空翻注册。本仓库已显式实现Go2资产/关节映射、动作CSV到NPZ转换、训练与独立动力学验证，不把G1任务称为Go2特技。

新技能手册：[侧空翻训练](../skills/sideflip/TRAINING.md)、[仿真](../skills/sideflip/SIMULATION.md)、[SDK方案](../skills/sideflip/DEPLOYMENT.md)；[站立行走训练](../skills/upright_walk/TRAINING.md)、[仿真](../skills/upright_walk/SIMULATION.md)、[SDK方案](../skills/upright_walk/DEPLOYMENT.md)。FK预处理、训练完成、动作成功和硬件约束分别保留证据，互不替代。
