# 统一创建实验与任务详情

在 Rob2Nav Studio 打开「创建实验」。点击顶部的横排卡片，可以在任务与权重、奖励与惩罚、核对与提交之间切换。完成预检后保存草稿，从原任务队列明确提交。

任务详情提供回放及各技能支持的分项曲线。拖动时间轴可定位 MuJoCo 回放和对应采样。点「带此项创建候选」可复制原实验配置并定位奖励项。

AS2-W 四轮起身支持额外奖励模板及受限算式。R14、R15、R16 盲走楼梯采用各自的固定奖励。Go2 使用注册的技能及参数，自定义 MoE 可选 Isaac 留出地形，单次前后翻保留独立 MuJoCo 协议和可从回放复算的分项。

## AS2-W 服务器资源

新 checkout 默认使用仓库内的 `skills/as2w/training`。它包含四个配方使用的训练脚本、R12 和 R17 初始化权重、`unitree_ros` 模型以及 `replay/fall-recovery-contract.json`。已有服务器可以设置 `ROB2NAV_AS2W_TRAINING=/root/as2w-training` 继续使用自己的训练资源。训练和 MuJoCo 依赖按服务器部署说明安装。

仓库内的评估入口和 MuJoCo XML 位于 `studio/as2w_eval`。已有的独立评估目录可通过 `ROB2NAV_AS2W_EVAL` 指定。使用 `ROB2NAV_AS2W_MUJOCO_PYTHON` 或 Studio 的 `--mujoco-python` 指定 MuJoCo 解释器。训练产物和资产转换缓存按实验隔离。

在「在线仿真 → 历史回放」搜索 AS2-W 可查看已发布的通过样本。每个视频的记录都保留种子、权重哈希与同轮评估的通过数。短程功能检查与正式训练分开标注。
