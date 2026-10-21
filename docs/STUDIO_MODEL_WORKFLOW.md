# 工作台技能证据与任务权重实时仿真

更新：2026-10-21。

## 工作台操作

点击“能力验证进度”的技能，左侧卡片原位切换视频；页面保持工作台，不弹详情窗口。视频下面显示录制来源、验收要求、仿真/部署/训练手册、对应候选权重详情入口、关联任务与方法来源。无本地视频时明确显示作者演示或暂无视频，不伪造验证结果。

“查看对应能力矩阵”是独立按钮：跳转能力与方案、定位并高亮对应技能，不自动弹窗。能力矩阵自己的“播放效果”按钮仍可打开详情，属于用户主动操作。

## 在线仿真：两种来源

在线仿真顶部“仿真权重来源”可选工作台技能或任务队列训练权重。

- 工作台技能使用已固定适配的 MuJoCo/专用导航或握手仿真。
- 任务队列选择训练任务后，默认选中已保存点中的最佳候选，也能选择其他已保存点。来源、选择准则、下载和证据入口一起显示。
- 当前原生任务检查点适配范围：Go2 / moe / moe_terrain_ppo。它使用 Isaac Sim/PhysX、实际检查点及其训练观测/传感器契约，不套用其他技能权重，不把 Isaac 检查点强行当作 MuJoCo 模型。
- 点击“启动所选权重实时仿真”才执行推理；显示持续更新的机身/关节物理状态，支持鼠标旋转/平移/缩放、跟随、暂停回放和停止任务。Isaac 冷启动和渲染可能慢于实时，视图显示的是实际返回状态；未产帧时如实等待或报错。
- 空队列/未保存检查点显示空态；不将手册复现的 CLI 权重伪装成队列训练结果。训练关闭时仍允许这个受限的纯推理入口，不启动训练 worker，也不消费其他排队训练任务。

当前“最佳”按准确保存点的 Train/mean_reward 排名；无分数时退回最终评估权重或最新保存点，并显示实际准则。它不是同协议、多种子独立评估已证明的最优模型。

接口 `POST /api/checkpoint-simulation` 仅接受已知训练任务及该任务已索引检查点，使用严格参数模型和原有同源写入保护。共用 gpu.lock，存在活动/待核对任务时拒绝启动。保存一个独立的仿真任务及命令/日志；取消走任务队列停止接口。单次推理不代表技能通过。

## 模型与证据：必要且完整的文件集合

旧索引仅收录 PT，现已补齐 ONNX、PPO ZIP，以及必要基础运动 MoE。下载使用已索引模型 ID 的专用接口，能下载 ONNX/ZIP，且不开放任意主机路径。

默认展示 14 项核心模型文件：后空翻训练检查点和 PT/ONNX 导出、侧空翻检查点和 PT 导出、直立失败基线检查点和导出、阶梯/高台/多地形手册复现检查点、导航 PPO ZIP、两份历史保留检查点和一份基础运动 MoE。任务最佳候选自动加入。中间权重与诊断导出保留在“包含历史归档和中间权重”中，不删除历史资源，也不把它们全部标为最佳。

模型详情显示格式、大小、原始路径、选择依据、验证限制、发布哈希、手册与配套配置/来源记录。评估只在 checkpoint_sha256 或 policy_sha256 与文件哈希匹配时自动关联；导出模型和训练检查点哈希不同，不能把其中一个的评估无说明地移给另一个。配置文件完整性检查采用标明的 LF 归一化文本内容哈希；模型文件始终使用原始字节 SHA256。

当前关键限制：本地后空翻严格动作/落地测试0/1，不能挪用上游模型演示成绩；侧空翻动作8/8但额定速度门槛0/8；直立模型没有真正迈步通过的最佳权重。SCAN 本身不需要神经网络权重，握手控制器也不需要专用训练模型；其底层运行依赖单列。

## 验证与复现

```bash
cd /root/gpufree-data/Rob2Nav
/root/gpufree-data/envs/rob2nav-studio/bin/python -m pytest studio/tests -q
# 不反序列化模型：核对必要模型/配套文件哈希、完整索引和逐文件下载
PYTHONPATH=$PWD /root/gpufree-data/envs/rob2nav-studio/bin/python scripts/audit_model_assets.py --output outputs/model_asset_audit_new.json
# 使用既有阶梯199检查点，隔离目录执行6秒真实Isaac物理推理；不训练，不写在线任务数据库
PYTHONPATH=$PWD /root/gpufree-data/envs/rob2nav-studio/bin/python scripts/verify_checkpoint_live.py --output outputs/checkpoint_live_validation_new
/opt/conda/bin/python scripts/verify_studio_workflow_web.py --output outputs/studio_workflow_web_new
```

原始证据：[实时检查点推理](evidence/studio_workflow_20261021/checkpoint_live.json)、[模型完整性与下载](evidence/studio_workflow_20261021/models.json)、[浏览器回归](evidence/studio_workflow_20261021/browser.json)。真实推理检查得到151帧、模拟时间6秒、精确输入哈希一致并产生视频；这是一轮集成检查，不是新的阶梯成功率评估。浏览器的队列最佳点默认选择使用界面测试夹具单独验证，与真实推理证据分开。
