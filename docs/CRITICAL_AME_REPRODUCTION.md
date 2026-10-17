# criticaldamped Go2 AME 过桩复现

目标是实际训练并验证路线通过能力。上游仓库没有成功过桩的 Go2 成品权重，不能通过下载它的 G1 权重完成 Go2 复现。

上游：[criticaldamped/AME_Locomotion](https://github.com/criticaldamped/AME_Locomotion)，固定版本 `15d02c891273d3f4c9b55c9afe3073bcd2d9152f`。使用其原始 Go2 网络、2124 维 actor 观测、奖励、动作、课程及两阶段地形；本目录提供运行、故障诊断和第一回合评估入口，不安装或覆盖服务器全局 RSL-RL。

## 场景与观测

Stage1 为基础粗糙地形，Stage2 为双列桩、交错桩、窄桥和沟隙等。该项目 Go2 Stage2 的桩边长课程为 45→25 cm，坑深 1.2 m；这与本仓库先前依据 SII 配置加入的 20 cm 窄桩、额外 12 cm 压力测试分别报告。

原项目 actor 使用 45 维本体状态和 33×21×3 理想 XYZ 地形扫描；critic 额外观察真实机体速度。它不是 D435i/Mid360 可见点学生。首先验证原任务，再考虑传感器蒸馏和 MuJoCo 迁移。

## 运行

服务器环境为 Isaac Sim 5.1 / Isaac Lab 2.3.2 / L40。下面路径是当前运行实例；其他机器应替换。

```bash
git clone https://github.com/criticaldamped/AME_Locomotion.git /root/gpufree-data/AME-criticaldamped
git -C /root/gpufree-data/AME-criticaldamped checkout 15d02c891273d3f4c9b55c9afe3073bcd2d9152f
cd /root/gpufree-data/Rob2Nav

/root/isaacsim/python.sh -m critical_ame.train --headless --repair-rays \
  --upstream /root/gpufree-data/AME-criticaldamped --stage 1 \
  --num-envs 512 --iterations 1000 --seed 42 --output /path/to/new_stage1_run

# 同阶段恢复：保留优化器、学习率、RNG、课程及下一轮编号。
/root/isaacsim/python.sh -m critical_ame.train --headless --repair-rays \
  --upstream /root/gpufree-data/AME-criticaldamped --stage 1 \
  --num-envs 512 --iterations 1000 --seed 42 --resume \
  --checkpoint /path/to/stage1/model_999.pt --output /path/to/new_resume_run

# 分项评估：只统计每个环境的首回合，避免反复重置重复计数。
python -m critical_ame.suite --repair-rays --stage 1 \
  --upstream /root/gpufree-data/AME-criticaldamped \
  --checkpoint /path/to/stage1/model_999.pt --levels 0 2 5 \
  --num-envs 32 --output outputs/critical_stage1_evaluation
```

阶段切换使用相同训练入口的 `--stage 2 --checkpoint ...`，省略 `--resume`，从网络权重开始新优化器和课程；须在基础地形分项验证后执行。上述命令不表示对应训练已经完成。

## 射线错误与最小运行时修正

原代码第一次长训练在 iteration 705 附近因单条高度射线未命中退出，最后有效检查点是 `model_700.pt`。从相同检查点恢复后再次捕获到方块地形上的未命中；机器人当时直立且仍位于通道内。

在同一地形、相同位姿重建该射线：起点 x=-1.97500002 m，恰在方块接缝。增加 5 m 起始高度仍不相交；x 平移 ±0.00005 m 都能与实际三角网格相交。

`--repair-rays` 仅对无效返回依次进行 ±x、±y 的 50 微米真实网格重查询。原本有效的返回保持不变；不补零、不合成高度、不改变地形、机器人、奖励或动作。仍未命中时继续由原始数值检查报错。此修正是本次实现相对于原始运行的明确差异，原始上游跟踪文件保持不变。

`ray_probe.py --verify-repair` 在原故障位姿的全部射线上验证：原有效值逐值不变，原唯一无效值恢复为真实交点。触发重查询时，其次数与样例保存在 `ray_requeries.json`。

## 验证口径

评估保留原来的场地、固定前向命令及 3.5 m 路线结束条件，并要求结束时直立、至少两足支撑、无失败终止。失败与超时分别记录。原 Play 配置可能使多个环境完全相同，N 个环境不自动等于 N 个独立随机试验；报告记录这一限制，后续需检查多个种子和初始条件。

目前已跑通两轮原始冒烟训练和六项上游网络/算法测试。ONNX 导出测试未运行（Isaac 环境缺少 onnxruntime）；依赖上游未提交原始文件的 G1 算法对照测试也未运行。TorchScript 导出已与原推理输出核对。

第 700 轮策略在 Stage1 低难度踏石评估中为 0/32：主要停留在出生平台，不具备进入复杂阶段的能力。训练恢复与进一步评估正在进行；未宣称复现过桩成功。

第一阶段有效训练已完成 3000 轮（保留 PPO 更新对应 36,864,000 次交互；失败后丢弃的更新另计），最终检查点为 `stage1_1000_3000/model_2999.pt`。评估增加 ±3 cm 初始位置和 ±0.04 rad 朝向扰动，并与上游零扰动配置分别记录。是否进入 Stage2 以基础地形分项路线通过率判断，不按奖励或时间直接切换。

另有[外部教师步态初始化对照](CRITICAL_AME_GAIT_INITIALIZATION.md)，其权重初始化、训练预算和结果与上述从零 PPO 恢复链分开。`critical_ame.evaluate` 保存 `policy_type` 和 `checkpoint_protocol`；后续 PPO checkpoint 继承该来源标签。模仿初始化权重只能 warm-start，包装器拒绝把它作为 PPO 状态直接 resume。

新评估同时保存 `initial_map.json`、`trace.json` 中的四足位置和接触力、各首回合的 `termination_terms`。可用 `python -m critical_ame.plot_trial /path/to/trial` 绘制实测足端轨迹。视频记录环境 0 至自动重置前，终止状态以重置前捕获的 trace 为准。

后续运行在 `manifest.json` 与 `source/` 保存启动时的包装器源码、参数、检查点哈希和上游版本。上游原始运行及早期诊断运行保留日志和参数，未声称它们具有当时尚未加入的包装器源码快照。
