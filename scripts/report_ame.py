"""Render the AME pilot report exclusively from published raw evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
base = root/"artifacts/ame"
manifest = json.loads((base/"manifest.json").read_text())
for item in manifest["checkpoints"].values():
    assert hashlib.sha256((base/item["path"]).read_bytes()).hexdigest() == item["sha256"]
kinds = ("stairs", "platform", "stones")
rows = []
results = {}
for label in ("control", "attention"):
    results[label] = {}
    for engine in ("native", "mujoco"):
        values = []
        for kind in kinds:
            data = json.loads((base/(label+"_eval")/(engine+"_"+kind)/"summary.json").read_text())
            values.append((data["successes"], data.get("total", len(data["episodes"]))))
        results[label][engine] = values
        rows.append("| "+label+" | "+engine+" | "+" | ".join(f"{s}/{n}" for s, n in values)+" |")
ablation = json.loads((base/"attention_eval/native_stones_without_branch/summary.json").read_text())
recheck = json.loads((base/"baseline_recheck/summary.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, engine in zip(axes, ("native", "mujoco")):
    x = np.arange(3)
    for j, label in enumerate(("control", "attention")):
        values = results[label][engine]
        bars = ax.bar(x+(j-.5)*.34, [100*s/n for s, n in values], width=.34,
                      label=label, color=("#999999", "#167bb6")[j])
        for bar, (s,n) in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{s}/{n}", ha="center", fontsize=9)
    ax.set_xticks(x, ["Stairs", "Platform", "Stones"])
    ax.set_title(engine); ax.set_ylim(0, 115); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
axes[0].set_ylabel("Supported full-course success (%)")
axes[1].legend(loc="upper right")
fig.tight_layout(); fig.savefig(base/"comparison.png", dpi=160); plt.close(fig)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fig, axes = plt.subplots(1,2, figsize=(10,3.5))
for label in ("control", "attention"):
    events = EventAccumulator(str(next((base/label).glob("events.out.tfevents.*")))).Reload()
    for ax, tag in zip(axes, ("Train/mean_reward", "Curriculum/terrain_levels")):
        samples = events.Scalars(tag)
        ax.plot([s.step for s in samples], [s.value for s in samples], label=label,
                color={"control": "#999999", "attention": "#167bb6"}[label])
for ax, title in zip(axes, ("Training return (not validation)", "Training curriculum level")):
    ax.set_title(title); ax.set_xlabel("PPO iteration"); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(base/"training.png", dpi=150); plt.close(fig)

attention_path = base/"attention_eval/native_stones/attention.json"
traces = json.loads(attention_path.read_text())
# Pick a time with substantial visible coverage, without selecting by success.
frame = min(traces, key=lambda r: abs(r["t"]-2.))
height = np.asarray(frame["height"]).reshape(11,17)
valid = np.asarray(frame["valid"]).reshape(11,17) > .5
weights = np.asarray(frame["weights"])
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
axes[0].imshow(np.where(valid, -height, np.nan), origin="lower", extent=[-.6,1.1,-.55,.55])
axes[0].set_title(f"Measured -scan, t={frame['t']:.1f}s")
for i, ax in enumerate(axes[1:]):
    ax.imshow(weights[i,:187].reshape(11,17), origin="lower", extent=[-.6,1.1,-.55,.55], cmap="magma")
    ax.set_title(f"Head {i+1}, null={weights[i,-1]:.2f}")
for ax in axes: ax.set_xlabel("forward x (m)")
axes[0].set_ylabel("left y (m)")
fig.tight_layout(); fig.savefig(base/"attention.png", dpi=150); plt.close(fig)

video_links = []
for label in ("control", "attention"):
    for kind in kinds:
        path = f"artifacts/ame/{label}_eval/mujoco_{kind}/episode_0.mp4"
        video_links.append(f"- [{label} / {kind}，seed 801 完整首回合]({path})")
text = """# Go2 地形注意力 PPO 增量实验

本次已实际训练原 MLP 对照与 AME 思路注意力策略，均使用模拟深度/雷达可见返回。它是对已有学生的架构增量试验，不是 AME 原论文复现，也不是 G1 权重迁移。

**本次增量训练未超过上一版稳定学生，不能替换现有模型。** 注意力仍值得用于稀疏落脚区的编码，但这次短预算、增量动作分支接法没有验证出可用收益。两组继训均发生部分遗忘，须同时阅读原学生历史成绩与同预算对照。

## 实验条件

两组从上一版发布的学生 checkpoint 初始化，同为 seed 44、512 环境、200 轮 × 24 步，即每组额外 **2,457,600** 次交互。全部 actor 与 critic 参数参与 PPO，使用同一传感器模型、课程、奖励和动力学设置；增加的参数与算力不相等。相同种子不保证不同网络消耗随机数的顺序相同，不能视为严格逐轨迹配对。

注意力组保留原 MLP，并加入 CNN、xy 坐标及本体状态查询的四头地形注意力动作分支。零初始化末层使训练前行为与原学生完全一致。输入仍为 419 维，未知区域保留掩码；没有用理想地形替代学生传感器。

第一次注意力尝试因新编码器的网格行列解释错误，在早期被停止，修正后从相同发布权重重新开始完整 200 轮；诊断日志与旧源码保存在 `artifacts/ame/diagnostics`，不计作已完成训练预算。两轮冒烟训练也不纳入正式成绩。

## 完整通过率

固定难度 0.5：台阶单级 10.25 cm，高台 21.5 cm；梅花桩支撑面约 46×41 cm、间隙 8.75 cm。Isaac 为 32 个初始状态（seed 801），MuJoCo 为 seeds 801–805 共 5 回合。通过要求完成包括下障碍的整条通道，并在终点保持直立及至少两足支撑。

| 模型 | 仿真器 | 台阶 | 高台 | 梅花桩 |
|---|---|---|---|---|
"""+"\n".join(rows)+f"""

![两组同预算成绩](artifacts/ame/comparison.png)

![训练过程，不是独立验证](artifacts/ame/training.png)

作为历史参考，上一版学生在相同任务上的 Isaac 成绩为 32/32、32/32、28/32，MuJoCo 为 4/5、5/5、2/5。历史权重没有本次额外训练量，所以架构比较应优先看上表两组。

当前评估入口重跑上一版学生，Isaac 台阶为 **{recheck['successes']}/{recheck['total']}**，证据见 `artifacts/ame/baseline_recheck`。

最终注意力网络只移除新增分支、保留最终基础 MLP 参数时，Isaac 梅花桩为 **{ablation['successes']}/{ablation['total']}**。这是同一最终权重的推理消融，不是重新训练的独立基线。

## 注意力检查

![梅花桩局部观测与四头注意力](artifacts/ame/attention.png)

图取首回合距 2 秒最近的采样；左侧为有效格上的负高度扫描值，属于相对高度表示。其余为地形 token 权重，各头独立归一化，标题列出 null token 权重。注意力图不能直接视为足端落点或安全约束。

## 完整视频与原始证据

所有视频均为 seed 801 的完整首回合，失败保留；不从测试中挑选成功回合替换。

"""+"\n".join(video_links)+"""

`artifacts/ame/manifest.json` 记录权重 SHA256、训练预算与全部评估计数；每组保留训练源文件快照、配置、完整日志、TensorBoard、最终 checkpoint、导出 actor、评估命令/退出码、轨迹和接触指标。`scripts/verify_ame.py` 校验源码字节、权重及视频帧数。

## 能力边界

单训练种子和每项 5 回合 MuJoCo 只支持初步比较，不能据此宣称稳定超过某篇论文。固定地形模板的多个初始状态不等于未见地形泛化；两种物理引擎表现也须分别报告。

本次采用重置优化器后的 PPO 继训、较少并行环境和较短预算。没有逐项隔离这些因素，不能将退化全部归因于注意力机制。若继续采用 AME 思路，更合理的下一项试验是从理想地形教师阶段训练编码器，再用传感器输入做动作蒸馏，最后以更小更新幅度微调，并加入未见地形与观测漂移验证。这是后续实验建议，未在本次执行。

这里仍是速度命令驱动的直行运动技能，尚未接入任意目标导航；没有新增宽深坑跳跃技能、真实 D435i/Mid360 驱动或实机测试。感知使用简化射线、理想位姿与一秒可见点记忆，传感器标定、时延、里程计漂移和真实雷达扫描仍需验证。

方法来源、Go2 适配及复现命令见 [AME 参考说明](docs/AME_REFERENCE.md)。
"""
(root/"AME_VALIDATION.md").write_text(text, encoding="utf-8")
print(json.dumps(results, indent=2))
