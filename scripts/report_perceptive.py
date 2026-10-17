"""Build the evaluation report directly from saved episode records."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
base = root/"artifacts/perceptive"
manifest = json.loads((base/"manifest.json").read_text())
kinds = ["stairs", "platform", "stones"]
names = {"stairs": "楼梯（上楼、平台、下楼）", "platform": "21.5 cm 高台（含走下）", "stones": "交错矩形梅花桩"}
metrics = {}


def read(group, kind, difficulty=.5):
    path = base/group/f"{kind}_{difficulty:.2f}"/"summary.json"
    if not path.exists(): return None
    d = json.loads(path.read_text())
    rows = d["episodes"]
    r = {"successes": d["successes"], "total": len(rows),
        "climbed": sum(bool(e.get("climbed_obstacle", e.get("climbed_platform", False))) for e in rows),
        "falls": sum(bool(e.get("fall", e.get("base_contact", False))) for e in rows),
        "off_lane": sum(bool(e.get("left_lane", e.get("off_lane", False))) for e in rows),
        "timeouts": sum(bool(e.get("timeout", False)) for e in rows),
        "calf_impulse_proxy_mean": float(np.mean([e.get("calf_impulse_proxy", 0.) for e in rows])),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path.relative_to(root)).replace("\\", "/")}
    metrics[f"{group}/{kind}_{difficulty:.2f}"] = r
    return r


def fraction(r):
    return f"{r['successes']}/{r['total']}" if r else "未完成"


native, mujoco = [], []
for group, label in [("teacher_native", "理想地形教师（特权输入）"), ("distilled_native", "可见传感器学生（仅蒸馏）"), ("student_native", "可见深度/雷达学生（PPO 后）")]:
    for kind in kinds:
        r = read(group, kind)
        if r:
            native.append(f"| {label} | {names[kind]} | {fraction(r)} | {r['climbed'] if kind != 'stones' else '—'} | {r['falls']} | {r['off_lane']} | {r['timeouts']} |")
for group, label in [("final_moe", "冻结 MoE"), ("final_ik", "MoE + 抬脚 IK"),
                     ("teacher_mujoco", "理想地形教师（特权输入）"), ("student_mujoco", "可见深度/雷达学生（PPO 后）")]:
    results = [read(group, k) for k in kinds]
    mujoco.append(f"| {label} | {' | '.join(fraction(r) for r in results)} |")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
for ax, groups, labels, title in [
    (axes[0], ["teacher_native", "student_native"], ["Ideal-height teacher", "Visible-ray student"], "Isaac native: full-course success"),
    (axes[1], ["final_moe", "final_ik", "teacher_mujoco", "student_mujoco"], ["MoE", "MoE + IK", "Ideal teacher", "Ray student"], "MuJoCo: same test model, full course")]:
    x = np.arange(3)
    width = .8/len(groups)
    for j, (group, label) in enumerate(zip(groups, labels)):
        scores = [read(group, k) for k in kinds]
        rates = [r["successes"]/r["total"] if r else np.nan for r in scores]
        ax.bar(x-.4+width/2+j*width, rates, width, label=label)
    ax.set_xticks(x, ["Stairs", "21.5 cm platform", "Stepping stones"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("Full-course success fraction"); ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=.2)
fig.savefig(base/"comparison.png", dpi=160)
plt.close(fig)

# Training reward is a diagnostic curve, not the independent success rate.
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    for stage in ["scan", "teacher", "student"]:
        paths = list((base/"training"/stage).glob("events.out*"))
        if not paths: continue
        events = EventAccumulator(str(paths[0])).Reload()
        tags = events.Tags()["scalars"]
        for axis, tag in zip(axes, ["Train/mean_reward", "Curriculum/terrain_levels"]):
            if tag in tags:
                data = events.Scalars(tag)
                axis.plot([e.step for e in data], [e.value for e in data], label=stage)
    axes[0].set_title("Training return (not test success)")
    axes[1].set_title("Mean curriculum terrain level")
    for ax in axes:
        ax.set_xlabel("PPO iteration"); ax.grid(alpha=.2); ax.legend()
    fig.savefig(base/"training.png", dpi=160); plt.close(fig)
except ImportError:
    pass

videos = sorted(base.glob("**/*.mp4"))
video_lines = [f"- [{str(p.parent.relative_to(base)).replace(chr(92), '/')} / {p.name}]({str(p.relative_to(root)).replace(chr(92), '/')})" for p in videos]
ablation_path = base/"student_sensor_ablation/summary.json"
ablation = ""
if ablation_path.exists():
    d = json.loads(ablation_path.read_text())
    normal = read("student_native", "stones")
    ablation = (f"同一 PPO 学生在梅花桩的输入消融：正常观测 {fraction(normal)}，高度与有效掩码全置零后 "
        f"{d['successes']}/{d['total']}。这是输入依赖检查，不是重新训练的盲策略，也不代表雷达独立增益。")
student_scores = [read("student_mujoco", k) for k in kinds]
moe_scores = [read("final_moe", k) for k in kinds]
ik_scores = [read("final_ik", k) for k in kinds]
if all(student_scores):
    student_total = sum(r["successes"] for r in student_scores)
    moe_total = sum(r["successes"] for r in moe_scores if r)
    ik_total = sum(r["successes"] for r in ik_scores if r)
    verdict = f"本轮统一 MuJoCo 测试，感知学生完整通过 {student_total}/15；冻结 MoE 为 {moe_total}/15，MoE + IK 为 {ik_total}/15。小样本仅描述这组场景；不能据此宣称普遍优越。"
else:
    verdict = "感知学生的统一 MuJoCo 测试尚未全部完成，不能给出优劣结论。"

text = f"""# 感知运动 RL 训练与对照报告

验证日期：2026-10-17。训练服务器：单 NVIDIA L40，Isaac Lab 2.3.2 / PhysX GPU；迁移测试：MuJoCo 3.2.3。

## 当前结论

{verdict}

理想地形教师与可见传感器学生分开报告。教师使用完整局部高度及真实机体速度；它的高成功率不是相机/雷达部署成功率。
本实验确实训练新权重，运行时直接输出关节目标，不调用 MoE、NJU BBC 或抬脚 IK。
目前仍是运动技能实验，没有把新权重替换进旧版 `active_run` 任意目标导航，也没有实机部署。

## 训练过程

1. 从零 PPO 尝试约 1,956 万采样步后终止，保留第 300 轮检查点。第 200 轮的中等难度原生探索测试三个场景均 0/32；没有把它称为收敛策略。
2. 使用 NVIDIA 官方 Go2 rough 检查点（外部第 1499 轮）初始化全部 actor/critic 权重，再执行 **400 轮 PPO，19,660,800 新采样步**。这是新越障课程微调，不是把官方文件改名。
3. 用教师在学生访问状态上的动作作标签，执行 **1,500 步 × 1,024 环境 = 1,536,000 次交互** 的混合教师/学生 roll-in 蒸馏。该阶段是模仿学习。
4. 可见深度/雷达学生再执行 **300 轮 PPO，7,372,800 新采样步**；读取保存的完成标记及检查点确认结束。

源字节快照、完整训练日志、TensorBoard 原始事件及检查点 SHA-256 见 [证据目录](artifacts/perceptive/) 与 [manifest](artifacts/perceptive/manifest.json)。
两轮 PPO 的模型参数均发生了更新，逐张量差异见 [参数更新校验](artifacts/perceptive/parameter_updates.json)。
检查点保存接口和显式 PD 参数导出中的调试失败保留为探索记录，未计作策略性能结果。

![训练回报与课程曲线](artifacts/perceptive/training.png)

## 独立原生测试

每场景 32 个初始状态，seed 801；固定难度 0.5：每级台阶 10.25 cm、高台 21.5 cm、梅花桩面 46×41 cm / 纵向空隙 8.75 cm / 坑底 -55 cm。
测试随机初始化与训练种子不同，但地形仍是相同模板族，不是未见几何分布的泛化测试。

| 策略 | 场景 | 完整通过 | 曾上台支撑 | 本体接触终止 | 越界/掉坑终止 | 超时 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(native)}

{ablation}

“曾上台支撑”指到达平台内部时，机体高度、朝向及至少两脚接触曾同时满足条件，不等于持续站稳或完成下台与终点任务。
完整通过要求到达约 7.5 m 终点、躯干朝上并至少两脚支撑；走出横向 ±0.85 m 区域或跌倒均不算通过。

## 同一 MuJoCo 模型对照

所有策略使用相同几何、相同种子 801–805、Go2 URDF 大腿范围、足端摩擦 0.8、关节力矩预算 ±23.5 N·m。
各策略使用自己的 PD 参数；训练策略导出的是显式电机的 25/0.5 增益，不能读取已禁用的 PhysX 关节驱动增益。
指令为固定前进速度 0.7 m/s 与航向反馈；这是受控运动技能测试，不是任意目标导航成功率。

| 策略 | 楼梯完整通过 | 高台完整通过 | 梅花桩完整通过 |
|---|---:|---:|---:|
{chr(10).join(mujoco)}

![原生与 MuJoCo 结果分开绘制](artifacts/perceptive/comparison.png)

两个引擎的资产与接触/电机细节仍有差异。原生成功、MuJoCo 失败时，应记为迁移未通过，不能直接部署实机。
探索阶段原始 RoboGauge XML 的不同关节/摩擦/力矩条件结果位于 exploration，未混入最终对照。
旧版 24 cm 高台单项成绩也不能与本表交叉排名：本轮包括走下平台与到达终点，接近距离、摩擦及力矩预算均不同。

## 录像

以下是完整仿真回合，不是拼接成功片段；`episode_0` 固定展示该组第一个种子，可能是失败回合。看视频时请同时核对相应 summary.json。
`student_success_replay` 是从主测试已成功的 seed 802 选取的楼梯和梅花桩完整成功回放，不另计入成功率；主测试 seed 801 的失败录像同时保留。

{chr(10).join(video_lines)}

## 能力边界与下一步

- 学生输入是可见射线构成的局部高度及掩码，不是原始 RGB/depth CNN；最多 1 秒已见点记忆使用理想仿真里程计。未知格没有用地形真值补全。
- 水平安装 Mid-360 对近处地面覆盖有限，当前近身高度主要依赖下倾深度相机；本轮未证明雷达独立增益。
- 相机和雷达是简化仿真传感器；尚无实机驱动、时间同步、标定误差、电机时延、载荷辨识及实机验证。
- 本轮奖励含非足端接触惩罚，仍没有整条小腿的硬避碰约束，不能保证零撞腿。
- 感知 RL 能联合调整落脚和全身动作，但是否比 MoE + IK 更好，需要同时通过感知退化、不同地形、不同种子和迁移测试；教师成功不足以作出这个结论。

参考 [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour)、[Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour) 和 [Vision Locomotion](https://vision-locomotion.github.io/)。本实验借鉴教师/学生与地形课程思路，未声称复现这些工作的完整方法或论文成绩。
复现入口、观测维度、地形参数见 [使用说明](docs/PERCEPTIVE_RL.md)。
"""
(root/"PERCEPTIVE_VALIDATION.md").write_text(text, encoding="utf-8")
(base/"report_metrics.json").write_text(json.dumps(metrics, indent=2))
print(verdict)
