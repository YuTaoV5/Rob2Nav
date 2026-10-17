"""Aggregate archived terrain evidence, without mixing navigation and locomotion tests."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / "artifacts/terrain"
    lines = ["# 多地形仿真评估", "", "日期：2026-10-17。MuJoCo 3.2.3 / RoboGauge MoE。", "",
             "| 测试 | 成功 | 碰撞 | 摔倒 | 入水 | 平均成功用时 |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["water", "stairs", "platform", "gap", "mixed", "mlp_platform24", "moe_platform24", "residual_platform24"]:
        p = root / name / "summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        times = [r["seconds"] for r in d["results"] if r["success"]]
        t = f"{statistics.mean(times):.2f} s" if times else "—"
        lines.append(f'| {name} | {d["successes"]}/{d["episodes"]} | {d["collisions"]} | {d["falls"]} | {d["water_entries"]} | {t} |')
    lines += ["", "前四项使用传感闭环导航；后三个 platform24 对照使用相同的前向速度和目标方向纠偏，不使用避障规划。Mixed 若列出，则是串联场景的实际结果。", "",
              "台阶：4×8 cm，总升高 32 cm。高台：24 cm 垂直升高。单项深坑：宽 30 cm、深 80 cm。种子改变传感噪声；未据此声称不同摩擦、载荷或实机均可通过。", "",
              "跨坑成功要求到达目标、保持直立且至少两足支撑，并记录技能激活时坑上方至少 60 ms 四足离地。跳跃时允许较大的临时俯仰，但躯干撞击、倒置或跌落仍算失败。", "",
              "残差模型使用 4 个环境训练了 32,768 个环境步，课程从低台增加到 24 cm。相同成功率不表示残差优于原 MoE。旧版上层导航 PPO 的 0/3 结果不属于这里的下层残差。", "",
              "跳跃是物理关节技能及参数校准，不是已训练收敛的端到端视觉跳跃 RL。所有视频来自 MuJoCo；未运行实机。", "",
              "原始数据和视频见 [artifacts/terrain](artifacts/terrain)。安装与运行见 [docs/TERRAIN.md](docs/TERRAIN.md)。"]
    (ROOT / "TERRAIN_VALIDATION.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
