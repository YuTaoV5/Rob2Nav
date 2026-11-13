# AS2-W 后双轮直立训练与部署手册

2026-11-13 · 研究复现版

## 1. 技能与当前证据

本技能使用独立策略。课程依次覆盖后双轮零速直立 10 秒、前后 ±0.10 至 ±0.20 m/s、左右转向与低速弧线，以及从四轮 fixed stand 接管、连续手柄变速与反打、松杆回稳、断流处置和受控回落至四轮 fixed stand。当前仓库快照覆盖第一阶段驻停。遥控移动和进出站姿的训练及验收记为 `not_run`。

| 第 3,000 轮，同一候选 | 实测结果 |
| --- | --- |
| Genesis 开发评测 | 4,096/4,096 跑满 10 秒，4,075/4,096 同时满足速度和轮载诊断阈值 |
| MuJoCo 标称诊断 | 跑满 10 秒，峰值前向速度 0.161 m/s，0.420 秒首次低于 10 N，驻停未通过 |
| Isaac Lab/PhysX 标称诊断 | 1.11 秒首次低于 10 N，4.815 秒触发机身姿态保护，峰值前向速度 0.629 m/s |

开发驻停阈值为从 0.26 秒起每个物理步的前向速度绝对值不高于 0.10 m/s、每侧后轮法向载荷至少 10 N，并持续到 10 秒。正式双仿真验收和真机状态均为 `not_run`。原训练初态在 Genesis 和 MuJoCo 存在机身与前大腿自碰。对称姿态、高增益和轮载奖励是独立实验线。正式验收应统一碰撞过滤和初态，冻结行为阈值、样本、随机种子及指令序列，并记录每项通过数和首个失效。

## 2. Genesis 训练

使用 Linux NVIDIA GPU。已使用 Genesis 1.4.6、PyTorch 2.11.0+cu130、TensorDict 0.14.2、rsl-rl-lib 5.5.1。评测分别使用带 MuJoCo 和 ONNX Runtime 的 Python，以及现有 Isaac Sim 5.1 / Isaac Lab 无头环境。下例从仓库根目录运行，环境变量指向对应解释器。

```bash
ROOT="$PWD"
BASE="$ROOT/training/as2w-backup/as2w-current"
SOURCE="$BASE/runs/rear-two-wheel/genesis-hold-development"
URDF="$BASE/assets/r237-train-root/official-model/robots/as2w_description/as2w.urdf"
export GENESIS_PY=/path/to/genesis-venv/bin/python
export MUJOCO_PY=/path/to/mujoco-venv/bin/python
export ISAAC_PY=/path/to/isaac-headless-python

# 冒烟与第一阶段长训各用一个尚未存在的输出目录
"$GENESIS_PY" "$SOURCE/train_hold_stage1.py" "$URDF" \
  "$SOURCE/pose-with-height.json" "$SOURCE/static-load.json" /tmp/as2w-hold-smoke 32 2 1
"$GENESIS_PY" "$SOURCE/train_hold_stage1.py" "$URDF" \
  "$SOURCE/pose-with-height.json" "$SOURCE/static-load.json" /tmp/as2w-hold-train 4096 6000 1000
```

最后三个参数依次为并行环境数、PPO 更新轮数和评测间隔。每阶段计划 6,000 轮，每 1,000 轮保存并评测。当前入口从随机初始化训练驻停策略。其他阶段需各自实现指令、奖励、退出条件和留出场景，升档以前一阶段评测为依据。

物理步长 5 ms，策略周期 20 ms。开发配置包含关节初态扰动、质量与惯量同比缩放 0.95–1.05、逐轮摩擦倍率 0.8–1.2、逐轴力矩倍率 0.9–1.1、5–10 ms 动作延迟和 5–20 N 水平推扰。范围为仿真代理。回合重置清空速度滤波、上一动作及两级力矩队列。角度软限位按四轮站姿代理参考向 URDF 硬限位取 90% 行程。力矩上限按开发配置各轴额定值的 90% 设置，扰动后再次限幅。实机可用力矩仍需标定。

## 3. ONNX 导出与仿真接入

```bash
RUN=/tmp/as2w-hold-train
MODEL="$RUN/hold-03000"
"$GENESIS_PY" "$SOURCE/export_hold_actor_diagnostic.py" "$MODEL.pt" "$MODEL.onnx"
"$MUJOCO_PY" "$SOURCE/verify_hold_ort.py" \
  "$MODEL.pt" "$MODEL.onnx" "$MODEL.vectors.npz" "$MODEL.ort.json"
(
  cd "$SOURCE"
  "$MUJOCO_PY" run_hold_mujoco.py "$URDF" "$RUN/pose-with-height.json" \
    "$MODEL.onnx" "$MODEL-mujoco-trace.json" > "$MODEL-mujoco.jsonl"
)
"$ISAAC_PY" "$SOURCE/run_hold_isaac_ground.py" "$URDF" "$RUN/pose-with-height.json" \
  "$RUN/static-load.json" "$MODEL.onnx" "$MODEL-isaac-ground.json"
```

两条闭环命令复现标称诊断，物理和接触参数以对应脚本为准。第 3,000 轮已记录 ONNX Runtime CPU 对 64 组合成输入的最大绝对误差 3.81×10⁻⁶。本次随文归档源码、配置和该数值校验报告，示例中的 PT 与 ONNX 由训练和导出命令生成。完整的观测与控制合同、最终目标及下一帧历史回放仍需随最终候选交付。

## 4. 部署步骤与映射

策略 ONNX 输入 `policy` 为 float32 `[N,68]`，输出 `raw_action` 为 float32 `[N,14]`。轴按 FL、FR、RL、RR 排列，各腿依次为 hip、thigh、calf、foot。按关节名映射到仿真器索引。DDS 槽位、正方向、零位和硬件降额留空，待 SDK 与实机只读标定核对。

| 输入索引 | 训练语义 |
| --- | --- |
| 0–11 | 腿关节角减参考角，rad |
| 12–27 | 16 轴速度乘 0.1 |
| 28–31 | 俯仰误差、横滚、滤波前向速度、相对起点前向位移 |
| 32–37 | 世界系机身角速度乘 0.25、线速度 |
| 38–51 | 上一次裁剪后动作 |
| 52–67 | 上一待发力矩除以逐轴力矩上限 |

输出先裁剪到 ±1。前 12 维按腿轴顺序乘 0.08 rad 后加参考角，再限幅。末两维按左后、右后轮顺序乘 2 N·m，再与闭环平衡力矩叠加。前轮在该模式下目标力矩为零。控制包装须复现姿态反馈、PD、力矩队列、限幅和历史复位。

实机接入按 v3 顺序执行。先校验文件哈希和离线逐帧回放，再完成 MuJoCo 与 Isaac Lab 同权重验收。部署脚本先提供只读影子模式，核对 LowState 数据龄、观测坐标、16 轴映射、最终命令和下一帧历史。正式闭环由 fixed stand 接管，覆盖断流和急停，退出回到 fixed stand，实机动作逐次取得现场授权。当前仅提供上述仿真诊断入口，真机部署脚本和标定待交付。

通用要求见 [训练守则](../../../training/as2w-backup/as2w-current/AGENTS.md)、[交付清单 v3](../../../training/as2w-backup/as2w-current/AS2-W-training-delivery-checklist-v3.md) 和 [AS2-W 部署入口](../../../deliveries/as2w-deploy/README.md)。本次随文发布 [训练快照与导出诊断源码](../../../training/as2w-backup/as2w-current/runs/rear-two-wheel/genesis-hold-development/)。
