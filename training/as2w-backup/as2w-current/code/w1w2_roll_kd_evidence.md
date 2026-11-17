# W1W2 滚动线 kd 带载对照 — 精简证据索引

- 任务：AS2-W 滚动线「同目标四轮 kd 带载对照」（`wheel_action_scale=5.0`、gyro OFF、腿控/姿态固定、
  同一份物理单位四轮目标序列、`±13.5 N·m` 预算；**只有轮端 kd 变化**：0.5 / 1.5 / 2.0）
- 执行：as2w-roll-kd，2026-11-16 23:18–00:05 (+08:00)
- 交付状态：**research / deployable: false**（仿真诊断，不是 §5 正式双仿真验收：无策略、无手柄、无随机化）
- 原始长日志/逐帧 trace/视频：**不在仓库**，只在 `/root/gpufree-data/as2w-runs/w1w2-loaded.4Pz917/`

## 1. 源码（本任务新增，`w1w2_` 前缀）

| 文件 | 作用 |
|---|---|
| `code/w1w2_roll_isaac_loaded.py` | Isaac/PhysX 侧：悬空 one-hot 校准 + 接地序列；下发点与逐物理子步录制；13.5 N·m 上限读回 |
| `code/w1w2_roll_mujoco_loaded.py` | MuJoCo 侧：逐字消费同一份序列；速度环真正消费的目标 + 请求/下发力矩分离 |
| `code/w1w2_roll_compare.py` | 双端逐帧目标一致性 + 逐用例对照表 + 较差侧统计 |
| `code/w1w2_roll_seq.json` | 版本化四轮目标序列（FL/FR/RL/RR，rad/s），两端共用 |

## 2. 启动命令

```bash
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/code"
unset LD_LIBRARY_PATH
cd /root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current
/isaac-sim/python.sh .../w1w2_roll_isaac_loaded.py --mode grounded --kd 1.5 --scale 5.0 --states A,B \
  --seq .../w1w2_roll_seq.json --usd .../kit3-as2w.usd/as2w/as2w.usda --urdf .../as2w.urdf \
  --out <dir> --label kd1p5 --device cuda:0
```

解释器必须是 `/isaac-sim/python.sh`；`/opt/isaaclab-venv/bin/python` 只是 Kit python 的软链接，直接用会 ABI 冲突
（`_ctypes ... undefined symbol: _PyErr_SetLocaleString`），清 `LD_LIBRARY_PATH` 后再用则报 "Isaac Sim is not installed"。

## 3. 生效值（实例化后读回，不是配置）

- 轮关节索引 FL/FR/RL/RR = 12/13/14/15（**按名字推导**后校验）；轮体/接触索引 13/14/15/16
- 腿 PD：hip/thigh/calf stiffness **55.0**、damping **2.0**（逐组读回）；轮 stiffness 0.0、damping **=kd**
- 轮力矩上限 **13.5 N·m** ×4（=0.9×URDF 15，`effort_limit_sim` 读回）；轮速上限 98.47 rad/s
- 动作项 `DelayedGyroVelocity`，scale 5.0，`as2w_action_delay_max_steps=0`，lag 0；gyro `AS2W_GYRO_KP=0`
- 整机 26.000 kg / 255.06 N

## 4. 结论（逐用例见 run 目录 `compare.log`/`compare.json`）

- **双端逐帧目标一致**：每个 kd 臂 4900 帧，跨仿真器目标偏差 **0**，策略步内重写偏差 **0**，帧集合无缺失。
- **支撑**：三组 × 两端 × 14 个 case/state，`support=1.000`、逐轮最长连续卸载 **0 采样**、最少承载轮 **4**。
- **无失败终止**：无摔倒、无非轮触地、无角度/膝面越界。3 个被标记窗口全部是"基线未达标"（静止窗口
  `mean|dq|` 0.053/0.057/0.084 > 0.05），不是失稳。
- **共模 0.25 m/s 档**：Isaac `kd=0.5` 只有 **0.58** 交付比（不达标），`kd=1.5` **0.89–0.91**，`kd=2.0` **0.92–0.94**；
  MuJoCo 三组都 ≈1.00。
- **0.05 m/s 档**：Isaac 三组都 ≈0（静摩擦门槛在 0.58–2.9 rad/s 之间）；MuJoCo ≈0.98。
- **左右镜像差动**：Isaac 三组都基本不交付（`(FR+RR−FL−RL)/2` 实测 ≤0.07，目标 0.385/0.769，`wz≤0.011`）；
  MuJoCo 随 kd 单调变好：0.005(kd0.5) → 0.157(kd1.5) → 0.305(kd2.0)。**差动在 Isaac 是结构性缺口，换 kd 不能修**。
- **刹停**：Isaac 停稳时间 kd1.5 0.22–1.48 s 最快，kd0.5 0.54–1.24 s，kd2.0 0.56–2.36 s。
- **力矩**：请求力矩峰值 1.57 / 2.21 / 3.66 N·m = 预算的 **12% / 16% / 27%**，全程无饱和，无需截断。
- **选增益**：推荐 **kd=1.5**。唯一同时满足"Isaac 共模 0.25 m/s 档 ≥0.88"与"所有 case ≤1.5 s 停稳"，
  支撑零损失、力矩占 16% 预算。kd=2.0 只在 MuJoCo 侧换来差动权限（Isaac 不复制），代价是刹停变慢一倍、
  多用 2 个 void 静止窗口；kd=0.5 因共模跟踪 0.58 被直接排除。

## 5. 未验证 / 不纳入

- 无真机证据；本轮不是正式双仿真验收。
- 差动在 Isaac 全部 `not_delivered`（0.10 与 0.20 rad/s 两档都测了）。
- 上一轮旧权重在 MuJoCo 的 `1/11、2/11、1/11` 仍只是换合同诊断，**未**纳入本任务结论。
- 未触碰其他 agent 的未提交改动与任何 `.bak` 文件。
