# AS2-W 滚动 · 实机部署包

> **版本** `deploy-as2w-v1` ｜ **日期** 2026-11-17 ｜ **技能** AS2-W 四轮滚动
> **状态 `research` ｜ `deployable: false`**
> **建议用途：受控研发首测（R3 只读影子回放）**，不作为验收通过的部署授权。

---

## ⚠️ 部署前必读（安全前置）

**本包不构成实机闭环授权。** 以下为**已知未通过项**，部署方必须逐项确认：

| 项 | 状态 | 含义 |
|---|---|---|
| **双仿真正式验收** | ❌ **未通过** | 四项纯原地转（`turn_left/right ±0.05/±0.10`）在 Isaac 与 MuJoCo **两端都不达标** |
| **R3 CPRO 只读** | ❌ **`not_run`** | 未在 ARM64 / 真实 LowState 上做影子回放 |
| **R4 有界实机** | ❌ **`not_run`** | 需现场人员按姿态/急停/监测计划逐级授权 |
| **C4 输出到控制目标** | ⚠️ **partial** | **无硬件 DDS 槽位映射**；`tau/ff` 遥测曾恒读 0（已修，见证据） |
| **MuJoCo 正式矩阵** | ⚠️ 仅 6/11 | 转向类仅 `arc_right` 在两端同时通过 |

**预检拒绝条件**：若 `runtime/preflight.py` 报告的硬件槽位未映射、或 `deploy.yaml` 中任一关节方向/零位未核实，**必须拒绝闭环启动**。这是 `AGENTS.md` 第 6 节的硬性要求。

**可执行的第一阶段只有：`runtime/preflight.py` + 只读影子回放**（不发电机命令）。

---

## 1. 交付物

| 项 | 值 |
|---|---|
| **选中权重** | `weights/armD_turn65_final.pt` |
| **权重 SHA-256** | `de61de58fd68efc356efbd48a0b9683153361fc87b6fb32c7b9c83584cbd6926` |
| **部署模型** | `weights/policy.onnx` |
| **ONNX SHA-256** | `087c91bc4c613271d3b841ac001ab6c169e7f6e62264671c5c2a4e4e23dc9821` |
| **ONNX 大小 / opset** | **175,066 B** / **17**（单文件，无外置 sidecar） |
| **checkpoint↔ONNX 对齐** | **`PARITY_PASS`，4096 样本，`max_abs_err = 1.907e-06`**（`weights/ONNX_PARITY.json`） |

### 模型接口

| 项 | 值 |
|---|---|
| 输入 | `obs`，**float32，1×65**（动态 batch 轴） |
| 输出 | `action`，**float32，1×16** |
| 控制周期 | **20 ms**（物理 5 ms × decimation 4） |
| 内置状态 | **无**（MLP，无 RNN/历史）；历史与外层包装状态见第 3 节 |

### 动作布局（16 维）

| 索引 | 含义 | 尺度 |
|---|---|---|
| 0–11 | 腿关节**位置**（顺序 `FL/FR/RL/RR` × `hip/thigh/calf/foot`） | ×0.35 + 默认站姿 |
| 12–15 | 四轮**速度**（`FL_foot/FR_foot/RL_foot/RR_foot`，前滚为正） | ×**16.0** |

---

## 2. 该候选的实测表现（分母与失败范围）

**冻结 11 项矩阵**：hold、turn ±0.05、turn ±0.10、straight ±0.25、arc ±0.25、brake ±0.25。

| 仿真器 / 口径 | 通过 | 备注 |
|---|---|---|
| **Isaac Lab，gyro OFF，两种子交集** | **6/11** | 种子 7393 单种子 7/11 |
| **Isaac Lab，gyro ON(kp60/cap4)，两种子交集** | **6/11** | 种子 7393 单种子 **8/11** |
| **MuJoCo 正式矩阵** | **6/11** | 转向类仅 `arc_right` PASS（1.09） |
| **双仿真共同通过** | **6 项** | hold、straight×2、brake×2、**arc_right** |
| 实机 | **`not_run`** | R3/R4 未执行 |

### 逐例（Isaac，种子 7393，gyro ON）

| 用例 | 结果 | 关键量 |
|---|---|---|
| hold | PASS | sup 1.000, \|vx\| 0.0005 |
| straight_fwd / bwd | PASS | 跟踪 0.87 / 0.85 |
| arc_left | PASS | 偏航跟踪 0.82 |
| arc_right | PASS | 偏航跟踪 0.87 |
| brake_fwd / bwd | PASS | — |
| **turn_left_0.05** | **FAIL** | 0.15 |
| **turn_right_0.05** | **FAIL** | 0.18 |
| **turn_left_0.10** | **FAIL** | **−0.03（方向反转）** |
| **turn_right_0.10** | **PASS** | 0.87（**仅单一种子**） |

**→ 四项纯转中三项不达标；唯一通过的 `turn_right_0.10` 只在 1 个种子上成立。**

---

## 3. 观测合同（65 维，逐段）

| 索引 | 内容 | 说明 |
|---|---|---|
| `0:3` | 速度指令 − (0.4, 0, 0) | 机体系 vx/vy/wz |
| `3:6` | 机身角速度 | 来自 IMU，机体系 |
| `6:9` | 投影重力方向 | 机体系 |
| `9` | 常数 0.4 | 占位/尺度项 |
| `10:26` | 关节角 − 默认站姿 | **轮轴按 4π 周期回绕** |
| `26:42` | 关节角速度 | 腿 + 轮 |
| `42:58` | **上一帧动作**（包装前） | 用于历史一致性 |
| `58:59` | 相位 sin | 周期 0.62 s |
| `60:64` | `(Δ轮速/0.20).clamp(±3) × zero_flag` | 轮速差分项 |
| `64` | zero_flag | 零指令标志 |

**单位**：角度 rad、角速度 rad/s、线速度 m/s。**坐标系**：机体系，前 +x、左 +y、上 +z。

**不可用仿真真值的字段**：本策略观测**全部**可由实机 LowState + IMU 构造，无特权信息。

---

## 4. 输出到控制目标（完整链路）

```
policy(obs) -> action[16]
  ├ 腿 0..11 : q_target = action*0.35 + default_stance          (位置 PD, kp=55, kd=2)
  └ 轮 12..15: dq_target = action*16.0                          (速度模式, kp=0, kd=6)
       └ (可选) gyro-1 前馈: ff = S*clip(kp*(wz_cmd-wz_meas), ±4 Nm), S=(-1,+1,-1,+1)
                             tau = kd*(dq_target-dq) ; tau = clip(tau+ff, ±13.5)
                             dq_dispatch = dq_meas + tau/kd
```

| 项 | 值 |
|---|---|
| 动作裁剪 | **±6.0** |
| 轮端力矩预算 | **13.5 N·m**（= 已核实硬件上限 15 N·m 的 90%） |
| 腿端力矩预算 | hip/thigh **54**、calf **81**（同为 90% 口径） |
| 关节顺序 | `FL/FR/RL/RR` × `hip/thigh/calf/foot`（part-major） |
| 轮序 | `FL_foot / FR_foot / RL_foot / RR_foot`，**前滚为正** |

**gyro-1 状态**：训练时**开启**（`kp=60 cap=4 gain=0`），评测必须显式传 `--yaw-kp 60 --ff-cap 4`，否则会**静默退回基线**（已加 `AS2W_GYRO_GUARD` 拒跑保护）。

---

## 5. 入口与退出

| 项 | 说明 |
|---|---|
| **允许启动姿态** | 四轮着地、`fixed stand`、数据新鲜 |
| **首帧历史** | `action[42:58]` 置 0；相位由回合计数从 0 起 |
| **成功条件** | 指令跟踪进入门槛且四轮支撑 ≥0.95、无终止、upright ≥0.85 |
| **失败处理** | 减速停稳 → 回 `fixed stand`；**不把策略终点当作整机站稳** |
| **回站方式** | **四轮着地回站**（单腿抬升在本站姿下几何不可行，见证据） |
| **控制权交回** | 由外层状态机负责；本包不实现状态机 |

---

## 6. 复现仿真

### 6.1 MuJoCo

```bash
python runtime/r11_mj_matrix_v2.py \
  --xml simulation/r11_mjcf_excl.xml \
  --ckpt weights/armD_turn65_final.pt \
  --out out/ \
  --contract contracts/wheel_contract_v1.json --contract-name r11_legacy \
  --states 3.0
```

- MJCF 的 `meshdir` 指向仓库内已跟踪的 STL：`training/as2w-backup/as2w-current/assets/r237-train-root/official-model/robots/as2w_description/meshes`
- `timestep` 用 **0.000625 × 32 = 0.020 s**（脚本体 XML 内为 0.00125，脚本会覆写）
- 已验证：从 GitHub 干净克隆可直接加载（`nq=23 nv=22 ngeom=19 nu=16`）

### 6.2 Isaac Lab

```bash
/isaac-sim/python.sh runtime/r11_acceptance_ff.py \
  --usd <as2w.usda> --urdf <as2w.urdf> --out out/ \
  --checkpoints "armD=weights/armD_turn65_final.pt" \
  --seeds 7393,311 \
  --contract contracts/wheel_contract_v1.json --contract-name r11_legacy \
  --yaw-kp 60 --ff-cap 4 --ff-gain 0
```

**必须**带 `--yaw-kp`，否则 `AS2W_GYRO_GUARD` 会拒跑（防静默退回基线）。

### 6.3 合同一致性

`contracts/wheel_contract_v1.json` 是**唯一来源**。训练、Isaac、MuJoCo 三方都必须读同一个 entry。
脚本会打印生效值供核对：

```
AS2W_CONTRACT_APPLIED {"name":"r11_legacy","wheel_kd":6.0,"wheel_scale":16.0,"gyro_enabled":false}
AS2W_MJ_CONTRACT_APPLIED {...}
```

差速/对角公式**只用**以下定义（`contracts/as2w_diff_formula_selftest.py` 已自检 9/9）：

```
左右差速 = (FR + RR − FL − RL) / 2
对角分量 = (FL + RR − FR − RL) / 2
```

### 6.4 视频

```bash
MUJOCO_GL=egl python runtime/r11_demo_video.py \
  --xml simulation/r11_mjcf_excl.xml --ckpt weights/armD_turn65_final.pt \
  --out evidence/videos/ --width 640 --height 480
```

---

## 7. 部署步骤（分阶段，逐阶段授权）

| 阶段 | 内容 | 是否需要现场授权 |
|---|---|---|
| **S0 静态校验** | `SHA256SUMS` 校验、ONNX 哈希、`contracts/` 一致性 | 否 |
| **S1 离线回放** | 用录制观测跑 ONNX，核对逐帧动作 | 否 |
| **S2 只读影子** | 接真实 LowState + IMU，**只计算不发送** | **需** |
| **S3 悬空测试** | 四轮离地，验证下发链路与保护 | **需** |
| **S4 有界实机** | 按姿态/急停/监测计划逐级 | **需，逐档单独授权** |

**S2 之前必须完成**：`deploy.yaml` 中每个关节的**电机槽位、方向、零位**核实。当前**未核实**，`preflight.py` 会因此拒绝闭环。

---

## 8. C1–C8 / R0–R4

| 项 | 状态 | 说明 |
|---|---|---|
| C1 版本溯源 | ✅ | 权重/ONNX/初始化 checkpoint 三份哈希 + 源码快照 + 解析配置 |
| C2 模型接口 | ✅ | opset 17，`max_abs_err 1.907e-06` |
| C3 观测合同 | ✅ | 第 3 节逐段 |
| C4 输出到控制目标 | ⚠️ **partial** | 链路完整；**无硬件 DDS 槽位映射** |
| C5 入口退出 | ✅ | 第 5 节 |
| C6 仿真与资产 | ✅ | MJCF + URDF + STL 均在仓库，可干净克隆复跑 |
| C7 逐例证据 | ✅ | `evidence/` |
| C8 最小回放 | ✅ | 逐帧对齐 `max_abs_err 5.96e-07` |
| R0 收包 | ✅ | |
| R1 离线对齐 | ✅ | |
| R2 sim2sim | ⚠️ partial | MuJoCo 6/11，未通过正式双仿真门槛 |
| R3 CPRO 只读 | ❌ **`not_run`** | |
| R4 有界实机 | ❌ **`not_run`** | |

---

## 9. 已知缺陷与撤回声明

1. **四项纯转不达标**；两端失效方向**相反**（Isaac 欠转、MuJoCo 超调），机理未定位
2. **左右不对称在两仿真器上方向相反** → 不能归因于策略固定偏置
3. **单腿抬升几何不可行**：IK 只能到 0.25–1.73 mm（目标 10 mm），实测为负 → 沿用四轮着地回站
4. **姿态偏置已被否决**：MuJoCo 改善但 Isaac 零指令漂移恶化 65–350 倍
5. **力矩余量曾恒读 0**：根因是评测脚本不读 `AS2W_GYRO_KP` 环境变量（只认 `--yaw-kp`），**此前所有 Isaac 结果都是 gyro-OFF 基线**；已修并加拒跑守卫
6. **`applied_torque` 不是求解器施力**：Isaac 隐式驱动实际饱和率标 **`not_measured`**
7. **本包无任何实机证据**

完整教训见仓库 `skills/as2w/ENGINEERING_LESSONS.md`。
