# AS2-W 滚动交付 · C1–C8 状态装配（研究候选，未通过验收）

日期 2026-11-16 03:0x · 对应 goal round 4
**状态：`research`（未通过验收）** —— 不得作为可部署版本发布

---

## 0. 一句话结论

**驻停（10 s）、±0.25 m/s 直线（前进/倒车）、松杆刹停已在 Isaac Lab 通过；转向在两端均未通过（门槛 0.70）**，因此本候选标记 **`research`**，不进入实机（R3/R4 = `not_run`）。

---

## 1. C1 版本溯源

### 1.1 候选权重全量 SHA256

| 名称 | 路径（相对 `/root/gpufree-data/as2w-runs/r11_fanziqi_lab/`） | SHA256 |
|---|---|---|
| `m799` | `train-20261115-224755-17426/model_799.pt` | `19c3c9f70f65cdb22dc7102a1e7a2c0c44ee0d0e8a2dee2a984d1961f3f01008` |
| **`m1999`** | `resume-20261115-204347-8722/model_1999.pt` | `f54751faddc211b4cfa7498ac35e95e4f2617779bcdb8b3b5de5904c9da72985` |
| `m2800` | `resume-20261115-230617-42771/model_2800.pt` | `28206dc8cb2cb7933dec79e377eb30da715361c8f9c1d20893e08c49329754f9` |
| **`m3998`** | `resume-20261115-230617-42771/model_3998.pt` | `92c44107bc16e4783e8b8cacc6c1d012f3f86e6700860a6a85badedbe86dfd66` |
| `turn-pilot` final | `turn-pilot.xM3K4M/train2000-20261116-005804-49391/r11-final.pt` | `f5b6c03a12967f6a2acb2b0fa77535d2418a67513a08a0fa38ca02269095ec6f` |
| `turn-pilot` m3998 | 同上目录 `model_3998.pt` | `3bef0e1496f2aa65c1b73b7fa2912d3ec6a81b140eab334a3490976a0f4e3494` |
| **`turn-fix` m3998** | `turn-fix.xM3K4M/train2000fix-20261116-013650-65418/model_3998.pt` | `617b252ca3b3520ff4e77dcfbb03adcc9d9f61c6f8a45c85c61d24675ffd2339` |
| `turn-fix` final | 同上目录 `r11-final.pt` | `d00070f823e3ab1bba2b5397e55c6d772e76284bc57fdb51faa0ae954bbdcf02` |

### 1.2 初始化哈希链

| 训练线 | 初始化权重 | 追加轮次 | 备注 |
|---|---|---|---|
| `m799` | 从零训练（`train-…-7745` 0→1999 的前段） | — | — |
| `m1999` | 同上 run 续训 | — | — |
| `m3998` | `resume-…-42271` 从 `m799` 续训 | 799→3998 | — |
| `turn-pilot` | **`m1999`（`f54751fa…`）** | +2000（1999→3998） | **延迟 DR 开启（语义缺陷）** |
| `turn-fix` | **`m1999`（`f54751fa…`）** | +2000（1999→3998） | **延迟 DR 关闭（`as2w_action_delay_max_steps=0`）** |
| `rolling-maneuvers@1199` | `e66a1bbac8cdae8cadf1d7c58226c1ac99178ad4b4a1ac7e98dbc334256caadc` | — | 归档；**输出** `81cdea1e65d6631a2118d83eed5c52cd6acc91de0a81bb0d7409c471a4cc644a`；`expanded` profile，**未评测** |

### 1.3 源码与资产

| 项 | 值 |
|---|---|
| 仓库 | `YuTaoV5/Rob2Nav` @ **`abde5f49330a467db800e7cdcfe6e991c88b4c9b`** |
| 训练守则 AGENTS.md | `a6a4e5e09c5218741e2b4a6f627d9b16ebf757027bec9725dafec211459faad5` |
| 训练入口 | `turn-fix.xM3K4M/train_r11.py` = `27d6f9644de6fdc4741e60a4f764346759e44767bb11bcd68432ee64c549ee9a` |
| **环境源码（已改）** | `code/as2w_isaac_robust_env.py` = `1af438db2ff3babc064a3a56bc8825880a5d3640b2d0f76174b7faa35233e24f`（改前 `fa648f7e764312df…`） |
| URDF | `696aec2c0915ec1e06bcc91a63104d8234c51c0eb94331fa5496bf426c51e926` |
| USD | `0c5786834c5deda135d3195062033dfe8312c9abf7daad689070925e89d3a55c` |
| MJCF（原始，未改） | `bd86aae1d8047a7be67aef8340c1303950f136af5fb5506dfeb451a1c94d0ce2` |
| MJCF（r11 专用副本） | `dc8a4d33a687327e021fc6f655164aad67b1c25fa997a94fd81277cec38e8bd9`（含 `<exclude base_link↔FL/FR_thigh>`，绝对 mesh 路径） |

### 1.4 ⚠️ 未提交的源码改动（必须在交付前处理）

```
git -C /root/gpufree-data/Rob2Nav status --short
 M training/as2w-backup/as2w-current/code/as2w_isaac_robust_env.py
?? training/as2w-backup/as2w-current/code/as2w_isaac_robust_env.py.bak-actiondelay-20261116-012918
```

**改动内容**：关闭动作延迟域随机化（`as2w_action_delay_max_steps = 0`），因为 `apply_actions()` 每物理子步执行一次（每策略步 4 次），5 槽环被写 4 次导致索引混叠，标称 0–4 策略步延迟实际塌缩为 0–20 ms 子步级。

**处置要求**：按 goal「不把新 run 整目录覆盖到 `as2w-current`」的原则，**该改动不应直接提交到 `as2w-current`**。交付时应把补丁作为独立 patch 文件（或在 `skills/as2w/rolling/` 的入口中显式设参）迁移，`.bak` 文件不入库。

---

## 2. C2 模型接口

| 项 | 状态 | 证据 |
|---|---|---|
| 输入 | **obs 65 维**（`1×65`） | 训练入口断言 `wrapped.get_observations()["policy"].shape[-1] != 65 → RuntimeError` |
| 输出 | **动作 16 维**（`1×16`） | 同上断言 `num_actions != 16` |
| 控制周期 | **20 ms（50 Hz）** | `env.step_dt = sim.dt(0.005) × decimation(4)` |
| dtype | `float32` | — |
| **ONNX 导出** | ❌ **`not_run`** | 未导出 |
| **ONNX/Torch 同输入逐帧对比** | ❌ **`not_run`** | 未做 |

**→ C2 未完成。** 但已有等价证据：**三解释器 actor parity**（Isaac Sim venv / MuJoCo 3.12.0 / MuJoCo 3.2.3）在同一批 41 帧观测上输出 **max|diff| = 0.0**，权重加载路径本身无歧义（`mjparity-20261115-222824/`）。

---

## 3. C3 逐维观测合同

| 索引 | 内容 | 缩放/偏置 | 状态 |
|---|---|---|---|
| `0:3` | 命令 `[vx, vy, wz]` | **减偏置 (0.4, 0, 0)** | ✅ 已验证（obs[0] 实测 −0.4） |
| `3:6` | `root_ang_vel_b` | 原值 | ✅ |
| `6:9` | `projected_gravity_b` | 原值 | ✅ |
| `9` | 常量 | **0.4** | ✅ |
| `10:26` | `q − default`（16 轴） | 轮 **`(q+2π)%(4π)−2π`** | ✅ 回绕实测通过 |
| `26:42` | `dq`（16 轴） | 原值 | ✅ |
| `42:58` | **`prev_action` = `env.policy_action`** | — | ✅ 用 `action_manager.action` 误差 4.06e−01 → 改后 **0.0** |
| `58,59` | `sin/cos(2π·ep_len·dt/0.62)` | 周期 0.62 s | ✅ |
| `60:64` | `clamp(Δwheel/0.20, ±3) × zero_flag` | 缩放 **0.20**、clamp **±3** | ✅ |
| `64` | zero flag `(command == 0).all()` | — | ✅ |

**默认站姿**：hip 0.0 / thigh 0.8 / calf −1.6 / foot 0.0（canonical 顺序 hip×4, thigh×4, calf×4, foot×4）

**⚠️ 口径警告**：任务描述里的 `obs[6]=2×cmd_vx`、`obs[8]=0.25×cmd_wz` **在本栈不存在**（已核实）。

---

## 4. C4 完整输出链路

```
actor raw(16)                          ← 网络输出
  → ×6.0 clip（wrapper `clip_actions`）
  → processed_actions:
        腿 0-11: × 0.35 + 默认站姿
        轮 12-15: × 16.0
  → 动作项 clip（腿：90% 角度带；轮：URDF 速度 100%，实际非绑定）
  → DelayedTarget（已关闭随机化，lag=0 → 直通）
  → set_joint_position_target_index / set_joint_velocity_target_index
  → PhysX 隐式驱动：腿 kp=55/kd=2；轮 **kp=0/kd=6**、`effort_limit=13.5 N·m`（实测 `maxForce=13.5`）
```

| 项 | 值 | 状态 |
|---|---|---|
| 关节顺序 | `JOINTS = [f"{leg}_{part}" for part in ("hip","thigh","calf","foot") for leg in ("FL","FR","RL","RR")]` | ✅ |
| 腿 scale / offset | 0.35 / 默认站姿 | ✅ |
| 轮 scale | 16.0 | ✅ |
| `clip_actions` | 6.0 | ✅ 实测从未触发（\|raw\|max ≪ 6） |
| 轮增益 | **kp=0 / kd=6** | ✅ 实测生效 |
| 轮力矩上限 | **13.5 N·m**（0.9×15） | ✅ 实测 PhysX `maxForce=13.5` |
| 轮速安全 clip | 98.47 rad/s（URDF 100%） | ⚠️ **非绑定**（`6×16=96 < 98.47`） |
| **包装版本** | `r11-train-v1`（无 PI、无低通、延迟 lag=0） | 需升版记录 |

**已知口径不一致（未改，需在交付文档列明）**：
1. 轮支撑奖励用**世界系 z 分量 + 2 N**；验收用 **‖F‖ + 5 N**
2. 后腿角：硬终止 1.40 / 软惩罚 1.30 / 90% 角度带内高达 3.851
3. 膝部：原点 45 mm / 包络 15 mm / 软 100 mm（**两个不同几何量**，原点→面偏移 ≈36.6 mm）
4. 非轮触地：训练 >4 N / MuJoCo 评测任意接触

---

## 5. C5 入口与退出

| 项 | 状态 | 证据 |
|---|---|---|
| 允许的姿态/速度 | ±0.25 m/s、±0.15 rad/s（cfg）；评测用 ±0.25 / ±0.05,0.10 | ✅ |
| 历史初始化 | 前 2 s（`episode_length_buf < 100`）零指令 + `hold_action` 斜坡 | ✅ |
| **hold 驻停 10 s** | ✅ **PASS**（Isaac vx −0.0002、wz +0.0012；MuJoCo 同） | `acceptance_v2.json` / `mujoco_matrix_v2.json` |
| **松杆刹停** | ✅ **PASS**（±0.25 末段 \|vx\| 0.000–0.013，距离 0.03–0.07 m） | 同上 |
| **fixed stand 中转** | ❌ **`not_run`** | 未做权重切换流程 |
| 断流恢复 / 失联 | ❌ **`not_run`** | — |
| 失败条件 | 终止项：`angle_violation` / `stance_violation` / `nonwheel_contact(>4N)` / `knee_surface` | ✅ 已记录 |
| 退出目标 | fixed stand | ❌ `not_run` |

---

## 6. C6 仿真与资产

| 项 | 状态 |
|---|---|
| URDF / USD / MJCF 哈希锁定 | ✅ 见 §1.3 |
| MuJoCo 合同 | ✅ 3.12.0、0.625 ms ×32 = 20 ms、轮 kd=6、±13.5 N·m；**MJCF 自带 timestep 0.00125 必须由入口覆盖并 assert** |
| 无穿透初态 | ✅ `base_z = 0.381395`（四轮间隙 0），加载平衡 0.3415–0.3450 |
| r11 专用 MJCF | ✅ 含 `<exclude base_link↔FL/FR_thigh>`，**原资产未改** |
| 训练仿真复跑入口 | ✅ `launch2000fix.sh`（含 init + 环境源码双 sha256 校验） |
| 延迟/随机化 | **动作延迟 DR 关闭**；观测延迟 `observation_lag ∈{0,1}` 保留 |
| actor parity（三解释器） | ✅ **max\|diff\| = 0.0** |

---

## 7. C7 逐例证据

### 7.1 Isaac Lab v1 冻结矩阵（4 候选 × hold + ±0.05/±0.10 × 2 种子）

`results/eval/frozen-20261116-0055/`

| 候选 | hold | 0.05L | 0.05R | 0.10L | 0.10R | 较差侧 |
|---|---|---|---|---|---|---|
| m799 | PASS | 12% | **−5%（反向）** | 25% | **−2%（反向）** | −8.4% |
| m1999 | PASS | 16%（seed311 **0.00**） | 41% | 25% | 29% | 0.0% |
| m2800 | **FAIL（1 次保护终止）** | 6% | 6% | 7% | 8% | 3.4% |
| **m3998** | PASS | **19%** | **9%** | **20%** | **8%** | **3.7%（四格全正）** |

### 7.2 Isaac Lab v2（含 0.25 m/s 档，m1999 / m3998）

`results/eval/accept-v2-20261116-012446/` —— **各 5/11**
- PASS：hold、straight_fwd/bwd_0.25、brake_fwd/bwd_0.25
- FAIL：4 个纯转、2 个弧线
- 22 用例**终止 0、支撑 ≥0.995、upright ≥0.995** → 失败全为跟踪类

### 7.3 MuJoCo v2（同 11 用例 × 3 站态，m3998）

`results/mujoco/matrix-v2-20261116-025950/` —— **4/11**
- PASS：hold、straight_fwd/bwd_0.25、brake_bwd_0.25
- **`brake_fwd` 一个站态摔倒**（upright −0.988、支撑 0.000、27 次非轮接触）→ 安全边界
- `turn_left_0.10` yaw 0.99–1.15（带内）但**支撑 0.49–0.69**

### 7.4 双仿真一致 / 分歧

| 一致 | 分歧 |
|---|---|
| hold、±0.25 直线、brake_bwd | **brake_fwd**（Isaac PASS / MuJoCo 摔倒） |
| **arc_left 两端都在 0.63–0.69** | turn_0.10 失效模式不同 |
| **arc_right 两端都几乎无偏航（0.01 / −0.04）** | — |

### 7.5 逐例首次违规

| 来源 | 结果 |
|---|---|
| `chain` 最差 run（575 帧） | **第 6 帧轮力矩饱和**（落步瞬态）、**第 14 帧 \|ddq_target\| 超几何差速** |
| Isaac v2 22 用例 | 终止 0 → 无硬违规 |
| MuJoCo v2 | `brake_fwd` 一态摔倒，**首帧未定位** → 待补 |

---

## 8. C8 最小回放

| 项 | 状态 | 证据 |
|---|---|---|
| actor 重推理（记录 obs → raw） | ✅ **max err 3.58e−07** | `results/chain/main-20261116-011159/` |
| raw → clip → processed → 最终轮/腿目标 | ✅ **全部 max err 0** | 同上 |
| raw → 下一帧 obs[42:58]（prev_action） | ✅ **max err 0** | 同上 |
| 轮角增量观测通道 | ✅ **max err 0** | 同上 |
| 观测链其余通道 | ✅ ≤1.05e−5（clock float32 相位精度） | 同上 |
| **首处分歧** | ✅ 只在「名义 0–4 控制步延迟」这一**错误模型**上（第 0 帧差 0.193 rad/s）；真实子步语义下无分歧 | 同上 |
| 固定延迟下回放 | ✅ 21 个 run | 同上 |
| **ONNX Runtime 同输入对比** | ❌ **`not_run`** | 未导出 ONNX |

---

## 9. 验收结论（按已通过的实际包络填写）

| 能力 | 范围 | 状态 |
|---|---|---|
| 零速驻停 10 s | \|vx\|<0.02、\|wz\|<0.02 | ✅ **通过**（Isaac + MuJoCo） |
| 直线前进 | **0.25 m/s**（跟踪 0.86–0.91 Isaac、1.04–1.11 MuJoCo） | ✅ **通过** |
| 直线倒车 | **0.25 m/s**（0.86–0.88 Isaac、1.04–1.08 MuJoCo） | ✅ **通过** |
| 松杆刹停 | 0.25 m/s（末段 \|vx\| 0.000–0.013、距离 0.03–0.07 m） | ✅ Isaac PASS / ⚠️ **MuJoCo 前进刹车一态摔倒** |
| 原地左转 | ±0.05/±0.10 | ❌ **未通过**（最好 0.43） |
| 原地右转 | ±0.05/±0.10 | ❌ **未通过**（最好 0.39） |
| 左弧线 0.25 | vx 0.87–0.92 ✅ / yaw 0.63–0.69 | ❌ **未通过**（差 0.01–0.07） |
| 右弧线 0.25 | vx 0.88–1.09 ✅ / **yaw 0.01 / −0.04** | ❌ **未通过（两端几乎无偏航）** |
| 0.6 / 1.0 m/s | — | **`not_run`**（按 goal 逐档解锁，低速未全过不前推） |
| 固定负载 0/+6 kg | — | **`not_run`** |
| 时延 / 摩擦 / 推扰 | — | **`not_run`** |
| 腿轮联合配置 | — | **`not_run`**（无对应权重） |
| ONNX | — | **`not_run`** |
| R3 ARM64/CPRO 只读 | — | **`not_run`** |
| R4 实机 | — | **`not_run`**（需现场授权） |

---

## 10. 交付判定

**`research` / 未通过。**
- **不进入实机**：转向为零，且 MuJoCo 出现摔倒
- **可发布为研究候选**：驻停 + 0.25 m/s 直线 + 刹停有双仿真证据
- **R4 待现场执行**，所缺证据：转向能力、ONNX 一致性、R3 只读预检、现场授权

---

## 11. 诚实声明

1. 本文是**证据装配**，非新增测量
2. **ONNX（C2/C8）、fixed stand 中转（C5）、0.6/1.0 m/s、固定负载、时延/摩擦/推扰、腿轮联合、R3/R4 全部 `not_run`**
3. Isaac 侧 2 种子、MuJoCo 侧 3 个确定性站态（**口径不同**，比较需注意）
4. `brake_fwd` MuJoCo 摔倒的**首帧未定位**
5. 源码改动（动作延迟 DR 关闭）**尚未提交**，`.bak` 文件在仓库内为未跟踪状态
6. 未改奖励/阈值/摩擦
