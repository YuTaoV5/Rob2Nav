# AS2-W 滚动 armD（转向曝光 65%）研究交付包

> 版本 `rolling-armD-turn65-research` ｜ 日期 2026-11-17 ｜ 负责人 AS2-W 滚动 agent
> 状态 **research / deployable=false** —— 见第 6 节 R0–R4。

## 1. 一句话结论

`armD` 是目前**最好的 Isaac 候选**：种子 7393 **7/11**，且**左右弧同时通过**（对照组 arc_left 是 FAIL）。但**四项纯转在两端都不达标**，因此**尚无验收通过权重**，本包不构成部署授权。

## 2. 能力与输入输出

| 项 | 值 |
| --- | --- |
| 技能 | AS2-W 四轮**滚动**（独立技能，与踏步/楼梯/起身隔离） |
| 权重 | `weights/armD_turn65_final.pt`，sha256 `de61de58fd68efc356efbd48a0b9683153361fc87b6fb32c7b9c83584cbd6926` |
| 部署模型 | `weights/policy.onnx`，opset 17，175066 B，sha256 `680e0c720750fdaf3a3403232151a9e0301d1646fd45c95449dbdf74f7d315e7` |
| 对照权重 | `weights/armE_control_reference.pt`，sha256 `4bdf4d26b076d7c717e9c151946dd42f8b3cf9dbbf592c5d6aac1ed5f6615866` |
| 输入 | `obs` float32 **1×65** |
| 输出 | `action` float32 **1×16**（12 腿轴位置 + 4 轮轴速度） |
| 控制周期 | **20 ms**（physics 5 ms × decimation 4） |
| 动作包装 | gyro-1（`yaw_kp=60`, `ff_cap=4`, `ff_gain=0`, `kd=6`），见 C4 |

## 3. 通过结果（各场景分母）

| 场景 | 分母 | 结果 |
| --- | --- | --- |
| Isaac Lab，种子 7393 | 11 用例 | **7/11 PASS** |
| Isaac Lab，种子 311 | 11 用例 | **6/11 PASS** |
| Isaac Lab，两种子 min 口径 | 11 | **6/11** |
| MuJoCo 矩阵（对照权重） | 11 | 7/11（armD 未跑 MuJoCo 正式矩阵） |
| 实机 | — | **not_run**（R3/R4 需现场授权） |

### 逐例（种子 7393，armD@1k）

| 用例 | 结果 | 关键量 |
| --- | --- | --- |
| hold | PASS | sup 1.000, |vx| 0.0005 |
| straight_fwd 0.25 | PASS | 跟踪 0.87 |
| straight_bwd 0.25 | PASS | 跟踪 0.85 |
| **arc_left 0.25** | **PASS** | 偏航跟踪 **0.83**（对照 FAIL 0.50） |
| **arc_right 0.25** | **PASS** | 偏航跟踪 **0.76** |
| brake_fwd / brake_bwd | PASS | — |
| turn_left_0.05 | FAIL | 偏航跟踪 0.02（门槛 ≥0.70） |
| turn_right_0.05 | FAIL | 0.08 |
| turn_left_0.10 | FAIL | 0.35 |
| turn_right_0.10 | FAIL | 0.02 |

**失败范围：纯原地转（±0.05 / ±0.10 rad/s）四项均不达标。**

## 4. 目录

```
rolling-armD-turn65-research/
├── README.md              本文件
├── manifest.json          机器可读索引（C1–C8、R0–R4、哈希）
├── SHA256SUMS             包内校验清单（不含自身）
├── weights/               部署权重、ONNX、对照权重
├── runtime/               动作包装、映射、部署/回退脚本、评测脚本、MJCF
├── training/              启动配方、训练源码、解析后配置
└── evidence/              双仿真逐例结果、回放、诊断、视频
```

## 5. C1–C8 逐项状态

| 项 | 状态 | 证据 |
| --- | --- | --- |
| **C1 版本溯源** | ✅ pass | 权重 sha `de61de58fd68efc3`；初始化 checkpoint `4bdf4d26…`（sha `4bdf4d26b076d7c717e9c151946dd42f8b3cf9dbbf592c5d6aac1ed5f6615866`）；run 目录 `training/resolved_config_armD.txt`；启动命令 `training/launch_armD.sh`；训练源码 sha 见 `manifest.json` |
| **C2 模型接口** | ✅ pass | obs 1×65 → action 1×16，opset 17，导出脚本 `runtime/export_policy_onnx.py`；**导出经干净检出实跑，字节级复现** sha `680e0c72…`；同输入差值 max_abs_err **1.43e-06** |
| **C3 观测合同** | ✅ pass | `evidence/C3_OBSERVATION_CONTRACT.md`（逐维索引/单位/坐标系/缩放/裁剪/滤波/复位） |
| **C4 输出到控制目标** | ⚠️ partial | 包装源码在 `runtime/`；**但验收脚本的 `tau_max`/`ff_max` 遥测读数恒为 0，力矩余量项标 `not_measured`**（见 `evidence/EVAL3_BIAS_SWITCH.md` 与本包 `manifest.json` 的 caveats） |
| **C5 入口/退出** | ✅ pass | 零指令驻停、四轮着地回站、固定杆位切换；姿态偏置为**实验开关、默认关闭** |
| **C6 训练仿真与资产** | ✅ pass | MJCF `runtime/r11_mjcf_excl.xml`（**已入库、相对 meshdir、从干净克隆可加载**）；URDF/USD/STL 均在 `training/as2w-backup/`；见 `evidence/MUJOCO_ASSETS_UPLOADED.md` |
| **C7 已有验证证据** | ✅ pass | 逐例结果在 `evidence/isaac/`、`evidence/mujoco/`；失败样例、首次失败帧、诊断链齐全；**armD 未跑 MuJoCo 正式矩阵，标 `not_run`** |
| **C8 最小回放** | ✅ pass | `evidence/replay/`；C8 逐帧对齐 max_abs_err **5.96e-07** |

## 6. R0–R4 状态

| 项 | 状态 | 说明 |
| --- | --- | --- |
| **R0 收包** | ✅ pass | C1–C8 齐备，SHA 一致，权重溯源正确 |
| **R1 离线对齐** | ✅ pass | checkpoint↔ONNX 1.43e-06；C8 回放 5.96e-07 |
| **R2 sim2sim** | ⚠️ **partial** | MuJoCo 侧对**对照权重**有完整矩阵（7/11）；**armD 权重的 MuJoCo 正式矩阵未跑**（`not_run`） |
| **R3 CPRO 只读** | ❌ **not_run** | 需现场设备与授权；影子回放脚本在 `runtime/` |
| **R4 有界实机** | ❌ **not_run** | 需现场人员逐档授权；本包**不构成**实机部署授权 |

## 7. 已知失败与未验证范围（如实声明）

1. **四项纯转不达标**（0.02–0.35 vs 门槛 0.70）；两端一致，无一端通过
2. **两端对纯转的失效方向相反**：Isaac 欠跟踪，MuJoCo 超调（1.41–1.56）—— 机理尚未定位
3. **右向不对称加深**：armD@6.6k 时 arc_left 0.93 / arc_right 0.17
4. **力矩余量未测**：`tau_max`/`ff_max` 遥测读数恒 0；Isaac 隐式驱动的实际饱和率标 **`not_measured`**
5. **长转 dwell 从未实现**：`AS2W_AB_TURN_DWELL_*` 只被打印未被读取；armB/armC 的「长转」实为**仅配比变化**
6. **单腿抬升在现有站姿下几何不可行**：IK 只能到 0.25–1.73 mm（目标 10 mm），实测为负；正常退出沿用四轮着地回站
7. **姿态偏置已被否决**：MuJoCo 改善但 Isaac 零指令漂移恶化 65–350 倍（0.7 mm → 22–30 cm）
8. **本包无实机证据**

## 8. 复跑入口

```bash
# MuJoCo 矩阵（需 MJCF 与 mesh 目录）
python runtime/r11_mj_matrix_v2.py --xml runtime/r11_mjcf_excl.xml \
  --ckpt weights/armD_turn65_final.pt --out out/

# Isaac 验收矩阵（两种子）
/isaac-sim/python.sh runtime/r11_acceptance_ff.py \
  --usd <as2w.usda> --urdf <as2w.urdf> --out out/ \
  --checkpoints "armD=weights/armD_turn65_final.pt" --seeds 7393,311

# 视频
MUJOCO_GL=egl python runtime/r11_demo_video.py --xml runtime/r11_mjcf_excl.xml \
  --ckpt weights/armD_turn65_final.pt --out videos/ --width 640 --height 480
```

## 9. 相关文档

| 文档 | 内容 |
| --- | --- |
| `evidence/AB_ARM_B_TWO_SEED_ISAAC.md` | B 组两种子结果 |
| `evidence/ARM_C_ARM_D_RESULT.md` | armC/armD 配比消融与因果判断 |
| `evidence/DUAL_END_DIVERGENCE.md` | 两端分岔定位（首个分岔点候选） |
| `evidence/DEGRADATION_MULTI_CASE.md` | 卸载先兆的多工况否证 |
| `evidence/LIFT_IN_PLACE_VERDICT.md` | 单腿抬升几何不可行 |
| `evidence/BIAS_ISAAC_VERDICT.md` | 姿态偏置被否决 |
| `evidence/SAFETY_DENOMINATOR_CORRECTION.md` | 90% 力矩口径更正 |
| `evidence/EXPORT_FIXED.md` | 导出入口修复与字节级复现 |
| `evidence/MUJOCO_ASSETS_UPLOADED.md` | MuJoCo 资产入库与可移植性 |
| `evidence/VIDEOS.md` | 效果视频说明 |

> 工程教训与自查表见仓库 `skills/as2w/ENGINEERING_LESSONS.md`。
