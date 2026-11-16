# rolling-r11-turn-gyro-research

**状态：research · 不可部署（deployable=false）**
日期 2026-11-16 · 负责人：服务器 Agent（lead）
最终权重：`weights/armE_support2p0_final.pt`
SHA256：`4bdf4d26b076d7c7`…（完整值见 `weights/SHA256`）
包装版本：**gyro-1**（本轮新增，见 `runtime/README.md`）

**汇总报告见 [FINAL_REPORT.md](FINAL_REPORT.md)** —— 实际通过的速度/转向/刹停/载荷范围与尚缺的现场证据。

## 能力声明（按实际通过的包络填写）

同一权重 `armE_support2p0_final.pt` 在两端的实测结果：

| 用例 | MuJoCo（5 站态） | Isaac（2 种子） |
| --- | --- | --- |
| 零指令驻停 10 s | PASS | PASS |
| 直线前进 0.25 m/s | PASS | PASS |
| 直线倒车 0.25 m/s | PASS | PASS |
| 弧线 0.25 m/s / +0.05 rad/s（左） | PASS | **FAIL（0.50–0.55）** |
| 弧线 0.25 m/s / −0.05 rad/s（右） | PASS | **PASS（0.78 / 0.87）** |
| 前进松杆刹停 | PASS | PASS |
| 倒车松杆刹停 | PASS | PASS |
| 原地转 ±0.05 / ±0.10 rad/s（4 项） | **FAIL** | **FAIL** |
| **合计** | **7/11** | **6/11**（基线 5/11） |

- **实际载荷范围：0 kg（无载）通过；+6 kg 不通过**（6/11 → 4/11，支撑 0.59–0.91 并出现终止）。
  质量读回：base_link 16.58 kg / 总 32.00 kg。见 `evidence/LOAD_TIER.md`。
- **整帧丢失**：IID 5%/15% 保持 6/11；但**突发连续 10/30 帧降为 5/11** —— 约 1,100 帧丢失时 IID 无退化而突发退化，**结构比比例更重要**。见 `evidence/FRAME_DROP.md`、`evidence/BURST_LOSS.md`。
- **通讯时延：未建立容限**（20 ms → 4/11、60 ms → 6/11，非单调；左右不对称随时延翻转）。见 `evidence/LATENCY_TIER.md`。
- **实际速度包络：仅 0.25 m/s。** 0.60 / 1.00 m/s 档实测不通过（四轮支撑 0.63–0.85、终止 2–4 次），
  策略饱和在约 0.40–0.52 m/s。见 `evidence/SPEED_ENVELOPE.md`。
- MuJoCo：`r11_mj_matrix_v2.py`，站态 2/3/4/5/6 s，`7/11`，双侧弧线均通过。
- Isaac：`r11_acceptance_ff.py`（gyro-1 包装，`--yaw-kp 60 --ff-cap 4`），种子 7393 与 311 **均为 6/11**，零终止。
- **双仿真同时通过：无。** 左弧线与 4 个纯转在 Isaac 上不通过，因此本包**不满足验收门槛**。

## 失败范围（必须与能力声明一并阅读）

1. **纯转在 Isaac 上全部不通过**：偏航跟踪 0.01–0.39，门槛为 [0.70, 1.30]。
2. **Isaac 左弧线不通过**（0.50–0.55）。提高反馈上限于 4→8 N·m **没有**改善（0.50→0.47），
   说明左侧不是力矩受限，而是结构性阻力。
3. **MuJoCo 侧的纯转是被四轮支撑判死**（支撑 0.487–0.927，门槛 0.95），不是偏航不足。
4. **MuJoCo 的 7/11 不能外推到 Isaac**：同一权重 Isaac 为 6/11，且两端卡在不同约束上。

## 目录

```
weights/   armE_support2p0_final.pt + policy.onnx + SHA256
runtime/   评测脚本（含 gyro-1 包装）+ MuJoCo 矩阵 + 执行链探针 + 口径工具
training/  armE 的训练入口（支撑惩罚 -2.0）
evidence/  isaac/、mujoco/、c8/ 的原始数据，C1–C8、双仿真对照与 C8 回放报告
```

## 复跑

```bash
# MuJoCo 11 用例矩阵（5 站态）
python3 runtime/r11_mj_matrix_v2.py --xml <r11_mjcf_excl.xml> \
  --ckpt weights/armE_support2p0_final.pt --out /tmp/mj --states 2.0,3.0,4.0,5.0,6.0
# Isaac 11 用例矩阵（gyro-1 包装）
python runtime/r11_acceptance_ff.py --usd <as2w.usda> --urdf <as2w.urdf> --out /tmp/isaac \
  --checkpoints "armE=weights/armE_support2p0_final.pt" --seeds 7393,311 --ff-gain 0 --ff-cap 4 --yaw-kp 60
```

## C1–C8 / R0–R4

见 `evidence/C1_C8_DELIVERY_STATUS.md`。摘要：

| 项 | 状态 |
| --- | --- |
| C1 版本溯源 | **pass**（选中/部署/初始化权重哈希、run ID、启动命令、实际解析配置均已记录；训练期环境源码已固化为不可变提交 `6d938eb5`，见 `evidence/C1_SOURCE_SNAPSHOT.md`） |
| C2 模型接口 | **pass**（见下） |
| C3 观测合同 | **pass**（`evidence/C3_OBSERVATION_CONTRACT.md`） |
| C4 输出链路 | partial（`evidence/C4_OUTPUT_CHAIN.md`；无实机 DDS 映射） |
| C5 入口与退出 | **partial（逐项枚举见 `evidence/C5_STATUS_ENUMERATION.md`）**：已测 8 项（松杆减速/停稳/回站 1 项 pass，失联/丢包/时延/交接首帧 7 项有数据但有边界）；未测 5 项（**急停**、**观测冻结/时间戳跳变/重复帧**、自动恢复、跨策略切换、控制权释放）。跨策略切换当前条件不具备（无第二个学习策略），控制权释放需现场 |
| C6 仿真与资产 | **pass**（`evidence/C6_ASSETS.md`；USD 展开引用层 9 个已逐个哈希见 `evidence/C6_USD_LAYERS.md`；URDF 引用的 36 个网格已逐个哈希见 `evidence/C6_MESH_HASHES.md`；训练复跑入口 recipe 已 pin 并通过校验。运行期 physics variant 的实际选中未验证，不影响资产锁定） |
| C7 逐例证据 | **pass**（`evidence/isaac/`、`evidence/mujoco/`） |
| C8 最小回放 | **pass**（600 帧真实 rollout，5.96e-07） |
| R0 | pass；R1 partial；R2 partial（无候选双端同时通过） |
| **R3 / R4** | **not_run（需现场授权）** |

## C2 模型接口（已通过）

`weights/policy.onnx` opset 17，175,066 B

- 输入 `obs`：`["batch", 65]` float32　输出 `action`：`["batch", 16]` float32　周期 0.02 s
- 结构：MLP 65-128-128-128-16，ELU 隐层，**输出为确定性均值**（`distribution.std_param` 采样用的 log-std 不进图）
- 无内部状态
- 同输入一致性：4096 样本，**max_abs_err 1.43e-06**、mean 1.01e-07、相对 1.34e-07，容差 1e-5 → **PASS**
- 证据：`evidence/C2_ONNX_PARITY.json`；导出脚本 `runtime/export_policy_onnx.py`
