# C1–C8 与 R0–R4 状态

| 项 | 状态 | 证据 / 缺口 |
| --- | --- | --- |
| C1 版本溯源 | **partial** | `evidence/C1_PROVENANCE.md`；权重/初始化/环境源码/配置哈希齐全；**训练入口已入库** `skills/as2w/rolling/train_armE_support2p0.py` 与配方 `recipes/armE-support2.sh`；仍缺：训练期环境源码的 git 快照（延迟 DR 改动按设计只以补丁交付） |
| C2 模型接口 | **pass** | `weights/policy.onnx` opset 17；输入 obs[65] → 输出 action[16]，周期 0.02 s；4096 样本 max_abs_err 1.43e-06（容差 1e-5）；证据 `evidence/C2_ONNX_PARITY.json`，脚本 `runtime/export_policy_onnx.py` |
| C3 逐维观测合同 | **pass** | `evidence/C3_OBSERVATION_CONTRACT.md`；65 维逐段按代码核对（索引/单位/坐标系/缩放/来源/复位）；缺口：时间戳与数据龄在仿真内不建模（实机 not_measured） |
| C4 完整输出链路 | **partial** | `evidence/C4_OUTPUT_CHAIN.md` + `runtime/r11_acceptance_ff.py`；增益/缩放/裁剪/gyro-1/力矩保护逐项列明；**未提供实机 DDS 映射**（R3 前置） |
| C5 入口与退出 | **partial** | `evidence/C5_HANDOVER.md`；仿真内 5 阶段（stand→ramp→hold→ramp→stand）零终止、支撑 1.000、退出漂移 0.0002/0.0001；缺口：**仅仿真**、零指令站姿非独立站姿控制器、**未测策略间控制权交接**、未测策略间控制权交接与急停；**通讯失效已测**（`evidence/C5_COMMLOSS.md`：失联即归零比冻结最后指令少走 27.3 倍距离） |
| C6 仿真与资产 | **partial** | `evidence/C6_ASSETS.md` + `evidence/C6_MESH_SHA256.txt`（18 个 STL 逐个哈希）；URDF/USD/MJCF/三个环境源码哈希与版本已附；缺口：USD 引用层展开内容未单独哈希 |
| C7 逐例证据 | **pass** | `evidence/isaac/`、`evidence/mujoco/` 含逐例 JSON；门槛与窗口在脚本内冻结 |
| C8 最小回放 | **pass** | `evidence/C8_REPLAY.md`；600 帧真实 rollout，`raw_actor_output` 逐帧记录，ONNX 重算 max_abs_err 5.96e-07、0/600 帧超差、无首次分歧；脚本 `runtime/r11_c8_dump.py` + `runtime/r11_c8_verify.py` |
| R0 收包 | **pass** | 权重与脚本已归档并校验 sha |
| R1 离线对齐 | **not_run** | 需 PYTHONPATH/仿真器环境，未出独立记录 |
| R2 sim2sim | **partial** | 双仿真均已跑，但**无同一候选双端同时通过** |
| R3 ARM64/CPRO 只读 | **not_run** | 需现场授权与设备 |
| R4 有界实机 | **not_run** | 需现场授权；**禁止以仿真替代** |

## 结论

本包**不满足验收门槛**（纯转与 Isaac 左弧不通过），仅作为研究交付：记录了双仿真分歧的机制、6 种已否证的修复路线，
以及一个在 Isaac 上可复现 +1 例（5/11 → 6/11）的有界陀螺偏航反馈包装。
