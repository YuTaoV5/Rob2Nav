# armD 转向课程 → R4000 续训迁移

> **阶段二已启动（2026-11-17 01:12 操作员指令「为啥还不启动」）**。阶段一交付物保持原样，演练与自检仍可独立复跑。
>
> - 授权记录写在 `configs/ab_curriculum.json` → `status.authorisation`（含原话与时间）；`training_authorised=true`、`phase=stage2_running`
> - 顺序：R4000 起点原评测入口 → A → B（`set -e` 链，A 成功才跑 B）
> - 实测：**16000 env 显存 19,977 MiB / 46,068 MiB**、利用率 91%，无 OOM；**2.00 iteration/min**，每臂约 **24.9 小时**，两臂约 50 小时
> - 运行期状态与 manifest 在 `work/AB-run/`（`work/` 不入库），`STATUS.md` 每次恢复先核真实进程与产物
> - **起点基线（自身评测入口，24 env，两个留出种子）**：0.20 m/s 前进 track 9–10/12、track_brake_hold 5/12、平均 0.173–0.174 m/s；倒车 10/12、4–9/12、0.169–0.171 m/s；**左转/右转/左弧/右弧在两种子上均 0/6 通过**（有符号偏航 左转 +0.052/+0.068、右转 −0.050/−0.046、左弧 +0.041/+0.041、右弧 −0.038/−0.046），终止 0–2 条/类
> - **起点已存在的轮端上限**：实测轮差交付仅 **0.30–0.48**（下发差分只兑现三到五成），这是转向失效链条里"带载轮速"一环的既有天花板，课程迁移只能在它之上挪动
> - 阶段二对照脚本 `scripts/stage2_compare.py`：只消费官方评测输出、不重新打分；给出左/右分别与**最差侧**归约，并明确标注首秒/3–5 s/末 3 s 与"最长连续卸载"需要逐帧记录器
> - 仍保持 `research / deployable: false`，不做实机

- **状态**：`research / deployable: false`，`phase: stage2_running`，`training_started: true`
- **入口**：`scripts/dry_run.py`（默认 dry-run，只有显式 `--run` 才训练，且需先过自检与授权位）

## 1. 交付物（绝对路径）

根目录 `/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/turn-curriculum-migration/`

| 文件 | 作用 |
| --- | --- |
| `configs/ab_curriculum.json` | 单一事实源：起点、合同、共享预算、A 课程、B 课程、差异清单、越界请求处理 |
| `scripts/turn_schedule.py` | B 课程决策核心（纯 Python，可离线运行，被自检直接验证） |
| `scripts/b_adapter.py` | 环境适配器：把 `TurnSchedule` 的决策映射到每环境张量，不新增采样逻辑 |
| `scripts/train_ab_entry.py` | **两臂共用入口**：import 钩子在仿真启动后强制 `num_steps_per_env`，B 另重绑定 `RollingSpeedCommands`，再用 runpy 跑**未改动的** R4000 入口 |
| `scripts/selfcheck.py` | 离线自检（18 项全过才允许启动） |
| `scripts/replay_contract.py` | 离线回放：状态序列、滤波复位、0.018 rad/步、驻留截断归因 |
| `scripts/dry_run.py` | dry-run 打印 + 受控启动器 |
| `evidence/selfcheck.json` | 自检结果（18 项）与从源码抽取的合同事实 |
| `evidence/replay.json` | 离线状态序列回放与驻留请求/实际/截断统计 |
| `evidence/dry_run_A.json`、`evidence/dry_run_B.json` | 两组 dry-run 记录（含完整启动命令） |
| `evidence/EVIDENCE_INDEX.md` | 证据索引：每条结论对应哪个文件/行/命令 |

## 2. 起点锁定（R4000）

- 文件：`deliveries/as2w-deploy/rolling-4000-stage020-research/weights/model_3999.pt`
- SHA256 **`e46954b697ae8c4de2530ebae8f7c520f8f222bdd6a117195ff57bc4d3327769`**、1,042,741 B —— 与现场记录**逐字节一致**（自检 `start_checkpoint_identity`）
- 结构：`iter=3999`；actor `mlp.0.weight (128,65) → mlp.6.weight (16,128)`；critic `(128,65) → 1`；命名与当前 runner 的 `mlp.*` 一致
- 优化器：`optimizer_state_dict` **存在**，17 个参数项、1 个 param group，`exp_avg` 形状与 actor+critic 参数一一对应，**结构上可加载**
- **决策：A/B 两组都只加载模型参数，不恢复优化器与迭代数。** 理由：R4000 的原始入口 `train_as2w_isaac_robust.py` 自己就是 `runner.load(..., load_cfg={"actor": True, "critic": True})`，既不读优化器也不读 `iter`；两组沿用同一恢复方式即无差异，也不需要发明新机制。此决定写在配置 `start.optimizer_*` 与自检 `optimizer_policy` 中。

## 3. armD 侧：转向课程的**实际生效**事实（剔除了无效证据）

来源：`results/train/armD-turn65-1k-20261116-145902/launch.sh` + `resolved_config.txt` + 基础代码 `as2w_ab_curriculum.py` + `as2w_isaac_robust_env.py`。

| 项 | 实际值 | 证据 |
| --- | --- | --- |
| 分段配比 | turn 0.40 / arc 0.25 / hold 0.15 / straight 0.10 / brake 0.10（转向合计 0.65） | `launch.sh` 的 `AS2W_AB_*_FRACTION` |
| 纯转幅值 | `sign * (0.05 或 0.10)`，符号与幅值各 50/50 | `as2w_ab_curriculum.py` `_resample` |
| 弧线 | **仅正向** `vx=+0.25`、`wz=±0.05` 50/50 | 同上（`cmd[m,0]=0.25`） |
| 直行 | `vx ~ U(-0.25,+0.25)`、`vy=0` | 同上 |
| 刹停 | `vx=±0.25` 50/50，松杆归零由基类 `_update_command` 的 lerp 完成 | 同上 + `FlatCommands._update_command` |
| 真实驻留 | **U(8,12) s，且对所有类别一律生效** | `as2w_isaac_robust_env.py`：`resampling_time_range=(lo,hi)` 取 `AS2W_AB_TURN_DWELL_MIN/MAX_S` |
| 训练时 gyro | ON：kp=60、cap=4、ff_gain=0 | `launch.sh` 的 `AS2W_GYRO_*` |
| 评测时 gyro | **曾为 OFF（缺陷）**，修正后单种子 8/11 | `evidence/GYRO_WAS_OFF_IN_EVAL.md`（armD 交付包） |

**无效证据剔除**：`as2w_ab_curriculum.py` 里 `AS2W_AB_TURN_DWELL_*` 只被**打印**、未被该模块用于分类别驻留；`armD` 的 `resolved_config.txt` 也自注了这条 KNOWN_DEFECT。因此「按类别给转向单独长驻留」在 armD 中**从未实现**；真实驻留是**全局** 8–12 s。本迁移据此设计，不把打印值当证据。

## 4. R4000 侧：A 组标称课程（A 组的全部课程内容）

来源：运行快照 `runs/rolling/filtered-4000-env4096-stage020.ovvRL3/source/`。

- 分段结构（`as2w_isaac_teleop.py`，`kind=U(0,1)` 按回合抽取）：`kind<0.40` 基础采样 40%、`0.40≤kind<0.80` 混合课程 40%、`kind≥0.80` park/回站 20%
- 混合分支内（`as2w_isaac_speed_range.py` `RollingSpeedCommands.sample_motion`，`--rolling-low-speed-smooth`）：**pure_turn 0.40 / arc 0.15 / straight 0.45**
  - 纯转：`vx=0`、`wz=sign*(0.035+0.085u)` → |wz| ∈ [0.035, 0.120]，双向
  - 弧线：**仅正向** `vx=0.05`、`wz=sign*(0.035+0.085u)`
  - 直行：`vx=sign*(0.03+0.17u)` → |vx| ∈ [0.03, 0.20]
- 真实驻留：**U(1.1, 3.4) s，全局**（`configure_speed` 覆盖了更早的 `(1.8,3.)` 与 `(.65,2.6)`）
- yaw 命令惯性：`carried=0.88*previous`，并按平面速度 `yaw_scale=((0.18-|v|)/0.08).clamp(0,1)` 衰减
- 回合 23 s、前 4 s 零指令、park 7 s、`--pure-turn-common-scale 0`、种子 311、4096 env、lr 5e-5

## 5. B 组设计（只改课程）

| 项 | A（R4000 标称） | B（迁移） |
| --- | --- | --- |
| 混合分支配比 | turn .40 / arc .15 / straight .45 | **pure_turn .30 / turn_long .15 / arc .25 / straight .30** |
| 持续转向 | 无 | **新增 `turn_long`，驻留 U(8,12) s** |
| 松杆段 | 无（仅 park/刹车语义） | **每个 turn_long 之后强制零指令 U(3.0,3.5) s** |
| 弧线 | 仅正向 0.05 m/s | **双向**，`vx=0.20`，`wz=sign*{0.02,0.04,0.06}` |
| 纯转 | ±U(0.035,0.120)，短驻留 | 不变（短驻留保留，单独评测） |
| 基础/ park 分支 | 40% / 20% | **不变**（驻停、前后直行、刹停、回站样本覆盖不变） |

离线统计（`selfcheck.json` → `b_statistics`，20000 段）：段占比 arc .222 / pure_turn .263 / straight .264 / turn_long .126 / release .126；按策略步占比 turn_long 0.374、release 0.122、pure_turn 0.176、arc 0.149、straight 0.178。即**持续转向约占混合分支步数的 37%、整体步数约 15%**，与 A 的纯转步数占比相当，量级有界。

**越界请求已拒**：您指定优先覆盖 `vx=0.25 m/s` 弧线，但冻结的 R4000 档位 `speed_cap=0.20 m/s`（采样上限也是 0.20），0.25 会**同时扩大速度域**，与本轮"不同时扩大速度/偏航/扰动"冲突。因此 B 用 **0.20 m/s**（档内最大），0.25 记为 `out_of_range` 并留给后续单因素速度实验；自检 `requested_arc_vx_gate` 会在有人把 B 改成越界值时**拒跑**。

## 6. 动作语义核对（并行保留 R4000 语义）

自检从快照源码/启动脚本**抽取**并与配置比对，16 个常量全部一致（`contract_from_source`）：

| 项 | 值 | 来源 |
| --- | --- | --- |
| 策略周期 / 物理步长 / decimation | 0.02 s / 0.005 s / 4 | 快照与运行合同 |
| 动作限幅 | ±6 | `clip_actions=6.` |
| 轮动作尺度 | 16 rad/s（`wheel_target = vx/(0.086*16)`） | `rolling_fixed_exit.py` |
| 腿动作尺度（动作项） | 0.35 rad/单位 | 限速表达式除子 |
| 腿残差预缩放（包装内） | `.18 * motion` | `rolling_fixed_exit.py`（**注意：与 0.35 是两个不同量**） |
| 腿目标限速 | 0.9 rad/s = **0.018 rad/策略步** | `step_limit = .9*step_dt/.35`，自检换算 `0.35×0.051429=0.018` ✓ |
| 腿残差滤波 | 5 Hz 一阶（α=0.4665） | `filter_leg_residual` + `--rolling-leg-filter-hz 5` |
| 轮执行器 kd | 2.0（damping，stiffness 0） | `actuator(... damping=2.)` |
| 开环转向补偿 | `-.66*(1+.25*steering)*yaw_drive*side` | `rolling_fixed_exit.py` |
| 环境侧 gyro 修正 | `.8*(yaw_cmd-ωz)` 限 ±0.10，\|yaw\|≥0.035 接入 | 同上 |
| 力矩 90% | FRACTION=0.90 × URDF effort → hip/thigh 54、calf 81、wheel 13.5 N·m | `as2w_training_safety.py` + `as2w.urdf` |

**包装读回**（离线数值，`wrapper_readback`，12 项全过）：直线四个轮目标相等；`.18` 腿残差随 motion 门控；正 yaw 命令下左轮后退、右轮前进；gyro 过转减小转向、欠转增大转向；输出 ±6 限幅。

**历史合同 vs 现入口的首个分歧（未修，按评测标注）**：armD/r11 线用 `DelayedGyroVelocity` 动作项（`sp = dq + tau/kd`，kd=6、scale=16）+ 可开关的 gyro-1 前馈；R4000 用**环境内开环映射**（`rolling_fixed_exit`）+ PhysX `damping=2`，且 gyro 修正在 `--rolling-fixed-exit` 下**恒开**。两者在第 0 帧就会给出不同的轮目标，**策略权重不可互换**。因此本轮 A/B 都固定在 R4000 映射上；评测也必须用 R4000 的 `train_as2w_isaac_robust.py --evaluate`，不能沿用 armD 的验收脚本，否则 gyro 开关口径不同。

## 7. 复跑命令

```bash
cd /root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/turn-curriculum-migration
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH -u VIRTUAL_ENV -u PYTHONHOME /root/gpufree-data/mujoco-venv/bin/python"

# 1) 离线自检（16 项，全过才允许启动）
$PY scripts/selfcheck.py

# 2) dry-run：打印起点哈希/初始化方式/轮控合同/A-B 课程差异/PPO 批量/环境步预算
$PY scripts/dry_run.py --arm A
$PY scripts/dry_run.py --arm B

# 3) 仅在此后、且配置 status.training_authorised=true 时才会真正训练
$PY scripts/dry_run.py --arm A --run
$PY scripts/dry_run.py --arm B --run
```

离线状态序列回放：

```bash
$PY scripts/replay_contract.py     # 写 evidence/replay.json
```

dry-run 输出要点（按操作员 2026-11-17 指定预算）：`16000 × 128 = 2,048,000` 转移/迭代（128 由 `train_ab_entry.py` 显式强制；R4000 入口本身没有该开关，默认是 24，两臂强制值相同），3000 迭代 = **6,144,000,000 环境步**（R4000 的 **15.625 倍**）；PPO 5 epochs × 4 minibatches（**批量 512,000**，从已安装 runner 配置读取而非假设）。两组完全同预算。

**两点必须提前记录的偏离/风险**（不是缺陷，但会影响与 R4000 的可比性和启动可行性）：

- **PPO 批量形变**：每迭代 2,048,000 转移是 R4000 的 5.33 倍（R4000 为 4096×24=98,304）。两组共享同一批量，所以 A/B 对照仍然干净；但**与 R4000 原基线本身的横向比较要注明批量已变**。
- **16000 并行环境在单张 L40 44 GiB 上从未跑过**（本项目历史最大 4096）。启动前必须先测显存，分配失败时该臂**不得启动**；3000 迭代的总工作量是 R4000 的 15.625 倍，工期预期要用实测首轮吞吐推算，不能估算。

## 6b. 离线状态序列回放（fixed stand → 接管 → 运动 → 松杆 → Brake → 回站）

`scripts/replay_contract.py`（无仿真器，纯 torch/纯 Python）复刻 R4000 的腿流水线：5 Hz 一阶残差滤波（α=0.4665）→ 限速（`.9*dt/.35` 动作单位，×0.35）→ 关节目标；并按 `TakeoverCommands`（前 100 步零指令）、`as2w_motion_start_s=4`、`park_until_s=7`、命令滤波 lerp 0.12 与死区（0.008 m/s、0.024 rad/s）回放整条状态序列。自检结论：

| 项 | 结果 |
| --- | --- |
| 稳态腿目标每步变化 | **0.018000 rad/步**，等于 0.9 rad/s × 0.02 s 的限值（不是"小于"而是正好被限住） |
| 运动前 / 接管窗口内的非零命令 | 0 帧 |
| 松杆窗口 | 最长 11.24 s 零指令（脚本序列中的 release+brake+回站区间） |
| 滤波与限速状态复位 | `reset()` 后 `filtered/slewed/joint_offset` 全为 0，与 `_reset_idx` 一致 |

**驻留请求 vs 实际（23 s 回合，400 回合）**：安装段 `{straight 966, pure_turn 1262, arc 783, turn_long 211, release 211}`；**松杆被回合末尾截断 = 0**；普通末段被回合边界裁剪 275 次（预期行为，单独统计）；因"剩余回合时间装不下 dwell + 最大松杆"而**降级为短转 251 次**。

这里暴露一个必须报告的课程设计张力：**8–12 s 长转 + 3.5 s 松杆与 23 s 回合长度相冲突**，约一半长转抽样会被降级（实际约 0.53 个长转/回合）。本阶段的处理是**宁可降级也不截断**，从而保证"每个安装的长转后必有完整松杆"；若要让长转密度更高，需要**两臂同时**延长回合长度或缩短反应窗口——那是独立的一步，不能在 B 里偷偷改。

## 8. 训练授权后的评测协议（已写入配置，本阶段不执行）

同一入口评 **R4000 起点 / A / B**；起点直接加载 `model_3999.pt`，不用可能被后续 run 覆盖的 `model_0.pt`。固定留出种子与初态；分报接管、驻停、前后直行、前后刹停、回站；弧线与纯转按正负方向分别报。真实长转按首秒 / 3–5 s / 末 3 s 报带符号偏航、实际轮差、四轮支撑、最长连续卸载。先判基础动作与硬保护，再比转向；用**最差侧**与各留出种子选权重。单边提升、单种子提升、卸载后空转、终止后样本都**不能**作为扩训依据。若 B 失败，按"策略输出 → 最终下发 → 带载轮速 → 偏航与支撑"定位最早失效点，gyro/增益另立版本做单因素。

## 9. 待核验事项（本阶段无法闭合）

1. **`vx=0.25` 档**：需单独立项扩速度域，不能在 B 里顺带打开。
2. **40% 基础分支的实际分布**：`TakeoverCommands`/`GOALS` 作用于 `teleop_speed_cap` 的细节未逐帧核对，仅确认结构占比。
3. **回合事件级统计**：本阶段只做离线采样器统计，真实 rollout 的按回合/命令事件分布要在训练后从日志复核（届时必须能在真实 rollout 中看到 8–12 s 长转，而不是只在采样器里）。
4. **`--rolling-maneuvers` 的对称增强**（`mirror_flat` + mirror loss 0.15）对 B 的双向弧线是否等价，需在首轮日志确认镜像样本未引入左偏。
5. **实机 DDS 槽位/符号/零位**仍未映射（沿用既有结论），不影响仿真续训，但阻塞闭环。
6. **`FRACTION` 抽取**依赖 `as2w_training_safety.py` 的写法；若上游改写法，自检会变 SKIP 而不是假装通过。
7. **16000 env 显存预检**未做（本阶段禁止 GPU 压测）：启动前必须实测，分配失败则该臂不得启动。
8. **回合长度 vs 长转驻留**的张力（见 6b）：当前策略是降级而不是截断，长转实际密度约 0.53/回合；若要提高需两臂同时改回合长度，单独验证。

## 10. 停止条件

本阶段到此为止：不启动训练、不续训、不做实机。启动前需现场明确授权，并把 `configs/ab_curriculum.json` 的 `status.training_authorised` 置为 `true`——脚本会在未授权时**拒跑**（退出码 5）。
