# 证据索引（armD 转向课程 → R4000 续训迁移）

每条结论 → 哪个证据文件 / 哪行源码 / 哪条命令产生。全部为本任务新增，未改动他人工作区文件。

## A. 起点身份

| 结论 | 证据 | 复现命令 |
| --- | --- | --- |
| `model_3999.pt` 与现场记录逐字节一致 | `evidence/selfcheck.json` → `checks[start_checkpoint_identity]`（sha256 `e46954b6…`，1,042,741 B） | `python scripts/selfcheck.py` |
| actor 65→16、critic 65→1、`iter=3999` | 同上（检查输出内联形状） | 同上 |
| 优化器状态存在且仅"结构可加载" | `evidence/selfcheck.json` → `checks[optimizer_policy]`；`configs/ab_curriculum.json` → `start.optimizer_*` | 同上 |
| A/B 均"仅权重"初始化（与 R4000 原入口一致） | 源码 `runs/rolling/filtered-4000-env4096-stage020.ovvRL3/source/train_as2w_isaac_robust.py` 第 104 行 `runner.load(..., load_cfg={"actor": True, "critic": True})` | 阅读源码 |

## B. armD 课程（实际生效值）

| 结论 | 证据 |
| --- | --- |
| 配比 turn .40 / arc .25 / hold .15 / straight .10 / brake .10 | `results/train/armD-turn65-1k-20261116-145902/launch.sh`（`AS2W_AB_*_FRACTION`）+ `resolved_config.txt` |
| 纯转 `sign*(0.05 或 0.10)`、弧线仅正向 `vx=+0.25, wz=±0.05`、直行 `U(-0.25,0.25)`、刹车 `±0.25` | 基础代码 `code/as2w_ab_curriculum.py` `_resample` |
| 真实驻留 U(8,12) s 全局生效 | `code/as2w_isaac_robust_env.py`（`AS2W_AB_TURN_DWELL_*` → `resampling_time_range`） |
| 分类别长驻留**从未实现**（打印值不得作证据） | `code/as2w_ab_curriculum.py` 仅打印；`resolved_config.txt` 的 KNOWN_DEFECT 自注 |
| 训练 gyro ON / 评测曾 OFF 且已修正 | `launch.sh`；armD 交付包 `evidence/GYRO_WAS_OFF_IN_EVAL.md` |

## C. R4000 标称课程（A 组）

| 结论 | 证据 |
| --- | --- |
| 分段结构 40% 基础 / 40% 混合 / 20% park | `source/as2w_isaac_teleop.py`（`kind` 与 `_resample_command` 的 `mixed` 掩码） |
| 混合内 pure_turn .40 / arc .15 / straight .45；wz=sign*(0.035+0.085u)；vx=sign*(0.03+0.17u) | `source/as2w_isaac_speed_range.py` `RollingSpeedCommands.sample_motion` |
| 真实驻留 U(1.1, 3.4) s | 同上 `configure_speed`（覆盖 `_all.py` 与 `teleop.py` 的更早赋值） |
| yaw 命令 0.88 惯性 + 速度衰减 | 同上 `_update_command` |
| 回合 23 s / 前 4 s 零指令 / park 7 s | `source/train_as2w_isaac_robust.py`（`--rolling-maneuvers` 分支） |
| 完整启动命令 | `runs/rolling/filtered-4000-env4096-stage020.ovvRL3/run.sh` |

## D. B 课程不变量（离线可验）

| 结论 | 证据 | 复现 |
| --- | --- | --- |
| 段/步占比、长转 2519 个、**每个长转后必接松杆**、松杆零指令、驻留区间、左右符号平衡、包络内 | `evidence/selfcheck.json` → `checks[b_schedule_statistics]`、`checks[b_schedule_invariants]`（含 `sign_balance` 数值） | `python scripts/selfcheck.py` |
| 模块默认值 = 配置声明值 | `checks[b_schedule_matches_config]` | 同上 |
| R4000 参考采样器仍在自身包络内（A/B 对比基线） | `checks[a_reference_envelope]`、`r4000_reference` | 同上 |
| `vx=0.25` 请求越界被拒、B 用 0.20 | `checks[requested_arc_vx_gate]`；`configs/ab_curriculum.json` → `B.requested_arc_vx_*` | 同上 |

## E. 动作语义与包装读回

| 结论 | 证据 |
| --- | --- |
| 16 个合同常量与快照源码一致（±6、0.35、0.18、0.086、16、kd=2、0.9 rad/s、5 Hz、0.66/0.25、0.8/±0.10/0.035、23/4/7 s、cap .20、pure-turn 0） | `evidence/selfcheck.json` → `checks[contract_from_source]`、`extracted_contract` |
| 限速换算自洽：`0.35 × 0.051429 = 0.018 rad/步 = 0.9 rad/s` | `checks[rate_limit_units]` |
| 90% 力矩由 `FRACTION × URDF effort` **推导**：54/54/81/13.5 N·m | `checks[torque_90pct_derived]` |
| 包装 12 项数值读回（直线四轮一致、`.18` motion 门控、开环转向补偿读回、正偏航=左轮后退、gyro 双向、±6 限幅） | `checks[wrapper_readback]` |
| B 安装不重写任何快照文件、不触碰观测/包装/滤波 | `checks[install_path_wiring]`、`checks[install_path_no_wrapper_touch]` |

## F. 预算与阶段闸门

| 结论 | 证据 |
| --- | --- |
| `4096 × 24 = 98,304` 转移/迭代；400 迭代 = 39,321,600 环境步 = R4000 的 10% | `checks[env_step_budget]`；`evidence/dry_run_A.json` `budget` |
| PPO 5 epochs × 4 minibatches、批量 24,576（从已安装 runner 配置**读取**） | `evidence/dry_run_*.json` → `budget.ppo_*`、`ppo_cfg_verified=true` |
| A/B 结构占比相同，仅混合分支内容不同 | `checks[ab_structure_shared]` |
| 未授权时拒跑；本阶段 `training_started=false` | `checks[phase_guard]`；`dry_run.py --run` 退出码 5 |
| 两组完整启动命令与 PYTHONPATH | `evidence/dry_run_A.json`、`evidence/dry_run_B.json` → `launch_command`、`pythonpath`、`env` |

## G. 复现顺序

```bash
cd /root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/turn-curriculum-migration
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH -u VIRTUAL_ENV -u PYTHONHOME /root/gpufree-data/mujoco-venv/bin/python"
$PY scripts/selfcheck.py                 # 写 evidence/selfcheck.json，必须 SELFCHECK_PASS
$PY scripts/dry_run.py --arm A           # 写 evidence/dry_run_A.json
$PY scripts/dry_run.py --arm B           # 写 evidence/dry_run_B.json
```

`evidence/*.json` 由上述命令生成，本索引描述的是生成后的内容，不是事后手写。
