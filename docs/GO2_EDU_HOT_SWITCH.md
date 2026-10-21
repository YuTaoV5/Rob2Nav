# Go2 Edu：FIXSTAND 动作与 MoE 步态的统一热切换

本页是统一运行器的操作入口，详细设备准入见 [统一部署](UNIFIED_GO2_DEPLOYMENT.md)。截至本页修订，**没有 Go2 Edu 实机验收记录，仓库所有配置仍为 `hardware_qualified=false`**。键盘、状态机和训练合约适配可离线测试；不要把仿真通过、训练完成或修改资格布尔值当成实机可用。

## 权重与用途

| 按键 | 配置 | 权重/状态 | 控制模式 |
| --- | --- | --- | --- |
| 1 | `stand` | 标准四足站姿，未取得设备资格 | 单一低层控制器 |
| 2 | `walk_moe` | RoboGauge 项目原始 `go2_moe_cts_137k_0.6739.pt`，45维本体观测；同一权重用于平地、上/下楼梯，已在仓库固定 SHA256 | 50 Hz，关节偏移×0.25，Kp20/Kd0.5 |
| 3 | `sideflip69` | 当前发布权重未满足新的 FIXSTAND 首尾目标；新权重训练/评估后需更换 | 69维模仿，参考残差两端、原始输出中段 |
| 4 | `backflip60` | 当前发布权重未满足新的 FIXSTAND 首尾目标；新权重训练/评估后需更换 | 60维观测，延迟起跳控制 |
| 5 | `upright52` | 预留键；只有训练通过并新增带哈希的配置后才可选 | 52维观测，首尾回站姿残差 |

按用户选定的路线，平地与楼梯**持续使用同一份原始 MoE 权重**，不创建楼梯专用权重切换。上游 RoboGauge 的 `go2_moe.flat`、`go2_moe.stairs_fd`、`go2_moe.stairs_bd` 都引用 `Go2MoETerrainConfig` 的这份模型；这说明模型接口相同，不等于已验证 Go2 Edu 上任意楼梯的通过率。本仓库另有 2304 维感知扩展楼梯模型及 [独立部署合约](../skills/stairs/DEPLOYMENT.md)，它不参与本方案，也不能误装进45维 `walk_moe` 进程。

## 从控制主机连接和操作

机载电脑与 GPU 训练服务器不是同一设备。先从 Go2 Edu 的实际网络配置获取机载电脑的地址、账户和机器人侧网卡；在控制主机上执行 `ip -br addr`、`ip route`，按已配置的凭据用 `ssh <onboard-user>@<onboard-ip>` 登录。`--interface` 填控制程序连接机器人 DDS 的本地物理网卡。安装固定 SDK2 Python 提交及仓库依赖，核对机器人关节/IMU/足力校准、模型和独占低层控制。参见 [统一部署](UNIFIED_GO2_DEPLOYMENT.md) 的设备资格记录说明。

先离线看配置，再验证未接电机的门禁：

```bash
python -m rob2nav.deployment profiles
python -m rob2nav.deployment plan --from walk_moe --to stand
python -m rob2nav.deployment run --profile stand --interface <robot-nic> \
  --sdk-root <pinned-sdk> --calibration <device-calibration.json> \
  --qualification <device-qualification.json> --model <verified-go2.xml> \
  --output outputs/go2edu_preflight
```

只有实际设备资格链完成后，才使用同一个 Linux 本地终端入口：

```bash
python -m rob2nav.deployment.teleop -- run --profile stand \
  --interface <robot-nic> --sdk-root <pinned-sdk> \
  --calibration <device-calibration.json> --qualification <device-qualification.json> \
  --model <verified-go2.xml> --output outputs/go2edu_teleop_001 --enable-motors
```

`W/S` 前后、`A/D` 横移、`Q/E` 转向，松键约0.18秒归零；空格立即请求零速度。`1` 站立、`2` 普通 MoE、`3` 侧翻、`4` 后翻、`5` 未来直立；原子技能先装载/预热/进入 `ARMED`，再按 `T` 开始。`Esc` 停止。状态每秒打印 `phase/active/pending/stage_error/fault`；未具资格、缺少配置或旧权重无法满足合约时会拒绝切换。键盘仅提交意图，电机仍由唯一 500 Hz 服务持有。

## 长任务状态机与交接边界

```text
CAPTURE → NEUTRAL → READY → ENTRY → WARMUP → BLEND → ACTIVE (步态)
                                             └→ ARMED → T → ACTIVE (原子动作)
ACTIVE → [满足完整动作结束和稳定站立] → ENTRY 或 EXIT → ENTRY
任一故障 → FAULT → 阻尼停机
```

同一技能连续片段只调整速度和驻留时间；平地→楼梯→平地持续保持 `walk_moe`，不会重载模型、清空策略历史或执行站立过渡。仿真中的 `stand ↔ walk_moe` 等已测组合可省略重复 `EXIT → NEUTRAL`；Go2 Edu 仅在设备资格记录 `compact_switch_pairs` 精确列出旧/新完整合约摘要时启用这条短路径，否则走完整恢复入口。短路径仍保留实际关节姿态、增益收敛、支撑和策略历史预热检查。原子动作不能中途抢切，落地和回站姿才允许离开。

更新统一运行器代码会改变 `runtime_bundle_sha256`，修改策略字段会改变完整合约摘要；已有设备资格记录必须按新版本重新生成，不可沿用旧摘要。`compact_switch_pairs` 初始为空，仅为现场确实做过该方向切换验收的有序旧/新摘要对添加记录。

[平地→楼梯→平地示例](../deployment/sequences/moe_flat_stairs_flat.json)连续使用三个 `walk_moe` 步骤。驻留秒数只是演示占位，不能代替真实路径感知；越过楼梯后若要切到站立或特技，须在有水平交接区域的位置发出切换请求。

**倾角要分清用途。** RoboGauge 原始 MuJoCo 配置的 `projected_gravity_rad=2.5` 是其评估截断；本仓库另训的感知楼梯任务用 `bad_orientation(limit_angle=1.0)` 作为跌倒终止，两者并非同一个策略的热切换阈值。统一热切换的四足站稳判据更严，默认要求 `gravity_z < -0.98`，原子入口还核对与约定重力方向的差 `<0.12`。不能把 1.0 或 2.5 rad 当成切换许可。长任务在计划离开地形步态时，若机身仍倾斜，就继续当前步态指令，直到重力向量距水平站姿 `<0.18`、角速度 `<0.35 rad/s`、至少三足接触，才开始交接；接着核心再等真正低速且四足站稳。每段可设置 `handover_extension_s`（默认60秒，上限300秒），到期报错、归零，避免永久挂起。空翻入口必须在水平、稳定、符合该动作起始姿态的位置，不能从台阶中间起跳。

## FIXSTAND 模型接入要求

训练导出后核对 `policy.json`、`fixstand.json`、模型和参考轨迹哈希，生成新 `deployment/skills/<id>.json`。`action_parameterization` 必须与训练一致：侧翻 `reference_residual_v2` 需要 `reference`、`residual_limit_rad`、`bridge_residual_limit_rad`；后翻 `backflip_entry_v2` 或低扰动 `backflip_entry_v3`（后者还需 `entry_hold_scale`）；直立 `upright_endpoint_v2` 或收紧首尾残差的 `upright_endpoint_v3`（后者还需 `endpoint_residual_limit_rad`），观测都是 `upright52`。每个新模型先做四脚标准站姿起点、动作完成、回到标准站姿及独立种子复核，再以统一运行器验证 `stand → 动作 → stand` 和 MoE 长任务切换。模型未通过时保持 `simulation_enabled=false` / `hardware_qualified=false`，网页和实机都不得展示为已验收。

离线回归：

```bash
python -m unittest tests.test_fixstand_deployment tests.test_deployment_core -q
python -m rob2nav.deployment.verify_mujoco --root . --asset source \
  --sequence <reviewed-sequence.json> --atomic-cycle --max-seconds 300 \
  --output outputs/fixstand_switch_review --seed 9315
```

这个 MuJoCo 命令仅在所引用配置可运行且验收条件已满足时才是有效验证；它不会证明 Go2 Edu 的传感器、马达或地形适配完成。
