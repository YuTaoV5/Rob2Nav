# Go2 统一部署与权重热切换

## 快速定位：网页编排、仿真与实机

网页“部署框架 → 拖拽编排与长期仿真”将 `stand`、`walk_moe`、研究侧翻等配置拖入时间线，可调整驻留时间、前进速度和最多4轮循环。浏览器调用 `POST /api/live-skills/composition/start`，服务器生成受限的序列JSON，由 `rob2nav.deployment.verify_mujoco` 运行**真实自由机身MuJoCo动力学**，同一 `SkillSequence`/`UnifiedRuntime` 完成所有切换。实时JPEG与相机角度经只读帧/受限相机接口传输；停止按钮或页面失联8秒会停止进程。上限32步、300秒。侧翻必须显式启用“研究仿真”；训练指标、电机速度或落地失败会如实留在 `outputs/live_skills/<session>/evaluation/summary.json` 和 `process.log`。网页没有实机电机路由。

可离线复现同一编排：在网页导出 `unified-composition.json`，复制到仓库目录，运行：

```bash
python -m rob2nav.deployment.verify_mujoco --root . --asset source \
  --sequence unified-composition.json --atomic-cycle --max-seconds 300 \
  --output outputs/composition_001 --seed 9315 --video
```

`--atomic-cycle` 仅允许隔离中的研究原子动作在仿真里执行，不代表实机认证。此仿真所需的精确XML、16个可视网格和上游MIT许可现已放入 `deployment/models/`，侧翻参考数据为 `deployment/reference/sideflip.npz`；发布权重位于 `artifacts/release_20261021/weights/`，因此全新克隆可直接核对SHA并运行。模型原始来源为固定 `GO2_backflip` 提交 `954aa61ceb62c515689677f44a21e32109f6b62d`，XML SHA256为 `fb81da82a8b5949bc0a45d5bf3a5003507c53d731cd43caf690cf729d05479d3`；自定义XML可通过 `--source-model` 指定，但需重新进行模型指纹核对。组合需要从 `stand` 开始、以 `stand` 结束；原子动作显式 `trigger=true`。当前侧翻和步态循环在自由机身仿真有通过记录，但峰值关节速度超额定，**不能据此对Go2电机放行**。

## 从工作站连接 Go2 机载电脑

这里的GPU服务器 `120.209.70.195:30752` 是研究服务器，并非默认的Go2机载电脑地址。先通过设备提供的机载电脑登录方式/网络配置取得实际IP、用户名和网口，不在脚本中猜测固定口令或IP。将开发机或机载电脑接入Go2允许的有线网络；在运行控制程序的Linux机器上执行 `ip -br addr`、`ip route`，确认与机器人相连的接口名（示例 `enp2s0` 只是占位）。从工作站按已配置的设备凭据执行 `ssh <onboard-user>@<onboard-ip>`；在机载电脑上再确认 `ip -br addr`、SDK低层只读状态可见、时间同步和CPU/网络能力。若控制程序在外置Linux主机运行，SSH进入该主机，传给 `--interface` 的仍是**该主机与Go2相连的接口**，不是SSH虚拟接口。官方SDK示例也要求替换实际网卡名称并在低层控制前处理运动服务独占，见 [Unitree Python SDK说明](https://github.com/unitreerobotics/unitree_sdk2_python) 与 [Go2低层示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/go2/low_level/go2_stand_example.py)。

在控制主机的独立Python环境安装仓库依赖和固定提交 `814556d15970dd2ecf1c9984e845ca02ab07e206` 的 `unitree_sdk2_python`；把SDK checkout路径传给 `--sdk-root`。先运行 `python -m rob2nav.deployment profiles` 检查每个技能的 `hardware_qualified`/`blocker`，再用下面命令验证静态门禁：

```bash
python -m rob2nav.deployment run --profile stand --interface <robot-facing-nic> \
  --sdk-root <pinned-sdk-checkout> --calibration <device-calibration.json> \
  --qualification <device-qualification.json> --model <verified-go2.xml> \
  --output outputs/device_preflight_001
```

此命令**不初始化DDS也不发送电机命令**。仓库自带的 `calibration.example.json` 和 `qualification.example.json` 故意不可用作实机放行。要进行真实发送，需按下文准入链完成设备序列号/固件、12关节符号零位、足力和IMU标定、编译模型/COM、完整策略契约、独立看门狗、电源与温度限制、运动服务独占、支架验收，并由负责人员生成有期限的资格记录。当前发布的每份技能配置都为 `hardware_qualified=false`，所以即使加上 `--enable-motors` 也会在DDS初始化之前拒绝。不要把示例布尔值改为true当作验收。

资格完成后，受限键盘入口为：

```bash
python -m rob2nav.deployment.teleop -- run --profile <qualified-gait> \
  --interface <robot-facing-nic> --sdk-root <pinned-sdk-checkout> \
  --calibration <device-calibration.json> --qualification <device-qualification.json> \
  --model <verified-go2.xml> --output outputs/teleop_001 --enable-motors
```

该入口只在Linux本地终端工作。按住 `W/S` 前后、`A/D` 横移、`Q/E` 转向，空格请求零速度，Esc请求停止；按键停止约0.18秒后客户端自动发送零速度，控制器本身在0.25秒后也会把过期速度目标渐降为零。键盘只是给统一控制器发速度意图，不能绕开电机资格门。初次站立/步态只在支架、急停和独立看门狗就绪时执行；可先以不带 `--enable-motors` 的命令观察门禁。

实机服务的状态机为 `CAPTURE → NEUTRAL → READY → ENTRY → WARMUP → BLEND → ACTIVE`（步态），原子技能经 `WARMUP → ARMED → 显式trigger → ACTIVE`；切换时 `ACTIVE → EXIT → NEUTRAL → ENTRY`。任何状态时效、接触、姿态、热/电压、力矩或资格条件失效进入 `FAULT` 并请求阻尼停止。键盘入口只用于已获准的步态；编排序列可用 `run --sequence <json>` 交给同一个服务，但序列中的**每一份**技能合同仍需单独获得资格。当前没有任何发布的特技满足该条件。

实现位于 `rob2nav/deployment/`，配置位于 `deployment/skills/`。目标是同一个低层发送器完成步态切换，以及“步态 → 单次 flip → 落地恢复 → 步态”，避免分别启动多个部署程序争抢电机。当前只执行离线和 MuJoCo 验证；**没有初始化真实机器人DDS，没有发送实机电机指令。所有随仓库发布的配置均为 `hardware_qualified=false`。**

完整资源索引见 `artifacts/release_20261021/`，历史任务与权重状态见 `handover.md`。通过加载或编码检查不等于策略通过实机验收。侧翻选定模型的动作验证为8/8，但速度门限仍失败；后空翻和直立行走也不能直接放行实机。

## 1. 一个电机控制所有者

```mermaid
flowchart LR
    SDK[SDK LowState] --> CAL[符号/零位/IMU/足力校准]
    CAL --> OBS[状态与接触约束观测]
    OBS --> SERVO[统一500Hz控制核心]
    PRE[异步加载/哈希/维度检查] --> WORK[独立策略进程 25/50/100Hz]
    OBS --> WORK
    WORK --> BOX[带epoch和时间戳的目标]
    BOX --> SERVO
    SEQ[技能编排/速度命令] --> SERVO
    SERVO --> LIMIT[PD/承重补偿/执行器包络]
    LIMIT --> CODEC[唯一LowCmd编码器与CRC]
    CODEC --> DDS[唯一SDK发布器]
```

SDK电机顺序固定为FR、FL、RR、RL，每腿hip/thigh/calf。所有模型输入输出按关节**名称**重排，必须是12个关节的完整双射。关节符号、机械零位与SDK原始量的变换由设备校准文件处理，不假定所有导出器顺序相同。

`LowCmdCodec`使用固定SDK提交 `814556d15970dd2ecf1c9984e845ca02ab07e206` 的真实IDL及CRC。会话固定选择torque或hybrid_pd模式，不在热切换中变更。torque发送端 `kp=kd=0`；hybrid_pd发送原生目标/增益与扣除PD后的残余前馈，避免重复计算PD。原生策略保留原来的内环增益与频率契约。协议20个槽中只给Go2的前12个电机施加目标力矩，其余槽保持停止值和零力矩。

SDK运动服务与低层策略不能同时拥有控制。发送前检查 `MotionSwitcherClient.CheckMode`，有其他服务时拒绝；不会自动调用StandDown/ReleaseMode。主机文件锁只防止本框架的多个写入者，不能证明网络上不存在其他发布器；现场仍须完成运动服务及其他控制端的独占仲裁。

## 2. 热切换事务

```mermaid
stateDiagram-v2
    [*] --> CAPTURE
    CAPTURE --> NEUTRAL: 实测姿态与已有承重力矩接管
    NEUTRAL --> READY: 接触/姿态/速度稳定
    READY --> ENTRY: 候选已加载且契约通过
    ACTIVE --> ACTIVE: 旧策略继续运行，后台预加载候选
    ACTIVE --> EXIT: 当前技能可退出且支撑有效
    EXIT --> ENTRY: 已恢复中立站姿
    ENTRY --> WARMUP: 姿态与增益过渡完成
    WARMUP --> BLEND: 步态历史及输出连续有效
    WARMUP --> ARMED: 原子动作入口及预热有效
    ARMED --> ACTIVE: 显式trigger，不渐混首个翻转动作
    BLEND --> ACTIVE: 原子提交新epoch
    ACTIVE --> FAULT: 状态/时效/执行器保护触发
    ENTRY --> FAULT: 入口未在时限内达标
    WARMUP --> FAULT: 新策略不稳定或过期
```

1. **准备**：后台验证权重SHA256、关节顺序、输入契约、缩放、默认姿态、参考数据与频率，再在独立进程加载。执行一次丢弃的推理探针，然后重置模型内部历史。失败不替换当前策略。
2. **旧技能退出**：步态的速度目标按加速度上限逐渐归零。静止判据不只看指令为零，还看实测姿态、角速度、关节速度与接触。部分策略零速时仍踏步，步态允许经过验证的低速三足支撑窗口；必须有有效的速度估计与质心投影余量。没有这些量时不采用该回退条件。
3. **姿态和PD桥接**：捕获实际关节和当前力矩，先过渡到统一中立站姿，再进入新技能入口。使用五次时间曲线、目标速度/加速度限制、增益斜率限制及承重补偿。不能只把新默认角度与新PD一次写入。
4. **承重连续性**：捕获/匹配前一周期的输出力矩；站姿阶段利用接触和固定动力学模型估计支撑力矩，并采用有界、仅在有效支撑时更新的校正。承重模型正则强度以`support_model_gain`纳入完整技能契约：当前普通入口为0，侧翻研究入口为2；低Kp站姿不能持续被错误名义载荷拉回，严格翻转入口也不能直接沿用未验证的纯关节积分器。新学习策略渐入时逐步退出额外承重偏置，避免把站姿积分器叠加到原策略之上。
5. **历史预热**：步态上一动作按当前实际施加的目标在新模型默认姿态/缩放下重新表示；原子动作按源训练重置契约清零逻辑动作历史，预热后进入ARMED，显式trigger才开始相位；不沿用另一个策略的裸12维动作。新策略在当前真实观测下预热，切换时不重置机器人根状态。ONNX/TorchScript各自使用对应观测适配器。
6. **提交**：只在渐入完成后更新当前策略。旧epoch或迟到响应被拒绝。保留有限的非活动模型缓存，便于flip后回到步态；缓存复用前仍重置并重新预热，清理进程在控制循环外完成。

切换阶段的默认限值为：目标参考速度2rad/s、加速度20rad/s²；Kp变化15/s、Kd变化2/s；力矩变化150N·m/s，即500Hz时每周期最多0.30N·m。这些是当前仿真桥接配置，不是对任意Go2或任意特技的通用认证参数。

**学习策略正式执行后保留其原生PD语义。** 不把为切换设计的低通滤波器永久套在所有步态/flip动作上，否则会改变训练分布甚至消除步态。原生动作阶段仍受执行器包络约束，但动态动作自己的快速力矩变化与“切换瞬间的冲击”分别统计。新策略激活边界额外保留短暂力矩连续性保护。

## 3. flip后恢复步态

flip标记为atomic。步态返回请求可以提前预加载、排队，但不会因为收到新权重就中断腾空动作，也不会在空中把两个网络输出简单插值。

参考相位只在真正激活后前进。到达动作末尾后继续由该策略恢复和站稳；只有动作时长/恢复窗结束且实际四足接触、姿态和速度全部合格，才进入EXIT。随后捕获落地后的实测姿态和力矩，通过中立姿态、步态入口、历史预热和渐入恢复步态。落地失败、姿态倒置或迟迟不能恢复时保持故障/拒绝切换，不能强行插值到“站立角度”。

侧翻入口是经核对的预策略实测关节姿态与重力方向，与训练默认角度、参考首帧分别记录。参考姿态、相位、yaw对齐、数据哈希与网络权重必须配套。候选加载期间旧步态可能转向，故yaw在实际入口预热时重新锁定，不能使用开始加载时的旧yaw。当前侧翻使用native_pd研究路径进行连续编排试验；它不解除速度约束失败带来的实机隔离。

直立双足行走的退出需要独立验证的降身/四足恢复控制器。当前没有通过验证的退出器，因此明确拒绝该技能进入SDK热切换，不对倾斜的机器人直接插值四足站姿。

## 4. 观测与策略契约

| 适配器 | 输入与状态 | 部署状态 |
|---|---|---|
| robogauge45 | gyro×0.25、重力、缩放速度、相对默认q、dq×0.05、上一动作；模型自身历史 | MLP/MoE仿真热切换 |
| backflip60 | gyro、重力、q/dq、两步动作历史、6维相位编码 | 可离线加载；严格动作/电机验收失败，默认隔离 |
| mimic69 | 参考q/dq、锚点方向、gyro、q/dq、动作历史 | 固定参考和SHA必需；默认隔离，可显式仿真研究 |
| upright48 | **机身坐标COM线速度**、gyro、重力、指令、q/dq、上一动作 | 不能用零向量冒充LowState未提供的线速度；退出控制器未验证 |
| stand | 显式站姿、PD及支撑补偿 | 用于接管/中立桥/停步 |

Isaac感知地形检查点还包含专用感知/记忆/融合契约，不能当成45维TorchScript直接加载。它们已归档，训练与仿真手册保留；在补齐并验证适配器前不接入SDK执行器。导航/SCAN输出的是有时效的速度意图，由当前步态执行，不再创建另一个LowCmd发布器。托爪控制也须经同一个执行器所有权接口接入，其接触控制不等价于一个可替换的普通12维网络。

## 5. 状态估计、保护与设备准入

`KinematicObserver`使用实测q/dq、IMU与足端力，通过固定Go2模型的FK/Jacobian得到接触约束速度和静态支撑多边形余量。多个接触足给出的速度不一致时拒绝估计；不虚构全局位置。共同滑移无法仅凭这些量观测出来，因此低摩擦、坡面、足力标定和载荷/COM变化必须单独验证。模型和校准SHA是设备准入的一部分。模型SHA覆盖编译后的完整MuJoCo模型、版本和根XML，包含被include引入的质量/COM等参数；只核对根XML不够。

默认保护包括状态年龄30ms、策略响应年龄60ms、控制间隔超过10ms、重复SDK tick、非有限值、关节/速度包络、入口和预热超时、参考/权重SHA错误及旧epoch。电压、温度、电机丢包及恢复阻尼采用设备资格文件中的已验参数，示例数值不代表已经标定。

实机发送必须同时满足：

- 设备序列号、LowState的原始version二元组、固件记录、SDK固定提交及实际导入包路径一致；机械符号/零位、IMU与足力校准已经核对。
- 整份技能契约获准，包括权重、默认姿态、PD、观测、缩放、相位与执行方式；只有权重SHA相同不够。
- 固定运动学/COM模型和控制代码bundle哈希一致，准入未过期。
- 独立机器人端看门狗、恢复阻尼和独占控制已在实际设备验证。

资格文件是受控运维记录，不是密码学认证。所有仓库示例都故意不可启用实机。故障时优先保留有界本地控制；状态丢失或主机/网络彻底失效时，Python进程本身无法保证电机行为，必须依赖已验证的设备端保护。

Python/普通Linux与DDS不能据本项目测试宣称硬实时500Hz。策略进程、加载、日志与网页被移出控制核心；但实际设备仍须完成CPU隔离、调度、通信与尾延迟验收。离线CRC耗时和MuJoCo渲染循环的耗时分别记录，不用平均帧率代替最大延迟。

## 6. 离线使用和验证

```bash
cd /root/gpufree-data/Rob2Nav
PY=/root/gpufree-data/envs/go2-mapless/bin/python
$PY scripts/verify_release_resources.py
$PY -m rob2nav.deployment profiles
$PY -m rob2nav.deployment plan --from walk_moe --to sideflip69
$PY -m unittest discover -s tests -p test_deployment_core.py -v
$PY -m rob2nav.deployment.verify_mujoco --output outputs/unified_switch_new --seed 9401 --video
```

侧翻研究编排要求按[侧翻训练手册](../skills/sideflip/TRAINING.md)从固定源生成参考，把经核对的reference.npz放入 `deployment/reference/sideflip.npz`。SHA必须为 `12877865c884becf477077a750fe1db3cdb01270c34a09ab35661d36ec375981`。该外部参考不混入SDK，也不改变真实机器人姿态。

```bash
$PY -m rob2nav.deployment.verify_mujoco --asset source --atomic-cycle \
 --output outputs/unified_atomic_new --seed 9402 --video
```

SDK独立环境使用Python3.11、NumPy1.26.4、CycloneDDS0.10.2及上述SDK提交。CycloneDDS C库0.10.2安装在独立前缀，避免覆盖ROS环境。真实IDL/CRC离线检查：

```bash
CYCLONEDDS_HOME=/root/gpufree-data/tools/cyclonedds-0.10.2 \
LD_LIBRARY_PATH=/root/gpufree-data/tools/cyclonedds-0.10.2/lib \
 /root/gpufree-data/envs/go2-sdk-check/bin/python scripts/verify_sdk_codec.py \
 --sdk-root /root/gpufree-data/Rob2Nav-upstreams/unitree_sdk2_python \
 --output outputs/sdk_codec_new.json
```

`deployment/calibration.example.json` 与 `qualification.example.json`仅说明字段。CLI `run`默认只给出未启用状态；只有显式 `--enable-motors` 且整个设备资格链有效才会进入连接流程。当前配置会在连接前拒绝，不提供绕过资格的force开关。速度命令超过250ms未续期则归零。侧翻入口使用记录的预策略实测关节姿态与重力方向，不能把默认姿态或参考首帧混为实际入口。接触力采用8N接通/5N断开的迟滞，仍须设备标定。现场资格齐备后，服务使用标准输入JSON接受 `{"switch":"walk_moe"}`、`{"velocity":[0.2,0,0]}`、`{"stop":true}`、`{"trigger":true}`，网页不直接承担实时电机控制。

## 7. 验证证据

最新证据见[统一部署归档](../artifacts/unified_deployment_20261021/README.md)和[逐项验收记录](../artifacts/unified_deployment_20261021/release_acceptance.json)。v12/v13五配置切换两轮通过；共用序列器在0°、+30°、-30°三个固定朝向均完成stand→MoE→sideflip→MoE→stand。交接段命令力矩步长≤0.300001Nm、Kp步长≤0.030001；v13普通切换最大倾斜约4.353°。这不是随机地形/载荷鲁棒性证明。

三次翻转最高速度比约1.0842、1.0851、1.0993，仍不通过额定速度验收；实机资格全部保持false。保留v10/v11失败、旧v9成功与v13确切源码、逐周期NPZ和视频。v14连续两次翻转返回通过（7次激活、总滚转-12.7139rad），当前全部部署模块与v14源码AST一致。缓存最多保留当前、候选和一个非活动记录，淘汰的参考NPZ及时关闭；18项核心测试含资源生命周期回归。核心18项、工作台62项、真实SDK IDL/CRC及模型指纹检查通过。SDK编码p99约0.174ms、最大0.296ms，仅为离线编码微基准，不证明DDS或整环硬实时。

连续返回示例见`deployment/sequences/repeated_flip_return.json`；仿真使用`--max-seconds 120`提供完整序列预算，不放宽单次入口/退出时限。该回归最高速度比1.08440，仍不获实机资格。

## 8. 共用序列器与设备准备

`deployment/sequences/flip_return.json`显式描述初始站姿、步态、原子动作触发、返回步态和最终站姿。`SkillSequence`由仿真与设备服务共用；不直接拥有电机发布器，原子动作结束条件仍由控制核心执行。序列期间只接受人工stop覆盖，避免人工速度与序列争抢。追加最终stand，拒绝不带显式trigger的原子步骤及超范围速度。

```bash
# 在与既有CPU仿真相同的Python 3.11环境中运行；准备外部参考及源资产后：
$PY -m rob2nav.deployment.verify_mujoco --asset source --atomic-cycle \
 --sequence deployment/sequences/flip_return.json --initial-yaw-deg 30 \
 --output outputs/unified_sequence_new --seed 9501 --video

# 设备服务预检示例：没有 --enable-motors，不打开DDS、不发电机命令。
$PY -m rob2nav.deployment run --profile stand --interface eth0 \
 --sdk-root /path/to/pinned/unitree_sdk2_python \
 --calibration deployment/calibration.example.json \
 --qualification deployment/qualification.example.json \
 --model /path/to/go2_isaac.xml --output outputs/device_session_new \
 --sequence deployment/sequences/flip_return.json
```

从仓库根目录执行，CPU依赖见`requirements.txt`和`skills/requirements-stunts-cpu.txt`；SDK/CycloneDDS需在独立前缀安装固定版本，不覆盖ROS。源Go2 XML路径和外部参考准备见侧翻训练手册。仿真研究允许动作候选运行，不解除设备门禁。当前示例即使加上电机启用参数也会在连接前拒绝。

设备接入依次完成：离线关节/IMU/足力标定与完整模型核对；独立看门狗/运动服务独占验证；支架上低力矩接管与PD变化验证；四足站姿和低速步态切换；最后仅对电机包络已通过的动态权重做受保护动作验收。每阶段保存实测时延、丢包、力矩/速度/接触数据；失败不推进资格。现有特技候选没有通过该设备验收链，不把仿真通过当作实机认证。

官方接口依据：[固定SDK低层示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/go2/low_level/go2_stand_example.py)。本实现复用了其协议接口，未执行该示例或其运动服务接管动作。
