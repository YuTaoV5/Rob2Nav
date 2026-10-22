# Go2 Edu空翻仿真与实机部署手册

## 1. 目标与当前边界

本手册将下载模型、仿真验收、机载环境与统一控制串成一条路径。网站单技能演示、统一热切换仿真、实机验收是三个阶段，不能相互代替。后翻已有v8仿真记录，侧翻及前翻仍有未解决的完整动作/物理约束问题；没有本项目Go2 Edu实机验收记录。实机落地不稳定反馈不能因仿真通过而关闭。

当前结论与模型编号请读[后翻参考连续性与踏步验收记录](后翻参考连续性与踏步验收记录.md)、[空翻与直立研究目标和证据](空翻与直立研究目标和证据.md)，并核对网站模型卡显示的实际哈希。旧v8侧翻/v6后翻候选保存在[历史归档](../artifacts/fixstand_best_flips_20261022/README.md)，不再称其为网站当前最佳。

## 2. 环境与前置条件

先完成[环境准备与路径约定](环境准备与路径约定.md)，在仓库根目录执行`source scripts/docs_env.sh`。仿真使用Linux x86_64主机的`CPU_PY`；训练使用`ISAAC_PY`；Go2机载电脑按自身ARM/x86架构另建环境，不拷贝主机venv。

在操作者电脑设置本机参数：`SSH_HOST`是机载IP、`SSH_USER`是设备账号、`SSH_PORT`是SSH端口。用`export 变量名=实际值`赋值，认证交互输入，不放入脚本或仓库：

```bash
ssh -p "${SSH_PORT:?请设置机载SSH端口}" "${SSH_USER:?请设置机载账号}@${SSH_HOST:?请设置机载IP}"
```

登录后运行`uname -m`、`cat /etc/os-release`、`ip -br addr`和`ip route`，记录架构、系统与机器人侧有线网卡。机器人侧网卡是机载电脑连接DDS的接口，不是Windows网卡或SSH隧道。

## 3. 准备模型、SDK与控制合约

先在仿真主机按[后翻仿真](../skills/backflip/仿真手册.md)、[侧翻仿真](../skills/sideflip/仿真手册.md)或[前翻仿真](../skills/frontflip/仿真手册.md)重放固定模型。归档权重和元数据必须成套保存；下载后核对模型SHA256，不能只复制同名`policy.pt`。模型训练命令见各自训练手册，不在机载电脑运行Isaac。

SDK沿用仓库记录的固定提交。仅在尚未创建这些源码目录的机载环境执行；已有目录先检查版本。在机载电脑克隆本项目、source环境后：

```bash
mkdir -p "$ROB2NAV_UPSTREAMS"
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$ROB2NAV_UPSTREAMS/unitree_sdk2_python"
git -C "$ROB2NAV_UPSTREAMS/unitree_sdk2_python" checkout 814556d15970dd2ecf1c9984e845ca02ab07e206
git clone --branch 0.10.2 --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git "$ROB2NAV_UPSTREAMS/cyclonedds-0.10.2"
cmake -S "$ROB2NAV_UPSTREAMS/cyclonedds-0.10.2" -B "$ROB2NAV_UPSTREAMS/cyclonedds-0.10.2/build" \
  -DCMAKE_INSTALL_PREFIX="$HOME/tools/cyclonedds-0.10.2"
cmake --build "$ROB2NAV_UPSTREAMS/cyclonedds-0.10.2/build" --parallel 2
cmake --install "$ROB2NAV_UPSTREAMS/cyclonedds-0.10.2/build"
export CYCLONEDDS_HOME="$HOME/tools/cyclonedds-0.10.2"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
```

需要Git、CMake与C编译器。然后创建设备专用Python环境，将`CPU_PY`指向其解释器，安装架构兼容的NumPy、策略推理依赖及SDK；不要在ROS系统Python中直接升级依赖。SDK安装方法以[固定源码README](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206)为准。以下检查不发送电机命令：

```bash
"$CPU_PY" -m pip install -e "$ROB2NAV_UPSTREAMS/unitree_sdk2_python"
"$CPU_PY" scripts/verify_sdk_codec.py --sdk-root "$ROB2NAV_UPSTREAMS/unitree_sdk2_python" \
  --output outputs/reproduction/go2edu_sdk_codec.json
"$CPU_PY" -m rob2nav.deployment --root "$ROB2NAV_ROOT" profiles
"$CPU_PY" -m rob2nav.deployment --root "$ROB2NAV_ROOT" plan --from stand --to walk_moe
```

策略逐项核对观测维数、关节名/顺序、动作历史、单位、频率、参数化、参考数据、Kp/Kd、截断和电机速度曲线。MoE是45维本体观测；侧翻69维；后翻60维。SDK是FR/FL/RR/RL逐腿，策略顺序按导出关节名重排。后翻必须核对`export.json.control_contract`、action_clip=8与真实入口YAML；不能用全关节同一限速或其他模型的PD替代。

## 4. 设备接入与按键操作

完整参数与状态机见[统一部署框架与状态机](统一部署框架与状态机.md)及[Go2 Edu按键热切换指南](Go2Edu按键热切换指南.md)。只有设备、模型和代码版本对应的标定/资格记录确实完成后，才运行电机入口。先设置`ROBOT_INTERFACE`、`SDK_ROOT`、`CALIBRATION_FILE`、`QUALIFICATION_FILE`、`MODEL_XML`为实测资源：

```bash
"$CPU_PY" -m rob2nav.deployment.teleop -- run --profile stand \
  --interface "${ROBOT_INTERFACE:?请设置机器人网卡}" --sdk-root "${SDK_ROOT:?请设置SDK路径}" \
  --calibration "${CALIBRATION_FILE:?请设置实际标定记录}" --qualification "${QUALIFICATION_FILE:?请设置实际资格记录}" \
  --model "${MODEL_XML:?请设置核对过的机器人模型}" --output outputs/go2edu_session_001 --enable-motors
```

当前未获设备资格的配置会拒绝运行，不能改布尔值冒充验收。去掉`--enable-motors`只是非连接说明，不会完成实机预检。

| 按键 | 含义 |
|---|---|
| 1 / 2 | 站立 / 原始MoE，平地与楼梯同一策略 |
| 3 / 4 / 5 | 请求侧翻 / 后翻 / 预留直立，是否可用由实际合约决定 |
| T | 触发已进入ARMED的原子动作 |
| W/S、A/D、Q/E | 前后、横移、转向；无速度键约0.18秒归零 |
| 空格 / Esc | 归零速度 / 停止服务 |

状态流为`CAPTURE → NEUTRAL → READY → ENTRY → WARMUP → BLEND → ACTIVE`；原子动作先进入`ARMED`等待T。动作结束并实际站稳后才交接，故障进入`FAULT`。精确版本/方向经过验证的短路径才省略重复过渡；所有电机写入归一个本地控制进程。

## 5. 输出与验收

先完成固定模型独立四种子的完整动作、首尾FIXSTAND、逐物理步URDF速度/限位和机身接地验收；侧翻还检查准备段抽动及左右方向。再在同一统一运行器验证`stand → 动作 → stand`和`MoE → 动作 → MoE`重复序列，保存完整原速录像。

实机另记录关节零点/符号、IMU与足力、固件、负载、DDS独占、推理年龄、发送周期、q/dq/tau和停止/恢复行为。模型、参考、SDK、标定、资格与完整运行时代码摘要一起归档。没有设备记录就标为未验证，不能由原项目视频推导。

## 6. 常见问题与下一步

模块不存在先核对解释器与仓库根目录；SDK导入失败先核对架构和CycloneDDS动态库；资格摘要不符先核对代码/模型是否更换。落地不稳定应保留实测日志，比对入口姿态、PD、动作历史、时延与摩擦/质量，不用增加任意等待时间掩盖问题。

训练与仿真细节回到对应技能的三本手册；历史候选记录用于解释失败，不作为新机器默认安装步骤。
