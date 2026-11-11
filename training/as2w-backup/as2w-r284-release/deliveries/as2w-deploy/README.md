# AS2-W 策略部署

参考 `unitree_cpp_deploy` 的模型独立打包、观测重建、控制状态和 SDK2 通信分层。本目录提供 AS2-W 的 C++17 CPU 推理、关节与轮速目标适配、MuJoCo 闭环回放和 SDK2 只读探测。真机电机写入待对应固件的官方低层契约和实机标定确认。 [部署参考](https://github.com/wty-yy/unitree_cpp_deploy)

## 已转换的权重

| 模型 | 用途 |
| --- | --- |
| `recovery-r09` | 7000 轮翻正站稳对照 |
| `recovery-r12` | 当前保留的翻正与四轮站稳策略 |
| `recovery-r13` | 混合地形恢复实验 |
| `stairs-r14` | 盲爬楼基线 |
| `stairs-r17-2000`、`stairs-r17-3000` | 模仿初始化后的盲爬楼检查点 |

`models/manifest.json` 记录源检查点、SHA-256 和关节顺序。原始 PPO 权重及优化器在训练交接包。楼梯检查点仍是实验策略，最新权重尚未通过五级盲爬楼。R18 只有候选代码。历史踏步与后双轮回放依赖各自的残差观测和外部控制逻辑，未套用本目录的 58 维接口。

## 构建与回放

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

C++ 核心只用标准库。每份权重有 60 帧来自训练导出器的参考输入和目标，覆盖恢复接管边界和轮角回绕。另有控制器测试覆盖失联锁定、恢复速度来源缺失、传感器异常和电机顺序、方向及零位转换。

装好训练归档所列 MuJoCo、NumPy 与 PyTorch 依赖后，从同一个 `as2w-results` 目录导出或复跑：

```bash
python export_models.py /path/to/as2w-results models
python verify_mujoco.py build/as2w_deploy /path/to/as2w-results replay.json
```

闭环回放由 C++ 从原始观测生成动作与目标，MuJoCo 执行原来的物理模型、固定种子和通过门槛。Windows 下使用 `build/as2w_deploy.exe`。

本地 7 项 CTest 均通过。360 帧对照中动作和目标的最大绝对差为 1.24e-5。六个权重共复跑 96 场，95 场与原通过标记一致。R09 和 R13 均为 16/16。R12 为 15/16，原结果为 16/16。不同的一场是右侧倒地、种子 12002，连续站稳 0.98 s，低于 1 s 门槛。三个盲爬楼权重均为 0/16，R17 3000 轮的种子 27001 仍达到第二级承重。原始结果见 `mujoco-native-report.json`。

六个 `.onnx` 文件通过了 ONNX checker 与 48 帧 PyTorch actor 对照。安装 ONNX Python 包后，导出时附加 `--onnx` 可重建静态输入 `obs[1,58]` 与输出 `actions[1,16]` 的模型。ONNX 仅包含 actor，观测与电机目标映射仍由本目录的 AS2-W 契约负责。本地 C++ 闭环使用已验证的 `.bin` 后端。

## 控制接口

控制周期为 20 ms。顺序为四腿各一组髋、股、胫及四轮，每组内按 FL、FR、RL、RR 排列。使用标定后的绝对编码器弧度和角速度。IMU 输入为机体角速度及 body-to-world、wxyz 四元数。四轮角度按训练的 4π 周期回绕。原版恢复模型需要实时机体系线速度估计。恢复模型前 2 s 保持被动并更新动作历史。盲爬楼固定前向指令 0.4 m/s，线速度槽填零。

## 无机身线速度的恢复学生

`train_recovery_student.py` 让原检查点在训练仿真中使用真值线速度给出动作标签，学生只看本体观测。后续轮次用学生驱动仿真，在学生访问到的状态上继续收集教师标签。开发集用于选模，正式评估沿用原来的六组场景和 1 秒连续站稳门槛。

```bash
python train_recovery_student.py /path/to/as2w-results TEACHER.pt STUDENT.pt
python train_recovery_student.py /path/to/as2w-results TEACHER.pt STUDENT.pt \
    --evaluate student --report student.jsonl
python export_models.py /path/to/as2w-results recovery-students --onnx \
    --student-checkpoints STUDENT.pt
```

学生的 C++ 模型在内部将前三个线速度槽置零，沿用恢复动作历史与启用时序。学生 ONNX actor 的输入为 `obs[1,55]`，从机体系角速度开始，后续顺序与原 actor 一致。ONNX 调用方需按恢复协议重建观测、限幅动作并生成电机目标。

服务器复测使用 MuJoCo 3.17.0，目录包含选出的学生权重、正式场景与新种子的逐场结果、训练配置和 R12 备选的补测。 [学生模型与证据](recovery-students/selected/) · [配对评估汇总](recovery-students/selected/comparison.json)

下表为相同场景和验收门槛下的通过数，箭头从各自的有速度教师指向无速度学生。

| 模型 | 原 240 场 | 新种子 A（96 场） | 新种子 B（96 场） |
| --- | ---: | ---: | ---: |
| R09 | 230 → 229 | 78 → 77 | — |
| R12 主选 | 232 → 233 | 85 → 78 | 84 → 78 |
| R12 备选 | 232 → 228 | 85 → 85 | 84 → 82 |
| R13 | 232 → 234 | 89 → 88 | 91 → 87 |

R13 学生是替换原 R12 起身策略的首选仿真候选，在三组评估中分别多通过 2、3、3 场。真机通过率和碰撞峰值仍待验收。 [R13 无速度 ONNX](recovery-students/selected/recovery-r13-no-velocity.onnx)

腿使用位置 PD，轮使用速度 PD。腿的训练增益为 55/2，轮为 0/2，前馈扭矩为零。控制器在输入异常、状态包超过 50 ms、或超出单回合时限时锁回被动，重新开始需显式调用 `arm()`。此处输出的是模型坐标下的目标，低层发送端应使用对应固件定义的被动或制动语义和硬件限值。

## SDK2 与真机接入

公开 SDK2 的 As2 示例使用 `unitree_go::msg::dds_::SportModeState_` 读取 `rt/lf/sportmodestate`。仓库中的公开 As2 示例目录提供 Sport 与音频，没有可核对 AS2-W 电机索引和接管语义的低层例程。Go2-W 的 `unitree_go::LowState_` 只用于可配置的只读协议探测。 [As2 状态示例](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/as2/sport/as2_sport_state.cpp) · [Go2-W 低层示例](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2w/go2w_stand_example.cpp)

官方资料说明，关闭内置运控后真机无法读取 `SportModeState`。公开的 HG 和 Go2 低层消息都没有机身线速度字段。 [运控关闭后的消息行为](https://github.com/unitreerobotics/unitree_mujoco/blob/main/readme.md#supported-unitree-sdk2-messages) · [HG LowState](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/idl/hg/LowState_.hpp) · [Go2 LowState](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/idl/go2/LowState_.hpp)

```bash
cmake -S . -B build -DAS2W_SDK2_SHADOW=ON -DCMAKE_PREFIX_PATH=/path/to/sdk2/install
cmake --build build --parallel
build/as2w_sdk2_shadow sport eth0
build/as2w_sdk2_shadow low eth0 VERIFIED_LOWSTATE_TOPIC models/stairs-r17-3000.bin calibrated.map --imu-wxyz-body-to-world
```

第二条探测命令只在固件确认兼容该消息类型后使用。`calibrated.map` 按清单中的 16 个关节依次各写一行，每行是关节名、DDS 槽位、方向（+1 或 -1）、编码器零位（弧度）。槽位不可重复。转换为 `q_sim = sign * (q_sdk - offset)`，速度按同一方向转换。需实测关节正方向、零位和 IMU 安装方向。程序只订阅和打印预演目标。

实际发令前还需对应 AS2-W X/EDU 固件的 `LowCmd/LowState` 定义、16 轴索引、轮速语义、CRC、接管与急停接口。原版恢复策略另外需要在低层接管后持续可用的机体系速度估计。当前可选 SDK2 程序尚未在 Linux SDK 或实机上运行。

- [r284-500 平地 PPO 仿真预览](flat-r284-500-preview/README.md) · Isaac PPO、65 维 ONNX、rl_deploy、C++ 回放与视频。
