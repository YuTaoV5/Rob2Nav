# AS2-W R152 平地仿真候选

R152 使用运动指令、IMU 和 16 轴关节反馈，按 50 Hz 输出 12 个腿关节位置目标和 4 个轮子角速度目标。部署不需要机身线速度话题。模型用于平地仿真和部署接口验证。楼梯、实机 DDS、电机映射及 ARM 板端运行仍待验证。

| MuJoCo 平地测试 | 跟踪与停车通过 | 同时满足接触和步态检查 |
| --- | ---: | ---: |
| 固定 52 例，ONNX | 44 | 21 |
| 独立 52 例，ONNX | 25 | 9 |

104 例均无跌倒或前轮换边。严格检查要求前轮各在一侧，自碰撞力低于 1 N，非轮体与地面接触峰值低于 50 N。仿真仅排除默认站姿就重叠的两对前大腿与机身网格。原 Python R152 的严格成绩分别是 24/52 和 9/52。闭环结果以 ONNX 实测为准。

[ONNX 平地视频](videos/flat-commands.mp4)选取慢速前进、倒车、左移和左转各一个通过的回合，每段随后停车。各回合种子和指标见 [视频记录](evidence/video-rollouts.json)。

## 在 unitree_cpp_deploy 中回放

复用上游 `OrtRunner`，使用独立的 AS2-W 16 轴目标。需要 Linux、C++17、CMake、yaml-cpp 和解压后的 ONNX Runtime。`delivery` 指向本目录的绝对路径，`ORT_ROOT` 指向 ONNX Runtime 的安装目录。

```sh
git clone --branch go2 https://github.com/wty-yy/unitree_cpp_deploy.git
git -C unitree_cpp_deploy checkout 4f71bf3d7133f3f20f1a4830730d30bb0d63c91e
cp -a "$delivery"/unitree_cpp_deploy/. unitree_cpp_deploy/
cmake -S unitree_cpp_deploy/deploy/robots/as2w -B as2w-r152-build \
  -DONNXRUNTIME_ROOT="$ORT_ROOT"
cmake --build as2w-r152-build -j
./as2w-r152-build/as2w_r152_replay \
  unitree_cpp_deploy/logs/as2w/r152/params/deploy.yaml \
  "$delivery/evidence/r152-trace.csv"
```

在 Linux x86-64 和 ONNX Runtime 1.22.0 上，750 帧 C++ 回放与 Python ONNX Runtime 输出逐帧一致，腿轮目标映射和反馈超时锁止均通过。原 Python 策略与 ONNX 的最大动作绝对误差为 `4.49e-6`。[C++ 回放记录](evidence/cpp-replay.txt) · [逐帧验证](evidence/parity.json)

## 模型接口与接入

| Float32 张量 | 形状 | 内容 |
| --- | --- | --- |
| `obs` | `[1,41]` | 机身坐标系指令 `vx, vy, wz`、IMU 角速度、投影重力、16 轴绝对位置和 16 轴速度 |
| `state` | `[1,844]` | 控制器进入时全零，随后使用上一帧的 `next_state` |
| `actions` | `[1,16]` | 归一化动作，顺序和限幅见 `deploy.yaml` |
| `next_state` | `[1,844]` | 下一帧控制器历史 |

`r152_controller.hpp` 提供 `Controller::step` 和 `Controller::reset`。使用已标定的传感器轴顺序、方向、零位和 IMU 四元数约定，将输入映射为 `SensorState`。`step` 返回腿位置和轮角速度目标。驱动侧需要按实机低层协议接入这些目标，并对无效或超时反馈执行停用。R152 的状态维数为 844，加载时请配套使用 R152 的 C++ 目标、模型和 YAML。

启用 CMake 选项 `-DAS2W_SDK2_SHADOW=ON` 可构建可选的 SDK2 只读探针。探针需要已确认的 LowState 话题、网络接口和显式电机映射表。映射表按 `deploy.yaml` 顺序每行填写 `关节名 DDS索引 方向符号 零位弧度`。探针只读取反馈并计算目标。当前交付的执行验证为离线回放，实机写控制需接入经标定的驱动接口后另行验证。
