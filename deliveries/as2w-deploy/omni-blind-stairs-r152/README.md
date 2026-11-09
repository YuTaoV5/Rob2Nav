# AS2-W R177 平地与盲楼梯仿真候选

这一版替换同目录的 R152 平地模型，保留原有部署路径。组合权重与评审视频一致，平地修正使用 R177 第 24,576 步，楼梯使用 R159 循环学生和 R171 第 81,920 步的修正。每 20 ms 输出 12 个腿关节位置目标和 4 个轮子角速度目标。推理使用指令、IMU 和编码器，无需机身实测线速度话题。

| 标准 MuJoCo 测试 | Python 原策略 | ONNX Runtime |
| --- | ---: | ---: |
| 平地固定集 | 36/52 | 35/52 |
| 平地独立集 | 17/52 | 16/52 |
| 五级楼梯 | 2/80 | 3/80 |

平地通过要求速度跟踪、停车、腿轮同侧、非轮碰撞峰值低于 50 N 和自碰撞峰值低于 1 N。ONNX 平地 104 个案例中跌倒、前轮跨中线和楼梯误触发均为零。ONNX 通过的楼梯案例均为 6 厘米，登顶后四轮稳定支撑 3.2–4.08 秒。8 和 10 厘米在本批次尚未通过。闭环成绩以 ONNX 实测为准。当前验证覆盖 Linux x86-64 和仿真，实机 DDS 映射及 ARM 板端运行待验证。

[ONNX 运行视频](videos/simulation-onnx.mp4)展示前进、倒车、横移、弧线、原地转向、停车与五级盲楼梯。视频选自通过案例，种子与指标见 [视频记录](evidence/onnx-video-rollouts.jsonl)。[评审用 Python 视频](videos/simulation.mp4)保留原演示的斜移和楼梯回合。[完整评估摘要](evidence/summary.json)

## unitree_cpp_deploy 与 rl_deploy

复用上游 `OrtRunner`，增加 AS2-W 16 轴有状态适配器。需要 Linux、C++17、CMake、yaml-cpp 和 ONNX Runtime。设置 `delivery` 为本目录的绝对路径，`ORT_ROOT` 为 ONNX Runtime 解压目录。

```sh
git clone --branch go2 https://github.com/wty-yy/unitree_cpp_deploy.git
git -C unitree_cpp_deploy checkout 4f71bf3d7133f3f20f1a4830730d30bb0d63c91e
cp -a "$delivery"/unitree_cpp_deploy/. unitree_cpp_deploy/
cmake -S unitree_cpp_deploy/deploy/robots/as2w -B as2w-rl-build -DONNXRUNTIME_ROOT="$ORT_ROOT"
cmake --build as2w-rl-build -j
./as2w-rl-build/rl_deploy \
  unitree_cpp_deploy/logs/as2w/r152/params/deploy.yaml \
  "$delivery/evidence/r177-trace.csv"
```

`rl_deploy` 命令执行离线回放。实际低层控制程序链接 `as2w_r152`，从 `r152_controller.hpp` 接入 `Controller::arm()`、`Controller::step()` 和 `Controller::stop()`。`arm()` 会清零 1193 维状态，`step()` 在 50 Hz 循环里接收已标定反馈和反馈时延。接收端根据 `active` 状态决定是否使用返回的腿轮目标。反馈超过 50 ms 或输入无效会锁止。连接机器人前需验证电机索引、方向、零位、IMU 四元数约定和驱动停用逻辑。可选 `-DAS2W_SDK2_SHADOW=ON` 构建 SDK2 只读探针，使用已确认的 LowState 话题和显式电机映射。

| Float32 张量 | 形状 | 内容 |
| --- | --- | --- |
| `obs` | `[1,41]` | 机身坐标系指令 `vx, vy, wz`、三轴 IMU 角速度、三轴投影重力、16 轴绝对位置和 16 轴速度 |
| `state` | `[1,1193]` | 进入控制模式时全零，之后逐帧传入上一步 `next_state` |
| `actions` | `[1,16]` | 归一化动作，前 12 轴为腿，后 4 轴为轮 |
| `next_state` | `[1,1193]` | 下一帧的控制历史、楼梯检测与循环隐状态 |

关节顺序和限幅见 [deploy.yaml](unitree_cpp_deploy/logs/as2w/r152/params/deploy.yaml)。腿的位置目标为默认角加 `0.35 × action`，轮角速度目标为 `16 × action`。原路径中的 `r152` 保留用于兼容部署配置。该路径的新模型、YAML 和 C++ 适配器需成套更新。

Linux x86-64 上，C++ 与 Python ONNX Runtime 的 1,200 帧回放逐帧一致，并通过腿轮目标与超时锁止检查。原 Python 策略与 ONNX 同帧最大动作误差为 `3.12e-6`，楼梯模式切换逐帧一致。[C++ 回放](evidence/cpp-replay.txt) · [逐帧对照](evidence/parity.json)

## 复现

提供五份 [训练权重](weights/) 与 [导出脚本](runtime/export_r177_onnx.py)。`check_server.py` 接收仿真资产根目录、交付目录和输出目录，支持 `--parity`、`--standard`、`--transitions` 和 `--video`。资产根目录需包含 `fall-baseline` 的关节契约、`local-step-diagnostic` 的 MuJoCo 场景及官方机器人网格。
