# AS2-W 无实测线速度联合策略 · R41

同一份权重在 MuJoCo 中执行平地 `vx / vy / yaw rate` 指令、停车和盲上楼梯。策略使用 IMU、关节与轮端反馈、历史动作和指令，从轮端运动学与历史观测估计平面速度。推理无需订阅官方运控的基座线速度话题。

- [R41 ONNX](unitree_cpp_deploy/logs/as2w/r41/exported/policy.onnx) · [PyTorch 权重](weights/policy.pt)
- [平地九种指令与停车 · 54 秒](videos/flat-commands.mp4)
- [五级 6 cm 楼梯 · 9 秒](videos/stairs-06cm.mp4) · [五级 8 cm 楼梯 · 9 秒](videos/stairs-08cm.mp4)
- [视频种子与指令](evaluation/video-rollouts.json)

| MuJoCo 场景 | Python 固定 | ONNX 固定 | Python 独立 | ONNX 独立 |
| --- | ---: | ---: | ---: | ---: |
| 平地指令与停车 | 41/52 | 40/52 | 20/52 | 20/52 |
| 五级 6 cm | 0/5 | 0/5 | 10/21 | 5/21 |
| 五级 8 cm | 1/6 | 1/6 | 0/21 | 0/21 |
| 五级 10 cm | 0/5 | 0/5 | 0/22 | 0/22 |

相比 [R21](https://github.com/YuTaoV5/Rob2Nav/tree/d480b82e70e5fb62a2afdd3e19a50c7485df0bff/deliveries/as2w-deploy/omni-blind-stairs)，固定平地测试从 34/52 提升到 41/52，独立平地从 16/52 提升到 20/52。R41 两组平地测试均无跌倒，停车均达标。固定测试中，前进、倒退及左右转各通过 4/4，纯左移 1/4、纯右移 2/4、左右斜移各 2/4。倒退混合指令的专项独立测试从停步基线的 4/48 提升到 14/48。 [专项校准](evaluation/reverse-mixed-calibration.json)

Python 版楼梯逐场结果与 R21 一致。视频使用本次发布的 PyTorch 权重重新录制，展示通过的仿真种子。8 cm 视频对应固定测试中通过的一次。ONNX 闭环中，两组平地测试均无跌倒，停车均达标，固定测试少通过一个纯右移种子，独立测试少通过五个 6 cm 楼梯种子。平地任意方向跟踪和楼梯复现率仍未达到稳定验收要求。 [ONNX 固定验收](evaluation/fixed-onnx.json) · [ONNX 独立复测](evaluation/confirmation-onnx.json)

## unitree_cpp_deploy

该适配使用 [上游 go2 分支的 OrtRunner](https://github.com/wty-yy/unitree_cpp_deploy/tree/4f71bf3d7133f3f20f1a4830730d30bb0d63c91e)，为 AS2-W 提供独立的 16 轴控制器。12 个腿关节输出位置目标，4 个轮子输出角速度目标。模型包含步态、停车、楼梯路由及估速历史。控制器以 50 Hz 运行，从 IMU 四元数计算投影重力。模式进入时将历史状态清零，每帧将 `next_state` 传入下一帧的 `state`。上游通用 Go2 控制器的 12 轴位置输出不适用于这个模型。

| Float32 张量 | 形状 | 内容 |
| --- | --- | --- |
| `obs` | `[1,41]` | 机身坐标系的指令 `vx, vy, wz`、IMU 角速度、投影重力、16 轴绝对位置和 16 轴速度 |
| `state` | `[1,835]` | 首帧全零，随后由上帧 `next_state` 提供 |
| `actions` | `[1,16]` | 归一化动作，顺序以 `deploy.yaml` 为准 |
| `next_state` | `[1,835]` | 更新后的策略历史 |

在 Linux 上运行以下命令。需要 C++17、CMake、yaml-cpp 和解压后的 ONNX Runtime。将 `delivery` 设为本目录绝对路径，将 `ORT_ROOT` 设为 ONNX Runtime 安装目录。

```sh
git clone --branch go2 https://github.com/wty-yy/unitree_cpp_deploy.git
git -C unitree_cpp_deploy checkout 4f71bf3d7133f3f20f1a4830730d30bb0d63c91e
cp -a "$delivery"/unitree_cpp_deploy/. unitree_cpp_deploy/
cmake -S unitree_cpp_deploy/deploy/robots/as2w -B as2w-r41-build -DONNXRUNTIME_ROOT="$ORT_ROOT"
cmake --build as2w-r41-build -j
./as2w-r41-build/as2w_r41_replay \
  unitree_cpp_deploy/logs/as2w/r41/params/deploy.yaml \
  "$delivery/evaluation/r41-trace.csv"
```

控制器接口接收策略关节顺序的传感器数据。低层报文的电机索引、方向、零位和固件接口需要按实机标定。适配层提供离线回放和可选的 SDK2 只读探针，当前未连接 AS2-W 实机，尚未验证 DDS 兼容性及 ARM 板端运行。探针可通过 `-DAS2W_SDK2_SHADOW=ON` 构建，需安装 SDK2，启动时传入已确认的 LowState 话题和显式映射表。映射表按 `deploy.yaml` 顺序每行填写 `关节名 DDS索引 方向符号 零位弧度`。具体参数可通过无参运行查看。

沿原 Python 策略轨迹逐帧比较了 67,950 次控制更新，ONNX 最大动作绝对误差为 0.000166。Linux x86-64 上使用 ONNX Runtime 1.22.0 编译并运行 C++ 回放，750 帧与 ONNX 参考输出逐帧相同，验证了腿部和轮端目标映射以及反馈超时停用。 [逐帧验证](evaluation/onnx-parity.json) · [C++ 回放记录](evaluation/cpp-replay.txt)

## 复现

在此目录中运行。环境使用 Python 3.10+、MuJoCo 3.17.0、NumPy、SciPy、PyTorch、Gymnasium、Stable-Baselines3。导出还需要 ONNX、ONNX Runtime 和 PyYAML。录制还需要 Pillow、FFmpeg 和 MuJoCo 的 EGL 或其他可用渲染后端。

将 [Unitree ROS 模型仓库](https://github.com/unitreerobotics/unitree_ros/tree/4005a1b0c3f464f515f380be4145cc617fbca725) 的所列版本放在 `assets/official-model`，其中应有 `robots/as2w_description/as2w.urdf` 和该模型的 `meshes/`。

```sh
sha256sum -c SHA256SUMS
python runtime/evaluate_omni_joint.py assets weights/policy.pt replay --audit-inputs
python runtime/export_r41_onnx.py assets weights/policy.pt replay/exported/policy.onnx replay/onnx-evidence --deploy-yaml replay/params/deploy.yaml --full
MUJOCO_GL=egl python runtime/render_omni_policy.py assets weights/policy.pt evaluation/fixed.json evaluation/confirmation.json replay-videos
```

完整逐场记录见 [固定验收](evaluation/fixed.json) 和 [独立复测](evaluation/confirmation.json)。输入审计覆盖 67,200 次控制更新，移除仿真状态句柄后动作完全一致。 [输入审计日志](evaluation/input-audit.txt)
