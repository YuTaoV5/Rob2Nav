# AS2-W R214 ONNX 与 `rl_deploy`

R214 在 R177 平地与盲楼梯模型上加入低站姿轮速差转向和首阶俯仰修正。单个 ONNX 同时处理平地 `vx / vy / wz` 指令、停车和五级盲楼梯。模型使用指令、IMU 和关节编码器，无需机身实测线速度或外部地形输入。

| 同一台本机的 MuJoCo 测试 | R177 ONNX | R214 ONNX |
| --- | ---: | ---: |
| 平地固定集 | 35/52 | 39/52 |
| 平地独立集 | 16/52 | 17/52 |
| 标准五级楼梯 | 1/80 | 1/80 |

R214 楼梯通过项为 6 cm，登顶后四轮支撑 2.16 秒。独立的 64 组随机切换中，33 组通过安全检查，0 组在所有指令段达到跟踪容差。数据来自本机 Windows x86-64 的 MuJoCo 和 ONNX Runtime，Linux C++ 重编译、ARM 运行及实机验证待完成。

[ONNX 模型](unitree_cpp_deploy/logs/as2w/r152/exported/policy.onnx) · [控制配置](unitree_cpp_deploy/logs/as2w/r152/params/deploy.yaml) · [完整评估](evidence/onnx-standard.jsonl) · [随机切换评估](evidence/onnx-transitions.jsonl) · [ONNX 仿真视频](videos/simulation-onnx.mp4)

## 编译和离线回放

依赖 Linux、C++17、CMake、yaml-cpp 和对应架构的 ONNX Runtime。下列命令复用已适配的上游 `OrtRunner`。

```sh
git clone --branch go2 https://github.com/wty-yy/unitree_cpp_deploy.git
git -C unitree_cpp_deploy checkout 4f71bf3d7133f3f20f1a4830730d30bb0d63c91e
cp -a "$delivery"/unitree_cpp_deploy/. unitree_cpp_deploy/
cmake -S unitree_cpp_deploy/deploy/robots/as2w -B as2w-rl-build -DONNXRUNTIME_ROOT="$ORT_ROOT"
cmake --build as2w-rl-build -j
./as2w-rl-build/rl_deploy \
  unitree_cpp_deploy/logs/as2w/r152/params/deploy.yaml \
  "$delivery/evidence/r214-trace.csv"
```

`delivery` 是本目录的绝对路径，`ORT_ROOT` 是 ONNX Runtime 解压目录。原 `r152` 路径用于兼容部署目录结构。R214 的模型、YAML 和 C++ 适配器需要成套复制。

| Float32 张量 | 形状 | 内容 |
| --- | --- | --- |
| `obs` | `[1,41]` | `vx, vy, wz`、三轴角速度、三轴投影重力、16 轴绝对位置和 16 轴速度 |
| `state` | `[1,1196]` | 首帧清零，之后逐帧传入上一步的 `next_state` |
| `actions` | `[1,16]` | 12 个腿关节目标和 4 个轮速目标的归一化动作 |
| `next_state` | `[1,1196]` | 下一帧的策略状态 |

50 Hz 控制周期中，腿目标是默认角加 `0.35 × action`，轮目标是 `16 × action` rad/s。关节顺序和限位见 `deploy.yaml`。`rl_deploy` 是离线回放入口。接入机器人的低层程序链接 `as2w_r152`，通过 `Controller::arm()` 清零状态、`Controller::step()` 更新动作、`Controller::stop()` 结束控制。下发端按返回的 `active` 判断是否使用动作。反馈超过 50 ms 或输入无效时，适配器会锁止。实机接入前需核对 DDS `LowState` 话题、电机映射、符号、零位和 IMU 约定。可选 `-DAS2W_SDK2_SHADOW=ON` 构建 SDK2 只读探针。

视频选自通过场景，逐场记录见 [视频记录](evidence/video-rollouts.json)。[回放记录](evidence/replay.json)包含三段共 1,050 帧 ONNX 输入和参考输出的检查结果。
