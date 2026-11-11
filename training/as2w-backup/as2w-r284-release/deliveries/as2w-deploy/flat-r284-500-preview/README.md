# AS2-W r284-500 平地仿真预览

此版本是 Isaac Lab / PhysX 平地 PPO 从随机参数训练的检查点索引 500，保存时完成 501 次更新、12,312,576 个环境步。Actor 只读取遥控指令和本体数据。训练契约中的迭代预算为 2,500。

- [ONNX](unitree_cpp_deploy/logs/as2w/r284-500/exported/policy.onnx)
- [确定性 PyTorch 权重](weights/actor-500.pt)
- [40 秒 ONNX 平地回放](videos/r284-500-onnx-flat.mp4)
- [ONNX 闭环分场景结果](evidence/onnx-flat.json)
- [rl_deploy 配置](rl_deploy.yaml)

## 仿真结果

每组使用四个场景种子。表中的速度是沿各段指令方向的投影均值。转向两段按每个种子的实际指令方向计分。两项零速分别独立运行 10 秒，在第 1 秒后统计。合格要求位移 ≤2 cm、偏航漂移 ≤3°、残速 P95 ≤0.02 m/s，且无非轮触地或终止。

| 场景 | 目标 | 标准 | 留出 |
| --- | --- | --- | --- |
| 前进 | 0.10 m/s | 0.136 m/s | 0.137 m/s |
| 后退 | 0.10 m/s | 0.090 m/s | 0.089 m/s |
| 两个横移段 | 0.10 m/s | 0.035 / 0.058 m/s | 0.036 / 0.059 m/s |
| 四个斜移段 | 约 0.10 m/s | 0.073–0.101 m/s | 0.071–0.102 m/s |
| 两个转向段 | 0.30 rad/s | 0.178 / 0.255 rad/s | 0.174 / 0.255 rad/s |
| 静止零指令 | 10 秒合格场景 | 2/4 | 1/4 |
| 运动后松杆 | 10 秒合格场景 | 0/4 | 0/4 |
| 非轮触地 | 帧数 | 0 | 2 |

松杆后最大位移的场景均值为 4.14 cm 和 4.63 cm。动态后腿偏离默认角的 P95 在纯横移段最高，场景均值约 0.61 rad。所有回合完成。此版本供仿真和只读联调，实机和正式驻停验收待验证。

## 接入 unitree_cpp_deploy

本包采用上游的 `logs/as2w/<版本>/exported` 与 `params` 目录布局。在已有 AS2-W 16 电机接入层新增 r284 策略状态，路径指向 `logs/as2w/r284-500`。根据 [deploy.yaml](unitree_cpp_deploy/logs/as2w/r284-500/params/deploy.yaml) 配置推理，再调用 [C++ 观测和目标映射](unitree_cpp_deploy/deploy/robots/as2w/r284_observations.hpp)。上游 Go2 的 12 电机策略状态需要保留独立配置。

1. 从 HG LowState 解码机体系 IMU 角速度、投影重力及 16 个电机的角度与角速度。由姿态计算投影重力时，将世界向下单位向量旋转到机体系。按照配置中髋、腿、膝、轮的四组顺序重排，每组 FL、FR、RL、RR。轮角若以 2π 周期返回，先展开连续角度，再按 4π 周期映射到模型区间。
2. 从遥控器获取机体系 `vx, vy, wz`，将中心死区映射到精确零。每 20 ms 调用 `Observations::push`。预览配置将平移模长限制为 0.10 m/s，偏航限制为 ±0.30 rad/s，以 0.12 系数滤波。目标为零且滤波后三轴绝对值都小于 0.005 时置零。
3. 将返回的 `obs[1,65]` 送入 ONNX Runtime。调用 `Observations::remember` 保存上次限幅动作，再用 `targets` 将 `actions[1,16]` 映射成腿关节位置和轮速目标。模型图内动作限幅为 `[-6,6]`。仿真腿部 `kp=55, kd=2`，轮部 `kp=0, kd=2`。
4. 进入策略、控制器重置或传感器流失效时调用 `Observations::reset`，清空滤波命令、上次动作、时钟和轮角参考。电机槽位、轮方向、IMU 方向、数据新鲜度及实际执行器增益由 AS2-W 接入层核对。

特征 `60:64` 是过滤后首次进入零指令时捕获的四轮相对角位移。差值按 4π 回绕，除以 0.20 rad 后限幅到 `[-3,3]`。非零指令时这四维和第 64 维都置零，并准备捕获下一次驻停。第 64 维是零指令标志。机身线速度只用于仿真评估，ONNX 输入不读取 SportModeState。

## 复现 C++ 只读回放

下列回放不连接 DDS。输入文件每行包含复位标志、60 维当前帧、16 维 PyTorch 动作。程序核对轮角记忆、动作限幅、ONNX 数值，并检查 C++ 遥控滤波、复位、角度回绕和目标映射的代表性用例。

```bash
gzip -dc evidence/action-replay.txt.gz > /tmp/r284-action-replay.txt
g++ -std=c++17 -O2 -I"$ORT/include" \
  -Iunitree_cpp_deploy/deploy/robots/as2w runtime/replay_r284_ort.cpp \
  -L"$ORT/lib" -Wl,-rpath,"$ORT/lib" -lonnxruntime -o /tmp/replay-r284
/tmp/replay-r284 unitree_cpp_deploy/logs/as2w/r284-500/exported/policy.onnx \
  /tmp/r284-action-replay.txt
```

Linux x64 ONNX Runtime 1.23.2 在 16 次复位、16,000 帧上的最大动作绝对误差为 `1.073e-6`。Python ONNX Runtime 的最大误差为 `9.537e-7`。录像使用同一 ONNX 闭环生成。
