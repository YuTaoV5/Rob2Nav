# AS2-W r270 · 122,880 步仿真预览

本包使用两段录像对应的同一检查点。r270 的学生和价值网络从随机参数训练，学生使用 480 ms 的本体历史。

- [ONNX 策略](unitree_cpp_deploy/logs/as2w/r270-122880/exported/policy.onnx)
- [PyTorch 检查点](weights/model_122880.pt)
- [平地动态遥控视频](videos/r270-122880-flat.mp4) · 12 秒
- [6 cm 五级楼梯视频](videos/r270-122880-stairs.mp4) · 35 秒
- [平地评估](evidence/dynamic-preview-122880.json)

平地标准与留出场景的平均速度跟踪误差分别为 0.188 和 0.190 m/s。动态松杆保持 0/8，非轮触地 0 帧。楼梯录像中两只前轮到达第一阶，后轮留在地面。这是 MuJoCo 预览，仅供仿真和只读联调。

## ONNX 与 unitree_cpp_deploy 接入

模型沿用 unitree_cpp_deploy 的 `logs/as2w/<版本>/exported` 和 `params` 目录布局。使用 [rl_deploy.yaml](rl_deploy.yaml) 和 [数值契约](unitree_cpp_deploy/logs/as2w/r270-122880/params/r270_contract.json) 配置 AS2-W 专用观测与目标生成。旧策略使用不同输入形状。接入时给 r270 单独配置观察组和推理状态，并在进入策略、控制器复位或数据流失效时清空历史、前次动作和步计数。

每 20 ms 从遥控器和 HG LowState 构造一个 60 维 float32 帧。遥控器提供机体坐标系指令 `vx, vy, wz`。按预览设置，平移指令限制在 0.15 m/s 的模长内，转向限制在 ±0.5 rad/s，每步用 0.12 系数向目标更新。目标为零且三个分量的绝对值均小于 0.005 时将过滤值置零。第一组三维在滤波后再减 `[0.4, 0, 0]`。

| 零基区间 | 当前帧内容 |
| --- | --- |
| 0:3 | 滤波后的遥控指令减偏置 |
| 3:6 | 机体坐标系角速度，rad/s |
| 6:9 | 机体坐标系投影重力 |
| 9 | 常数 0.4 |
| 10:26 | 关节及轮角减默认角。轮角先按 4π 周期映射到 [−2π, 2π) |
| 26:42 | 与角度同序的关节和轮角速度，rad/s |
| 42:58 | 上一帧经过限幅的策略动作。复位首帧填零 |
| 58:60 | 0.62 秒周期的 sin、cos。相位从第 0 步开始 |

[C++ 历史缓冲](runtime/r270_history.hpp) 将当前帧与过去第 2、4、…、24 帧的 `[3:58]` 依次拼接。启动缺帧时重复最早可用帧。输入为 `obs[1,720]`，输出为 `actions[1,16]`。动作限幅 `[-6,6]` 已在图内。按数值契约，将前 12 维映射为默认角加 `0.35 × action` 的腿部位置目标，将后 4 维映射为 `16 × action` 的轮速目标，再分别使用契约中的目标限位。关节逻辑顺序为髋、腿、膝、轮四组，每组 FL、FR、RL、RR。预览采用相同的裸策略动作路径，包含零指令本身的策略响应。

AS2-W 接入从官方 HG LowState 读取 IMU 和电机反馈，从遥控器读取指令，无需 SportModeState 的机身线速度。按目标固件核对 DDS 类型、16 个槽位、IMU 与轮方向、角度展开及遥控器刷新。数值契约的限位来自仿真。实机侧参数和完整低层控制链仍需分别验证。

## 复现 C++ 只读回放

在本目录设置 `ORT` 为 Linux ONNX Runtime 解压目录，然后运行以下命令。回放输入由两段视频相同的仿真场景生成。每行包含复位标志、60 维当前帧和 16 维原模型动作。该程序不连接 DDS。

```bash
gzip -dc evidence/action-replay.txt.gz > /tmp/r270-action-replay.txt
g++ -std=c++17 -O2 -I"$ORT/include" runtime/replay_r270_ort.cpp \
  -L"$ORT/lib" -Wl,-rpath,"$ORT/lib" -lonnxruntime -o /tmp/replay-r270
/tmp/replay-r270 unitree_cpp_deploy/logs/as2w/r270-122880/exported/policy.onnx \
  /tmp/r270-action-replay.txt
```

Linux x64 ONNX Runtime 1.23.2 的 2,350 帧回放完成两次历史复位，最大动作绝对误差为 1.49e-7。Python ONNX 参考运行时的误差为 2.98e-8。回放从 60 维仿真帧开始，验证历史拼接和策略推理。录像使用 PyTorch 检查点生成。SDK 解码、执行器绑定与实机运行尚未验证。
