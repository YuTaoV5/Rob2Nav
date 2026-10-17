# Go2 无地图导航仿真（RoboGauge 本体运控）

分层实现：实时 D435i 深度 / Mid-360 点云 → 无地图导航 → 机体系速度指令 → **RoboGauge 原版 Go2 类和 TorchScript 策略** → 关节 PD → MuJoCo 刚体与接触动力学。不是把机器人当作平面质点移动。

上游：[wty-yy/RoboGauge](https://github.com/wty-yy/RoboGauge)，初始核查版本 `add2c7ee0185ea8fd93d242f86e1c8cac6e9913b`。上游源码和模型在 `vendor/RoboGauge`，使用其 MIT 许可；本目录不修改上游代码。

## 服务器

工作目录：`/root/gpufree-data/go2-mapless`。

独立 Python：`/root/gpufree-data/envs/go2-mapless/bin/python`。现有 Isaac Lab 环境不变。此版本选用 MuJoCo 3.2.3，遵守 RoboGauge 对 `<3.3.0` 接触动力学版本的要求。CPU 执行小批量本体策略和 MuJoCo，EGL 可使用 L40 渲染。

```bash
cd /root/gpufree-data/go2-mapless
bash run.sh -m unittest discover -s tests -v
bash run.sh -m go2_nav.run --scene slalom --episodes 3 --video --output outputs/slalom
bash run.sh -m go2_nav.run --scene random --episodes 10 --seed 100 --output outputs/random_eval
```

每个输出目录包含 `summary.json`、`trajectory_N.json`；加 `--video` 后生成 `episode_N.mp4`。成功、碰撞、摔倒、超时分别记录。控制器不读取碰撞模型或场景障碍列表；几何真值仅供物理、传感器仿真和评估使用。

`bash run.sh -m go2_nav.snapshot` 生成场景、深度和局部点云三联图。评估完成后用 `bash run.sh -m go2_nav.report` 汇总 `VALIDATION.md`。

## 可训练导航

```bash
bash run.sh -m go2_nav.train --steps 100000 --envs 4 --scene random --output outputs/ppo
bash run.sh -m go2_nav.run --policy ppo --checkpoint outputs/ppo/navigation.zip --scene random --seed 100 --episodes 10 --video --output outputs/ppo_eval
```

训练 PPO 的是**上层导航**，下层 RoboGauge 运控保持冻结。默认观测为 72 个雷达距离扇区、12×16 深度、目标方向/距离、本体速度及上一速度指令，共 273 维。图像目前降采样后输入 MLP，并非原始 RGB/CNN；可用 `--sensors depth` / `lidar` / `both` 做消融，训练和评估必须使用相同模式。消融只向策略和反应式控制器屏蔽另一传感器，底层仍计算两路传感数据，不能用于比较单传感器计算耗时。

`reactive` 是不需要训练的局部速度采样基线，`straight` 是不避障对照。PPO 训练入口已提供不等于策略已经收敛；请以独立随机种子评估报告为准。

## 传感器与假设

所有参数在 `config.json`：位置单位米、角度单位度，机器人机体系 +X 前、+Y 左、+Z 上。D435i 正前方针孔深度，Mid-360 为 360° 方位角的多仰角理想射线。深度输出是光轴深度；点云以当前机体系给出，只保留局部测量，不累积占据地图或 ESDF。

- 安装位置是初始假设，**不是你实机外参标定值**，需要用实际安装位姿替换。
- 当前 Mid-360 模型不是 Livox 非重复扫描的硬件级仿真；D435i 也未复现双目匹配、运动模糊、材质失效或 IMU 偏置。
- 添加了距离噪声与随机丢点；默认控制/物理/导航分别为 50/500/10 Hz，两个传感器在导航周期更新。
- 导航使用仿真提供的理想本体姿态、速度和目标相对方向，代表未来 odometry/VIO/LIO 接口；**不声称免定位**。未实现真实 ROS2 或硬件通信。
- 默认额外载荷为 0 kg，便于先核对原版运控；`sensor_payload_kg` 可做质量扰动，但只是增加基体质量，不是传感器与支架完整惯量模型。
- 场景装配时为 actuator 显式设置上游 `go2` / `knee` 默认类，继承非零扭矩范围；关节控制策略、观测、PD 增益与动作映射均来自上游。
- 环境包含空场、交错障碍、随机障碍及完全封闭的墙；反应式方法可能局部困住，不提供任意迷宫的全局完备性保证。

本项目当前目标是服务器仿真，不会向你的实机发送运动命令。

## 文件

- `go2_nav/env.py`：场景、真实关节物理、RoboGauge 接口、两类传感、Gymnasium 环境。
- `go2_nav/planner.py`：只用当前点云和目标方向的无地图速度采样基线。
- `go2_nav/run.py`：评估、视频、轨迹与结果。
- `go2_nav/train.py`：Stable-Baselines3 PPO 上层导航训练/续训。
- `tests/test_simulation.py`：传感契约、行走控制、动作限制与迎墙行为检查。

新环境安装：Python 3.11 后先安装所需 PyTorch，再 `pip install -r requirements.txt`。远端实际依赖版本见 `environment.lock.txt`。
