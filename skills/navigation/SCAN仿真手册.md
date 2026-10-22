# SCAN 导航仿真操作与复现手册

先阅读[新手复现总览](../../docs/新手复现总览.md)与[环境准备与路径约定](../../docs/环境准备与路径约定.md)。本指南的环境变量均由当前机器配置；每个新终端先在仓库根目录source `scripts/docs_env.sh`。


新增[台阶后过窄门与带台阶的路线导航](../../docs/SCAN台阶与窄门任务说明.md)：网页默认加载台阶窄门场景，支持在地图上选择台阶后的目标。场景尺寸、低点云过滤边界和独立验收记录见新说明。

更新：2026-10-21。配套[实机完整部署手册](SCAN实机部署手册.md)、[网页桥接实现与原始证据](../../docs/SCAN网页仿真桥接说明.md)。

新增[多房间动态行人与指定人物跟随任务](../../docs/SCAN动态避障与人物跟随说明.md)：导航页默认SCAN，选择任务/场景/人物/难度后重置。动态任务基于固定上游提交增加了**可选占用老化补丁**，不属于未经修改的原版；必须按新任务手册构建。人物ID来自仿真标签，低阶梯和刚性沙地近似的验证范围见该手册及归档，不能由旧静态导航成绩推导。

## 1. 当前交付与边界

本项目已经运行的是 SCAN 原生 ROS2 规划器 + 原生闭环轨迹控制器 + RoboGauge MoE 低层策略 + MuJoCo 自由基座接触动力学。点云来自 MuJoCo 射线，机身里程计来自仿真真值；没有实现 LIO，也没有连接机器人。SCAN 是经典规划器，不需要训练导航权重；MoE 权重是另一层的运动策略。

上游社区 ROS2 移植提交固定为 `d0b921c9b05a6d291d144d60882b2e0e88d2c0e0`。上游原生 `run.launch.py` 仿真使用运动学模拟器；本网页改用 MuJoCo，不启动该运动学模拟器。ROS1 原版和这个 ROS2 移植版不是同一运行环境。

## 2. 网页操作

1. 打开 `http://127.0.0.1:8766/#navigation`，选择“SCAN 原生规划器 + MuJoCo”。在线仿真页也可选择“SCAN 原生规划仿真”。独立窗口 `/scan-live`。
2. 等待 READY，点击地图空地设置目标，点击开始。地图点击用于目标，渲染画面拖拽用于视角。
3. 画面左键拖动旋转、右键拖动平移、滚轮缩放。观察实际位置、轨迹、规划次数、指令数和目标距离。
4. 点击停止后核对状态；新任务前重置。重置创建新 session、清空规划地图并重启对应规划子进程。
5. ARRIVED 是网页距离阈值约 0.40m；不代表实机厘米级停靠。FAULT 时先看日志，不能用反复开始掩盖定位或规划错误。

浏览器关闭/失联 2.5 秒停止仿真导航；桥接指令超过 0.3 秒过期或桥未就绪时下发零目标速度。它们是仿真保护，不是已验证的实机停车距离保证。

## 3. 启动服务与远程访问

先按[环境准备与路径约定](../../docs/环境准备与路径约定.md)设置当前机器的变量；以下命令从仓库根目录执行。

```bash
"$WEB_PY" -m studio.api --root "$ROB2NAV_ROOT" --port 8765
# 保持上面的服务运行，在另一个终端检查（桥和worker也须启动）：
curl -fsS http://127.0.0.1:8765/scan-live-api/api/state
```

Web、SCAN worker与ROS桥分别启动后应各自就绪；本仓库不要求某台服务器的Supervisor配置。端口：Studio 8765，SCAN worker 8769，ROS/仿真 UDP 桥 127.0.0.1:8771。用户本地 8766 是 SSH 隧道端口，并非服务器监听端口。客户端建立隧道（认证交互输入，不写入文档或仓库）：

```bash
ssh -N -L 8766:127.0.0.1:8765 -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}"
```

若 8766 已有隧道，不要重复绑定。不要把仿真 UDP 桥或控制 worker 直接暴露到公网。

## 4. 环境与从源码恢复

已验证服务器：Ubuntu 24.04 + ROS Jazzy；CPU 环境 Python 3.11、MuJoCo 3.2.3、torch 2.7 CPU、numpy 1.26.4。ROS 由系统 `/usr/bin/python3` 运行，MuJoCo/MoE 由独立环境运行。不要混用 ROS Python 和策略环境依赖。

上游 README 原目标是 Ubuntu 22.04/Humble；当前 Jazzy 是本项目加兼容补丁后的实测组合。全新机器先按 ROS 官方文档安装对应发行版及 rosdep/colcon；以下只覆盖 SCAN 工作空间，不声称是一条安装全部 Studio、模型和系统驱动的万能命令。现有机器不要覆盖已有 upstream 工作区。

```bash
# 新建工作区时执行；目录已存在时先核对提交，不重复clone
mkdir -p "$ROB2NAV_WORKSPACE/scan-reproduction"
cd "$ROB2NAV_WORKSPACE/scan-reproduction"
git clone https://github.com/wuyi2121/SCAN-Planner.git upstream
cd upstream
git checkout d0b921c9b05a6d291d144d60882b2e0e88d2c0e0
source /opt/ros/jazzy/setup.bash
# Jazzy 三处 cv_bridge 头文件兼容补丁；先检查再应用
git apply --check "$ROB2NAV_ROOT/artifacts/release_20261021/source_patches/scan-jazzy-compatibility.patch"
git apply "$ROB2NAV_ROOT/artifacts/release_20261021/source_patches/scan-jazzy-compatibility.patch"
rosdep update
rosdep install --from-paths src --ignore-src -r -y
sudo apt install libarmadillo-dev libglew-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 pkg executables scan_planner
```

必须出现 `scan_planner_node` 和 `closed_loop_controller`。若只做上游运动学复现，可执行：

```bash
ros2 launch scan_planner run.launch.py is_real_world:=false navi_mode:=1 sensor_type:=lidar controller_mode:=closed_loop use_gpu:=false
# 另一个已 source 的终端：
ros2 launch scan_planner rviz.launch.py
```

此路径不是网页 MuJoCo 验证。两种模拟器不能同时向相同话题发布机身状态。

网页MuJoCo手动启动（相同服务已启动时不要重复）：准备三个终端，每个终端先进入仓库并`source scripts/docs_env.sh`。终端一启动ROS桥并保留日志：

```bash
mkdir -p outputs/scan_web
set -o pipefail
bash deployment/start-scan-bridge.sh 2>&1 | tee outputs/scan_web/bridge.log
```

终端二启动物理仿真worker：

```bash
cd "$ROB2NAV_ROOT"
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$CPU_PY" -m skills.navigation.live --planner scan --port 8769
```

终端三启动本节前面的Web服务；如已运行则保留原服务。三个进程保持运行后，再用另一个终端执行curl检查。停止本次手动启动的服务可在其终端按Ctrl+C。

模型及依赖按仓库环境配置恢复；发布资源索引为 [model_index.json](../../artifacts/release_20261021/model_index.json)。运行 `python scripts/verify_release_resources.py` 核对归档完整性，不能把特技最佳权重当成 SCAN 模型。

## 5. 数据流与诊断

| 环节 | 当前输入/输出 | 核查方法 |
| --- | --- | --- |
| MuJoCo | 实际射线点云、body/sensor Odometry | worker state、传感器点数 |
| ROS 桥 | `/quad_0/cloud`、`/quad_0/body_pose`、`/quad_0/lidar_pose` | ROS topic info/hz |
| 目标 | `/move_base_simple/goal` PoseStamped | 订阅者数、goal_acknowledged |
| SCAN | `/planning/bspline` | plans 应增长 |
| 原生闭环控制器 | `/quad_0/cmd_vel` Twist | commands 与新鲜度 |
| MoE + MuJoCo | 关节力矩、接触与实际轨迹 | 到达、碰撞、跌倒记录 |

ROS 诊断终端需 source Jazzy 与 upstream/install，并设置 `ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`。点云已在 world 坐标，当前仿真 `cloud_is_world=true`；显示地图障碍真值不作为 SCAN 点云输入。传感器参数见 [config.json](config.json)，不等同于真实雷达标定。

```bash
tail -n 80 "$ROB2NAV_ROOT/outputs/scan_web/bridge.log"
tail -n 80 "$ROB2NAV_ROOT/outputs/scan_web/planner.log"
ros2 topic info /quad_0/cloud -v
ros2 topic hz /quad_0/body_pose
ros2 topic info /move_base_simple/goal -v
```

READY 但无轨迹：查目标订阅者/ack、ROS domain、点云与 odom，再查 planner.log。端口占用：查是否重复启动 Supervisor 与手工 worker。黑屏：核对 EGL/MuJoCo 环境。到达误差大：区分规划轨迹、底层跟踪误差和网页到达阈值。

## 6. 可复现实验与验收

```bash
cd "$ROB2NAV_ROOT"
python3 scripts/verify_scan_mujoco_web.py --output outputs/scan_web_check_new
```

会重置当前 SCAN 任务；使用新输出目录。已发布固定测试：空旷目标 [2,0]、绕障目标 [6,1.6]，2/2 到达，最终距离 0.377m/0.371m，两次均无碰撞/跌倒。见[原始摘要](../../docs/evidence/scan_mujoco_web/summary.json)及[修复前失败记录](../../docs/evidence/scan_mujoco_web/previous_failure.json)。两次固定测试不代表全面鲁棒性。

扩展验收至少覆盖多随机种子、多目标方向、窄通道、目标不可达、动态障碍、点云掉帧/噪声、定位延迟、规划进程退出、网页断线、连续重置。逐例记录成功率、最小间距、碰撞/跌倒、路径长度、终点误差和停止时间；未测试项标为未验证。
