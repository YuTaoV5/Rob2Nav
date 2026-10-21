# Go2 四足机器人 SCAN 完整导航实机部署手册

硬件准备先读[D435i/MID360安装、标定与上线总手册](../../docs/GO2_D435I_MID360_INSTALL_CALIBRATION.md)，包含安装供电、ROS驱动、LIO/相机外参及完整现场步骤。

更新：2026-10-21。配套[仿真手册](SCAN_SIMULATION.md)。本手册是针对当前项目的完整集成方案；当前验证到 MuJoCo，尚未完成实机传感器、LIO 和速度执行桥的联调，没有执行机器人通信或电机指令。下文实机操作须由现场人员在各阶段验收后实施。

新增的[动态行人与人物跟随实验](../../docs/SCAN_DYNAMIC_FOLLOW_TASKS.md)使用仿真标签和可选地图占用老化，**不能直接迁移为实机识人或人员安全功能**。上线前须另做人物检测/身份跟踪、米制3D观测、遮挡处理和独立停车；楼梯与真实软土还需要地形运动控制器验收。实机配置保持地图老化关闭，不能直接照搬网页动态任务参数。

## 1. 应部署的完整能力

推荐第一版使用 **真实三维雷达 + LIO 定位 + SCAN 局部避障 + 宇树原厂运动控制器**。先完成平地目标点导航，再接全局路线与任务管理。网页是目标/状态入口，导航计算和失联停车必须留在机器人随身计算机上，公网服务器不能承担最终停车闭环。

```text
MID360 等三维雷达 + IMU → 驱动/时间同步 → LIO 连续里程计
                                       ├→ 机身与传感器位姿
三维点云（去畸变、坐标明确）───────────────┴→ SCAN 滑动地图/轨迹规划
地图定位/路线服务 → 目标或 initial_path ─────→ SCAN Bspline
SCAN closed_loop_controller → 带时间戳/有效期的速度安全门
                            → SportClient.Move(vx,vy,wz)
                            → 原厂步态控制器 → 机器人
实际里程计/状态/故障 ──────────────────────→ 闭环反馈与停车
```

SCAN 提供局部空间碰撞规划，不自动提供雷达驱动、LIO、重定位、全局路线搜索、楼层切换、任务调度或运动 SDK 桥。完整“指定地点→自主抵达”还要部署地点地图、起点定位、全局路线服务与任务状态机；地图中有路线也不等于机器人具备爬楼运动能力。

## 2. 当前交付与待完成项

| 组件 | 当前状态 | 实机落地要求 |
| --- | --- | --- |
| SCAN ROS2/闭环控制器 | 固定源码已编译并接 MuJoCo | 同版本移植到随身计算机，校准参数 |
| 雷达/相机 | 仿真射线已实现 | 确认实际型号、安装、驱动与时间戳 |
| 位姿 | 仿真真值 | 接入并验收 LIO/VIO，不能以 Go2 IMU 代替完整定位 |
| 坐标/消息适配 | 仿真 UDP 桥 | 新建真实传感器适配节点；仿真桥不能复用为硬件接收器 |
| 速度执行 | MoE 驱动 MuJoCo | 新建带安全门的 ROS→Sport SDK 执行服务 |
| 全局路线/定位 | 上游 initial_path 接口 | 接入地图、重定位和路线生成/管理 |
| 网页 | 仿真目标与渲染 | 新建认证后的硬件任务 API/遥测，禁止复用 sim start 假装实机 |
| 实机验收 | 未执行 | 分阶段日志、停车距离、跟踪与避障验收 |

当前仍需确认机器人具体版本/固件/SDK 权限、随身计算机架构、实际雷达或相机、网卡名、外参和工作地形。不能默认所有 Go2 型号均允许相同控制接口。硬件配置未确认前，不能给出真实可用的 IP、外参矩阵或雷达启动文件名。

## 3. 硬件与软件选型

- 随身计算机：运行 Linux、ROS2、LIO、SCAN 与本地安全门；先测整链路 CPU/内存/温度和延迟，再决定配置。云端仅做展示/记录。
- 感知优先方案：MID360 类三维雷达及其 IMU，安装刚性支架、独立可靠供电，标定雷达/IMU/机身关系。不能把 Go2 原厂雷达自动视为 MID360 或直接套 Livox 驱动。
- 相机方案：D435i 对齐深度可输入 SCAN，但仍需可用的连续 VIO/LIO/融合定位；弱纹理、强光、玻璃等单独验收。不能仅靠识别图像或 IMU 得到稳定的世界坐标位姿。
- 本项目 SCAN 环境为 Ubuntu 24.04/Jazzy + 兼容补丁；上游社区版本说明是 22.04/Humble。雷达/LIO 若仅支持其他 ROS 发行版，应选择并验证同发行版端到端构建，或明确消息桥接和时钟开销，不声称已经跨版本验证。
- SDK 固定 `814556d15970dd2ecf1c9984e845ca02ab07e206`。已有离线 SDK/CRC 检查不等于 DDS 通信或运动能力通过。

参考驱动：[Livox ROS2 驱动](https://github.com/Livox-SDK/livox_ros_driver2)、[RealSense ROS](https://github.com/realsenseai/realsense-ros)、[FAST_LIO 原始项目](https://github.com/hku-mars/FAST_LIO)。这些是外部候选，不是本仓库已验证的 Jazzy 一键 LIO 依赖。选用的 ROS2 LIO 移植必须额外固定仓库/提交并保留标定和 bag。

## 4. 坐标与消息契约：先通过离线回放

统一米、秒、弧度，机身 x 前、y 左、z 上；SDK Move 的方向要在现场与该契约逐项核实。固定全链路时间来源，记录设备时钟到 ROS 时间偏移。时间戳必须是测量时间，不能用“接收时刻”掩盖传感器延迟。

建议维护连续 `odom→base_link`，雷达/IMU/相机是经过标定的固定外参。若全局 SLAM 有回环跳变，使用 `map→odom` 处理全局修正，不能让局部控制位姿突然跳变。SCAN 的 world 命名和 TF 必须与实际选择的规划坐标一致。

固定上游 real launch 默认接口：

| 话题 | 类型 | 契约 |
| --- | --- | --- |
| `/LIO/odom_vehicle` | nav_msgs/msg/Odometry | 连续规划坐标中的机身位置/方向 |
| `/LIO/odom_imu` | nav_msgs/msg/Odometry | 与点云变换一致的传感器/IMU位姿；须核对外参实现 |
| `/LIO/clouds_lidar` | sensor_msgs/msg/PointCloud2 | real 默认期望非 world 点云 |
| `/camera/aligned_depth_to_color/image_raw` | sensor_msgs/msg/Image | depth 模式；还需匹配标定/深度单位 |
| `/move_base_simple/goal` | geometry_msgs/msg/PoseStamped | 与规划坐标一致的目标 |
| `/initial_path` | nav_msgs/msg/Path | navi_mode=3 的外部全局路线 |
| `/planning/bspline` | scan_planner_msgs/msg/Bspline | SCAN 到轨迹控制器 |
| `/cmd_vel` | geometry_msgs/msg/Twist | 控制器输出，需安全门接收 |

上游 real 模式设置 `grid_map.cloud_is_world=false`、`grid_map.need_extrinsic=true`。必须检查所选 LIO 输出到底是雷达局部系、IMU系还是已变换世界系，以及 SCAN 源码的外参方向；不能仅改 frame_id。已在 world 的点云若再次变换会产生重影/错位。仿真配置 world 点云为 true，不能直接复制到硬件。

real launch 中的 D435 内参是预设数值，不能视为本机标定。按实际分辨率、对齐目标和 CameraInfo 更新 fx/fy/cx/cy，核对米/毫米转换与深度有效范围。用静止墙面、旋转机身、前进后退的 bag 检查地图不随传感器转动、地面不抬升、障碍尺度正确后再进入运动阶段。

只读诊断（在已启动驱动/LIO 的离线回放环境执行）：

```bash
ros2 topic list -t
ros2 topic info /LIO/odom_vehicle -v
ros2 topic hz /LIO/odom_vehicle
ros2 topic hz /LIO/clouds_lidar
ros2 topic echo /LIO/odom_vehicle --once
ros2 run tf2_ros tf2_echo odom base_link
```

记录频率、P95/P99 延迟、点数、定位状态与漂移。频率合格不代表定位正确；传感器融合失效必须输出健康标志供安全门停车。

## 5. 构建与规划器接入

构建与 Jazzy 补丁步骤见[仿真手册](SCAN_SIMULATION.md)。使用相同固定提交，保存安装依赖、编译日志与参数副本。下列命令仅启动规划器/轨迹控制器，不包含雷达、LIO 或 SDK 电机执行服务；先在 bag 回放/无执行器环境核对：

```bash
source /opt/ros/jazzy/setup.bash
source /root/gpufree-data/scan-reproduction/upstream/install/setup.bash
# 单机规划进程可保持本机发现；多机需要单独配置受控 DDS 网络
export ROS_DOMAIN_ID=88
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch scan_planner run.launch.py is_real_world:=true use_sim_time:=false sensor_type:=lidar controller_mode:=closed_loop navi_mode:=1
```

若驱动/LIO 位于另一台机器，LOCALHOST 将无法发现它们；必须明确 ROS domain、网络接口、发现配置与访问控制。SDK 自身 DDS 的 domain/网卡配置独立核对，不能照抄 SCAN 的 88。

navi_mode=1 是目标点；2 是 YAML 路点序列，需 keypoints_file；3 接外部 initial_path。部署地图地点导航时，先全局定位，再将地点解析为规划坐标中的目标/路线。路径协议和航点推进、失败重试、取消、定位重置都由任务层负责。

目前上游默认规划最大速度约 0.75m/s、加速度 0.5m/s²；闭环控制器默认 vx 0.75、vy 0.35m/s、yaw 1rad/s。这是源码默认值，不是本机安全限值。初始现场验收应采用更低限速，并同时限制 planner、controller 和最终安全门，按实际延迟、制动距离、机身含载荷外形调整地图膨胀/地面高度/传感器盲区。

## 6. 必须实现的速度安全门与 SDK 执行服务

推荐首版使用原厂 SportClient 高层速度控制，保留原厂步态闭环。固定 SDK 的 `Move(vx,vy,vyaw)` 和 `StopMove()` 见[源码](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/go2/sport/sport_client.py)。参数为物理速度，不套用 PPO 的 273 维观测或归一化缩放。

本仓库尚无可直接投入真实导航的该执行服务，因此没有虚构一条“启动即上机”的 CLI。实现时必须具备以下契约，先离线/mock SDK 验收，再现场授权：

1. 单一控制权：人工遥控优先；只有导航激活且定位、感知、机器人状态健康时才接收 SCAN。禁止多个进程同时发送运动命令。
2. Twist 没有时间戳：接收时用单调时钟记录年龄，并核查上游 odom/cloud 原始时间戳；新鲜的 Twist 不能掩盖陈旧点云。协议增加任务/session 和有效期，拒绝旧任务指令。
3. 速度有限值检查、限速、加速度/角加速度限制，周期发送。初始建议上限 vx 0.20、vy 0.10m/s、wz 0.30rad/s，发送 50Hz、Twist 超时 0.30s；这些是待现场测量调整的工程起始值，不是厂家安全保证。
4. 任何未激活、取消、目标到达、无有效路线、定位异常、感知过期、SDK 返回错误、心跳超时、进程退出都进入停止锁存；按 SDK 接口执行 StopMove 并核对实际速度下降，人工确认后才解锁。
5. 独立看门狗/人工急停：执行进程被 kill 后不能依赖其 finally 来停车；验证 SDK/机器人对命令失联的真实行为，并配置独立仲裁或硬件停车路径。StopMove 也不等于急停或零制动距离。
6. 记录命令值、年龄、限幅前后值、执行返回码、实际速度、停止原因。上线默认 disarmed，不自动起身/解除原厂运动服务。

伪代码只描述待实现逻辑，不是可运行的上机程序：

```text
每个发送周期：
  若人工接管/任务取消/健康失败/输入过期/SDK故障：锁存停止；请求 StopMove
  否则若已人工启用：校验有限值→物理限速→斜率限制→Move
  保存接收与发送时间、返回码、实际速度
独立看门狗：执行器心跳消失→触发独立停车；禁止自动复位
```

另一条路线是“SCAN→MoE→LowCmd”。当前统一 SDK 框架仍在验证，物理切换检查未通过、所有技能未获得硬件资格；不能用离线 CRC 或仿真成功替代资格。首版导航不依赖低层热切换，更不能在导航时直接切翻转特技。详见[统一部署状态](../../docs/UNIFIED_GO2_DEPLOYMENT.md)。

## 7. 从离线到实机的分阶段验收

| 阶段 | 操作与通过条件 | 留存材料 |
| --- | --- | --- |
| A 仿真 | 固定两场景复现；补随机目标、延迟/掉包/退出故障注入 | 参数、种子、轨迹、碰撞/停止记录 |
| B 真实 bag | 仅驱动/LIO/SCAN 回放，无 SDK 发送；验证时间、外参、尺度、地图和轨迹 | bag、TF、延迟/漂移报告 |
| C 原厂运动基线 | 现场人员按厂家流程检查模式/控制权；低速各方向与停止测试 | 固件/序列号、网卡、方向、停止时间/距离 |
| D 闭环目标点 | 空旷平地，从短距离前进/侧移/转弯到静态绕障；全程可人工接管 | 指令与实测速度、终点误差、最小间距 |
| E 失效测试 | 断点云/定位、杀规划器/执行器、断网页/网络、取消任务；全部可靠停车且不自启 | 每项触发时间、停车时间/距离、锁存状态 |
| F 完整路线 | 建图/重定位、地点/航点、多段路线、障碍阻断、任务取消与恢复 | 路线完成率、重定位误差、失败原因 |

每阶段由现场负责人签认后才升级；当前 A 仅有固定两场景证据，B–F 未执行。楼梯、坡地、透明/低矮障碍、动态人群和户外强光须各自专项验收，不能由平地测试推断。

实际安全距离应覆盖机身含传感器/载荷轮廓、定位误差、感知盲区、最坏链路延迟移动距离和实测制动距离。使用测量的高分位延迟和最大停车距离确定限速/膨胀；不把 planner 默认半径当完整安全包络。

## 8. 部署运维、网页与交接

随身端拆分服务：sensor-driver、lio、sensor-adapter、scan-planner/controller、velocity-gate、watchdog、telemetry/task-api。规划/感知可由 Supervisor/systemd 重启；运动服务重启必须回到 disarmed，不能自动恢复旧任务。启动顺序：传感器→时间/TF→LIO健康→SCAN就绪→任务层→人工启用速度门；关闭先取消任务并确认停止，再关执行与感知。

网页硬件入口应单独显示“实机”、机器人身份、定位/点云年龄、控制权、速度门状态、任务与停止原因；开始/取消通过认证任务 API，断网页不能使机器人失控。MuJoCo `/scan-live-api` 及 UDP8771 只用于仿真。网页视频可替换为机载摄像头，不能把仿真渲染冒充真实观测。

每次发布记录：项目/SCAN/SDK/驱动/LIO提交、环境、机器人固件、传感器序列号、外参/内参/时间标定、地图和参数哈希、所有验收 bag/视频/指标、已知失败与回滚版本。当前发布资源见[交接文档](../../handover.md)；还没有可宣称为“当前 Go2 完整导航实机通过”的报告。

## 9. 故障定位与资料

- 地图随机器人旋转/重影：检查点云坐标和 cloud_is_world，避免重复外参变换；查时间同步与去畸变。
- 有点云无轨迹：检查 odom有效性、目标坐标、订阅者、可通行空间和规划日志。
- 有轨迹不走：检查任务是否启用、速度门锁存、控制权、SDK返回码；不要绕开保护。
- 路线正确却擦碰：检查实际轮廓、定位误差、底层跟踪/延迟和制动距离，降低速度后重新验收。
- 重定位跳变：取消旧轨迹、锁存停止，重新对齐 map/odom 和路线，再人工恢复。

固定来源：[SCAN ROS2 源码](https://github.com/wuyi2121/SCAN-Planner/tree/d0b921c9b05a6d291d144d60882b2e0e88d2c0e0)、[real/sim launch](https://github.com/wuyi2121/SCAN-Planner/blob/d0b921c9b05a6d291d144d60882b2e0e88d2c0e0/src/planner/plan_manage/launch/run.launch.py)、[宇树 SDK](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206)。上游 README 明确 LIO、相机和宇树驱动为外部依赖；本手册的限速、安全门与验收步骤是项目集成设计，尚未经过该机器人的现场验证。
