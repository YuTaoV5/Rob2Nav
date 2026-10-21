# Go2：D435i / MID360 安装、标定与上线总手册

更新：2026-10-21。适用当前 Rob2Nav 的 SCAN 导航与伸手托爪方案。配套：[SCAN实机部署](../skills/navigation/SCAN_DEPLOYMENT.md)、[真实相机托爪完整流程](../go2_nav/paw_contact/REAL_CAMERA_DEPLOYMENT.md)。

本文给出现场安装和集成步骤；本次未连接传感器或机器人，未完成真实标定，也没有发送电机命令。这里的配置样例、误差预算和频率建议不是本机测量结果。

## 1. 先分清两个功能需要什么

| 功能 | 必需输入/执行 | D435i作用 | MID360作用 |
| --- | --- | --- | --- |
| SCAN 雷达导航 | 三维点云、连续机身/传感器位姿、轨迹执行 | 可选近场补充/视频；不必同时启用深度模式 | 推荐承担点云与雷达惯性定位 |
| 原地伸手托爪 | 米制掌面位置/法向、关节/姿态/支撑状态、接触力与柔顺执行 | 自然手掌 RGB-D 定位的主要传感器 | 非必需；可提供环境感知/机身定位，不能代替精细掌面和接触力 |
| 移动到人前再托爪 | 上述两条链 + 任务/控制权仲裁 | 到位后跟踪手掌 | 到位前导航/定位 |

可在同一 Go2 上固定安装两者，共用标定与时间基准。不要把“驱动已出图/点云”误认为“导航/托爪已经完成”。D435i 内置 IMU、MID360 内置 IMU 和 Go2 IMU 是不同传感器，不能直接拼接当作同一个 IMU。

## 2. 安装前登记和准备

填写机器人版本/序列号/固件、允许的 SDK 控制模式、随身计算机型号/架构/系统/内核、相机和雷达序列号/固件、供电接口规格、网卡名、安装图和负载质量。机器人参数未知时不能确定供电线序、螺钉长度和最终坐标矩阵。

准备：刚性可重复定位支架、合适螺钉与防松件、短且可靠的 USB3 数据线、MID360 配套电源/网线、经核算的稳压与保险、标定板/已知尺寸标记、测量夹具、水平/角度量具、软质测力台和独立急停/人工接管。实际接人手前先用假手或软垫测力台。

建议计算机、雷达与 Go2 控制网络分清接口。示意：Go2专用网口 ↔ 随身计算机；MID360 ↔ 第二网口/受控交换机；D435i ↔ USB3；网页 ↔ Wi-Fi/管理网。不要让更改雷达IP的操作改断机器人控制链。

## 3. 机械安装与供电

### 3.1 MID360

安装在机身上部的刚性位置，避开头部/腿/线束遮挡；按厂家 CAD 和视场模型检查周向及下缘盲区。固定到机身而非会相对晃动的软壳；隔振结构如有明显弹性位移，需要重新评估外参稳定性。不要遮挡散热或把外壳定位面当作点云坐标原点。

[MID360官方手册](https://www.livoxtech.com/mid-360/downloads)给出的供电范围是9–27V DC，推荐12V；启动功率高于稳态。按启动负载、电缆压降和瞬态留裕量，确认最大输入不超限。不能未经核对直接接 Go2 电池，也不能把普通 RJ45 数据连接当成已支持标准 PoE。使用厂家连接定义或经检验的线束；不要凭线色猜线序。

安装孔以本机版本CAD为准；MID360手册标明底部四个M3安装孔、孔深5mm，按支架厚度选择有效嵌入长度，不让螺钉顶到孔底。Go2侧使用经厂家确认的载荷安装位，不直接在电池/机身外壳钻孔。

断电安装，检查极性/短路/机械干涉后再给传感器供电。线缆做应力释放，关节全运动范围内不拉扯、不进入脚部。记录新增质量、安装重心与外形，更新导航碰撞包络。

### 3.2 D435i：向下60°如何定义

这里“向下60°”指在机身标准站姿下，光轴相对**机身水平向前方向**向下偏60°；若你指相对竖直60°，等价角度不同，必须在图纸里明确。支架上的角度刻度不等于相机光学坐标精确姿态。

相机固定在前上方，使左右手目标区域及抬爪路径尽可能处于彩色和有效深度的共同视野。避免鼻部、腿、雷达或支架挡住镜头；检查左右前爪接近手掌后会不会遮住整个掌面。先做可调支架，再根据真实视野确定前伸量/高度；不能只凭60°决定安装位置。

相机使用其正式安装孔/三脚架接口和防旋转支架；[D435i官方数据表安装图](https://www.realsenseai.com/wp-content/uploads/2023/03/Intel-RealSense-D400-Series-Datasheet-March-2023.pdf)列有M3安装位，核对最大嵌入长度和扭矩后再选螺钉。安装孔中心不是光学原点，不能把尺量到外壳的距离直接当光学外参。

USB3直连可靠端口，查看 `lsusb -t` 的链路速度；线缆固定并给散热留空间。普通 D435i 及其连接不能因为 MID360 的防护等级就被当成整套户外防水。相机与雷达固定后拍照、做定位标记；拆装、碰撞、支架滑动后重新检查外参。

## 4. 软件安装：先只读传感器，不接电机执行器

当前 SCAN 已验证环境是 Ubuntu24.04/ROS2 Jazzy；随身端应选所有驱动/LIO都支持的统一发行版。以下以**已安装 ROS2 Jazzy 的 Ubuntu 主机**为例。Jetson、定制内核或其他发行版先按厂商匹配依赖，不能直接照搬 x86 DKMS 安装。

### 4.1 RealSense

[ROS官方驱动仓库](https://github.com/realsenseai/realsense-ros)提供二进制与源码安装路径；选择一条并固定版本，避免 apt 库和源码工作区互相覆盖。

```bash
source /opt/ros/jazzy/setup.bash
sudo apt update
apt-cache policy ros-jazzy-realsense2-camera ros-jazzy-realsense2-description
# 确认有适合本机的候选版本后安装：
sudo apt install ros-jazzy-realsense2-camera ros-jazzy-realsense2-description
ros2 pkg prefix realsense2_camera
ros2 launch realsense2_camera rs_launch.py --show-args
```

如包不可用，按该仓库 release 与对应 librealsense 支持矩阵选择源码构建，使用 rosdep/colcon。底层工具、udev、内核支持按[librealsense Linux安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md)处理；不要在未核对内核时盲装 DKMS。连接后先运行 `rs-enumerate-devices` / `realsense-viewer`（工具安装后），确认序列号、固件、彩色/深度可用和支持的流配置，关闭 Viewer 后再启动 ROS，避免双进程争用。

示例启动：明确命名，彩色与深度开启并对齐。流配置先以设备列出的支持项为准；640×480×30只是起始例子。

```bash
ros2 launch realsense2_camera rs_launch.py \
 camera_namespace:=camera camera_name:=d435i \
 enable_color:=true enable_depth:=true \
 rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30 \
 align_depth.enable:=true enable_sync:=true
```

该命名下预期话题如下，实际以 `ros2 topic list -t` 为准：

```bash
ros2 topic list -t
ros2 topic hz /camera/d435i/color/image_raw
ros2 topic hz /camera/d435i/aligned_depth_to_color/image_raw
ros2 topic echo /camera/d435i/color/camera_info --once
ros2 topic echo /camera/d435i/aligned_depth_to_color/camera_info --once
ros2 topic info /camera/d435i/color/image_raw -v
```

检查编码、分辨率、CameraInfo、时间戳、QoS以及对齐边缘；需要IMU时按驱动支持项单独启用 gyro/accel，不能认为 enable_sync 就完成了相机—雷达—机器人跨设备硬同步。不要同时启动 librealsense 直接采集和 ROS 驱动争抢同一相机。

### 4.2 MID360

先用 Livox Viewer2发现并确认本机雷达IP/序列号/固件和点云，退出Viewer后启动ROS驱动。雷达与主机数据接口使用同一规划子网、无IP冲突；驱动JSON中的每个 host_net_info 地址都要填本机实际网卡地址，lidar_configs.ip 填真实雷达地址。[官方示例JSON](https://github.com/Livox-SDK/livox_ros_driver2/blob/master/config/MID360_config.json)的192.168.1.5/192.168.1.12是示例，不能当作你的设备信息。

```bash
ip -br address
ip route
# 将已确认的雷达IP替换进去：
ping -c 3 <MID360_IP>
```

在新工作目录安装 SDK2，再安装 ROS 驱动。先确定适合本机的 tag/SHA 并执行 checkout，下面省略未知的现场版本号；记录每个 `git rev-parse HEAD`，生产启动不自动跟随 master。

```bash
mkdir -p ~/go2_sensor_ws/src ~/go2_sensor_sources
cd ~/go2_sensor_sources
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
# git checkout <已选定并验证的SDK2_TAG_OR_SHA>
cmake -S . -B build
cmake --build build -j4
sudo cmake --install build
sudo ldconfig
cd ~/go2_sensor_ws/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
cd livox_ros_driver2
# git checkout <已选定并验证的DRIVER_TAG_OR_SHA>
source /opt/ros/jazzy/setup.bash
./build.sh jazzy
source ~/go2_sensor_ws/install/setup.bash
```

上述 jazzy 构建入口来自[当前官方驱动说明](https://github.com/Livox-SDK/livox_ros_driver2#23-build-the-livox-ros-driver-2)；固定老版本可能不同，先看选定提交的 build.sh。修改该版本实际读取的 `MID360_config.json`，核对 install 内的配置路径并重建/安装；不能只改 src 后假设运行时已更新。

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
# 另一个已source的终端：
ros2 topic list -t
ros2 topic info /livox/lidar -v
ros2 topic hz /livox/lidar
ros2 topic hz /livox/imu
```

`msg_MID360_launch.py`通常输出 Livox 自定义点消息，含逐点时序；LIO需要这类信息时保持该格式。SCAN最终需要 PointCloud2，但不能因此在LIO之前丢掉去畸变所需的逐点时间字段。实际话题名/消息类型以选定驱动为准。JSON外参预变换、LIO外参和TF不能重复应用。

### 4.3 冻结软件组合

保存 `uname -a`、`lsb_release -a`、相机/雷达固件、ROS包版本、librealsense/SDK2/驱动/LIO提交、配置文件及哈希、设备序列号。官方 FAST_LIO 主项目主要是ROS1接口；不能把它的 roslaunch 命令当成 Jazzy ROS2 可运行入口。选择维护中的ROS2 LIO移植并单独固定/构建/回放验证，再接SCAN；本项目目前没有经过实机验收的一键LIO安装器。

## 5. 标定清单与顺序

| 项目 | 如何获得 | 输出与验收 |
| --- | --- | --- |
| D435i彩色/深度内参、畸变、内部深度到彩色外参 | 先读工厂参数/CameraInfo；标定板与量距复核，异常再按厂家流程校准 | 分辨率匹配的K/D、对齐验证、深度单位/偏差报告 |
| 相机到机身外参 | 下面的已知标定板夹具/外部定位方案 | `T_base_color_optical`、误差与独立验证点 |
| MID360到机身外参 | 厂家坐标原点+安装测量，使用多面目标/外部参考精校 | `T_base_lidar`，水平面与墙面验证 |
| 雷达到其LIO所用IMU外参 | 核对内部IMU厂家定义或外置IMU联合标定 | 明确方向/单位的 `T_imu_lidar`；去畸变和静态重力检查 |
| 相机/雷达相互外参 | 两者到base的变换可推导；融合时再联合标定 | `T_color_lidar = inv(T_base_color) @ T_base_lidar` |
| 时间 | 测量设备时间域/偏移/漂移，配置同步 | 时钟来源、偏移、P95/P99延迟、丢帧率 |
| Go2关节/足端/接触 | SDK关节顺序、零位、FK及独立测力台 | 足端位置误差、接触力/滞回、负载模型 |

仅将MID360用于LIO、D435i用于相对手掌定位时，不必先做紧耦合雷达—相机联合标定才能开始只读验证；两者各自到机身的外参仍需确认。接入LIVO/彩色点云或跨传感器融合时再增加相互外参和时间的联合优化。不要用LIO在线估计的雷达—IMU外参，冒充相机—机身外参。

## 6. 固定机身相机是否需要手眼标定

**需要相机到机器人机身的外参标定；不一定需要机械臂式 AX=XB 手眼算法。** 相机固定在Go2机身上，相对base近似刚性，所需量为 `^B T_C`，把相机光学坐标的点变到base。人体手掌是被观测目标，不是具有已知机器人FK的末端，不能拿检测到的人手轨迹直接套 `calibrateHandEye`。

统一约定 `^A T_B` 将 B 系点变换到 A 系。base：x前/y左/z上；color optical：x右/y下/z前。ROS约定见[REP103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)。相机驱动通常已发布 camera_link 到各 optical frame 的内部TF，只再发布 base_link 到 camera_link，避免同一TF有两个发布者；若离线文件存的是base到color optical，要先与驱动内部变换求解到camera_link后发布。

### 6.1 推荐：已知位置标定板/夹具

1. 使用打印尺寸经实测的 AprilTag/ChArUco/棋盘格刚性板；明确角点编号、单位和板坐标原点。不要把显示屏缩放后的格长当印刷实际尺寸。
2. 用刚性测量夹具把板放在相对base已知的姿态 `^B T_T`。在选定实际流配置下，以彩色内参/畸变PnP估计 `^C T_T`。
3. 计算 `^B T_C = ^B T_T @ inv(^C T_T)`。使用多个**各自已测量板位姿**覆盖视野/距离/倾角，联合优化固定外参。重复同一位置的100帧不能冒充100个独立姿态。
4. 无法精确测板到base时，改用外部定位系统提供 `^W T_B(i)` 和板的 `^W T_T`，则 `^B T_C = inv(^W T_B(i)) @ ^W T_T @ inv(^C T_T(i))`。另一个方案是经验证FK的机器人腿带标记做手眼标定，但关节零位、腿柔性、足垫偏移会进入误差，不作为首选捷径。
5. 求解后使用**未参与拟合**的左右/中间位置与不同高度验证三维点和掌面法向。检查平移单位、矩阵方向、四元数顺序及旋转行列式；变换一个“相机前方点”和“相机右侧点”做符号验证。

[OpenCV手眼与PnP文档](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html)说明各输入变换方向；如果选用AX=XB，必须先按实际安装推导输入方向和可观测运动，不能照搬eye-in-hand示例。

### 6.2 向下60°的理想几何，仅供查错

无偏航/横滚时，从color optical到base的理想旋转为：

```text
R_B_C = [[0, -sin(alpha),  cos(alpha)],
         [-1, 0,          0],
         [0, -cos(alpha), -sin(alpha)]]
alpha = 60 degrees
p_B = R_B_C p_C + t_B_C
```

真实安装还有平移、支架误差、内部相机frame偏移和机身姿态变化。该矩阵不是现场标定结果；严禁仅输入60°后直接开启触手动作。相机高度H、掌面高度h时，光轴在该高度平面的前方交点约为 `x_camera+(H-h)/tan(60°)`；它只说明中心射线落在哪里，不代表整个左右工作区可见。

用实际 `K` 将可达区网格变到相机再投影，检查彩色边界、深度有效区、最小Z和抬爪遮挡。D435i的最小有效深度与分辨率有关，见[官方D400数据表](https://www.realsenseai.com/download/21345/?tmstv=1778232235)；手很近时不能统一假设“所有配置0.3m内都可用/都不可用”。例如表中D435i 640×480的Min-Z为175mm、1280×720为280mm，实际稳定掌面测量还要留余量并实测。

## 7. MID360 / LIO标定与时间

从厂家点云原点到base测量初值，不从雷达外壳中心随意估计。扫描水平地面、前方/侧方墙面和有厚度的目标，分别检查高度、roll/pitch与yaw；只看单一地面不能充分约束yaw和平面内平移。需要精校时用具有多个非平行面的目标/外部基准联合优化，再在不同位置和转动数据上验收。

优先先使用MID360自身IMU完成LIO；依据选定LIO对外参 `extrinsic_T/R` 的数学定义填写。官方手册公开了内部IMU位置/轴向，但该数值需要按LIO输入方向换算，不能不看方向就复制或认为平移为零。引入D435i/Go2外置IMU时必须额外做相对时延、轴向、尺度/偏置和外参标定。

MID360支持PTPv2/GPS时间同步；按厂商协议、网卡能力和本机时钟架构设置，并验证确实锁定。`ethtool -T <雷达网卡>`查看时间戳能力。单机初步读数可以先测未同步偏移，但不能把“有ROS时间戳”当同步已通过。相机enable_sync主要处理该相机的图像流；它不会同步MID360或SDK。跨机chrony/NTP也不自动等于传感器硬件时钟对齐。

标定/静止初始化先让传感器稳定并按所选LIO要求静置。保存静态IMU平均/方差和运动bag；不要给已包含重力/坐标旋转的数据再重复变换。验收地图在机器人转动时不出现明显重影、水平地面不倾斜、闭环或退化时健康状态能可靠触发停止。

## 8. 采集、离线检查与下一步

先按实际话题名录包；输出目录使用新名字。仅采传感器不代表执行器已获许可。

```bash
ros2 bag record -o sensor_calibration_run01 \
 /camera/d435i/color/image_raw /camera/d435i/color/camera_info \
 /camera/d435i/aligned_depth_to_color/image_raw \
 /camera/d435i/aligned_depth_to_color/camera_info \
 /livox/lidar /livox/imu /tf /tf_static
```

不要用640×480图像配另一分辨率内参。ROS Image为16UC1时查发布者定义的深度单位，32FC1通常以米表示但仍需确认；librealsense原生Z16用设备depth_scale转米。对齐到彩色的深度使用对应color optical投影模型/对齐后的CameraInfo，而不是原始depth K。含畸变图像先用匹配模型校正或SDK反投影；不能忽略非零畸变直接套针孔公式。

新增[离线检查模板](https://github.com/YuTaoV5/Rob2Nav/blob/main/deployment/sensors/calibration.template.json)与[检查工具](https://github.com/YuTaoV5/Rob2Nav/blob/main/scripts/check_sensor_calibration.py)只检查外参刚体合法性、配对点误差、时间摘要和针孔视野几何，不采相机、不解算外参、不连接机器人，也不证明真实深度精度或避障安全。未标定模板会拒绝通过。

离线使用（在已获取本仓库的本机执行，不访问传感器）：

```bash
# 填写模板的真实测量数据后检查；不修改原始模板
python3 scripts/check_sensor_calibration.py --input /path/to/measured_calibration.json --output outputs/sensor_calibration_check.json
# 仅检查60°理想合成例子；必须显式允许合成数据，输出hardware_release始终为false
python3 scripts/check_sensor_calibration.py --input deployment/sensors/calibration.synthetic_60deg.json --allow-synthetic --output outputs/sensor_calibration_synthetic_check.json
```

通过离线检查之后仍需执行下面的实际传感器、状态与控制验收。

### 8.1 SCAN分支

1. 固定驱动、LIO和标定版本；只读回放验证odometry、去畸变点云、TF和健康状态。
2. 编写传感器适配，把实际LIO结果映射到SCAN所需的body/sensor Odometry和PointCloud2；同时正确配置world/local点云和外参，具体接口见[SCAN实机手册](../skills/navigation/SCAN_DEPLOYMENT.md)。本项目仿真UDP桥不接真实传感器。
3. 先只显示地图/轨迹、不启动SDK执行；验证目标坐标和碰撞包络。深度路线还需替换SCAN real launch硬编码相机内参及话题；雷达路线不需要为了装了D435i就强行切depth模式。
4. 通过带TTL/限速/控制权/失联停车的Sport SDK速度门，按空旷低速→静态绕障→故障注入→全局路线逐级验收。当前该真实适配与导航执行闭环尚待实现/现场联调。

### 8.2 托爪分支

按[真实相机托爪流程](../go2_nav/paw_contact/REAL_CAMERA_DEPLOYMENT.md)依次实现：RGB-D目标检测→米制掌面→base/交互锚点转换→只读目标可视化→台架力/支撑控制→假手→有限现场真人测试。没有摄像头识别模块、精确力反馈和实机柔顺控制适配时，安装标定完成也不能直接进入真人托爪。

## 9. 每次开机与变更管理

开机：检查支架/线缆/传感器窗口→供电→确认序列号与USB/以太网→启动驱动→核对时间/TF/标定ID→LIO或手掌目标只读自检→任务与本地保护就绪→现场按对应功能验收流程启用执行。关机先结束任务并确认四足稳定，再关闭执行服务与感知；启动失败不能自动恢复旧目标。

支架松动、相机分辨率/对齐模式/固件改变、传感器拆装、底层机器人模型/足垫或负载改变后，重新评估相关标定与阈值。保留旧标定以便对比，但禁止将旧参数静默用于新硬件。

交接目录建议包含：设备清单、安装CAD/照片/线束图、软件锁定文件、CameraInfo、T矩阵及协方差、标定原始bag、拟合/独立验证点、时间报告、力/FK校准、每阶段通过/失败记录。未填序列号/真实矩阵/验证报告的示例文件不能命名为“实机标定已通过”。

## 10. 本次文档与工具检查

服务器68项Studio测试通过，两份新手册入口已由浏览器实际点击核验。离线工具的[合成60°数学检查](evidence/sensor_calibration_plan_20261021/synthetic.json)、[未标定模板拒绝记录](evidence/sensor_calibration_plan_20261021/template_rejection.json)和[检查范围](evidence/sensor_calibration_plan_20261021/verification.json)一并保存。这里没有真实传感器标定数据，合成数据的零误差不能当作硬件精度。
