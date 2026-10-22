# Go2 EDU：Noetic / Foxy 双环境实机导航操作手册

> 文档类型：研究记录。新环境先阅读[环境准备与路径约定](环境准备与路径约定.md)，从当前仓库根目录设置变量。历史报告中的成绩、模型编号和运行状态属于记录当时；当前是否可执行、是否通过，以对应技能中文手册及实际模型/评估文件为准。原始配置和日志仍保留在归档中，不能原样执行其中的机器路径。


更新：2026-10-22。

适用硬件：你的 **Go2 EDU + 机载 Ubuntu 20.04 + ROS1 Noetic / ROS2 Foxy + MID360 + 可选 D435i**。先按第2节读取系统信息；如果实际不是20.04，或是不同JetPack/架构，先核对依赖，不能拿服务器Jazzy环境覆盖机载系统。

本文交付部署步骤和配套参考实现；**尚未在你的Go2 EDU上编译、连接传感器或完成行走验收**。配套工具通过的离线检查见末节，不能代替现场停车、坐标和控制权测试。

## 1. 先选定一条主线

**首版全程使用Noetic：MID360驱动 → Elevator-LIO → 本仓库坐标适配 → SCAN ROS1 → 速度门 → Unitree SDK2 SportClient → 原厂步态。** Foxy保留，不卸载、不升级，也不需要在两个ROS之间桥接规划数据。SDK2通过自己的DDS通道访问Go2，不要求导航节点使用ROS2。

这样选择是因为[SCAN原版](https://github.com/wuyi2121/SCAN-Planner/tree/348e8a590a50a5a6bbab8d8c6dcfd171f009be26)与[所选LIO](https://github.com/xiaofan4122/Elevator-LIO/tree/1d79af77f3d9747ea57ef52a9b01d326a8ec561a)提供Noetic入口，而本网站现有SCAN ROS2仿真基于另一条社区分支/Jazzy。它们不能只替换一行`source`就视为Foxy兼容。

首个现场目标是：**平地、低速、短距离目标导航与静态绕障**。随后再做保存地图/复定位/分段路线。人物视觉跟随、标准楼梯、真实沙地、托爪和网站实机远控分别验收；安装两个传感器不会自动获得这些能力。

阅读顺序：第2–7节只做一次；每天运行看第8–10节；地图看第11节；问题查第14节。所有Linux命令都在**机载电脑**执行，不是在Windows或研究服务器上执行。

## 2. 记录硬件，保留厂家环境

```bash
lsb_release -ds
uname -m
python3 --version
ls /opt/ros
ip -br address
ip route
cat /etc/nv_tegra_release 2>/dev/null || true
```

保留输出、Go2固件版本、机载电脑型号、MID360序列号/固件及传感器照片。`aarch64`表示ARM64，要在机载电脑原生编译，不能复制服务器x86二进制；上游LIO对ARM64未给出与x86相同的回归声明。保留厂家运动服务和遥控接管，首版不使用LowCmd、不关闭sport_mode、不刷机。

建立目录并取本仓库：

```bash
cd ~
git clone https://github.com/YuTaoV5/Rob2Nav.git
mkdir -p ~/go2edu_nav/{config,logs,bags,maps,generated}
bash ~/Rob2Nav/deployment/go2edu/open_noetic.sh
```

若`~/Rob2Nav`已经存在，使用原有checkout并核对`git status`，不要覆盖本地工作。**以后每个新终端先运行最后一条命令**；它只为当前子shell加载Noetic，清除继承的Foxy/Conda路径，不修改厂家的`.bashrc`。本手册不要求重新安装已存在的ROS。Noetic/Foxy均是旧发行版；apt源不可用时先恢复对应发行版的签名仓库或厂家离线包，不能混用Jammy/Noble的deb补洞。

## 3. 安装MID360和D435i

### 3.1 安装与接线

可参考[SCAN配套Go2 EDU双传感器支架CAD](https://github.com/zhechen003/GO2-EDU-sensor_layout)。先试装，检查螺钉啮合、刚度、防松、线缆应力释放、俯仰运动时净空及视野；CAD中的尺寸/外参不是你装机后的标定值。

MID360固定在机身前上方，尽量避免机壳、把手、相机和线束挡住扫描窗口。首版建议采用容易测量的安装姿态；如果倾斜安装，将实际完整旋转填入外参，不能只修改点云frame_id。需要俯视近地障碍时，用实测视野决定倾角，不强行把D435i的向下60°复用给MID360。

接线：MID360航空线的**电源端接经过核算的稳压供电、RJ45接机载以太网**；D435i接可靠USB3口。MID360厂家规定9–27V、推荐12V，不能把Go2电池原始电压直接当成可用电源，也不能将普通PoE直接接MID360 RJ45。核对极性、启动/低温电流、保险和散热，按[Livox手册](https://www.livoxtech.com/mid-360/downloads)对应线序施工。

推荐两个网络接口：一个保持Go2控制链路，另一个USB以太网专接MID360，避免更改SDK网卡后SSH掉线。D435i向下60°可服务托爪，但并不保证前向导航/人员ReID视野。MID360导航不依赖相机，先跑通雷达链。

### 3.2 只配置雷达专用网卡

以下以MID360仍使用192.168.1.0/24网段为前提；已改IP时使用实际网段。用Livox Viewer 2核对设备IP，不只根据序列号猜测。**不要填写当前SSH/Go2控制网卡**。

```bash
ip -br link
read -rp 'MID360专用网卡名，例如 enx...: ' LIDAR_IFACE
read -rp 'Viewer确认的MID360当前IP: ' LIDAR_IP
export LIDAR_IFACE LIDAR_IP
export LIDAR_HOST_IP=192.168.1.50
sudo ip link set dev "$LIDAR_IFACE" up
sudo ip addr add "$LIDAR_HOST_IP/24" dev "$LIDAR_IFACE"
ip route get "$LIDAR_IP"
ping -c 3 "$LIDAR_IP"
```

若报`File exists`，先`ip -br address`核对已有地址；不要删掉其他网卡地址。上述为临时配置，重启后需恢复。现场验证后再由现有NetworkManager/netplan为**该专用接口**持久化相同地址，避免改全局默认路由。Viewer退出后才启动ROS驱动；同一雷达不要同时开ROS1驱动、Foxy驱动和Viewer采集。

## 4. 一次性安装和固定版本构建

本节全部在第2节的干净Noetic shell运行；每段失败先解决，再执行下一段。

### 4.1 系统依赖

```bash
test -f /opt/ros/noetic/setup.bash
sudo apt update
sudo apt install -y build-essential cmake git python3-pip python3-venv \
  python3-numpy python3-yaml python3-opencv python3-rosdep python3-pytest \
  libeigen3-dev libpcl-dev libopencv-dev libyaml-cpp-dev libtbb-dev \
  qtbase5-dev nlohmann-json3-dev libarmadillo-dev \
  ros-noetic-ros-base ros-noetic-catkin ros-noetic-roscpp ros-noetic-rospy ros-noetic-tf \
  ros-noetic-tf2-ros ros-noetic-rviz ros-noetic-pcl-ros \
  ros-noetic-pcl-conversions ros-noetic-std-srvs
rosdep --version
# 仅从未初始化rosdep的机器执行；已有文件时跳过这条
# sudo rosdep init
rosdep update --include-eol-distros
```

若`rosdep update`网络失败，不能把“忽略所有依赖错误”当作构建完成。ARM机器若内存紧张，以下所有`-j2`改`-j1`。

### 4.2 Livox SDK2及ROS1驱动

```bash
cd "$NAV_ROOT"
git clone https://github.com/Livox-SDK/Livox-SDK2.git
git -C Livox-SDK2 checkout c0796f04c143143899c87a773d9f6b7136453c0b
cmake -S Livox-SDK2 -B Livox-SDK2/build -DCMAKE_BUILD_TYPE=Release
cmake --build Livox-SDK2/build -j2
sudo cmake --install Livox-SDK2/build
sudo ldconfig

mkdir -p "$NAV_ROOT/ws_livox/src"
git clone https://github.com/Livox-SDK/livox_ros_driver2.git "$NAV_ROOT/ws_livox/src/livox_ros_driver2"
cd "$NAV_ROOT/ws_livox/src/livox_ros_driver2"
git checkout 21445540f0d100dc86a7e6df312dd70bbdb4afdf
./build.sh ROS1
source "$NAV_ROOT/ws_livox/devel/setup.bash"
```

按[官方驱动](https://github.com/Livox-SDK/livox_ros_driver2/tree/21445540f0d100dc86a7e6df312dd70bbdb4afdf)使用CustomMsg入口，不能用只发布PointCloud2的launch代替LIO原始输入。改接收地址（同一个终端保留第3节的两个IP变量；新终端重新填写）：

```bash
export MID360_CONFIG="$NAV_ROOT/ws_livox/src/livox_ros_driver2/config/MID360_config.json"
python3 - <<'PY'
import json,os,ipaddress
p=os.environ['MID360_CONFIG']
host=str(ipaddress.IPv4Address(os.environ['LIDAR_HOST_IP']))
lidar=str(ipaddress.IPv4Address(os.environ['LIDAR_IP']))
assert host!=lidar
c=json.load(open(p))
for key in ('cmd_data_ip','push_msg_ip','point_data_ip','imu_data_ip'):
    c['MID360']['host_net_info'][key]=host
c['lidar_configs'][0]['ip']=lidar
# 保持驱动点坐标为雷达原始坐标，安装外参交给LIO，不重复应用。
assert all(v==0 for v in c['lidar_configs'][0]['extrinsic_parameter'].values())
with open(p,'w') as f:json.dump(c,f,indent=2)
print(p,host,lidar)
PY
```

此文件指定的是收数地址，不会把物理雷达自动改成新IP。修改后还要核对launch确实加载此文件。

### 4.3 Elevator-LIO

```bash
mkdir -p "$NAV_ROOT/ws_lio/src"
git clone https://github.com/xiaofan4122/Elevator-LIO.git "$NAV_ROOT/ws_lio/src/Elevator-LIO"
cd "$NAV_ROOT/ws_lio/src/Elevator-LIO"
git checkout 1d79af77f3d9747ea57ef52a9b01d326a8ec561a
git submodule update --init --recursive
cd "$NAV_ROOT/ws_lio"
rosdep install --from-paths src --ignore-src --rosdistro noetic -y --skip-keys nlohmann-json-dev
catkin_make -j2 -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
rosmsg md5 livox_ros_driver2/CustomMsg
rosmsg md5 lio/CustomMsg
```

最后两行应相同。配套`livox_relay.py`会显式转换两个包的消息命名空间，并在MD5不同时拒绝启动，避免直接假设`lio/CustomMsg`等同驱动包消息。

### 4.4 SCAN ROS1原版

```bash
cd "$NAV_ROOT"
git clone https://github.com/wuyi2121/SCAN-Planner.git
cd SCAN-Planner
git checkout 348e8a590a50a5a6bbab8d8c6dcfd171f009be26
git submodule update --init --recursive
rosdep install --from-paths src --ignore-src --rosdistro noetic -y
catkin_make -j2 -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
rospack find scan_planner
```

这是与网页ROS2仿真**不同的源码锁定**，不应用网页的动态占用老化补丁；本手册后续使用生成的专用launch，避开上游默认仿真及预置安装外参。

### 4.5 Unitree SDK2的独立Python环境

使用[固定SDK](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206)，不依赖Foxy的Python overlay。先装库，**不运行厂家运动示例**。

```bash
cd "$NAV_ROOT"
git clone --branch 0.10.2 --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git
cmake -S cyclonedds -B cyclonedds/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$NAV_ROOT/cyclonedds/install"
cmake --build cyclonedds/build -j2
cmake --install cyclonedds/build
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git -C unitree_sdk2_python checkout 814556d15970dd2ecf1c9984e845ca02ab07e206
python3 -m venv --system-site-packages "$NAV_ROOT/sdk_env"
source "$NAV_ROOT/sdk_env/bin/activate"
python -m pip install 'pip<25' 'cyclonedds==0.10.2'
python -m pip install --no-deps -e "$NAV_ROOT/unitree_sdk2_python"
python -c "import rospy,numpy,cv2; from unitree_sdk2py.go2.sport.sport_client import SportClient; print('imports OK; no robot connection')"
deactivate
```

`--no-deps`保留系统ROS兼容的numpy/OpenCV，由上面的apt提供，避免在机载系统全局升级Python库。若SDK找不到Cyclone库，检查`CYCLONEDDS_HOME`和`LD_LIBRARY_PATH`是否来自`env.noetic`；不要在此shell再source厂家Foxy。

## 5. 一次性标定：只需要填你的真实参数

完整采集和独立验证步骤见[传感器安装标定总手册](D435i与MID360安装标定手册.md)。本节将结果接入程序，不用“相机60°”代替标定。

| 输入 | 如何得到 | 用在哪里 |
| --- | --- | --- |
| `T_base_lidar` | 确定Go2机身参考原点，测量雷达点云原点的xyz与安装旋转；用已知墙/地面和独立测点复核 | LIO机身外参，自动取逆 |
| `T_imu_lidar` | MID360内部雷达到内置IMU的厂家定义/对应标定；不能用机身IMU顶替 | LIO和传感器位姿 |
| `initial_base_height_m` | 标准四足站立时，机身参考原点到水平地面的实测高度 | 使nav_odom中地面接近z=0 |
| SDK网卡与雷达IP | `ip -br address`、厂家网络资料和Viewer | SDK控制接口、驱动配置 |

约定`T_A_B`满足`p_A=T_A_B*p_B`；平移单位米，旋转矩阵右手正交。表单使用roll/pitch/yaw角度，生成`Rz(yaw)*Ry(pitch)*Rx(roll)`。例：MID360内部IMU外参不能因为传感器倾斜就再叠加同一个机身安装角；外参方向填反会导致地图错位。

```bash
python3 "$ROB2NAV_ROOT/deployment/go2edu/calibration_form.py" \
  --out "$NAV_ROOT/config/calibration.json"
python3 -m json.tool "$NAV_ROOT/config/calibration.json"
```

表单不提供虚假的安装默认值，生成`reviewed=false`。核对坐标轴、量纲、矩阵方向和独立测量后，用编辑器将其改成`true`。这是现场标定记录，不是程序已经测得或验收了你的外参。`T_imu_lidar`可参考所锁定LIO传感器配置，但必须与当前MID360型号、驱动坐标设置一致。

生成专用LIO和SCAN配置：

```bash
python3 "$ROB2NAV_ROOT/deployment/go2edu/prepare.py" \
  --calibration "$NAV_ROOT/config/calibration.json" \
  --lio "$NAV_ROOT/ws_lio/src/Elevator-LIO" \
  --scan "$NAV_ROOT/SCAN-Planner" \
  --out "$NAV_ROOT/generated"
```

生成器会拒绝未填写/未审核参数或错误源码版本；生成`root_rob2nav.yaml`及`scan_go2edu.launch`，关闭电梯模式、轮速融合和仿真，不开启网页地图老化。保持上游原近场过滤配置，**现场必须核对0.8m球形盲区及机身自遮挡**；该盲区内不能依赖这条点云链检测障碍，必要时经bag验证调整过滤并加入其他近场感知。

## 6. 时间和坐标：此步骤通过后才能启用运动

三个适配问题已在配套代码中处理：

1. 所锁定LIO的`/LIO/clouds_lidar`实际由[publish.cpp](https://github.com/xiaofan4122/Elevator-LIO/blob/1d79af77f3d9747ea57ef52a9b01d326a8ec561a/src/node/publish.cpp)输出为**world点云**。适配器将其z整体平移到`nav_odom`并发布`/rob2nav/cloud_world`；SCAN设`cloud_is_world=true`，不能再次旋转。
2. 雷达原点从`world_T_imu * T_imu_lidar`计算；SCAN设`need_extrinsic=false`，避免上游硬编码外参再应用一次。
3. 该LIO机身Odometry的twist未填。适配器从带测量时间的真实机身位置差分世界系速度，供SCAN使用；不是从速度指令积分伪造里程计。此专用话题twist按SCAN约定为世界系速度，不应直接当作通用ROS child-frame twist输入其他融合器。

首次站立的实测高度用于平移所有位姿和点云；保持原x/y和重力对齐方向，发布`nav_odom → world` TF。地面应接近z=0，机身应接近你测量的高度。**LIO重新初始化、换地图或外参改变时，先停车，重启适配器和SCAN，重新设目标。** 不在运动中偷偷重设原点。

MID360点云与IMU必须共享正确测量时基。先用Viewer/固件及对应驱动核对时间模式；检查ROS时间戳与机载系统时间差、点云/IMU对齐和实际延迟。需要PTP时按设备和网卡支持配置，不能用NTP/chrony替代雷达硬件时间同步，也不要在适配器里把测量时间改成“现在”。超过时间门限时程序会拒收，应解决同步/处理延迟，不直接调大门限掩盖。

## 7. 可选：D435i驱动

主导航链可以暂时不启动D435i。要同时检查相机，在干净Noetic shell中：

```bash
sudo apt install ros-noetic-realsense2-camera ros-noetic-realsense2-description
roslaunch realsense2_camera rs_camera.launch align_depth:=true \
  enable_gyro:=true enable_accel:=true unite_imu_method:=linear_interpolation
```

另一个终端：

```bash
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
rostopic echo -n 1 /camera/color/camera_info
```

若Noetic相机deb无候选版本，参考[Noetic wrapper 2.3.2](https://github.com/realsenseai/realsense-ros/tree/2.3.2)及其匹配librealsense 2.50.0构建；Jetson按[官方Jetson说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md)匹配实际JetPack，不能盲装其他内核的DKMS。60°托爪机位仍需要`T_base_color_optical`及深度测距验证，详见[真实相机托爪方案](../go2_nav/paw_contact/真实手掌识别与部署方案.md)。本导航适配不包含真实手掌检测器。

## 8. 每天启动：先只读/预览，不连接执行器

每个终端先执行：

```bash
bash ~/Rob2Nav/deployment/go2edu/open_noetic.sh
```

通过机载桌面/VNC或SSH X转发运行RViz。当前ROS1只绑定本机master地址；若要笔记本原生RViz，需要另配双方可达ROS_IP/ROS_MASTER_URI，不能只改一端。首版优先在机载环境操作，减少网络变量。

**终端A：ROS master**

```bash
roscore
```

**终端B：MID360驱动**

```bash
roslaunch livox_ros_driver2 msg_MID360.launch
```

**终端C：消息命名空间转换**

```bash
python3 "$ROB2NAV_ROOT/deployment/go2edu/livox_relay.py"
```

**终端D：LIO，不自动打开RViz**。机器人保持标准站立且静止，让IMU完成初始化：

```bash
rosrun lio lio _config_path:=root_rob2nav.yaml
```

**终端E：真实坐标适配**

```bash
python3 "$ROB2NAV_ROOT/deployment/go2edu/pose_adapter.py" \
  --calibration "$NAV_ROOT/config/calibration.json"
```

**终端F：SCAN规划与闭环控制器**

```bash
roslaunch "$NAV_ROOT/generated/scan_go2edu.launch"
```

**终端G：不连接SDK的速度门预览**

```bash
python3 "$ROB2NAV_ROOT/deployment/go2edu/velocity_gate.py"
```

默认不import SDK、不打开Go2 DDS、不发送Move/StopMove；只检查ROS输入与显示限幅后的意图。

**终端H：观察与记录**

```bash
rostopic hz /livox/imu
rostopic hz /rob2nav/livox_custom
rostopic hz /rob2nav/body_pose
rostopic hz /rob2nav/cloud_world
rostopic echo -n 1 /rob2nav/body_pose
rosparam get /scan_planner_node/grid_map/cloud_is_world
rosparam get /scan_planner_node/grid_map/need_extrinsic
rostopic info /rob2nav/cmd_vel_raw
```

频率命令用Ctrl+C退出后执行下一条。最后两个参数必须分别为`true`、`false`；原始速度话题应只有SCAN控制器**一个发布者**。雷达典型点云10Hz，IMU看实际固件设置，机身位姿高频输出需稳定且时间戳正确。频率不代表定位成功；还要看静止漂移、旋转重影、尺度、地面高度及CPU温度/处理积压。

另开终端运行`rviz`：Fixed Frame选`nav_odom`，添加PointCloud2 `/rob2nav/cloud_world`、Odometry `/rob2nav/body_pose`、TF，并按`rostopic list`添加SCAN可视化Marker话题。不要用上游运动学仿真模型动画当实机反馈。

## 9. 预览目标设置和停车逻辑

**终端I：维持操作授权心跳**（只表示这个终端仍在运行，**不是硬件急停或真实手持deadman**）：

```bash
rostopic pub -r 10 /rob2nav/operator_heartbeat std_msgs/Bool 'data: true'
```

**终端J：启用预览并设置目标**

```bash
rosservice call /rob2nav/arm 'data: true'
python3 "$ROB2NAV_ROOT/deployment/go2edu/goal.py" --relative 1.0 0.0
rostopic echo /rob2nav/gate_status
```

返回`success: true`后，才发送新的目标；先前目标不会在重新启用后自动恢复。门控还等待时间戳不早于该目标的新Bspline，以及此后的速度指令，避免直接采用缓存速度；原生Twist没有任务ID，这仍不能替代完整带session的控制协议。机器人此时不会实际行走，因为G是预览模式。状态中应有`execute:false`、`armed:true`和受限速度。

相对目标`--relative 前方米数 左方米数`按**设置目标时的实际机身朝向**换算，避免把“世界x正方向”误当狗面前；例如`--relative 1.0 0.5`表示前1m、左0.5m。绝对目标使用：

```bash
python3 "$ROB2NAV_ROOT/deployment/go2edu/goal.py" --absolute 2.0 1.0
```

绝对坐标必须来自当前`nav_odom`或已确认对齐的保存地图，例子数字不是你现场的可通行点。也可在RViz使用**2D Nav Goal**，Fixed Frame必须为`nav_odom`。当前固定SCAN的交互模式取初次机身高度做目标z，**不保证目标最终朝向，也不靠2D目标自动做楼层切换**。

取消/停止：

```bash
rosservice call /rob2nav/arm 'data: false'
```

也可在I终端Ctrl+C：约0.4秒心跳门限后停止并锁存，不能把0.4秒当成实测制动时间。断点云、暂停LIO、关闭SCAN控制器也应分别触发停车锁存；恢复输入后必须重新arm并发新目标。

速度门还检查输入有限值、原始测量时间、定位跳变、单发布者、SDK状态/姿态及返回码。限速为总平移≤0.20m/s、侧移≤0.10m/s、偏航≤0.30rad/s，有加速度限制；到目标0.35m范围停车，任务最长120秒。这些是首轮工程起点，不是厂家或实测安全保证。

### 9.1 真实bag回放（仍不连接执行器）

先停真实雷达驱动、LIO、适配器、SCAN和速度门，关闭所有执行模式进程；保留roscore。C消息中继可以保留。另选历史bag后，只回放雷达原始输入，避免同时回放旧适配输出造成多个发布者：

```bash
rosparam set /use_sim_time true
read -rp '历史bag绝对路径: ' REPLAY_BAG
rosbag play --clock "$REPLAY_BAG" --topics /livox/lidar /livox/imu
```

在其他干净Noetic终端依次重新运行C/D/E，然后启动回放时钟模式的SCAN和速度门：

```bash
roslaunch "$NAV_ROOT/generated/scan_go2edu.launch" use_sim_time:=true
```

```bash
python3 "$ROB2NAV_ROOT/deployment/go2edu/velocity_gate.py" --replay
```

目标预览仍按I/J操作。`--replay`与`--execute`互斥；实机执行模式拒绝`use_sim_time=true`。回放中机器人位姿来自原始记录，不会响应新的速度意图，不能据此声称完成新的闭环到达。暂停bag时，门控按单调墙钟检测过期并锁存，不随ROS时钟一起冻结。

回到实时运行前停止全部回放节点，执行`rosparam set /use_sim_time false`，再按第8节从真实驱动开始；不要让bag与实机数据混流。

## 10. 现场启用Go2行走：只替换终端G

先完成真实bag回放、机身朝向/位置验证、遥控接管、SDK只读状态、空旷受控场地的停止测试。操作员在旁负责遥控急停。当前参考速度门**没有实现独立于进程的硬件看门狗、完整固件故障码解释或安全认证**；执行器被kill/机载电脑断电时不能依赖Python finally停车，必须先确认机器人通信失联行为和独立停车路径。未通过这些现场准入项时维持预览，不进行无人值守行走。

核对Go2 SDK网卡（与雷达专用网卡区分），只读读取状态：

```bash
read -rp '厂家确认的Go2 SDK网卡名: ' GO2_SDK_IFACE
export GO2_SDK_IFACE
ip -br address show dev "$GO2_SDK_IFACE"
source "$NAV_ROOT/sdk_env/bin/activate"
python "$NAV_ROOT/unitree_sdk2_python/example/high_level/read_highstate.py" "$GO2_SDK_IFACE"
```

Ctrl+C退出只读程序。停止原G预览进程，确认只剩一个速度门；重新开G、进入Noetic shell并填写网卡名后执行：

```bash
source "$NAV_ROOT/sdk_env/bin/activate"
read -rp 'Go2 SDK网卡名: ' GO2_SDK_IFACE
python "$ROB2NAV_ROOT/deployment/go2edu/velocity_gate.py" \
  --execute --interface "$GO2_SDK_IFACE"
```

此命令才会连接SDK，但仍默认disarmed，不自动起身、不切运动模式。**通过厂家流程进入可接受SportClient速度命令的状态，保持原厂sport服务运行**。停止其他自主控制进程，遥控接管按厂家流程核实，不能假定任何遥控输入都会自动压过SDK。

保持I心跳，再在J执行arm、新目标；第一目标仅0.5m：

```bash
rosservice call /rob2nav/arm 'data: true'
python3 "$ROB2NAV_ROOT/deployment/go2edu/goal.py" --relative 0.5 0.0
```

观察真实Go2与`gate_status`中的`execute:true`，核对移动方向、目标误差、StopMove返回和实际停止。然后依次测试前/后/左/右/转向、小绕障，记录最坏延迟和停车距离后再扩大距离。不要调用`StandUp/RecoveryStand`去掩盖异常状态。

建议录包（新终端）：

```bash
rosbag record -O "$NAV_ROOT/bags/acceptance_$(date +%Y%m%d_%H%M%S).bag" \
  /livox/lidar /livox/imu /rob2nav/livox_custom \
  /LIO/odom_imu /LIO/odom_vehicle /LIO/clouds_lidar \
  /rob2nav/body_pose /rob2nav/cloud_world /rob2nav/cmd_vel_raw \
  /move_base_simple/goal /planning/bspline /rob2nav/gate_status /tf /tf_static
```

退出顺序：J disarm → 确认真实站稳 → 停G/I → 停SCAN/适配器 → 停LIO（等待保存）→ 停雷达/相机 → 停roscore。不要先关定位再让机器人继续执行旧目标。

## 11. 保存地图、下次定位和多房间路线

### 11.1 保存地图

首次用原厂遥控低速采集，导航执行器保持disarmed。完成闭环/走廊采集后正常Ctrl+C退出LIO，等文件保存结束：

```bash
find "$NAV_ROOT/ws_lio/src/Elevator-LIO/PCD" -maxdepth 1 -name '*.pcd' -ls
read -rp '本次确认保存完成的PCD绝对路径: ' SAVED_PCD
cp "$SAVED_PCD" "$NAV_ROOT/maps/site_v1.pcd"
cp "$NAV_ROOT/config/calibration.json" "$NAV_ROOT/maps/site_v1.calibration.json"
sha256sum "$NAV_ROOT/maps/site_v1.pcd" "$NAV_ROOT/maps/site_v1.calibration.json"
```

同时拍照标记**启动位置、朝向和站姿**，保存实际位姿/TF、目标坐标、源码版本和bag。PCD本身不是可通行地图，也不自动包含全局路线。

### 11.2 从已知起始位姿复定位

该LIO的重定位不是任意位置一键全局匹配；首版回到采集起点附近、相同朝向/安装，按[上游说明](https://github.com/xiaofan4122/Elevator-LIO/blob/1d79af77f3d9747ea57ef52a9b01d326a8ec561a/yaml/README.md)复定位。不要在另一间房直接载入地图后就arm。

```bash
cp "$NAV_ROOT/maps/site_v1.pcd" "$NAV_ROOT/ws_lio/src/Elevator-LIO/PCD/rob2nav_site_v1.pcd"
python3 - <<'PY'
from pathlib import Path
import os,yaml
p=Path(os.environ['NAV_ROOT'])/'ws_lio/src/Elevator-LIO/yaml'
c=yaml.safe_load((p/'runtime/rob2nav.yaml').read_text())
c['relocation']['relocation_enable']=True
c['relocation']['pcd_load_name']='rob2nav_site_v1.pcd'
(p/'runtime/rob2nav_reloc.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
r=yaml.safe_load((p/'root_rob2nav.yaml').read_text());r['runtime_config']='runtime/rob2nav_reloc.yaml'
(p/'root_rob2nav_reloc.yaml').write_text(yaml.safe_dump(r,sort_keys=False))
PY
rosrun lio lio _config_path:=root_rob2nav_reloc.yaml
```

重新运行适配器和SCAN，核对已知墙角/地面与点云、机身位置误差，再发送保存的绝对目标。安装改变后重新标定并重建/核验地图。

### 11.3 多房间完整导航还需要什么

本套可运行入口实现的是**连续定位下的局部目标导航**；本仓库网页的建筑A*没有被自动迁移到实机。复杂房间先用若干已核验的门洞/走廊目标逐段导航：每段到达后重新arm、发下一目标。不要把隔墙后的终点直接输入，就宣称具备完整建筑级全局搜索。

要实现“选会议室→自己找门→绕过封闭路段→到达”，需在已验收基础上加：可通行2D/2.5D地图或拓扑图、全局定位/重定位、A*/全局路线、航点调度/取消/到达、无路可走处理及统一安全仲裁。SCAN的`navi_mode=3`可接外部`/initial_path`，但本参考速度门/launch只验收目标模式1；不能直接改成3就跳过任务层适配。楼梯、沙地还需要底层运动能力和对应地形规划，不能由规划路径保证可走。

## 12. Foxy应该怎么用

你现有Foxy及厂家功能可以保留。导航Noetic终端不sourceFoxy；SDK2的DDS domain 0及网卡由SDK初始化明确指定，不用ROS1 master地址替代。其他ROS2开发工作在独立Foxy终端进行；**同一时刻只让一个自主运动执行者控制机器人**。

若未来要求全ROS2：需把驱动、LIO、SCAN、适配器和速度门在同一发行版原生构建/验收。[官方Unitree ROS2](https://github.com/unitreerobotics/unitree_ros2)提供Foxy配置参考，但不因此保证所选SCAN/LIO ROS2版本支持Foxy。本次不把Humble/Jazzy命令冒充Foxy一键部署，也不要求你为本方案重装机载系统。

确需ROS1/ROS2互通时，单独隔离`ros1_bridge`终端和话题白名单；自定义`CustomMsg/Bspline`要匹配两端定义及桥接生成规则。不要将整个控制图无差别双向桥接，也不同时启动两套雷达驱动。本主线无需这一步。

## 13. 配套文件与验证范围

| 文件 | 用途 |
| --- | --- |
| `deployment/go2edu/open_noetic.sh` / `env.noetic` | 独立Noetic终端与固定工作空间 |
| `calibration_form.py` / `calibration.template.json` | 录入实测外参，缺项不会自动补单位阵 |
| `prepare.py` | 检查源码pin、生成LIO和低速SCAN配置 |
| `livox_relay.py` | ROS1驱动/LIO消息命名空间转换与MD5核对 |
| `pose_adapter.py` | 真实位姿、雷达射线原点、初始高度与world点云适配 |
| `velocity_gate.py` | 默认不连接SDK；新目标授权、过期停车、限速/斜率、SDK返回检查 |
| `goal.py` | 从实际机身朝向设置相对目标，或发送绝对目标 |

离线测试：

```bash
cd "$ROB2NAV_ROOT"
python3 -m pytest -q tests/test_go2edu_deployment.py
```

离线记录见[Go2 EDU验证归档](../artifacts/go2edu_deployment_20261022/README.md)。测试检查无标定拒绝、旋转/逆变换、大小端/行padding点云平移、新目标及新轨迹授权、失联/时间戳/NaN/多发布者/定位跳变停车、限速及到达逻辑，并检查默认main不导入SDK、执行/回放参数互斥；不启动ROS或机器人。主机离线通过不等于机载Noetic完整编译、实际ROS消息兼容、SDK网络、感知精度或停车距离验收。现场B–F阶段仍按[实机总手册](../skills/navigation/SCAN实机部署手册.md)完成并留存报告。

## 14. 常见问题快速定位

| 现象 | 先检查 |
| --- | --- |
| Noetic命令却加载Foxy/Conda库 | 用`open_noetic.sh`开干净shell，不改厂家系统库 |
| ping通雷达但无数据 | host_net_info是否为本机专用网卡地址、UDP端口/防火墙、Viewer或另一驱动是否占用；检查实际网卡UDP流量 |
| LIO没有订阅到点 | 启动C中继，核对两端MD5、CustomMsg入口和IMU时间 |
| 点云重复旋转/随狗移动 | 本锁定LIO输出world，必须走world适配和`cloud_is_world=true`，不能重复外参 |
| 地面抬高/机身在地下 | 校准高度和雷达/机身原点；同时检查位姿、world点云是否应用同一z平移 |
| arm失败 | 看`gate_status`的input_ages；核对I心跳、单一cmd发布者、点云/odom原始时间戳；实机还要新鲜SDK状态 |
| arm成功但不走 | 是否仍为dry-run；是否arm后发新目标；SDK返回码/运动模式/控制权；不要绕开速度门 |
| 机器人已经到点仍想转 | 最终朝向未实现；看实际位置和门控arrived，disarm确认；不要误读旧bspline |
| 关网页后机器人不停止 | 网页目前只控制仿真，实机执行不依赖网页；现场用disarm/授权心跳和独立接管，不能把网页关闭当急停 |
| 节点重启后自己继续旧任务 | 参考门默认disarmed且要求新目标；若发生，排查其他执行者或绕过门的SDK进程 |

首轮验收表应逐项记录：点云/定位频率与P95/P99年龄、静止漂移、已知距离误差、四向和转向、StopMove返回、实测停止时间/距离、断点云/断定位/杀规划器/杀执行器/失去心跳的行为、人工接管及锁存后不会自启。没有数据的项目标“未测”，不填写推测成功。
