# MID360：SCAN头部套件与Go2 EDU原厂安装的源码对照

审计日期：2026-10-28。SCAN上游HEAD与本项目锁定版本相同：`348e8a590a50a5a6bbab8d8c6dcfd171f009be26`。用户使用原厂标准安装，但尚无安装照片、支架型号、机载原厂外参或实测参数，因此本页不赋予其任何固定角度/平移。不能填入先前估计的7°。

## 先区分三个模型

1. **作者实物**：[SCAN README](https://github.com/wuyi2121/SCAN-Planner/tree/348e8a590a50a5a6bbab8d8c6dcfd171f009be26)链接了作者的[头部传感器CAD套件](https://github.com/zhechen003/GO2-EDU-sensor_layout)。该套件集成MID360与RealSense，明确属于head-mounted assembly；不能从图片推算你的支架外参。
2. **仿真感知**：CPU `src/simulator/local_sensing/src/pointcloud_render_node.cpp` 和GPU `opengl_render_node.cpp` 默认 `lidar_pitch=45.0`（度）、`vertical_fov=90.0`、`yaw_fov=360.0`、`min_raylength=1.0`（米）。雷达姿态计算叠加Ry(pitch)，但位置直接取body odometry，无真实安装杆臂平移。它不是原厂套件的精确数字孪生。
3. **实机估计/建图**：LIO计算真实IMU、雷达和机身位姿；SCAN使用传入的雷达位姿与点云。实机启动时仿真渲染器关闭，调整上述sim参数不会改变实机外参。

URDF里的`radar_joint`也不能直接当作MID360外参。仅修改URDF、静态TF或RViz显示，不会自动替换grid_map.cpp内部的矩阵计算。

## 坐标与两个不同的外参

统一记号：T_A_B把B坐标变到A；B为原运动控制器使用的机身原点，I为MID360内置IMU，L为原始雷达。Go2机身IMU、MID360内置IMU、base_footprint不能混为同一原点。

- `T_B_L`：雷达相对于机身的完整6自由度安装，换位置/倾角/朝向时需要重新确认。
- `T_I_L`：MID360内部雷达与其内置IMU的相对几何。整台雷达换支架并不改变它。
- `initial_base_height_m` / SCAN `body_height`：初始站立机身参考点离水平地面的高度，不是雷达离地高度。

SCAN [grid_map.cpp](https://github.com/wuyi2121/SCAN-Planner/blob/348e8a590a50a5a6bbab8d8c6dcfd171f009be26/src/planner/plan_env/src/grid_map.cpp#L54-L61) 的`lidar_extrinsic_`为单位旋转和平移`[-0.011,-0.02329,0.04412] m`。它与MID360内置IMU到雷达的名义几何一致；默认实机传入`/LIO/odom_imu`时，矩阵用于从该IMU位姿得到雷达位姿。它不是作者头部支架到狗身的完整外参，不应在此接口下替换成T_B_L。

LIO的`yaml/sensors/livox.yaml`另有`offset.lidar_R_body`和`offset.lidar_t_body`。其默认旋转含90°偏航，平移为`[-0.2,0,-0.15] m`，不是你的原厂安装值。源码`src/node/publish.cpp`按下式生成机身位姿：

`T_W_B = T_W_I × T_I_L × T_L_B`

因此应填`T_L_B = inverse(T_B_L)`；平移是`-R_B_L.T × t_B_L`，不能仅把xyz取负。检查机载原厂SLAM配置中是否已有`T_Dog2lidar`可作候选；必须先核对矩阵方向、Dog原点与当前B是否一致。原厂产品名称本身不足以确认某组数值。

## 实机必须修改或核对的项目

| 项目 | 源码/配置入口 | 按你的安装如何处理 |
|---|---|---|
| 安装6自由度 | 本仓库`data/calibration.json:T_base_lidar`；LIO `offset.lidar_R_body/lidar_t_body` | 写实测T_B_L；prepare.py自动求逆并生成LIO配置。保留前/左/上坐标约定 |
| 内部雷达–IMU关系 | `T_imu_lidar`；LIO `offset.imu_R_lidar/imu_t_lidar` | 同一MID360整机不因换支架修改；名义值还需验证原始驱动坐标 |
| 点云是否已在世界系 | SCAN `grid_map/cloud_is_world` | 本项目锁定LIO的`/LIO/clouds_lidar`实际上已做transLidar2World，必须true |
| SCAN是否再施加外参 | `grid_map/need_extrinsic`与`sensor_pose_topic` | 本项目已计算真实雷达位姿，使用`/rob2nav/lidar_pose`，设false |
| 机身参考高度 | `initial_base_height_m`、`grid_map/body_height` | 量B原点离初始水平地面的高度，不量雷达外壳顶 |
| 自身遮挡/近场 | LIO `yaml/runtime/mapping.yaml:downsample` | 审查`blind=0.8`、`use_box_blind`和`box_corner_*`，不能沿用作者旧位置的盒子 |
| 机器人占据包络 | SCAN `advanced_param.xml:grid_map/double_cylinder_radius/offset`、`obstacles_inflation_z_up/down` | 按狗身、外露支架和运动包络审查；原默认0.25/0.18和上下0.1米不是安装外参 |

额外核对`filter_size`和自适应`max_voxel`，避免低密度/粗体素抹掉16cm台阶；这些是点云保真与性能调参，不能修复错误外参。启用LIO的box盲区后该源码使用盒子路径替代球形blind；盒子定义在L坐标系，倾斜安装下不能直接照抄机身坐标的xyz范围。必须看原始点云，避免把前方踏面一起删掉。

## 上游run.launch的一个实际问题

[run.launch](https://github.com/wuyi2121/SCAN-Planner/blob/348e8a590a50a5a6bbab8d8c6dcfd171f009be26/src/planner/plan_manage/launch/run.launch#L20-L24)在实机分支把`cloud_is_world`设为false，而本项目锁定的Elevator-LIO版本发布的是世界系注册点云。直接照搬该launch会再做一次刚体变换。点云名字中有lidar不代表坐标就在L系，必须查看发布实现。

本项目生成的配置已修正此组合，不必为了换标准支架再改SCAN算法：

```text
body_pose_topic   = /rob2nav/body_pose
sensor_pose_topic = /rob2nav/lidar_pose
cloud_topic       = /rob2nav/cloud_world
grid_map/cloud_is_world = true
grid_map/need_extrinsic = false
grid_map/frame_id = nav_odom
```

pose_adapter从真实IMU位姿计算T_W_L；LIO按已配置安装外参计算T_W_B。机身、雷达、世界点云共同做相同世界z原点平移，不对点云重复旋转。

其他接法的条件：若继续直接传`/LIO/odom_imu`，可保留`need_extrinsic=true`和内部T_I_L；若改传机身位姿且要求SCAN再算雷达位姿，矩阵才应是T_B_L，但该版本不是可直接加载任意TF/矩阵的参数接口，需要参数化C++或上游适配。不要只发布TF后假设算法会自动读取。

## 现场更新顺序

1. 定义与MoE一致的B原点，取得安装座型号/照片与雷达坐标原点尺寸；水平静止、重力与墙面检查姿态，确认完整xyz/roll/pitch/yaw。
2. 复制`configs/calibration.template.json`到`data/calibration.json`，填写实测T_base_lidar和initial_base_height_m，保留正确的T_imu_lidar；独立审阅后才设reviewed=true。
3. 在scan-sim2real根目录运行：

```bash
python scripts/prepare.py --calibration data/calibration.json   --lio sources/lio --scan sources/scan --out generated --navi-mode 3
```

4. 启动生成的LIO/pose_adapter/SCAN配置，检查nav_odom下机身、雷达、地面、墙面与规划轨迹一致；转身后墙不绕错原点，机身姿态不带虚假的固定倾角。
5. 对比原始点云与LIO/SCAN输入，确认近台阶是否由视场不可见、0.8m盲区或体素过滤造成。先观察速度，不以“有点云”宣称实机导航通过。

原始bag不预旋转；已重力对齐的世界系PCD不再按安装角补转。新标定后宜从原始bag重新建图，或明确旧地图的配准版本，不能混用未知外参生成的地点坐标。

## 若需要仿真也与标准安装一致

只改`lidar_pitch`不够。两种renderer应统一支持并使用实测T_B_L（完整旋转和平移），同时修改雷达姿态发布、光线/裁剪的原点及方向。当前CPU/GPU代码均有body位置直接作为ray origin的路径，不能只改发布的odometry或URDF。

MID360[官方规格](https://www.livoxtech.com/mid-360/specs)为水平360°、垂直-7°至52°，不是关于水平对称的90°。把vertical_fov简单改成59°仍不能精确表达上下非对称边界，应在L系用独立俯仰上下限裁剪，再施加实际T_B_L；还需处理狗身/支架遮挡与机器人俯仰。真实Livox输入不需要调renderer视场。

几何示例，仅说明视场影响：假设雷达离地0.50m、雷达+x朝前、前向下倾pitch为θ，在水平地面前向截面内，最低-7°射线的最近地面交点约为`d=0.50/tan(7°+θ)`。θ=0°时约4.07m，θ=7°时约2.00m，θ=13°时约1.37m。这里13°只是算例，绝不代表已确认你的安装角。支架、机身遮挡会进一步改变可见范围。软件外参只能摆正已有数据，不能补回没有被扫描到的近场台阶。

## 源码核对点

- SCAN `grid_map.cpp`：54–61行外参开关与硬编码矩阵，820行起sensorPoseCallback，约894行起世界/传感器点云分支。
- SCAN `run.launch`：实机话题选择与cloud_is_world分支。
- SCAN `pointcloud_render_node.cpp`：212行起publishLidarPose，约1007行和1996–2037行仿真原点/姿态与默认参数；GPU对应opengl_render_node.cpp。
- SCAN `advanced_param.xml`：双圆柱、垂直膨胀、body_height。
- Elevator-LIO固定版本`1d79af77f3d9747ea57ef52a9b01d326a8ec561a`：`yaml/sensors/livox.yaml`、`yaml/runtime/mapping.yaml`、`src/node/publish.cpp`。

这是一份源码审计与参数映射，不是实测安装标定，也未在实机验收。
