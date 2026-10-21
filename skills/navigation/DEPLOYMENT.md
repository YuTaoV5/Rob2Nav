# Go2 导航 SDK2 部署手册（方案）

SCAN 导航请使用[专用实机部署手册](SCAN_DEPLOYMENT.md)与[仿真手册](SCAN_SIMULATION.md)。本页 PPO 观测与归一化参数不用于 SCAN。

本次未接机器人、未发送SDK命令、未做实机验证。网页MuJoCo worker不是硬件驱动。仅Go2本体，不依赖机械臂；真实导航还需要本体传感器或经标定的雷达/相机与里程计来源。

先完成[仿真验收](SIMULATION.md)并固定[训练配置](TRAINING.md)。规则导航与PPO头都输出3维归一化动作，转换为米/秒与弧度/秒：vx=action[0]×0.65，vy=action[1]×0.25，wz=action[2]×1.2。规则控制器内部输出物理速度，接口统一归一化一次，不能在SDK端再次缩放。

机器人控制计算机安装方案：

```bash
python3.11 -m venv ~/venvs/rob2nav
source ~/venvs/rob2nav/bin/activate
python -m pip install numpy scipy torch stable-baselines3==2.6.0
mkdir -p ~/src
cd ~/src
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
git checkout --detach 814556d15970dd2ecf1c9984e845ca02ab07e206
git rev-parse HEAD > ../sdk2-reviewed-commit.txt
python -m pip install -e .
ip addr
```

接机时锁定审查后的SDK提交、Go2型号/固件、DDS网卡、低层权限与网络配置。安装完成不证明本机兼容，当前项目没有可直接执行的实机导航发送节点；下面明确列出必须实现的适配，不提供虚构启动命令。

1. 在机器人本地运行激光/深度处理、时间同步与里程计。仿真lidar_position=(0,0,.18)m、120方位×9仰角、噪声/丢失模型不是实际雷达驱动；实机需按真实安装外参重采样为相同点云与72扇区。273维PPO观测中的深度项在lidar模式必须保持同样屏蔽语义；不要向同维数组填入不相容物理量。
2. 网页只发布目标和开始/停止；目标必须带frame_id、时间戳、序号和会话。机器人本地把目标变换到连续局部里程计坐标。网页/服务器不参与实时电机循环。
3. 高层SDK路线可将物理速度送入SportClient.Move，并用StopMove停止；此时使用厂家运动控制器，和MuJoCo MoE底层不同，必须重新评估速度跟踪与停止距离，不是策略原样迁移。
4. 保持同一MoE底层的路线需要另实现SDK2 LowState→与上游完全一致的5×45历史观测→MoE关节目标→500Hz有界PD/力矩LowCmd。固定关节映射、IMU四元数顺序、动作缩放、默认姿态、PD、推理延迟、CRC、状态时效与单写入仲裁。低层不能同时运行SportClient运动服务。不得把导航头的3维速度直接当12维电机目标。
5. 实机本地看门狗独立检查状态与速度指令超时、倾斜、电源、温度、足端接触异常和求解/推理数值。网页2.5秒超时仅用于演示，不直接用于实机。机器人端应按实际控制周期与制动能力设更短指令寿命并测试停止距离；失效时进入经过验证的站立/阻尼保护，避免简单断电失撑。

官方SDK参考：[高层速度与停止示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/go2/sport/sport_client.py)、[低层站立/CRC示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/go2/low_level/go2_stand_example.py)。不要直接运行示例里的翻转等菜单动作来替代本项目策略验证。

接机顺序：只读状态录制与离线回放→悬吊关节/符号校验（低层路线）→无导航速度跟踪和停止→空旷低速定点→固定软障碍绕行→遮挡/里程计漂移/网络断流测试。保存模型/SDK/标定哈希、目标与速度时序、低层状态和同步视频。验收至少检查到达误差、碰撞/跌倒、停止距离、状态年龄和本地急停接管。实机方案仍需驱动与传感适配开发；本次只交付方案及仿真证据。

本次方案的官方SDK源码固定为 `814556d15970dd2ecf1c9984e845ca02ab07e206`；初次为源码核对，随后已在独立go2-sdk-check环境验证真实SDK消息与CRC；未初始化DDS、未连接机器人。统一部署框架仍在物理验证中，见最新handover及docs/UNIFIED_GO2_DEPLOYMENT.md。实机型号/固件兼容性仍按各技能方案在后续实施时审查。
