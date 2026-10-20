# Go2 侧空翻 SDK2 部署方案

技能合约：69维actor、12动作、scale0.5、PD40/1、50Hz。观测依次为参考关节位置/速度24、anchor方向矩阵前两列6、机身角速度3、相对默认关节位置12、关节速度12、前一动作12。读取policy.json和参考NPZ的实际原生关节顺序。

参考相位由本地单调时钟按50Hz推进。启动时仅将参考的yaw对齐实测站立yaw，保留真实姿态、速度和动作历史；不重置机器人状态。参考末尾保持最后参考并观察落地恢复；再次触发须恢复四足站稳并单独验证重复动作。训练中随机参考相位重置只属于episode初始化，不能搬到实机动作中。

部署前冻结模型、参考、默认姿态和SDK提交。当前两轮模型没有翻转且速度约束失败，完整模型未验收。厂家高层LeftFlip固件动作是另一条实现路线，不能用它替代本策略训练/迁移证据。

本次范围为SDK2部署方案：未连接Go2、未发布LowCmd、未做实机验收。当前完整训练与独立仿真未完成，不应直接执行动态特技。

控制计算机准备（以下仅安装与离线文件操作）：

```bash
python3.11 -m venv ~/venvs/go2-skills
source ~/venvs/go2-skills/bin/activate
python -m pip install numpy PyYAML torch
mkdir -p ~/src
cd ~/src
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
git rev-parse HEAD > ../sdk2-reviewed-commit.txt
python -m pip install -e .
ip addr
```

SDK依赖CycloneDDS0.10.2。若安装时找不到DDS，先在~/src克隆eclipse-cyclonedds/cyclonedds的releases/0.10.x分支，用cmake -S/-B构建并安装到用户目录，设置CYCLONEDDS_HOME后重试；具体依赖见[官方SDK安装说明](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/README.md)。接机时记录并锁定审查后的SDK提交、Go2型号/固件、低层控制权限和网卡。本文未执行SDK安装或通信。

将[仿真手册](SIMULATION.md)导出的policy.pt和policy.json、配置清单复制到机器人本地控制计算机，核对SHA256。控制必须在机器人本地运行，网页/SSH不进入实时电机链。

本次Isaac原生关节顺序为FL/FR/RL/RR的hip组、thigh组、calf组；对应SDK2索引为`[3,0,9,6,4,1,10,7,5,2,11,8]`。每次仍以policy.json的joint_names按名称建立映射，不能假定所有Go2权重顺序一致。统一弧度、弧度/秒、N·m；IMU四元数转为wxyz并校验轴向。

机器人端实现两层：50Hz策略线程构建观测、TorchScript推理、校验有限数并生成12关节目标；独立500Hz低层线程读取最新状态和有时效的完整目标，计算`tau=40*(q_des-q)-dq`，按具体URDF及真实电机包络限幅，再构建带CRC的SDK2 LowCmd。使用位置PD或显式力矩模式只能二选一，避免电机端再次叠加PD。训练中的200Hz物理积分不是500Hz实机时延验证。

低层线程需状态时间戳、命令年龄、关节/速度/力矩/变化率、IMU、接触、电源和温度看门狗；策略卡顿或DDS断流由本地保护接管。高层运动服务与低层发送器只允许一个执行器写入者。起步先四足站稳，再平滑接管默认姿态，操作员明确触发一次动作；物理接管/急停独立于网页。具体保护模式需依据本机固件验证，不能无条件清零力矩导致失撑。

交付实机驱动时还需实现只读录制→离线回放/dry-run→悬吊映射与限幅→四足站立/断流接管→受保护动态动作的分阶段验收。记录原始q/dq/tau、IMU、命令、接触和控制周期P99/最大值。当前仓库提供训练和仿真代码，SDK发送节点属于本方案后续接机实现内容；没有虚构可直接运行的发送器命令。

官方接口依据：[Go2低层LowState/LowCmd与CRC示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/low_level/go2_stand_example.py)。该示例本身会控制电机，不能当作只读采集程序。实机策略加载前必须通过本次模型的MuJoCo任务与约束检查；仿真通过仍不等于硬件通过。
