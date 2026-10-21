# Go2 后空翻 SDK2 部署手册（方案）

范围：仅Go2本体、Unitree SDK2低层控制。本次不连接机器人、不发送电机命令、不做实机验证。

当前阻止实机使用的已知问题：上游预训练模型的MuJoCo回合虽完成约2π翻转并落稳，但关节速度39.10rad/s、越限0.1077rad，硬件约束失败；本次5000轮完整重训已经结束，自身最终模型同样存在约束失败；1ms积分对照仍未解决全部非足端接触和速度问题。必须先完成[仿真验收](SIMULATION.md)。下述命令是后续实施方案，不是已经执行的实机部署。

来源固定为954aa61ceb62c515689677f44a21e32109f6b62d的deploy_real/，包括独立配置加载、60维actor、50Hz策略与500HzLowCmd线程。只下载源码和策略，不依赖外接机械臂或轮式底盘。

在机器人本地控制计算机建立隔离环境：

```bash
python3.11 -m venv ~/venvs/go2-backflip
source ~/venvs/go2-backflip/bin/activate
python -m pip install numpy PyYAML onnxruntime==1.23.2
cd ~/src
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
git checkout --detach 814556d15970dd2ecf1c9984e845ca02ab07e206
git rev-parse HEAD > ../unitree-sdk2-reviewed-commit.txt
python -m pip install -e .
```

SDK具体提交必须在接机时审查并锁定，不能把上述安装当作固件兼容证明。将已验证的GO2_backflip源码与本次exported/policy.onnx复制到控制计算机，保存模型SHA256与配置。无机器人连接的合约检查：

```bash
cd ~/src/GO2_backflip
python deploy_real/deploy_real_backflip.py --check --onnx ./policy.onnx
sha256sum policy.onnx deploy_real/configs/go2_backflip.yaml
ip addr
```

--check只做配置、ONNX维度及一次推理，不打开DDS。核对实际有线网卡名、机器人型号、固件、低层控制权限、官方SDK依赖和通信网段。SDK电机顺序FR/FL/RR/RL与策略FL/FR/RL/RR要由源配置映射，不能直接写12维数组。

接机阶段先只读LowState，登记关节符号/零位/限位、IMU坐标、单位、CRC和时间戳；再做悬吊低幅检查、四足站立和看门狗/独立接管验证。部署程序会释放内置运动服务并接管关节，启动前须完成单写入仲裁。50Hz策略卡顿时500Hz输出线程必须独立转入经验证的阻尼策略。

只有在本次策略已通过仿真约束、台架和有保护支撑的控制验证之后，才使用以下官方源入口。网卡enp5s0是示例，必须替换实际设备；本项目未执行此命令：

```bash
cd ~/src/GO2_backflip
python deploy_real/deploy_real_backflip.py enp5s0 deploy_real/configs/go2_backflip.yaml --onnx ./policy.onnx
```

先保持当前姿态1秒，2秒过渡到默认站姿，校准足压，START启用actor，A在满足直立/低速度/接触门槛后触发；SELECT及看门狗进入阻尼。源码允许再次触发不代表重复翻转已经训练或验证，首次接机按单次回合验收。不得直接把仿真状态重置搬到真实机器人。

接机日志应包含实测q/dq/tau、IMU、接触、命令、线程周期P99/最大值、观测与动作年龄、电压/温度、急停原因。通过条件须分别覆盖完整旋转、落地稳定、无危险碰撞、硬限位/速度/力矩包络和失联保护。实机表现未知，当前卡片不得标为硬件通过。

依据：[固定版本部署说明](https://github.com/Robot-Nav/GO2_backflip/blob/954aa61ceb62c515689677f44a21e32109f6b62d/deploy_real/README.md)、[Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)。

本次方案的官方SDK源码固定为 `814556d15970dd2ecf1c9984e845ca02ab07e206`；仅作源码核对，未安装SDK、未打开DDS、未连接机器人。实机型号/固件兼容性仍按各技能方案在后续实施时审查。
