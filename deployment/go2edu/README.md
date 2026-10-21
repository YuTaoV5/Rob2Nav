# Go2 EDU机载适配工具

从[Noetic/Foxy逐条命令手册](../../docs/GO2_EDU_NOETIC_FOXY_NAVIGATION.md)开始。目标环境为Ubuntu20.04/Noetic/Python3.8；这些脚本独立于Rob2Nav的训练环境，不需要把整个项目pip安装到机载Python。

先完成MID360和LIO构建、实际外参及时间检查，再生成配置。所有缺失的安装标定值均保持未填写，`prepare.py`拒绝未经审核的配置。

`velocity_gate.py`默认只读ROS输入并预览速度，不导入SDK。`--execute --interface 网卡`才连接SDK，仍默认未启用；重新启用后须发新目标、接收其后的轨迹和速度。`--replay`只用于ROS时钟回放，并与执行模式互斥。

本目录是**参考适配实现**，已有离线测试和命令/配置静态检查；未在你的Go2 EDU上完成原生编译、真实传感器和行走验收。不能代替独立停车路径、人工接管或完整的硬件故障处理。真实部署状态见[离线记录](../../artifacts/go2edu_deployment_20261022/README.md)。
