# Go2 SDK2 方案源码核对

> 文档类型：研究记录。新环境先阅读[环境准备与路径约定](环境准备与路径约定.md)，从当前仓库根目录设置变量。历史报告中的成绩、模型编号和运行状态属于记录当时；当前是否可执行、是否通过，以对应技能中文手册及实际模型/评估文件为准。原始配置和日志仍保留在归档中，不能原样执行其中的机器路径。


> 后续验证更新（2026-10-21）：已在独立环境安装固定SDK与CycloneDDS0.10.2，仅运行真实IDL/CRC离线检查；未初始化DDS、未创建发布器或连接机器人。下方“仅源码审查”的说明保留为当时阶段记录。最新框架和失败边界见[统一部署文档](统一部署框架与状态机.md)。


本次仅核对官方源码，不执行SDK安装、导入、DDS通信或实机命令。七项能力的部署手册与握爪方案都固定同一提交，避免浮动master导致入口和依赖变化。

固定提交：814556d15970dd2ecf1c9984e845ca02ab07e206。该提交在官方仓库可访问；不是对某个Go2固件兼容性的测试结果。

- [官方固定提交](https://github.com/unitreerobotics/unitree_sdk2_python/commit/814556d15970dd2ecf1c9984e845ca02ab07e206)
- [安装说明](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/README.md)：SDK文档列出Python≥3.8、CycloneDDS0.10.2等依赖；缺少DDS库时需先构建其0.10.x来源并配置路径。各手册安装命令是后续方案，不代表本次执行。
- [Go2高层客户端源码](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/go2/sport/sport_client.py)：用于导航速度/停止方案的接口依据。高层厂家控制器与本项目MuJoCo学习式底层不同。
- [Go2低层站立/CRC示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/go2/low_level/go2_stand_example.py)：可核对LowState/LowCmd话题、CRC、运动服务仲裁与2ms发送周期；这不是建议直接运行动态特技。

前一版导航方案的高层客户端链接已更正为真实存在的库文件。所有实机动作仍停留在方案；仿真结果和SDK源码核对不能替代真实电机边界、关节符号、状态时效和保护接管的验证。
