# Go2 EDU部署方案离线验证（2026-10-22）

配套[操作手册](../../docs/GO2_EDU_NOETIC_FOXY_NAVIGATION.md)和[参考工具](../../deployment/go2edu/README.md)。

- `pytest.xml`：Windows主机10项离线测试通过；覆盖标定拒绝、逆变换、大小端/行padding点云、限速/停车、新目标与新轨迹、默认main不导入SDK及执行/回放互斥。未启动ROS、DDS或机器人。
- `validation.json`：研究服务器以实际固定提交的SCAN和LIO源码生成配置，检查Python3.8语法、36段手册Bash语法、消息字段一致性和静态展开参数。服务器有Jazzy、没有Noetic，**没有执行Noetic原生编译或roslaunch运行**。参数表是XML静态解析结果，不是机载实际参数读取。
- `synthetic_scan_go2edu.launch`：仅用于离线检查的生成结果。真正部署须由实际外参、站立高度和锁定源码重新生成；不能把测试目录当作现场标定。
- `frontend_build.log`、`web_smoke.json`、`manual.png`：生产构建通过；导航页手册链接、报告API及正文渲染通过，浏览器JS异常为0。只检查网站阅读入口，没有开启实机控制。
- `driver_schema/`：依据官方固定版本的[CustomMsg](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/msg/CustomMsg.msg)和[CustomPoint](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/msg/CustomPoint.msg)保留的规范化字段缓存（去注释/多余空格），与固定LIO消息字段比较。它不替代现场`rosmsg md5`及实际ROS连接测试。

核对结果：world点云模式true、重复外参false、机身/射线原点分别适配、规划/控制平移上限0.20m/s；未应用网页动态地图老化补丁。上游仓库的SDK、雷达驱动及Noetic发布兼容性由手册给出固定版本和现场命令，不能用这份离线记录宣称你的机载软件已经构建通过。

需要现场补齐：目标机器架构/JetPack、传感器供电/网络、测量外参/时钟、LIO精度与漂移、DDS/固件控制权、真实方向/停车距离、失效接管、独立看门狗以及地图复定位。未发送任何实机电机或高层运动指令；硬件资格未授予。
