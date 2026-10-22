# SCAN 网页实时 MuJoCo 仿真

新增[台阶后过窄门与台阶路线](SCAN_STEPS_AND_NARROW_DOOR.md)：5cm/8cm低台阶、0.95m窄门、可选手动目标高程补丁及规划/执行速度对齐；具体尺寸和验收范围以新说明为准。

完整操作/环境恢复见[SCAN仿真手册](../skills/navigation/SCAN_SIMULATION.md)，硬件导航集成见[SCAN实机部署手册](../skills/navigation/SCAN_DEPLOYMENT.md)。

网页入口：导航页选择「SCAN 原生规划器 + MuJoCo」，或在线仿真选择「SCAN 原生规划仿真」。独立窗口为 `/scan-live`。点击地图空地设置目标，再点击开始。左键拖动画面旋转、右键拖动平移、滚轮缩放。停止与重置只影响仿真。

## 实际执行链

`MuJoCo 射线点云 + 真实机身里程计 → ROS2 SCAN 地图/规划器 → Bspline → 原生 closed_loop_controller → Twist → RoboGauge Go2 MoE → 关节力矩 → MuJoCo 接触动力学`。

SCAN 使用已经编译的 ROS2 community 提交 `d0b921c9b05a6d291d144d60882b2e0e88d2c0e0`，仓库为 https://github.com/wuyi2121/SCAN-Planner 。Jazzy 的三个 cv_bridge 头文件兼容修改保留在服务器 `/root/gpufree-data/scan-reproduction/jazzy-compatibility.patch`。桥接不启动 `go2_kinematic_sim`，不通过积分 Twist 直接写机身位置。机器人初始化后由 MuJoCo 及低层策略推进。

ROS_DOMAIN_ID=88，仅本机发现；传感器/指令通过127.0.0.1 UDP 8771在系统ROS解释器与CPU策略环境间传输。每次重置生成新session，丢弃旧session指令，重建两个规划进程并清空地图。超过0.3秒没有新指令或桥接不可用时执行零速度。网页失联2.5秒停止导航；桥接失联10秒清理其规划进程。

规划器读取实际射线返回，显示地图的障碍列表仅用于绘图和拒绝明显无效目标。传感器配置、最大范围与噪声见 `skills/navigation/config.json`。目前的里程计是仿真真值，并非已实现实机LIO。桥接发布 world 点云与 body/sensor Odometry，不接收真实相机或雷达。

## 使用与复现

服务器已安装 ROS Jazzy、编译该固定 SCAN 工作空间，以及准备 CPU MuJoCo 环境。先核对：

```bash
cd /root/gpufree-data/Rob2Nav
git -C /root/gpufree-data/scan-reproduction/upstream rev-parse HEAD
test -x /root/gpufree-data/scan-reproduction/upstream/install/scan_planner/lib/scan_planner/scan_planner_node
/root/gpufree-data/envs/go2-mapless/bin/python -c 'import mujoco,torch;print(mujoco.__version__,torch.__version__)'
```

两个终端分别启动（已有 Supervisor 服务时不要重复执行）：

```bash
# 终端1：原生规划桥，自动按仿真session启动/停止规划与跟踪进程
bash deployment/start-scan-bridge.sh
```

```bash
# 终端2：真正的MuJoCo机器人动力学与网页worker
cd /root/gpufree-data/Rob2Nav
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /root/gpufree-data/envs/go2-mapless/bin/python -m skills.navigation.live \
 --planner scan --port 8769
```

Studio通过 `/scan-live-api` 代理worker；访问用户已连接的 http://127.0.0.1:8766/#navigation 。服务配置见 `deployment/scan-supervisor.conf`，需加入现有 Supervisor 主配置后 `reread`/`update`；主配置已包含同名进程时不重复追加。

```bash
supervisorctl -c /root/gpufree-data/deployment/rob2nav-supervisord.conf status
tail -f /root/gpufree-data/deployment/scan-bridge.log
tail -f outputs/scan_web/planner.log
curl -fsS http://127.0.0.1:8765/scan-live-api/api/state
```

状态包含 `scan.plans`、`commands`、`source_commit`、点数与目标发布/轨迹确认信息。目标发布等待ROS订阅者出现；未收到新轨迹时每2秒重发，避免DDS冷启动丢失首次目标。启动成功与规划成功、目标到达分别检查。

## 验证范围

桥接已实际完成空旷场景和绕障场景的集成检查；原始结果及修复前失败记录分开保留在 `outputs/scan_mujoco_web_validation*`。网页鼠标与技能切换检查位于 `outputs/live_skills_web_validation`。这些固定场景不构成广泛鲁棒性、复杂三维通行或实机安全证明。修复后的这两项为2/2，最终距离分别0.377m与0.371m；[原始摘要](evidence/scan_mujoco_web/summary.json)、[空旷轨迹](evidence/scan_mujoco_web/empty_trace.json)、[绕障轨迹](evidence/scan_mujoco_web/slalom_trace.json)。[修复前1/2记录](evidence/scan_mujoco_web/previous_failure.json)保留冷启动无轨迹的失败，未并入新测试分母。

重跑固定目标集成检查（会重置当前SCAN仿真；输出目录须不存在）：

```bash
cd /root/gpufree-data/Rob2Nav
python3 scripts/verify_scan_mujoco_web.py --output outputs/scan_web_check_new
```

SCAN 是经典规划器，没有神经网络训练阶段。MoE是独立的低层运动策略。实机只保留[SCAN实机完整部署手册](../skills/navigation/SCAN_DEPLOYMENT.md)，未执行机器人通信或电机指令。
