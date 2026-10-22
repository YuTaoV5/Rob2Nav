# SCAN台阶与窄门导航验收

最终代码版本的结果以`acceptance.json`、`summary.json`和未带前缀的六个`stairs_*_summary.json`为准：两场景分别种子200/201/202，共6/6到达，无碰撞、无跌倒。每个摘要必须具有`scan.goal_height_enabled=true`和`planner_max_velocity_mps=0.3`，台阶路线的静态规划余量为0.60m。

- `stairs_door`：门前2级×5cm、门净宽0.95m、门后下2级到达终点；3/3，约38.6–39.5仿真秒。
- `stairs_route`：上/下2级×8cm，顶部高16cm；3/3，约39.2–41.5仿真秒。
- `web/validation.json`：通过真实鼠标点击另选终点(约6.28,-1.20m)，实际上下台阶、过门、到达；相机拖动、10个场景选项和旧跟随场景加载验证。
- `*_trace.json`：逐次原始状态，包括机身三维位置、实际速度指令、SCAN计划次数与当前子目标，不是预先生成的理想轨迹。
- `initial_stairs_route_201_*`：初始版本平台家具附近超时，保留原始失败。修复了连续位姿落在空地、但栅格取整落入膨胀障碍的A*锚点问题。
- `before_speed_alignment/`：中间版本的运行记录，包含距终点约0.44m而停滞的情况；不纳入最终版本的6/6。没有放宽到达阈值、碰撞检查或动态可行性检查来消除失败。
- `scan_goal_height_applied.diff`、`source_fingerprints.json`：实际编译的SCAN语义差异（统一LF换行、零上下文），二进制和Python文件SHA256；指纹另含原始字节及LF标准化值。固定上游HEAD加明确的可选补丁，不能称为完全未修改的上游。
- `unit_tests.txt`：8项地图、A*回归、桥接和目标高度补丁契约测试通过。

`evaluate.py`通过当前服务器8769端口运行实际服务；`verify_web.py`通过8765网站代理和Chrome/Playwright测试。它们会重置当前仿真。复现前先按[完整说明](../../docs/SCAN_STEPS_AND_NARROW_DOOR.md)应用并编译目标高度补丁。无需新训练权重，低层仍用现有RoboGauge MoE。

以上仅为给定模型、低台阶尺寸和门宽的固定场景测试；不证明标准楼梯、任意窄门、未知低物体分类或Go2 Edu实机能力。新场景的0.20m支撑相对点云过滤阈值不是已验收的20cm台阶能力。
