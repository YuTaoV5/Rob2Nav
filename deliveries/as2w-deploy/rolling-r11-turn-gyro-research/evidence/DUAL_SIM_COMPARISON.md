# 双仿真对照（同一权重 armE_support2p0_final.pt）

## 结果对照

| 用例 | MuJoCo | Isaac |
| --- | --- | --- |
| hold | PASS | PASS |
| straight_fwd_0.25 | PASS | PASS |
| straight_bwd_0.25 | PASS | PASS |
| arc_left_0.25 | PASS（yaw 1.04–1.06） | FAIL（0.50–0.55） |
| arc_right_0.25 | PASS（yaw 0.84–0.88） | PASS（0.78–0.87） |
| brake_fwd / bwd | PASS | PASS |
| turn_left_0.05 | FAIL（支撑 0.760） | FAIL（yaw 0.22–0.39） |
| turn_right_0.05 | FAIL（支撑 0.927） | FAIL（yaw 0.01–0.35） |
| turn_left_0.10 | FAIL（支撑 0.580） | FAIL（yaw 0.18–0.24） |
| turn_right_0.10 | FAIL（支撑 0.487） | FAIL（yaw 0.02–0.29） |
| **合计** | **7/11** | **6/11** |

## 两端卡在不同约束

- **MuJoCo**：纯转由**四轮支撑**判死（0.487–0.927），偏航本身可以进入带内。
- **Isaac**：纯转由**偏航**判死（0.01–0.39），支撑几乎恒为 1.000。

## 机制证据（`runtime/r11_isaac_exec_probe.py`）

Isaac 轮驱**不是力矩受限**：13.5 N·m 上限下峰值仅 4.47 N·m，触顶比例 0.000。
但**小指令存在死区**：0.064 rad/s 时 kd 律已需 2.95 N·m；而目标到 2.5–3.4 rad/s 时 exec 达 0.67。

## 包装层修复（gyro-1）的效果与边界

`tau_ff = S × clamp( yaw_kp × (wz_cmd − wz_meas), ±4 N·m )`，`S = (−1,+1,−1,+1)`（左右差速），
无积分、无滤波、仅四轮、执行器仍 clamp ±13.5 N·m。

- 有效：把 Isaac 从 5/11 提到 **6/11（两种子一致）**，`arc_right` 由 0.57 升到 0.78/0.87 并转为 PASS。
- 无效：`arc_left`（0.50–0.55）与 4 个纯转。提高上限 4→8 N·m 未改善左弧（0.50→0.47）。
- 参数响应非单调（cap 4→0.78、6→0.69、8→0.92），说明策略闭环方差显著，**不能用插值选参**。
