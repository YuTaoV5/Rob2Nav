# C8 最小回放

## 内容

- 捕获：`c8/c8_capture_arc_right_0.25.json` —— **600 帧**真实 Isaac rollout（50 Hz，12 s）
- 固定初态：`env.seed(7393) + env.reset()` 后 **2.0 s 零指令整定**
- 固定指令：`vx=0.25, vy=0, wz=-0.05`（右弧线，该用例在 Isaac 上是 PASS 的）
- 每帧记录：`obs[65]`、`raw_actor_output[16]`、`wheel_target[4]`（动作管理器与裁剪后的**最终目标**）、
  `wheel_dq_measured[4]`、`yaw_rate`、`upright`、以及 `next_obs[65]`

## 校验（`runtime/r11_c8_verify.py`）

用 `weights/policy.onnx` 在**每一帧记录的 obs** 上重算，与记录的原始输出逐维比较：

| 指标 | 值 |
| --- | --- |
| 帧数 | **600** |
| **max_abs_err** | **5.960e-07** |
| mean_abs_err | 7.110e-08 |
| 最大误差位置 | 第 **459** 帧，输出维 **11** |
| 最大误差维度 | 11 |
| **首次分歧帧** | **无**（0/600 帧超容差） |
| 容差 | 1e-5 |
| **判定** | ✅ **PASS** |

## 复跑

```bash
# 1) 捕获（需要 Isaac Lab + USD/URDF）
python runtime/r11_c8_dump.py --usd <as2w.usda> --urdf <as2w.urdf> --out /tmp/c8 \
  --ckpt weights/armE_support2p0_final.pt --case arc_right_0.25 --seed 7393
# 2) 重算校验（只需 onnxruntime）
python runtime/r11_c8_verify.py /tmp/c8/c8_capture_arc_right_0.25.json weights/policy.onnx --tol 1e-5
```

## 边界（必须与结论一并阅读）

1. 回放覆盖**策略网络**这一段：`obs → actor 输出`。动作管理器、裁剪、gyro-1 反馈、
   90% 限位与腿滤波**不在 ONNX 图内**，它们属于 `runtime/r11_acceptance_ff.py` 的包装逻辑。
2. 捕获来自**仿真**，不是实机日志。
3. 只捕获了 **1 个用例 / 1 个种子 / 1 个初始状态**。
