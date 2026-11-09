# AS2-W 无实测线速度联合策略 · R21

同一份 PyTorch checkpoint 在 MuJoCo 中执行平地 `vx / vy / yaw rate` 指令、停车和盲上楼梯。策略使用 IMU、关节与轮端反馈、历史动作和指令，并从轮端运动学及历史观测估计平面速度。推理无需订阅官方运控的基座线速度话题。仿真真值用于评测。

- [R21 权重](weights/policy.pt)
- [平地六种指令与停车 · 36 秒](videos/flat-commands.mp4)
- [五级 6 cm 楼梯 · 9 秒](videos/stairs-06cm.mp4)
- [五级 8 cm 楼梯 · 9 秒](videos/stairs-08cm.mp4)
- [视频片段的种子与通过记录](evaluation/video-rollouts.json)

| MuJoCo 场景 | 固定验收 | 独立复测 |
| --- | ---: | ---: |
| 平地指令与停车 | 34/52 | 16/52 |
| 五级 6 cm | 0/5 | 10/21 |
| 五级 8 cm | 1/6 | 0/21 |
| 五级 10 cm | 0/5 | 0/22 |

视频选自实际通过的种子，且以发布的权重重新运行验证。固定验收中，前进、倒退、停车和左转各通过 4/4，纯左移 1/4、纯右移 2/4、右转 0/4。平地任意方向控制与楼梯成功率仍未达到稳定验收要求。当前交付为 Python/MuJoCo 仿真候选，实机和机器人端部署验证待完成。

## 复现

在此目录中运行。环境使用 Python 3.10+、MuJoCo 3.17.0、NumPy、SciPy、PyTorch、Gymnasium、Stable-Baselines3。录制还需要 Pillow、FFmpeg 和 MuJoCo 的 EGL 或其他可用渲染后端。

将 [Unitree ROS 模型仓库](https://github.com/unitreerobotics/unitree_ros/tree/4005a1b0c3f464f515f380be4145cc617fbca725) 的所列版本放在 `assets/official-model`，其中应有 `robots/as2w_description/as2w.urdf` 和该模型的 `meshes/`。

```sh
sha256sum -c SHA256SUMS
python runtime/train_omni_routed_residual.py screen assets weights/policy.pt fixed-replay.json
python runtime/train_omni_routed_residual.py screen assets weights/policy.pt confirmation-replay.json --confirmation
MUJOCO_GL=egl python runtime/render_omni_policy.py assets weights/policy.pt evaluation/fixed.json evaluation/confirmation.json replay-videos
```

完整逐场记录见 [固定验收](evaluation/fixed.json) 和 [独立复测](evaluation/confirmation.json)。
