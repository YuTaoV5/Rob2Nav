# 演示视频（**仅可视化，不是验收结果**）

按训练守则要求，**视频演示与正式验收分别标记**。本视频只含**两端都通过**的六个用例，
不包含任何仅在单端通过或失败的工况，避免用视频暗示更宽的能力。

- 生成脚本：`runtime/r11_demo_video.py`
- 权重：`weights/armE_support2p0_final.pt`（`4bdf4d26`）
- 仿真器：**MuJoCo 3.2.3**（录制端；不是 Isaac）
- 规格：640×480、30 fps、**71.7 s**、2150 帧
- 服务器产物：`results/video/demo2-*/as2w_rolling_best_demo.mp4`

## 分段（每段左上角有文字标签）

| # | 标签 | 时长 | 说明 |
| --- | --- | --- | --- |
| 1 | `hold zero command` | 5 s | 零指令驻停 |
| 2 | `straight_fwd 0.25 m/s` | 6 s | 直线前进 |
| 3 | `straight_bwd 0.25 m/s` | 6 s | 直线倒车 |
| 4 | `arc_right 0.25 m/s wz -0.05` | 6 s | 右弧线 |
| 5 | `brake_fwd 0.25 -> 0` | 10 s | 前进后松杆刹停（5 s 驱动 + 5 s 归零） |
| 6 | `brake_bwd -0.25 -> 0` | 10 s | 倒车后松杆刹停 |

## 地面网格

已跟踪的 MJCF 地面 geom **没有材质**，所以网格是**临时副本**加上 MuJoCo 内建 checker 纹理实现的：

- 临时文件写在**运行目录内**（`mjcf_grid.xml`），**已跟踪的 MJCF 资产未被修改**
- 纹理：`builtin="checker"`，`texrepeat 12 12`，浅灰配色
- 同时加强了 headlight（默认光照下机身偏暗）

## 复现

```bash
MUJOCO_GL=egl /root/gpufree-data/mujoco-venv/bin/python runtime/r11_demo_video.py \
  --xml <r11_mjcf_excl.xml> --ckpt <armE r11-final.pt> --out <dir> \
  --width 640 --height 480 --fps 30
```

要求：`mujoco`、`ffmpeg`、`MUJOCO_GL=egl`（本机 OSMesa 不可用）。
帧以 PPM 直出（该 venv 无 PIL/imageio），再由 ffmpeg 编码并叠加文字。

## 边界

1. **这是 MuJoCo 的可视化，不是 Isaac 结果，也不是验收证据**
2. 未叠加任何数值遥测；数值结论请看 `evidence/` 下的评测报告
3. 视频文件**未提交进仓库**（体积原因），由上面的脚本可复跑重现
4. 画面中不含真实硬件，**不得作为真机验收的替代**
