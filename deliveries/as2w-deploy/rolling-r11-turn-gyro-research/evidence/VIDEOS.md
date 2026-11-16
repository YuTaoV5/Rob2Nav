# 仿真效果视频：armD@1k（当前最好的 Isaac 候选）

| 项 | 值 |
| --- | --- |
| 权重 | `armD-turn65-1k/run/r11-final.pt`，sha `de61de58fd68efc3` |
| 仿真器 | **MuJoCo**（Isaac 侧无渲染管线） |
| 分辨率/帧率 | 640×480 / 30 fps |
| 合辑 | `as2w_rolling_best_demo.mp4`（2150 帧，含文字标注） |
| 分镜 | clip00–clip05 六个用例各一段 |

## 分镜内容

| 文件 | 用例 |
| --- | --- |
| `clip00.mp4` | hold  zero command |
| `clip01.mp4` | straight_fwd  0.25 m/s |
| `clip02.mp4` | straight_bwd  0.25 m/s |
| `clip03.mp4` | arc_right  0.25 m/s  wz -0.05 |
| `clip04.mp4` | brake_fwd  0.25 -> 0 |
| `clip05.mp4` | brake_bwd  -0.25 -> 0 |

## 说明与边界

1. 视频是**仿真演示**，**不是实机回放**；R3/R4 仍 `not_run`
2. 视频脚本内置的六例是「两端都通过」的历史集合（注释如此写），**不含四项纯转** —— 因为纯转目前两端都不达标
3. 视频用 `r11_demo_video.py` 生成，其中 MJCF 的 `meshdir` 被改为绝对路径（网格 XML 写在输出目录，相对路径会失效）
4. 中间产物（frames/、mjcf_grid.xml）不入库
