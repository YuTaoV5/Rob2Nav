# 全技能仿真与背景验收（2026-10-22）

最终结果见 `web_validation.json`。`initial_web_validation.json`保留限速改进前的首次跳跃记录；最终参数以`jump_summary.json`为准，不能用首次结果声明最终电机约束。

- 后翻、侧翻、直立：完整运行，原有未通过项如实保留。
- 最终跳跃：完整7秒，1/1；机身抬升0.1068m，腾空0.162s，无非足端接地，末端站稳，关节速度比0.607、限位超出0。
- 连续越障：完整运行，1/1、零碰撞、零跌倒。
- 左右爪：拖拽虚拟手后均进入接触HOLD。
- 楼梯、高台、多地形：仅启动与实时帧烟测，不声明完整通过。
- `composition/`：新背景下原有连续序列复测，驻留7秒；场景颜色与网格只作用于渲染。
- `python_tests.txt`：36项测试结果。
- `verify_web.py` / `verify_final.py`：浏览器验证脚本，需要Studio和握手服务、Chrome与Playwright。最终脚本调用仓库`artifacts/composer_timing_20261022/verify_web.py`；路径见源代码，可按本机目录调整。

截图与JPEG包含棋盘、0.25米网格及坐标轴；`jump_trace.json`与各`*_summary.json`保存真实物理运行结果。当前所有技能均未获Go2 Edu实机资格，独立场景切换会重置模型，不代表连续热切换。
