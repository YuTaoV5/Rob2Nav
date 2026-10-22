# 统一部署技能说明与计时验收

默认序列：站稳不动 2 秒 → 日常行走 3 秒 → 站稳不动 2 秒。
有效驻留为 7 秒；初始化、准备、切换、等待稳定与原子动作时间额外计入总时间。这次真实 MuJoCo 运行的具体结果见 `web_smoke.json`，逐次 API 采样见 `timing_samples.json`。

- `before_run.png`：直观中文名称、用途和计划时长。
- `completed.png` / `mobile.png`：桌面和手机页面最终计时。
- `verify_web.py`：Playwright 浏览器验收脚本；修改顶部服务器路径和端口后运行，需要已启动 Studio、Playwright 和 Chrome。

代码测试：在仓库根目录运行 `python -m unittest tests.test_deployment_core tests.test_sequence_progress -q`；在 `studio/frontend` 运行 `node --test tests/composer.test.mjs` 和 `npm run build`。

此变更仅增加只读进度，不缩短控制器姿态、稳定性和模型切换门槛。原子动作仍由控制器确保动作结束与落地；不使用人工填写的驻留秒数。循环后续轮次与上一轮末尾共用站立步骤，因此默认序列两轮为 12 秒有效驻留、四轮为 22 秒。

这些记录是仿真和网页验证，不是机器人硬件验收。
