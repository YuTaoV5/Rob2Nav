# 训练与复现状态

运行 ID 为 `filtered-4000-env4096-stage020.ovvRL3`，终点 `model_3999.pt` 采用从零开始的轮数索引。训练从 R1199 系列的 `rolling-knee-margin-formal/model_1199.pt` 初始化。作业使用 Isaac Lab、PhysX、expanded 配置、学习率 5e-5、种子 311、4096 个并行环境、4000 轮和 0.20 m/s 速度档。

[运行合同](run-contract.json) 和 [原始输入哈希](inputs.sha256) 来自原作业。输入哈希记录服务器当时使用的路径和字节身份。训练源码参考 commit 为 `9dffc8c9055f9df3783af3ed6901da782d950f6f`，后续改动的确切运行源码以原作业快照哈希为准。

原作业的训练入口为 `train_as2w_isaac_robust.py`。环境参数包含 `--profile expanded --num-envs 4096 --iterations 4000 --learning-rate .00005 --seed 311 --speed-cap .20 --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --pure-turn-common-scale 0 --park-policy-rad 0 --rolling-leg-filter-hz 5 --rolling-low-speed-smooth --rolling-maneuvers --rolling-knee-margin --safety`。原始 launcher、完整解析配置、确切版本依赖和与输入哈希匹配的源码及资产仍需从训练服务器补取。这份参数摘要只用于定位原作业。

本次发布中的两套评估合同与训练合同分别保存。Isaac 终点评估合同中的学习率是评估入口值，不是这次训练学习率。MuJoCo 的速度档、推扰、丢包和测试数在其评估合同中，导出和回放分别保存各自的权重身份。
