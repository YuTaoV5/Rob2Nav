# AS2-W 滚动 4000 轮研究归档

本包对应 4000 轮、4096 环境的 0.20 m/s 作业末权重。评估范围为前后直行、左右原地转向、低速弧线、松杆制动、10 秒驻停及 fixed stand 进出。状态为 **研究归档，双仿真未通过，实机闭环未授权**。

| 环境 | 已验证结果 |
| --- | --- |
| MuJoCo | 稳定 72/72，无非轮触地，驻停 39/72，完整通过 27/72 |
| Isaac Lab / PhysX | 行为与安全门槛均未通过，转向与弧线完整通过 0/24 |
| ONNX | opset 17，float32，obs 1×65 到 actions 1×16，控制周期 20 ms。实机日志观测回放最大动作差 4.77e-6 |
| 实机日志离线诊断 | 末权重在 487 帧运动观测上的四路轮残差均饱和。轮速越界惩罚实验在第 250、500 轮消除该批回放饱和，第 1000 轮再次出现高饱和 |

逐例结果和实际评估参数见 [evidence](evidence/)；PT 和 ONNX 见 [weights](weights/)；训练运行参数、原始输入哈希及重现缺口见 [training](training/)；观测、动作与映射状态见 [runtime](runtime/)。

运行时范围及交付状态见 [v3 manifest](manifest.json)。3 m/s、连续手柄全时序、控制权切换和实机硬件映射均待补证。踏步、楼梯和起身是独立策略。原始实机日志、逐帧观测和账号凭据均未纳入此归档。

交付依据见仓库现有的 [AS2-W 训练守则](../../../training/as2w-backup/as2w-current/AGENTS.md) 与 [v3 交付清单](../../../training/as2w-backup/as2w-current/AS2-W-training-delivery-checklist-v3.md)。
