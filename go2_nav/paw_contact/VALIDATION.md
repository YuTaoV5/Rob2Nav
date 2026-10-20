# Go2 托爪仿真验收记录

当前版本专项检查42/42、随机组合32/32，共74/74项。包括无效请求拒绝和失效撤回；不能称为74次独立人手握持实验。

已测最大手掌法向力 5.900 N；最大机身倾斜 0.965°。

专项判据：正常请求至少连续1秒保持1–8N接触、横向误差<4.5cm、倾斜<12°；全程力<15N、倾斜<15°、基座高度>0.25m；撤回后四足接地。无效请求必须拒绝；长时丢失必须撤回。

| 检查组 | 通过 | 总数 |
|---|---:|---:|
| nominal | 4 | 4 |
| position | 6 | 6 |
| moving | 4 | 4 |
| noisy_delayed | 4 | 4 |
| brief_dropout | 4 | 4 |
| push | 4 | 4 |
| low_friction | 4 | 4 |
| heavy | 4 | 4 |
| invalid_far | 2 | 2 |
| closed_palm | 2 | 2 |
| low_confidence | 2 | 2 |
| long_dropout | 2 | 2 |
| mixed | 32 | 32 |

浏览器实际拖拽改变目标，分别完成右爪与左爪保持，再撤回/停止恢复。记录的保持力分别为3.231N 和 3.255N。JavaScript异常0条；未带写入保护头请求返回403。

既有Studio后端回归测试44项通过；前端TypeScript/Vite构建通过。

首轮诊断37/42与失败trace完整保留在paw_contact_benchmark_v1。后续将物理手掌运动和测量噪声分离，并将手掌接触面调整为11×9cm。32个随机组合种子未用于这次修改。

数据目录：outputs/paw_contact_benchmark_v2、outputs/paw_contact_holdout_v1、outputs/paw_contact_demo_v1。参数范围、测量定义、可重跑命令和真实机器人缺口见DEPLOYMENT.md。

当前演示不使用摄像头，也未连接真实机器人；动力学、触觉误差、控制时延和人体接触实机效果都不能由本报告代替。
