# 物理子步恢复证据（2026-10-22）

本目录归档实际诊断和三项训练中的两项新任务配置，不表示训练完成或动作通过。原地直立配置和父权重见既有acrobatics_clean_goal与fixstand_upright_v3归档。

- sideflip_v10_physics_substeps：500Hz只读4000样本，控制帧速度比0.943，真实峰值1.749；保留原始trace、实际YAML和峰值位置。
- sideflip_v10_initialization_parity：两种初态均未解决MuJoCo落地失败。
- backflip_v7_capture_entry_minus15：统一触发时的观测运动学快照及轨迹；fixture仅保留配置，不复制模型。
- backflip_v7_native_observed_entry_minus15：原生重放同运动学状态，初始站姿仍失败；不含求解器热启动/隐藏执行器状态，不是完整状态克隆。
- fixstand_sideflip_substep_peaks_v11：正式与烟测实际配置、完成烟测及物理统计；正式未完成，无最终模型。
- fixstand_frontflip_positive_pitch_v1：独立前翻实际配置与完成烟测；正式未完成，无最终模型。
- training_snapshot.json：日志迭代快照，不作为最终验收或持续利用率。

SHA256SUMS覆盖全部文件（自身除外）。复现与验收边界见docs/PHYSICS_SUBSTEP_AUDIT_20261022.md。源权重继续复用此前归档，不重复上传。
