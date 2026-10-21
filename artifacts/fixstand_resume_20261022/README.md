# FIXSTAND 续训父权重与独立评估

这是 2026-10-22 服务器续训的可复现输入和失败证据，目录结构保留 `outputs/` 相对路径，便于复制回同名位置。三份 `model_*.pt` 都是**服务器训练得到的父检查点**，不是外部下载模型，也没有通过新的完整实机门槛。`exported/` 存放对应可推理文件及动作/观测合约；`seed_*/summary.json` 为自由基座 MuJoCo 独立种子结果。`fixstand_sideflip_terminal_v5_20261022` 和 `fixstand_sideflip_hip_taper_diagnostic_20261022` 只保存失败评估及消融元数据，未收录退化候选的权重副本。

`SHA256SUMS` 覆盖本目录下全部 55 份 `outputs/` 文件。恢复前核验哈希；后翻 v3 的 `evaluation.json` 已修正为速度约束失败，不能根据旧的动作通过记录宣称可用于 Go2 Edu。训练状态、每条续训命令及验收解释见 [续训报告](../../docs/FIXSTAND_CONTINUATION_20261022.md)。
