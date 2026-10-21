# 紧凑热切换对照记录

完整机制与复现命令见[说明](../../docs/COMPACT_HOT_SWITCH.md)。摘要保留物理后端、源码/模型/权重哈希及所有控制器事件。原始积分轨迹可用相同输入重新生成，未将其当作不可重算资源重复上传。

- `legacy_9411/9412.json` 与 `compact_9411/9412.json`：stand→MoE→stand配对对照。
- `legacy_atomic.json` 与 `compact_atomic.json`：相同seed9413的侧翻组合对照。
- `compact_yaw_30.json`、`compact_yaw_-30.json`：扩展朝向验证。
- `compact_atomic_v2.json` 和对应MP4：带视频复核；视频只对应这一回合。
- `compact_atomic_v3.json`：最终快捷路径名单限制后的标准组合回归，判据通过。
- `compact_profiles.json`：初版低刚度入口缩短失败。
- `compact_profiles_v2.json`、`legacy_profiles_v2.json`：恢复完整入口后，新旧路径都在低刚度站姿预热中失败。
- `compact_gaits_final.json`：未限制名单时MLP混合组合失败；`compact_gaits_v3.json` 与 `legacy_gaits_v3.json` 为限制名单后的匹配对照，两者仍未通过，不能计为通过技能。

`passed`只描述摘要中的组合控制判据，不能覆盖关节额定速度约束。侧翻最高速度比仍大于1，全部未获实机资格。不同版本以摘要内源码哈希区分，历史成功不自动继承给后续改动。
