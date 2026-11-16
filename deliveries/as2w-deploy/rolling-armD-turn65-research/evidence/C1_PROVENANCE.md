# C1 版本溯源

| 项 | 值 |
| --- | --- |
| 仓库 | Rob2Nav |
| **基线 commit** | `abde5f49330a467db800e7cdcfe6e991c88b4c9b`（"Index and pin AS2-W R4000 simulation assets"） |
| 选中输出权重 | `weights/armE_support2p0_final.pt`，sha256 `4bdf4d26b076d7c7…`（完整值见 `weights/SHA256`） |
| **初始化权重** | `model_3998.pt`，sha256 `92c44107bc16e4783e8b8cacc6c1d012f3f86e6700860a6a85badedbe86dfd66` |
| 训练入口 | `training/train_armE_support2p0.py` |
| 生效配置 | `yaw_weight 3.0`、`yaw_std 0.05`、`sym_weight 0`、**`rolling_wheel_support -2.0`**、LR **3e-4 自适应**、iters 600、envs 4096、save_interval 100 |
| 环境源码 | `as2w_isaac_robust_env.py` sha256 `1af438db2ff3babc064a3a56…`（含**动作延迟 DR 关闭**的改动） |
| ONNX | `weights/policy.onnx`，opset 17，sha256 见 `SHA256SUMS` |
| 运行记录 | 服务器 `/root/gpufree-data/as2w-runs/r11_fanziqi_lab/yaw-arms/armE-support-20261116-034514/` |

## 缺口

1. **训练时的代码未提交到 git** —— 上述 sha256 是当前文件，训练期间的不可变快照
   只存在于服务器工作树；基线 commit `abde5f4` 早于本轮全部训练。
2. `as2w_isaac_robust_env.py` 的延迟 DR 关闭**以补丁形式**交付（见上一版包
   `rolling-r11-turn-research/runtime/action-delay-disable.patch`），本包未重复携带该补丁，
   只记录其生效后哈希。
3. 未附训练 tfevents 摘要（留在服务器工作区，未进包）。
