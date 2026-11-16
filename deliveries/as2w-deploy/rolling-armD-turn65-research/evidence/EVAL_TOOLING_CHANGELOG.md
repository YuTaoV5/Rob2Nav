# 评测工具版本 eval-2：变更内容与为何不升动作包装版本

goal 要求「包装变化即升版，即使模型哈希相同」。本轮需要精确区分两种「包装」：

| 概念 | 版本 | 本轮是否变化 |
| --- | --- | --- |
| **动作包装**（部署契约：16 轴映射、尺度、偏置、限位、PD 增益、90% 保护） | `gyro-1` | ❌ **未变化** |
| **评测工具**（验收/分析脚本） | `eval-2` | ✅ **变化** |

**→ 因此动作包装版本保持 `gyro-1`，另立 `eval_version: eval-2`。若把评测工具的改动记成动作包装升版，会让读者以为部署契约变了，那是误导。**

## eval-2 相对 eval-1 的变更

| 变更 | 文件 | 影响 |
| --- | --- | --- |
| 新增 `--obs-freeze N` | `runtime/r11_acceptance_ff.py` | 观测冻结/重复帧用例 |
| 新增 `--obs-stale-steps K` | `runtime/r11_acceptance_ff.py` | 陈旧观测用例 |
| 新增双口径报告工具 | `runtime/r11_dual_basis_report.py` | mean / min 两口径并报 |
| 新增分段转向报告工具 | `runtime/r11_segmented_report.py` | 首秒 / 3–5 s / 末 3 s |

## 权重与 ONNX

**本轮没有任何权重或 ONNX 变化。**

| 项 | 值 |
| --- | --- |
| `weights/armE_support2p0_final.pt` | `4bdf4d26b076d7c7…`（未变） |
| `weights/policy.onnx` | opset 17、175,066 B（未变） |

## 评测专用性的保证

两个新参数**只在验收脚本里存在**，不出现在 `deploy.yaml` 的动作映射中，也不被预检脚本读取。`runtime/preflight.py` 的行为未变：硬件字段仍为 `not_measured` 时依旧**拒绝**起实机闭环。

## 边界

1. 观测畸变参数**已实现但未运行**（GPU 被训练占用），因此 eval-2 的这一部分**没有实测证据**2. 两个新工具都在真实数据上自检过（双口径工具首次运行即发现并修正一处我自己的判定错误）3. 动作包装版本的判定依据是「部署契约是否变化」，不是「文件是否被编辑」

## 6. 回归验证（改完必须验）

本项目此前已被自研 YAML 解析器坑过 3 次，因此改动 `deploy.yaml` 后必须回归：

| 检查 | 结果 |
| --- | --- |
| `_mini_yaml` 解析 `deploy.yaml` | ✅ 16 个顶层键，`version=gyro-1`、`eval_version=eval-2`、`deployable=False` |
| `preflight.py --mode dry` | ✅ exit 0，输出预期 WARN（19 个未验证硬件字段） |
| `preflight.py --mode hardware` | ✅ **exit 1（拒绝起实机）** —— 拒绝行为未被破坏 |

**→ 新增的 `eval_version` 键不影响解析器，预检的「硬件字段未测则拒绝」保护仍然生效。**

### 测试过程中的一处自身失误

第一次验证时我把 `_mini_yaml` 当成收**路径**的函数（它收**文本**），并把 `--mode` 写成 `offline`（实际是 `dry/sim/hardware`），因此得到两个假错误。**→ 报告工具/检查脚本的首次运行结果同样要先自证，不能直接当结论。**
