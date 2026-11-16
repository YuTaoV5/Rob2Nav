# 干净检出复核（goal 清单第 3 项）—— 并抓到一个真实缺陷

goal 的提交前审查要求：「**从干净检出复核路径、清单哈希、启动说明，至少跑适用的 CLI/冒烟或最小离线回放**」。此前我一直在工作树里校验清单，**从未从干净检出复核** —— 这是一处真实遗漏。本轮补齐，并因此抓到一个缺陷。

## 1. 🔴 抓到的缺陷：清单里混进了编译缓存

从本地仓库 `git clone --depth 1` 到临时目录后校验：

```
校验: 165/166
sha256sum: runtime/__pycache__/preflight.cpython-312.pyc: FAILED open or read
```

**根因**：我先前用 `importlib` 导入 `preflight.py` 做验证测试时，Python 在工作树生成了 `runtime/__pycache__/preflight.cpython-312.pyc`；随后重新生成 `SHA256SUMS` 时是「遍历所有文件」，于是**把这个缓存也收录进了清单**。

**它实际上从未被提交**（`git ls-files` 里查不到），所以干净检出里自然没有它 → 清单出现 1 项 FAILED。但**清单本身是错的**：它描述了一个不该存在的文件。这正是 goal 要求排除的「缓存」。

### 修法

1. 删除工作树里的 `runtime/__pycache__/`2. 重新生成 `SHA256SUMS` 时**显式排除** `__pycache__/`、`*.pyc`、`*.pyo`、`*.log`、`*.tmp`3. 提交修正后**再从干净检出复验**

## 2. 复验结果（干净检出，提交 `89c0280`）

| 检查 | 结果 |
| --- | --- |
| 文件数（不含 SHA256SUMS） | **165** |
| **清单校验** | **165/165 OK** |
| 缓存文件 | **0（无 `__pycache__`、无 `.pyc`）** |

## 3. 启动说明路径复核（干净检出内实际存在）

| 路径 | 存在 |
| --- | --- |
| `runtime/r11_acceptance_ff.py` | ✅ |
| `runtime/deploy.yaml` | ✅ |
| `runtime/preflight.py` | ✅ |
| `weights/policy.onnx` | ✅ |
| `README.md` / `SHA256SUMS` | ✅ |

## 4. 最小离线回放（在干净检出里跑）

```
$ python runtime/preflight.py --mode dry
policy file      : weights/policy.onnx
declared status  : research (deployable=False)
  WARN  unverified hardware fields (19): FL_calf, FL_foot, ...
dry run: no approval is granted by this mode
```

**→ 干净检出里的预检能正常加载配置与 ONNX 路径，并正确给出 19 个未验证硬件字段的警告，`deployable=False` 被读到。**

## 5. 顺带核对：课程模块两处副本一致

`training/as2w-backup/as2w-current/code/as2w_ab_curriculum.py` 与 `skills/as2w/rolling/as2w_ab_curriculum.py`**哈希一致（`f969e454a1b4096e`）** —— 索引指向的文件与训练实际使用的是同一份。

## 6. 结论

**goal 清单第 3 项的「干净检出复核 + 清单哈希 + 启动说明 + 最小离线回放」四项均已执行并留有本文件为证。**同时说明：**只在工作树里校验清单是不够的** —— 正是换成干净检出才暴露出缓存被收录的问题。

## 7. 边界

1. 干净检出用的是**本地仓库 clone**（`file://`），不是远端 fetch —— 因为本地领先远端 65 个提交且无推送授权2. 未在干净检出里跑完整 11 例矩阵（需 GPU，当时被训练占用）；只跑了 `preflight --mode dry`3. 未验证 `python -m` 方式调用（脚本按路径调用）
