# ✅ 上传完成

## 1. 推送结果

| 项 | 值 |
| --- | --- |
| 远端仓库 | `YuTaoV5/Rob2Nav` |
| **推送提交** | **`28cb7c506524ec9d0f09e83f3db63452bde97411`** |
| 范围 | `abde5f4..28cb7c5` → `main`（**普通快进，无 force**） |
| 提交数 | **80** |
| 本地 / 远端一致性 | 三方一致（本地 HEAD = ghssh/main = origin/main） |

## 2. 从远端重新 clone 的校验

| 检查 | 结果 |
| --- | --- |
| clone HEAD | `28cb7c5` ✅ |
| 包文件数 | 195 |
| `SHA256SUMS` | 见下方“清单修正”一节 |
| `weights/armE_support2p0_final.pt` | sha `4bdf4d26…` **与记录一致** ✅ |
| `weights/policy.onnx` | 175,066 B ✅ |

## 3. 关键入口在远端均已可取

| 路径 | 大小 |
| --- | --- |
| `skills/as2w/README.md` | 7,282 B |
| `skills/as2w/rolling/recipes/ab-gyro-loop.sh` | 4,661 B |
| `skills/as2w/rolling/as2w_ab_curriculum.py` | 4,101 B |
| `deliveries/as2w-deploy/rolling-r11-turn-gyro-research/README.md` | 6,108 B |
| `deliveries/.../weights/policy.onnx` | 175,066 B |
| `deliveries/.../weights/armE_support2p0_final.pt` | 1,042,859 B |

## 4. 🔴 一处我自己造成的缺陷（已修）

首次推送后从远端 clone 校验，`SHA256SUMS` 报 **194/195**，失败项是 `runtime/r11_isaac_target_replay.py`。

**原因：我提交了该脚本的支撑索引修正，却没有重生成 `SHA256SUMS`，导致清单与文件不一致。**
这正是「清单必须最后生成」这条规则 —— **我又违反了一次**。

**修法**：重生成清单 → 校验 195/195 → 追加提交并再次快进推送。

## 5. 授权方式记录

- 使用**临时 ed25519 deploy key**（仓库级，勾选 Allow write access），公钥指纹 `SHA256:zUBJ7OT74Y7VCucCZLIU9GWjJAIrx23K1/RjQs25RLc`
- 私钥仅存服务器 `/root/.ssh/rob2nav_upload_ed25519`，**未显示、未入库、未提交**
- `~/.ssh/config` 中 `Host github.com` 用 `IdentitiesOnly yes` 绑定该 key，不影响其它用途
- **建议：本次上传完成后从 GitHub 删除该 deploy key**（不再需要）
