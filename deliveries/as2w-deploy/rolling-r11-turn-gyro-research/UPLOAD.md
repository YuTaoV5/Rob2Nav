# 上传状态：**已授权、已审查、待凭据**

操作员已明确授权上传。本文件记录**上传前审查结果**、**具体阻塞**与**一键执行方式**。

## 1. 当前状态

| 项 | 值 |
| --- | --- |
| 本地 HEAD | `86fe5500e561dbf9d8a15fdfc97416b3f92edd65` |
| 领先远端 | **72 个提交** |
| 落后远端 | **0**（快进合法，无需 rebase） |
| 远端 main | `abde5f4`（与本地 `origin/main` 一致） |
| 出站内容 | **239 文件 / 6.02 MB** |
| 最大出站文件 | 1.94 MB（C8 回放 JSON），**远低于 GitHub 100 MB 单文件限** |
| 是否需要 LFS | **不需要** |

## 2. 🔴 唯一阻塞：主机无推送凭据

已实际尝试推送，失败于认证：

```
fatal: could not read Username for 'https://v6.gh-proxy.org': terminal prompts disabled
```

主机上没有：`credential.helper`、`~/.git-credentials`、`GITHUB_TOKEN`、`GH_TOKEN`、`gh` CLI、SSH 密钥。
**→ 这是环境限制，不是仓库问题；我不能代为创建凭据，也不会把凭据写进仓库或脚本。**

## 3. 上传前审查（全部通过）

| 检查 | 结果 |
| --- | --- |
| 出站 diff 含密钥/凭据 | ✅ 未发现（扫描 PAT/AKIA/私钥/password/api_key/token 模式） |
| 压缩包/日志/原始采集 | ✅ 无 |
| `__pycache__` / `.pyc` / `.bak` | ✅ 无 |
| whitespace 错误 | ⚠️ 1 处，在**旧包** `rolling-r11-turn-research/runtime/action-delay-disable.patch:23`（历史遗留，非本次新增） |
| 包清单校验 | ✅ **189/189 OK** |
| 权重文件 | 2 个 `.pt` + 1 个 `.onnx`，均属两个交付包自身，**非重复副本** |

## 4. 一键执行方式（三选一）

**方式 A —— 提供 token（推荐，脚本不会落盘）**

```bash
GH_TOKEN=ghp_xxxxxxxx bash /root/push_rob2nav.sh
```

脚本会：核对快进合法性（落后即**拒绝**而不是强推）→ 把 token 仅注入**本次** url → `git push` → 回读远端 main 校验。

**方式 B —— 交互式输入**

```bash
bash /root/push_rob2nav.sh      # git 会提示用户名与 token
```

**方式 C —— 从我方机器直接推**（无需服务器凭据）

```bash
git clone https://github.com/YuTaoV5/Rob2Nav.git && cd Rob2Nav
git remote add srv <服务器路径或 rsync 拉取>
git fetch srv main && git push origin srv/main:main
```
或在授权后由我出具 `git bundle`，由有凭据的一端 `git fetch` 后推送。

## 5. 推送后的校验清单（脚本已自动做前两项）

1. `git push` 返回 0
2. 回读 `git rev-parse origin/main` 等于本地 HEAD
3. 网页确认入口可达：`skills/as2w/README.md`、`deliveries/as2w-deploy/rolling-r11-turn-gyro-research/README.md`
4. 网页确认权重可取：`weights/policy.onnx`（175,066 B）、`weights/armE_support2p0_final.pt`（1,042,859 B）
5. 网页确认 `SHA256SUMS` 与本地一致（189 条）

## 6. 如实声明

- **本轮未能完成上传** —— 缺凭据，不是被拒绝也不是审查未过- 我不会把任何 token 写入仓库、提交历史或服务器脚本- 一旦提供凭据，执行 `/root/push_rob2nav.sh` 即可完成；全部审查已就绪
