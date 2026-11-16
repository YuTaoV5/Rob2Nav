# 上传授权：**临时 SSH deploy key 已生成，等待您授权**

操作员要求生成可授权的临时凭证。本文件记录授权步骤与授权后的一键推送方式。

## 1. 需要您做的事（两步，约 1 分钟）

### 第 1 步：打开这个链接

**https://github.com/YuTaoV5/Rob2Nav/settings/keys/new**

（仓库级 deploy key，权限最小，只影响这一个仓库；若该入口不可用，用账户级：https://github.com/settings/ssh/new ）

### 第 2 步：粘贴下面这一整行公钥

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPxPXV2FTNk9Phhcu1CaniEYSCyHpezH3OXda9wdUR/1 rob2nav-upload-20261116-110651
```

- **Title**：随意，建议 `rob2nav-upload-20261116`- **Key**：粘贴上面那一行- **✅ 必须勾选 `Allow write access`**（否则只能读、不能 push）- 然后点 `Add key`

公钥指纹（用于核对，防止粘错）：`SHA256:zUBJ7OT74Y7VCucCZLIU9GWjJAIrx23K1/RjQs25RLc`

> 私钥在服务器 `/root/.ssh/rob2nav_upload_ed25519`，**不会显示、不会进仓库、不会被提交**。> 该 key 只配给 `github.com`，且 `IdentitiesOnly yes`，不影响其它用途。

## 2. 授权后我会执行（一条命令）

```bash
bash /root/push_after_auth.sh
```

该脚本会：SSH 授权自测 → 核对快进合法性（**远端已前进则拒绝，不强推**）→ `git push ghssh HEAD:main` → 回读远端 main 与本地 HEAD 比对。

## 3. 备选方案：PAT（个人访问令牌）

若不希望加 SSH key，也可以给一个 **fine-grained PAT**（只需 `Contents: Read and write` 权限、仅限 `YuTaoV5/Rob2Nav`）：

1. 打开 **https://github.com/settings/personal-access-tokens/new**2. Repository access 选 `Only select repositories` → `YuTaoV5/Rob2Nav`3. Permissions → Repository permissions → **Contents: Read and write**4. 生成后执行：

```bash
GH_TOKEN=<你的token> bash /root/push_rob2nav.sh
```

该脚本**不落盘** token（仅注入本次 URL）。

## 4. 当前待推送内容（未变）

| 项 | 值 |
| --- | --- |
| 提交数 | **76** |
| 远端 main | `abde5f4`（与本地 `origin/main` 一致 → 快进合法） |
| 出站体积 | 239 文件 / 6.02 MB |
| 包校验 | 194/194 OK |
| 安全审查 | 已通过（无密钥/无原始采集/无缓存） |

## 5. 授权前自测结果（已确认通路可用）

| 检查 | 结果 |
| --- | --- |
| github.com HTTPS | **200** |
| github.com SSH (22) | **可达** |
| `ssh -T git@github.com`（授权前） | `Permission denied (publickey)` —— **符合预期，只差授权** |

## 6. 备选：bundle（无需任何服务器凭据）

已有 `Rob2Nav-upload-20261116.bundle`（2.98 MB，含全部提交，基线 = 远端 main）。在任意有凭据的机器上 `git fetch <bundle> main:upload && git push origin upload:main` 即可。
