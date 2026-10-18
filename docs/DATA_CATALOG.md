# 完整项目数据与恢复目录

[完整归档下载](https://github.com/YuTaoV5/Rob2Nav/releases/tag/full-archive-2026-10-19) · [新环境部署步骤](DEPLOYMENT.md) · [机器可读摘要](release-summary.json)

归档包含 **20,979个文件，7.24 GiB未压缩数据**，分为21个独立tar.gz包，压缩后共5.58 GiB。其中有616个PT文件、405个MP4文件，以及RL导航ZIP权重、TensorBoard、JSONL、评估记录和源码快照。

**数量含重复快照、中间阶段与校验模型，不是独立实验数。原AME两类20cm窄桩目标仍未完成。**

| --group | 恢复目录 | 文件数 | 未压缩MiB | PT | MP4 |
|---|---|---:|---:|---:|---:|
| `project-artifacts` | `Rob2Nav/artifacts` | 7,323 | 580.8 | 91 | 93 |
| `project-outputs` | `Rob2Nav/outputs` | 7,981 | 2762.0 | 273 | 78 |
| `moe-runs` | `MoE-runs` | 817 | 3268.9 | 105 | 0 |
| `ame-runs` | `AME-runs` | 227 | 94.7 | 44 | 0 |
| `ame-upstream` | `AME-criticaldamped` | 130 | 94.0 | 2 | 0 |
| `moe-upstream` | `MoE-upstream` | 116 | 65.1 | 3 | 0 |
| `legacy-initial` | `go2-mapless` | 262 | 55.1 | 4 | 7 |
| `legacy-terrain` | `rob2nav-terrain` | 1,127 | 166.0 | 6 | 161 |
| `legacy-perceptive` | `Rob2Nav-perceptive-63f5bcf` | 2,996 | 324.6 | 88 | 66 |

## 下载、恢复与验证

源码与精选资料已在Git主仓库；完整未筛选资料在同仓库Release，避免把数GB中间权重追加到普通Git历史。每个part均为可独立恢复的完整tar.gz，不能cat拼接。

```bash
git clone https://github.com/YuTaoV5/Rob2Nav.git /srv/robotics/Rob2Nav
cd /srv/robotics/Rob2Nav
python3 scripts/restore_release.py --destination /srv/robotics --list
python3 scripts/restore_release.py --destination /srv/robotics
python3 scripts/restore_release.py --destination /srv/robotics --verify-restored
```

`--destination`是Rob2Nav克隆的父目录。可加`--group project-outputs --group moe-runs`按需恢复，`--download-dir`指定缓存。每个文件的路径、大小、SHA-256和所属分包在`data-manifest.json`，压缩包摘要在`SHA256SUMS`。下载器校验附件/分包/内部文件摘要，拒绝路径穿越、软链接和不同内容的意外覆盖；已通过校验的下载可复用。

## 寻找对应实验

- 基座：`Rob2Nav/vendor/RoboGauge/resources/models/go2/`；NJU冻结技能：`Rob2Nav/vendor/NJUAgility/models/`。两者校验脚本在主仓库。
- 网页直接索引的精选证据：`Rob2Nav/artifacts/`；服务器原始证据：`Rob2Nav/outputs/`。
- MoE全部中间权重和训练记录：同级`MoE-runs/`与`Rob2Nav/outputs/moe_perceptive/`。
- AME原生训练/诊断：`AME-runs/`、`AME-criticaldamped/`和`Rob2Nav/outputs/critical_repro/`。
- 早期感知教师/学生/注意力：`outputs/perceptive/`、`outputs/ame/`及artifacts对应目录。
- 历史初始、越障、感知快照：`go2-mapless/`、`rob2nav-terrain/`、`Rob2Nav-perceptive-63f5bcf/`。旧绝对路径只作来源，不直接执行旧command.json。
- `artifacts/server_originals/`保留与Git最新版不同的服务器旧报告，原位置使用Git最新版，以免恢复冲突。
- Studio SQLite使用在线备份，保留任务历史；不恢复旧worker/PID，也不自动执行任务。

## 范围与实测边界

- [实际环境版本](environments/environment-observed.json)和Web/MuJoCo/Isaac的pip freeze均在主仓库与Release；新环境重建按部署文档，不直接重装跨平台freeze。
- manifest的excluded记录缓存、node_modules、嵌入Git、虚拟环境、凭据文件、SQLite临时journal和软链接。项目旧传输tar包保留，避免遗漏历史包内资料。
- NVIDIA完整安装二进制、驱动和远程机器人资产不打包，按上游安装与原条款获取。第三方代码/模型保留各自许可。
- 打包与恢复/路径/校验测试和Studio后台合计22项通过；发布逐包核对GitHub size与digest。新GPU训练与泛化不在归档校验范围。
- 详情见[第三方说明](../THIRD_PARTY_NOTICES.md)、[交接文档](../handover.md)及[V4后续诊断](../V4_VALIDATION.md)，区分基座/教师/学生/IK/NJU和原场景/前置。

## 分包下载

| 附件 | MiB | 文件数 | SHA-256 |
|---|---:|---:|---|
| [project-outputs.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/project-outputs.part001.tar.gz) | 436.8 | 2111 | `51387442eea5aae1d0f4b3ba232553438470e7bee5d93cfba5c29decfbbcfe01` |
| [project-outputs.part002.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/project-outputs.part002.tar.gz) | 442.9 | 1440 | `945e6bffaa689324624e6665d93053d30c7239bfcf445faf36cbf6f951fe2808` |
| [project-outputs.part003.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/project-outputs.part003.tar.gz) | 456.9 | 1413 | `37625cfac996117c7c6b5ea8b846a68040180d8bfb39c8182ff0cf6913dc3138` |
| [project-outputs.part004.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/project-outputs.part004.tar.gz) | 411.2 | 1075 | `7ef3f820128aa6ce02dc9ec060b5a7f8422677adf3c406960ca2237c287f25be` |
| [project-outputs.part005.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/project-outputs.part005.tar.gz) | 467.0 | 1391 | `e0f79011a001973925565f028e1dd51e771479b75ce7310b9d710f0324e0a3e1` |
| [project-outputs.part006.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/project-outputs.part006.tar.gz) | 202.7 | 551 | `3868d97df368bb5b36bcaf96bb0ea3a4b039ffbdce4364b60a77b554e9abac4a` |
| [moe-runs.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/moe-runs.part001.tar.gz) | 355.6 | 181 | `3e50ceb8d81c562f2bc16d6ae9e2c786ff60d47c44cab03ebe7969fed9c0c5e4` |
| [moe-runs.part002.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/moe-runs.part002.tar.gz) | 361.6 | 84 | `cdbd0f6dbb282cc04cbd55141279cc10edd0f09b5a281fb98e9143a1313471bf` |
| [moe-runs.part003.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/moe-runs.part003.tar.gz) | 359.0 | 110 | `4e3d7fea5ec9a7b831c37cf1b24fc3f890ecd412b12695e913111e2607c2cbad` |
| [moe-runs.part004.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/moe-runs.part004.tar.gz) | 382.1 | 140 | `a90ff384db85833227568cfbb7b2667acb75cf25a2d881c49050bfd36361ac7d` |
| [moe-runs.part005.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/moe-runs.part005.tar.gz) | 404.0 | 104 | `712678f74e33474258929319af47b1f7d3ff5255ecfa195352d7cbfba764702f` |
| [moe-runs.part006.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/moe-runs.part006.tar.gz) | 361.1 | 126 | `9fab3ddd5095db0b71e090d36d9bffb68e2f210783d53d2344bc0312fa12799f` |
| [moe-runs.part007.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/moe-runs.part007.tar.gz) | 155.1 | 72 | `a1f79fb127d0f0ee75e61ebcc3b2d1ac64fa0a30ac10c314381f328ee393b918` |
| [ame-runs.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/ame-runs.part001.tar.gz) | 78.8 | 227 | `6e7daaabd9bfab68a24d333731fa49aac7f0aa61382e25016f5a077deadc6871` |
| [ame-upstream.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/ame-upstream.part001.tar.gz) | 37.8 | 130 | `cc1843f8b5ff505d463cc3a667d3fa45cf511851dc1f4ca2122736ac9ecd873e` |
| [moe-upstream.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/moe-upstream.part001.tar.gz) | 24.4 | 116 | `dc7844e7030f689738847ed90d684baa88cf03756ca3d613efa88dcbc941310a` |
| [legacy-initial.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/legacy-initial.part001.tar.gz) | 29.3 | 262 | `00f983fa4dc72add7d2157bdec39974c8bf43272e1cde826230f85313999933f` |
| [legacy-terrain.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/legacy-terrain.part001.tar.gz) | 103.9 | 1127 | `607dc7c8c71c44a533dccf1dd159a1d069e18d0bb6a174bb6516e3acf087df89` |
| [legacy-perceptive.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/legacy-perceptive.part001.tar.gz) | 236.6 | 2996 | `909ee7b39928eeb7d60e2246f101bdf7086f897cc4a7076f0c828f36c6482182` |
| [project-artifacts.part001.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/project-artifacts.part001.tar.gz) | 346.3 | 4797 | `d03070278f28b23c20d08ab658bfce6aec79a1b2b5953d8346f4202f535112c6` |
| [project-artifacts.part002.tar.gz](https://github.com/YuTaoV5/Rob2Nav/releases/download/full-archive-2026-10-19/project-artifacts.part002.tar.gz) | 59.1 | 2526 | `38fa846e1399d7c362029ba9867de32e1d00c25ec104c984385b68c0e6ab0784` |
