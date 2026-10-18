完整项目资料归档：20,979个文件、7.24 GiB未压缩数据，21个独立分包。包含全部中间权重、日志、TensorBoard、评估JSON、视频、历史项目与上游快照，以及三个环境的观测依赖记录。

- [新环境部署](https://github.com/YuTaoV5/Rob2Nav/blob/main/docs/DEPLOYMENT.md)
- [数据目录与按组恢复](https://github.com/YuTaoV5/Rob2Nav/blob/main/docs/DATA_CATALOG.md)
- data-manifest.json：逐文件SHA-256与分包映射；SHA256SUMS：分包摘要。
- 使用scripts/restore_release.py恢复；destination是Rob2Nav克隆父目录。各part为独立tar.gz，不能拼接。

原AME20cm窄桩目标仍未完成，归档保留失败和停滞证据。恢复资料与打开网页不启动训练。环境二进制与NVIDIA资产按原条款另行安装。
