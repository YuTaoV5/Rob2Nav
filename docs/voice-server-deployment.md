# 新服务器部署与验收顺序

## 部署分工（用户已确认）

5090 与 Go2 位于用户自己的局域网，当前 agent 无法连接，由用户现场部署。不要再把提供5090 SSH作为等待条件，也不要尝试打通或暴露该局域网。agent 在已授权的现有服务器验证可复现的软件流程，用户在现场验证ROS/RViz、传感器、地图与机器人动作。

现场按以下顺序复现，每步保留日志：

1. 拉取本页指定的两个仓库/分支，安装语音环境并校验模型。
2. 用全新的demo标注文件启动语音服务，在另一个终端运行 benchmark_voice.py 与 smoke_voice.py；先复现真实ASR/TTS和合成任务回环。
3. 按导航仓库文档01–04准备ROS1 Docker/RViz，采集真实MID360数据，确认标定与地图配准；先观察规划速度。
4. 核对已有MoE环境、设备资格与唯一控制写入者，再按文档09接入live桥，现场验收停车和分段任务。

服务器已通过的demo回环不替代第3、4步；5090性能和真实麦克风效果由现场 benchmark 与录音测试填写。运行命令使用现场 NAV_ROOT/ROB2NAV_ROOT，不依赖测试服务器路径。

此流程适用正常 Ubuntu 22.04/24.04 服务器。由用户在现场确认GPU、机器人网段和本地登录方式；不能把旧 L40 服务器的检测当成新 5090。语音默认 CPU ONNX，不依赖 CUDA，5090 留给后续 GPU 语音对照或运动训练。磁盘预留模型、源码和 ROS 镜像空间，先跑 df/nvidia-smi/docker preflight。

## 两个根目录

```bash
git clone https://github.com/YuTaoV5/scan-sim2real.git
export NAV_ROOT="$PWD/scan-sim2real"
git clone --branch codex/voice-agent-capabilities https://github.com/YuTaoV5/Rob2Nav.git
export ROB2NAV_ROOT="$PWD/Rob2Nav"
cd "$ROB2NAV_ROOT"
# 安装 python3.11/venv 后：
bash scripts/install_voice.sh
.voice-venv/bin/python scripts/audit_voice_skills.py --root "$ROB2NAV_ROOT"
.voice-venv/bin/python -m voice_agent --mode demo --places data/voice/fresh-demo.json \
  --speech-config configs/speech.local.json
```

语音独立环境只装 numpy 和 Sherpa，不安装 ROS 或 Torch；runtime-command 中的 Python 必须换成原 Rob2Nav 实机环境。源码目录用当前集成分支；SDK 固定版本遵循原资格记录。先确认 audit 中权重摘要和 blocker，任何未通过技能不能在实机执行。

另开终端执行 benchmark_voice.py、smoke_voice.py；只使用 demo 模式。保存 data/voice 下 latency.json、roundtrip.json、events.jsonl，并记录 lscpu、Python 和模型 manifest。对用户真实麦克风再测至少20条地址/停止/技能短句，记录按键释放→ASR→调度→TTS生成→扬声器响起，区分客户端网络时间。目标例如 ASR p95<300ms、短回复合成<200ms只能作为验收目标，不能当作已测成绩。

Studio 使用自己的原环境：

```bash
cd "$ROB2NAV_ROOT/studio/frontend"
npm ci
npm run build
cd "$ROB2NAV_ROOT"
# 用已安装 studio/requirements.txt 的 Python（依仓库原 Studio 文档）
python -m studio.api --host 127.0.0.1 --port 8765
```

SSH 转发 `-L 8765:127.0.0.1:8765 -L 8096:127.0.0.1:8096` 后，浏览器打开 localhost:8765/#voice。页面通过 /voice-api 同源代理访问固定环回端口；VOICE_AGENT_PORT 可更改后端端口。live 代理额外需要服务端 VOICE_AGENT_ALLOW_LIVE_PROXY=1。服务独立进程，关闭浏览器不等于停止机器人；任务以状态机、bridge 和 runtime 心跳管理。

## ROS1 Docker 与 RViz

Docker 镜像、构建、source overlay 与 RViz 操作统一来自 NAV_ROOT 的 [01](https://github.com/YuTaoV5/scan-sim2real/blob/main/docs/01-new-server.md)，不要在两个仓库各复制一份未知配置。导航目录挂载 /work，另把 ROB2NAV_ROOT 挂到 /robot/Rob2Nav，并挂载已有运行环境/SDK所需路径。若 SDK 环境不在容器，需在同一个可达 ROS master 的主机运行 bridge；不能让示例绝对路径指向不存在的挂载。

容器使用 host network 只解决当前主机网络命名空间，不会让云服务器自动进入 Go2 以太网。推荐 LIO/SCAN/ROS桥和运动运行时放在机器人可直达的现场计算机，云 5090 只做语音（经现有可路由网络）；ROS1 master+所有节点的 ROS_IP/ROS_HOSTNAME 必须双向可达。ROS1 不能只转发11311端口就传输所有消息；未知现场网络前不能声称已连通。

远程 RViz 可用文档01的容器显示方案；先看合成/回放，只观察规划输出。现场传感器启动后验证 nav_odom、TF、时间与实测外参，再接入 live 模式。RViz 可视化和 GPU 语音不是同一个验收项。

## 分阶段证据

| 阶段 | 必须真实执行与保存 | 当前交付的上限 |
|---|---|---|
| 语音 | 官方模型加载、WAV识别、TTS输出、延迟与模型SHA | 本地可测；服务器需登录目标 |
| 任务回环 | TTS→ASR→地址/技能→状态机完成 | demo反馈不证明物理动作 |
| ROS | overlay编译、实时/回放话题、RViz截图 | 需正常Docker与现场数据 |
| SDK | 原设备标定、批准合同、watchdog、唯一writer、故障停车 | 未具资格不得电机使能 |
| 实机 | 平地、障碍、楼梯单段、顶部、下坡、完整路线 | 现场逐项验收并记录失败 |

首次 live 在平地、隔离区域按原资格流程：只读健康→零速度→低速单轴→停止→杀语音/桥/网络→独立watchdog→导航到实测近点→已资格技能。翻腾不能与楼梯导航并行；到达站立、落地与技能区校验后才能提交。未验证的倒立退出控制器继续保持阻止。

## 交给下一位 agent

先读当前 evidence/voice-validation.json 与 source锁；保留 unknown/null 安装值。更新测试时必须注明真实模型/合成反馈/物理仿真/实机的区别，提交命令、环境、日志和SHA。服务器与实机验收未完成时列缺项，不能仅修改报告的 passed 字段。登录密码、令牌、私有音频和地图留在运行目录，不提交Git。

## 2026-10-28 已执行的服务器测试

在用户已授权的现有服务器独立目录部署，硬件实际为 Xeon 8358 + L40，Python3.12.3。ASR/TTS使用CPU各2线程，未占GPU。5090位于用户局域网，由用户部署，当前没有5090性能实测。

- 官方5.592秒中文样例：预热后4次识别中位数约248ms。
- “正在前往会议室。”完整短WAV生成：预热后4次中位数约498ms；同文缓存约0.003ms，仅为服务内部生成/读取时间。
- 6条真实TTS→WAV→ASR→HTTP→状态机：启用、地址导航、后空翻、侧空翻、倒立、关闭，全部通过，最终DISABLED。运动侧是demo合成反馈，没有运行机器人运动权重或物理仿真。
- 初始短口令存在误识别并被拒绝；相关phrase-audit保留，不以最终6条通过外推真实麦克风准确率。
- Studio页面已在本机浏览器验证同源代理调度，前端构建通过。ROS、RViz、SDK、实机及楼梯均未执行。

结果和模型/部署源码哈希见 evidence/voice-*.json。现有语音服务监听服务器环回8096；从本机建立 `ssh -N -L 8096:127.0.0.1:8096 -p "$SSH_PORT" "$SSH_USER@$SSH_HOST"` 后打开 http://127.0.0.1:8096。使用你实际的SSH登录信息，不把密码写入脚本。此次环境未安装或修改ROS/Docker。
